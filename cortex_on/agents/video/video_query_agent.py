"""
Video Query Agent for CortexON

This agent handles natural language queries over video content,
performing semantic search and generating video clips with answers.
"""

import os
import json
import subprocess
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from fastapi import WebSocket
import logfire
import weaviate
from weaviate.classes.query import MetadataQuery, Filter

from utils.stream_response_format import StreamResponse


@dataclass
class VideoQueryDeps:
    """Dependencies for Video Query Agent"""
    websocket: Optional[WebSocket] = None
    stream_output: Optional[StreamResponse] = None
    weaviate_url: str = "http://weaviate:8080"
    output_dir: str = "/data/out"


class VideoQueryResult(BaseModel):
    """Result from video query"""
    question: str
    answer: str
    video_id: str
    clip_path: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: float
    evidence_chunks: List[str]
    status: str
    error: Optional[str] = None


# System prompt for the video query agent
VIDEO_QUERY_SYSTEM_PROMPT = """You are a video query specialist agent responsible for answering questions
about video content using semantic search and evidence-based reasoning.

Your capabilities:
1. Understand natural language questions about video content
2. Search vector databases for relevant transcript and visual information
3. Merge and rank search results by relevance
4. Generate precise video clips containing the answer
5. Synthesize natural language answers grounded in evidence

You work as part of a multi-agent system coordinated by an orchestrator agent.
Always provide clear progress updates and cite your sources.

When answering questions:
- Search both transcript and keyframe collections
- Merge temporally adjacent results for better context
- Generate clips that fully contain the answer
- Provide timestamps and confidence scores
- If the answer is not in the video, say so explicitly
- Never hallucinate information not present in the evidence
"""


# Initialize the agent
video_query_agent = Agent(
    model=AnthropicModel(os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-7-sonnet-20250219")),
    system_prompt=VIDEO_QUERY_SYSTEM_PROMPT,
    deps_type=VideoQueryDeps,
    result_type=VideoQueryResult
)


@video_query_agent.tool
async def search_transcript_collection(
    ctx: RunContext[VideoQueryDeps],
    question: str,
    video_id: str,
    collection_name: str = "VideoChunks",
    top_k: int = 8
) -> List[Dict[str, Any]]:
    """
    Search the transcript vector collection for relevant chunks
    
    Args:
        ctx: Runtime context with dependencies
        question: Natural language question
        video_id: Video identifier to filter results
        collection_name: Weaviate collection name
        top_k: Number of results to return
        
    Returns:
        List of relevant transcript chunks with metadata
    """
    try:
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(
                f"Searching transcript collection for: '{question}'"
            )
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        # Connect to Weaviate
        client = weaviate.connect_to_local(host=ctx.deps.weaviate_url.replace("http://", ""))
        
        try:
            collection = client.collections.get(collection_name)
            
            # Embed query locally — no OpenAI needed
            import sys as _sys
            _sys.path.insert(0, "/app/video_scripts")
            from embedding_service import get_embedding_service
            svc = get_embedding_service()
            query_vec = svc.embed_text(question)

            # Perform semantic search with local vector
            results = collection.query.near_vector(
                near_vector=query_vec,
                limit=top_k,
                filters=Filter.by_property("video_id").equal(video_id),
                return_metadata=MetadataQuery(distance=True),
                return_properties=[
                    "text",
                    "start_seconds",
                    "end_seconds",
                    "video_id",
                    "snippet_index",
                    "chunk_index"
                ]
            )
            
            # Convert results to dictionaries
            chunks = []
            for obj in results.objects:
                props = obj.properties or {}
                metadata = obj.metadata or {}
                
                chunks.append({
                    "text": props.get("text", ""),
                    "start_seconds": props.get("start_seconds", 0.0),
                    "end_seconds": props.get("end_seconds", 0.0),
                    "distance": metadata.get("distance", 1.0),
                    "video_id": props.get("video_id", ""),
                    "snippet_index": props.get("snippet_index"),
                    "chunk_index": props.get("chunk_index")
                })
            
            if ctx.deps.stream_output and ctx.deps.websocket:
                ctx.deps.stream_output.steps.append(
                    f"Found {len(chunks)} relevant transcript chunks"
                )
                await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
            
            logfire.info(f"Retrieved {len(chunks)} chunks from {collection_name}")
            return chunks
            
        finally:
            client.close()
            
    except Exception as e:
        logfire.error(f"Error searching transcript collection: {e}")
        raise


@video_query_agent.tool
async def merge_temporal_windows(
    ctx: RunContext[VideoQueryDeps],
    chunks: List[Dict[str, Any]],
    merge_gap: float = 8.0,
    max_window: float = 140.0
) -> Dict[str, Any]:
    """
    Merge temporally adjacent chunks into optimal time windows
    
    Args:
        ctx: Runtime context with dependencies
        chunks: List of retrieved chunks with timestamps
        merge_gap: Maximum gap in seconds to merge chunks
        max_window: Maximum window duration in seconds
        
    Returns:
        Best time window with merged chunks
    """
    try:
        if not chunks:
            return {
                "start_time": 0.0,
                "end_time": 0.0,
                "chunks": [],
                "score": 0.0
            }
        
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append("Merging temporal windows...")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        # Sort chunks by start time
        sorted_chunks = sorted(chunks, key=lambda x: x["start_seconds"])
        
        # Merge adjacent chunks
        windows = []
        current_window = {
            "start_time": sorted_chunks[0]["start_seconds"],
            "end_time": sorted_chunks[0]["end_seconds"],
            "chunks": [sorted_chunks[0]]
        }
        
        for chunk in sorted_chunks[1:]:
            gap = chunk["start_seconds"] - current_window["end_time"]
            proposed_end = max(current_window["end_time"], chunk["end_seconds"])
            window_duration = proposed_end - current_window["start_time"]
            
            if gap <= merge_gap and window_duration <= max_window:
                # Merge into current window
                current_window["end_time"] = proposed_end
                current_window["chunks"].append(chunk)
            else:
                # Start new window
                windows.append(current_window)
                current_window = {
                    "start_time": chunk["start_seconds"],
                    "end_time": chunk["end_seconds"],
                    "chunks": [chunk]
                }
        
        windows.append(current_window)
        
        # Score windows (prefer more chunks and lower distances)
        for window in windows:
            score = 0.0
            for chunk in window["chunks"]:
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = max(0.0, 1.0 - chunk.get("distance", 1.0))
                score += similarity
            # Bonus for more chunks
            score += 0.15 * len(window["chunks"])
            window["score"] = score
        
        # Return best window
        best_window = max(windows, key=lambda w: w["score"])
        
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(
                f"Best window: {best_window['start_time']:.1f}s - {best_window['end_time']:.1f}s "
                f"({len(best_window['chunks'])} chunks, score: {best_window['score']:.2f})"
            )
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        return best_window
        
    except Exception as e:
        logfire.error(f"Error merging temporal windows: {e}")
        raise


@video_query_agent.tool
async def generate_video_clip(
    ctx: RunContext[VideoQueryDeps],
    index_json: str,
    start_time: float,
    end_time: float,
    output_path: str,
    pad_seconds: float = 2.0
) -> str:
    """
    Generate a video clip for the specified time window
    
    Args:
        ctx: Runtime context with dependencies
        index_json: Path to snippets JSON file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path for output clip
        pad_seconds: Padding to add before/after
        
    Returns:
        Path to generated clip
    """
    try:
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(
                f"Generating video clip: {start_time:.1f}s - {end_time:.1f}s"
            )
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        # Add padding
        t1 = max(0.0, start_time - pad_seconds)
        t2 = end_time + pad_seconds
        
        # Call get_clip.py
        cmd = [
            "python3",
            "/app/video_scripts/get_clip.py",
            "--index-json", index_json,
            "--t1", str(t1),
            "--t2", str(t2),
            "--out", output_path
        ]
        
        logfire.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Clip generation failed: {result.stderr}")
        
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(f"Clip generated: {output_path}")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        return output_path
        
    except Exception as e:
        logfire.error(f"Error generating video clip: {e}")
        raise


@video_query_agent.tool
async def synthesize_answer(
    ctx: RunContext[VideoQueryDeps],
    question: str,
    evidence_chunks: List[Dict[str, Any]]
) -> str:
    """
    Synthesize a natural language answer from evidence chunks
    
    Args:
        ctx: Runtime context with dependencies
        question: Original question
        evidence_chunks: Retrieved and merged evidence chunks
        
    Returns:
        Natural language answer
    """
    try:
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append("Synthesizing answer from evidence...")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        if not evidence_chunks:
            return "The answer is not there in the video."
        
        # Format evidence with timestamps
        evidence_text = "\n\n".join([
            f"[{chunk['start_seconds']:.1f}s - {chunk['end_seconds']:.1f}s]: {chunk['text']}"
            for chunk in evidence_chunks
        ])
        
        # Use the agent's LLM to synthesize answer
        synthesis_prompt = f"""Based on the following evidence from the video transcript, 
answer this question: {question}

Evidence:
{evidence_text}

Provide a clear, concise answer that directly addresses the question.
Include relevant timestamps in your answer.
If the evidence does not contain enough information to answer the question,
say "The answer is not there in the video." """

        # This would typically call the LLM directly
        # For now, return a structured response
        answer = f"Based on the video content at timestamps {evidence_chunks[0]['start_seconds']:.1f}s - {evidence_chunks[-1]['end_seconds']:.1f}s, "
        answer += "the video discusses the requested topic. "
        answer += f"Specifically: {evidence_chunks[0]['text'][:200]}..."
        
        return answer
        
    except Exception as e:
        logfire.error(f"Error synthesizing answer: {e}")
        raise


async def run_video_query(
    question: str,
    video_id: str,
    index_json: str,
    collection_name: str,
    deps: VideoQueryDeps
) -> VideoQueryResult:
    """
    Main entry point for video query workflow
    
    Args:
        question: Natural language question
        video_id: Video identifier
        index_json: Path to video snippets JSON
        collection_name: Weaviate collection name
        deps: Dependencies including websocket
        
    Returns:
        VideoQueryResult with answer and clip
    """
    try:
        # Initialize stream output
        if deps.websocket:
            deps.stream_output = StreamResponse(
                agent="video_query_agent",
                status="in_progress",
                steps=[f"Processing question: {question}"]
            )
            await deps.websocket.send_json(deps.stream_output.model_dump())
        
        # Run the agent workflow
        prompt = f"""Answer the following question about video {video_id}:

Question: {question}

Steps:
1. Search the transcript collection for relevant chunks
2. Merge temporally adjacent results
3. Generate a video clip containing the answer
4. Synthesize a natural language answer with timestamps

Collection: {collection_name}
Index JSON: {index_json}"""

        result = await video_query_agent.run(
            user_prompt=prompt,
            deps=deps
        )
        
        # Update final status
        if deps.stream_output and deps.websocket:
            deps.stream_output.status = "completed"
            deps.stream_output.steps.append("Query completed successfully")
            await deps.websocket.send_json(deps.stream_output.model_dump())
        
        return result.data
        
    except Exception as e:
        logfire.error(f"Video query failed: {e}")
        
        if deps.stream_output and deps.websocket:
            deps.stream_output.status = "failed"
            deps.stream_output.steps.append(f"Error: {str(e)}")
            await deps.websocket.send_json(deps.stream_output.model_dump())
        
        return VideoQueryResult(
            question=question,
            answer="",
            video_id=video_id,
            confidence=0.0,
            evidence_chunks=[],
            status="failed",
            error=str(e)
        )
