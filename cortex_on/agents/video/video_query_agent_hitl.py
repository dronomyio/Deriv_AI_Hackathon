"""
Video Query Agent for CortexON (with Human-in-the-Loop)

This agent handles natural language queries over video content,
performing semantic search and generating video clips with answers.

UPDATED: Includes human-in-the-loop decision points for query disambiguation
and answer validation
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
    ask_human: Optional[callable] = None  # Human-in-the-loop callback


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
6. Consult humans when queries are ambiguous or results are uncertain

You work as part of a multi-agent system coordinated by an orchestrator agent.
Always provide clear progress updates and cite your sources.

When answering questions:
- Search both transcript and keyframe collections
- Merge temporally adjacent results for better context
- Generate clips that fully contain the answer
- Provide timestamps and confidence scores
- If the answer is not in the video, say so explicitly
- Never hallucinate information not present in the evidence
- Ask humans to clarify ambiguous questions
- Confirm with humans before generating expensive video clips
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
            
            # HUMAN-IN-THE-LOOP: Show results and ask if they look relevant
            if len(chunks) == 0:
                if ctx.deps.ask_human:
                    response = await ctx.deps.ask_human(
                        f"No results found for '{question}' in video {video_id}. "
                        f"Would you like to: (1) Rephrase the question, (2) Search all videos, or (3) Give up? "
                        f"Please respond with 1, 2, or 3."
                    )
                    if response.strip() == "1":
                        new_question = await ctx.deps.ask_human(
                            "Please provide a rephrased question:"
                        )
                        # Recursive call with new question
                        return await search_transcript_collection(
                            ctx, new_question, video_id, collection_name, top_k
                        )
                    elif response.strip() == "2":
                        # Search without video_id filter
                        logfire.info("Searching across all videos")
                        # Could implement cross-video search here
            elif len(chunks) > 0 and ctx.deps.ask_human:
                # Show preview of top result
                top_result = chunks[0]
                preview = top_result['text'][:150]
                response = await ctx.deps.ask_human(
                    f"Found {len(chunks)} results. Top result preview: '{preview}...' "
                    f"Do these results look relevant to your question? (yes/no)"
                )
                if response.lower() in ['no', 'n']:
                    new_question = await ctx.deps.ask_human(
                        "Would you like to rephrase your question? If yes, please provide the new question. If no, type 'skip'."
                    )
                    if new_question.lower() != 'skip':
                        return await search_transcript_collection(
                            ctx, new_question, video_id, collection_name, top_k
                        )
            
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
        
        # HUMAN-IN-THE-LOOP: Confirm window selection if multiple good options
        if len(windows) > 1 and ctx.deps.ask_human:
            top_windows = sorted(windows, key=lambda w: w["score"], reverse=True)[:3]
            window_descriptions = []
            for i, w in enumerate(top_windows):
                duration = w["end_time"] - w["start_time"]
                window_descriptions.append(
                    f"Option {i+1}: {w['start_time']:.1f}s - {w['end_time']:.1f}s "
                    f"(duration: {duration:.1f}s, {len(w['chunks'])} chunks, score: {w['score']:.2f})"
                )
            
            response = await ctx.deps.ask_human(
                f"Found {len(windows)} potential answer windows:\n" + 
                "\n".join(window_descriptions) + 
                f"\n\nI recommend Option 1. Do you want to use it, or choose a different option? (1/2/3)"
            )
            
            choice = int(response.strip()) if response.strip().isdigit() else 1
            if 1 <= choice <= len(top_windows):
                best_window = top_windows[choice - 1]
        
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
        # HUMAN-IN-THE-LOOP: Confirm before generating expensive clip
        if ctx.deps.ask_human:
            duration = end_time - start_time
            response = await ctx.deps.ask_human(
                f"About to generate a {duration:.1f}s video clip from {start_time:.1f}s to {end_time:.1f}s. "
                f"This may take some time. Proceed? (yes/no)"
            )
            if response.lower() not in ['yes', 'y']:
                logfire.info("User cancelled clip generation")
                return ""
        
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
1. Search the transcript collection for relevant chunks (ask human if no results)
2. Merge temporally adjacent results (ask human to confirm best window)
3. Generate a video clip containing the answer (ask human before processing)
4. Synthesize a natural language answer with timestamps

Collection: {collection_name}
Index JSON: {index_json}

Use human-in-the-loop when results are unclear or require confirmation."""

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
