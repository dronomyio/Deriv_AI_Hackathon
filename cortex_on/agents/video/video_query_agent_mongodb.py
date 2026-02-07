"""
Video Query Agent for CortexON (with MongoDB Storage)

This agent handles semantic search across video transcripts and keyframes,
retrieves relevant segments from MongoDB, and generates answers with evidence.

FEATURES:
- MongoDB storage integration for queries and results
- Weaviate semantic search
- Query history tracking
- Human-in-the-loop for result validation
"""

import os
import json
import subprocess
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import uuid

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from fastapi import WebSocket
import logfire

from utils.stream_response_format import StreamResponse
from storage.mongodb_storage import (
    MongoDBStorage,
    Query as QueryRecord
)


@dataclass
class VideoQueryDeps:
    """Dependencies for Video Query Agent"""
    websocket: Optional[WebSocket] = None
    stream_output: Optional[StreamResponse] = None
    storage: Optional[MongoDBStorage] = None
    weaviate_url: str = "http://weaviate:8080"
    output_dir: str = "/data/out"
    ask_human: Optional[callable] = None


class VideoQueryResult(BaseModel):
    """Result from video query"""
    query_id: str
    question: str
    answer: str
    evidence: List[Dict[str, Any]]
    confidence: float
    video_clips: List[str]
    status: str
    error: Optional[str] = None


VIDEO_QUERY_SYSTEM_PROMPT = """You are a video query specialist agent responsible for answering questions 
about video content using semantic search and MongoDB storage.

Your capabilities:
1. Search video transcripts using Weaviate vector database
2. Retrieve relevant segments from MongoDB
3. Generate accurate answers with evidence
4. Create video clips at relevant timestamps
5. Track query history in MongoDB
6. Consult humans when results are ambiguous

You work as part of a multi-agent system coordinated by an orchestrator agent.
Always provide clear, evidence-based answers with timestamps and confidence scores.
"""


video_query_agent = Agent(
    model=AnthropicModel(os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-7-sonnet-20250219")),
    system_prompt=VIDEO_QUERY_SYSTEM_PROMPT,
    deps_type=VideoQueryDeps,
    result_type=VideoQueryResult
)


@video_query_agent.tool
async def search_video_content(
    ctx: RunContext[VideoQueryDeps],
    video_id: str,
    question: str,
    collection_name: str = "video_transcripts"
) -> Dict[str, Any]:
    """
    Search video content using Weaviate and retrieve segments from MongoDB
    """
    try:
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(f"Searching for: {question}")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        # Check if video exists in MongoDB
        if ctx.deps.storage:
            video = ctx.deps.storage.get_video(video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found in database")
            
            if video['processing']['status'] != 'completed':
                raise ValueError(f"Video {video_id} is not fully processed yet")
        
        # Use the existing query_weaviate.py script
        cmd = [
            "python3",
            "/app/video_scripts/query_weaviate.py",
            "--collection", collection_name,
            "--query", question,
            "--video-id", video_id,
            "--top-k", "5"
        ]
        
        logfire.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # HITL: No results found
            if ctx.deps.ask_human:
                response = await ctx.deps.ask_human(
                    f"No results found for '{question}' in video {video_id}. "
                    f"Try rephrasing? (yes/no)"
                )
                if response.lower() in ['yes', 'y']:
                    return {"results": [], "needs_rephrase": True}
            
            raise RuntimeError(f"Search failed: {result.stderr}")
        
        # Parse search results
        search_results = json.loads(result.stdout)
        
        # Enrich with MongoDB data
        enriched_results = []
        for res in search_results.get('results', []):
            # Get full transcript segment from MongoDB
            if ctx.deps.storage:
                segments = ctx.deps.storage.get_transcript_segments(
                    video_id=video_id,
                    start_time=res['start_time'],
                    end_time=res['end_time']
                )
                
                if segments:
                    enriched_results.append({
                        "text": res['text'],
                        "start_time": res['start_time'],
                        "end_time": res['end_time'],
                        "score": res['score'],
                        "segments": segments,
                        "weaviate_id": res.get('weaviate_id')
                    })
        
        # HITL: Relevance check
        if enriched_results and ctx.deps.ask_human:
            preview = "\n".join([
                f"[{r['start_time']:.1f}s-{r['end_time']:.1f}s]: {r['text'][:100]}..."
                for r in enriched_results[:3]
            ])
            response = await ctx.deps.ask_human(
                f"Found {len(enriched_results)} results. Top matches:\n{preview}\n\n"
                f"Are these relevant? (yes/no)"
            )
            if response.lower() not in ['yes', 'y']:
                return {"results": enriched_results, "user_rejected": True}
        
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(
                f"Found {len(enriched_results)} relevant segments"
            )
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        return {
            "results": enriched_results,
            "result_count": len(enriched_results)
        }
        
    except Exception as e:
        logfire.error(f"Error searching video content: {e}")
        raise


@video_query_agent.tool
async def generate_video_clip(
    ctx: RunContext[VideoQueryDeps],
    video_id: str,
    start_time: float,
    end_time: float,
    output_name: Optional[str] = None
) -> str:
    """
    Generate a video clip for a specific time range
    """
    try:
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(
                f"Generating clip: {start_time:.1f}s - {end_time:.1f}s"
            )
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        # Get video file path from MongoDB
        if ctx.deps.storage:
            video = ctx.deps.storage.get_video(video_id)
            if not video:
                raise ValueError(f"Video {video_id} not found")
            
            video_path = video['files']['video_path']
        else:
            video_path = f"/data/downloads/{video_id}.mp4"
        
        # Generate output filename
        if not output_name:
            output_name = f"{video_id}_clip_{int(start_time)}_{int(end_time)}.mp4"
        
        output_path = Path(ctx.deps.output_dir) / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use the existing get_clip.py script
        cmd = [
            "python3",
            "/app/video_scripts/get_clip.py",
            video_path,
            str(start_time),
            str(end_time),
            str(output_path)
        ]
        
        logfire.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Clip generation failed: {result.stderr}")
        
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(f"Clip generated: {output_name}")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        return str(output_path)
        
    except Exception as e:
        logfire.error(f"Error generating clip: {e}")
        raise


@video_query_agent.tool
async def synthesize_answer(
    ctx: RunContext[VideoQueryDeps],
    question: str,
    search_results: List[Dict[str, Any]],
    video_id: str
) -> Dict[str, Any]:
    """
    Synthesize an answer from search results using LLM
    """
    try:
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append("Synthesizing answer...")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        # Prepare context from search results
        context = "\n\n".join([
            f"[{r['start_time']:.1f}s - {r['end_time']:.1f}s]\n{r['text']}"
            for r in search_results[:5]
        ])
        
        # Use LLM to generate answer
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        prompt = f"""Based on the following video transcript excerpts, answer the question.

Question: {question}

Transcript excerpts:
{context}

Provide a clear, concise answer with specific timestamps as evidence.
Also rate your confidence (0.0-1.0) in the answer."""

        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer_text = response.content[0].text
        
        # Extract confidence (simple heuristic)
        confidence = 0.8 if len(search_results) >= 3 else 0.6
        
        return {
            "answer": answer_text,
            "confidence": confidence,
            "evidence_count": len(search_results)
        }
        
    except Exception as e:
        logfire.error(f"Error synthesizing answer: {e}")
        raise


async def run_video_query(
    video_id: str,
    question: str,
    user_id: str,
    session_id: str,
    generate_clips: bool,
    deps: VideoQueryDeps
) -> VideoQueryResult:
    """
    Main entry point for video query workflow with MongoDB storage
    """
    try:
        # Initialize stream output
        if deps.websocket:
            deps.stream_output = StreamResponse(
                agent="video_query_agent",
                status="in_progress",
                steps=["Starting video query with MongoDB storage..."]
            )
            await deps.websocket.send_json(deps.stream_output.model_dump())
        
        # Generate query ID
        query_id = str(uuid.uuid4())
        
        # Create query record in MongoDB
        if deps.storage:
            query_record = QueryRecord(
                query_id=query_id,
                user_id=user_id,
                session_id=session_id,
                question=question,
                question_normalized=question.lower().strip(),
                video_ids=[video_id],
                collection_name="video_transcripts",
                search_results={},
                answer={},
                metrics={},
                feedback=None,
                created_at=datetime.now(timezone.utc),
                completed_at=None
            )
            deps.storage.insert_query(query_record)
        
        # Run the agent workflow
        prompt = f"""Answer the following question about video {video_id}:

Question: {question}

Steps:
1. Search the video transcript using semantic search
2. Retrieve relevant segments from MongoDB
3. Synthesize a clear answer with evidence
{"4. Generate video clips at relevant timestamps" if generate_clips else ""}

Provide timestamps and confidence scores."""

        result = await video_query_agent.run(
            user_prompt=prompt,
            deps=deps
        )
        
        # Update query record with results
        if deps.storage:
            deps.storage.update_query_result(
                query_id=query_id,
                answer={
                    "text": result.data.answer,
                    "confidence": result.data.confidence,
                    "evidence": result.data.evidence
                },
                metrics={
                    "evidence_count": len(result.data.evidence),
                    "clips_generated": len(result.data.video_clips),
                    "processing_time_seconds": (
                        datetime.now(timezone.utc) - query_record.created_at
                    ).total_seconds()
                }
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
            query_id=query_id,
            question=question,
            answer="",
            evidence=[],
            confidence=0.0,
            video_clips=[],
            status="failed",
            error=str(e)
        )
