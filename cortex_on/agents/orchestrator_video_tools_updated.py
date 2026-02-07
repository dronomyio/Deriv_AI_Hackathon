"""
Orchestrator Video Tools (Updated)

This module extends the CortexON orchestrator agent with video understanding tools.
These tools enable the orchestrator to delegate video processing tasks to specialized agents.

UPDATED: Script paths now point to /app/video_scripts/ instead of /app/
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

from pydantic_ai import RunContext
import logfire

from agents.video_ingest_agent import (
    run_video_ingestion,
    VideoIngestDeps,
    VideoIngestResult
)
from agents.video_query_agent import (
    run_video_query,
    VideoQueryDeps,
    VideoQueryResult
)
from utils.stream_response_format import StreamResponse


async def video_ingest_task(
    ctx: RunContext,
    video_url: str,
    output_dir: str = "/data/out"
) -> str:
    """
    Tool for orchestrator: Ingest a video and prepare it for querying
    
    This tool delegates to the Video Ingestion Agent to:
    - Download the video
    - Extract transcripts
    - Create structured JSON output
    
    Args:
        ctx: Runtime context from orchestrator
        video_url: URL of the video to process
        output_dir: Directory for output files
        
    Returns:
        JSON string with ingestion results
    """
    try:
        # Create stream output for this task
        task_stream = StreamResponse(
            agent="orchestrator_agent",
            status="delegating",
            steps=[f"Delegating video ingestion to Video Ingestion Agent: {video_url}"]
        )
        
        if ctx.deps.websocket:
            await ctx.deps.websocket.send_json(task_stream.model_dump())
        
        # Create dependencies for video ingestion agent
        ingest_deps = VideoIngestDeps(
            websocket=ctx.deps.websocket,
            download_dir="/data/downloads",
            output_dir=output_dir
        )
        
        # Run video ingestion
        result = await run_video_ingestion(video_url, ingest_deps)
        
        # Update orchestrator stream
        if result.status == "completed":
            task_stream.status = "completed"
            task_stream.steps.append(
                f"Video ingestion completed: {result.segment_count} segments, "
                f"{result.transcript_word_count} words"
            )
        else:
            task_stream.status = "failed"
            task_stream.steps.append(f"Video ingestion failed: {result.error}")
        
        if ctx.deps.websocket:
            await ctx.deps.websocket.send_json(task_stream.model_dump())
        
        # Return result as JSON string
        return json.dumps({
            "status": result.status,
            "video_id": result.video_id,
            "video_title": result.video_title,
            "snippets_json": result.snippets_json,
            "segment_count": result.segment_count,
            "duration": result.duration,
            "error": result.error
        }, indent=2)
        
    except Exception as e:
        logfire.error(f"video_ingest_task failed: {e}")
        return json.dumps({
            "status": "failed",
            "error": str(e)
        }, indent=2)


async def vector_index_task(
    ctx: RunContext,
    json_path: str,
    video_id: str,
    collection_name: str = "VideoChunks"
) -> str:
    """
    Tool for orchestrator: Index video transcript in vector database
    
    This tool delegates to the Vector Indexing Agent to:
    - Chunk the transcript semantically
    - Generate embeddings
    - Store in Weaviate collection
    
    Args:
        ctx: Runtime context from orchestrator
        json_path: Path to snippets JSON file
        video_id: Video identifier
        collection_name: Weaviate collection name
        
    Returns:
        JSON string with indexing results
    """
    try:
        # Create stream output
        task_stream = StreamResponse(
            agent="orchestrator_agent",
            status="delegating",
            steps=[f"Delegating vector indexing to Vector Indexing Agent: {video_id}"]
        )
        
        if ctx.deps.websocket:
            await ctx.deps.websocket.send_json(task_stream.model_dump())
        
        # Call the weaviate_ingest.py script from video_scripts directory
        import subprocess
        
        cmd = [
            "python3",
            "/app/video_scripts/weaviate_ingest.py",  # UPDATED PATH
            "--json", json_path,
            "--video-id", video_id,
            "--collection", collection_name,
            "--openai-model", "text-embedding-3-small",
            "--dimensions", "512",
            "--max-tokens", "350"
        ]
        
        logfire.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Vector indexing failed: {result.stderr}")
        
        # Parse output to get chunk count
        output_lines = result.stdout.strip().split('\n')
        chunk_count = 0
        for line in output_lines:
            if "chunks inserted" in line.lower():
                try:
                    chunk_count = int(line.split()[0])
                except:
                    pass
        
        task_stream.status = "completed"
        task_stream.steps.append(
            f"Vector indexing completed: {chunk_count} chunks indexed in {collection_name}"
        )
        
        if ctx.deps.websocket:
            await ctx.deps.websocket.send_json(task_stream.model_dump())
        
        return json.dumps({
            "status": "completed",
            "video_id": video_id,
            "collection_name": collection_name,
            "chunk_count": chunk_count
        }, indent=2)
        
    except Exception as e:
        logfire.error(f"vector_index_task failed: {e}")
        return json.dumps({
            "status": "failed",
            "error": str(e)
        }, indent=2)


async def keyframe_analysis_task(
    ctx: RunContext,
    video_path: str,
    output_path: str,
    fps: float = 1.0,
    k: int = 6
) -> str:
    """
    Tool for orchestrator: Extract and analyze keyframes from video
    
    This tool delegates to the Keyframe Analysis Agent to:
    - Extract representative frames
    - Generate visual descriptions using LLM
    - Create keyframes JSON
    
    Args:
        ctx: Runtime context from orchestrator
        video_path: Path to video file
        output_path: Path for keyframes JSON output
        fps: Frames per second to sample
        k: Number of keyframes to select
        
    Returns:
        JSON string with analysis results
    """
    try:
        # Create stream output
        task_stream = StreamResponse(
            agent="orchestrator_agent",
            status="delegating",
            steps=[f"Delegating keyframe analysis to Keyframe Analysis Agent: {video_path}"]
        )
        
        if ctx.deps.websocket:
            await ctx.deps.websocket.send_json(task_stream.model_dump())
        
        # Call the keyframes_describe.py script from video_scripts directory
        import subprocess
        
        cmd = [
            "python3",
            "/app/video_scripts/keyframes_describe.py",  # UPDATED PATH
            "--clip", video_path,
            "--out", output_path,
            "--fps", str(fps),
            "--k", str(k),
            "--max-hamming", "6",
            "--llm-model", "gpt-4.1-mini"
        ]
        
        logfire.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Keyframe analysis failed: {result.stderr}")
        
        # Load and validate output
        with open(output_path, 'r') as f:
            keyframes_data = json.load(f)
        
        keyframe_count = len(keyframes_data.get('keyframes', []))
        
        task_stream.status = "completed"
        task_stream.steps.append(
            f"Keyframe analysis completed: {keyframe_count} keyframes extracted"
        )
        
        if ctx.deps.websocket:
            await ctx.deps.websocket.send_json(task_stream.model_dump())
        
        return json.dumps({
            "status": "completed",
            "keyframes_json": output_path,
            "keyframe_count": keyframe_count
        }, indent=2)
        
    except Exception as e:
        logfire.error(f"keyframe_analysis_task failed: {e}")
        return json.dumps({
            "status": "failed",
            "error": str(e)
        }, indent=2)


async def video_query_task(
    ctx: RunContext,
    question: str,
    video_id: str,
    collection_name: str,
    index_json: str,
    output_clip: str
) -> str:
    """
    Tool for orchestrator: Query video content and generate answer clip
    
    This tool delegates to the Video Query Agent to:
    - Search vector database for relevant content
    - Merge temporal windows
    - Generate video clip
    - Synthesize natural language answer
    
    Args:
        ctx: Runtime context from orchestrator
        question: Natural language question
        video_id: Video identifier
        collection_name: Weaviate collection to search
        index_json: Path to snippets JSON
        output_clip: Path for output video clip
        
    Returns:
        JSON string with query results and answer
    """
    try:
        # Create stream output
        task_stream = StreamResponse(
            agent="orchestrator_agent",
            status="delegating",
            steps=[f"Delegating video query to Video Query Agent: '{question}'"]
        )
        
        if ctx.deps.websocket:
            await ctx.deps.websocket.send_json(task_stream.model_dump())
        
        # Create dependencies for video query agent
        query_deps = VideoQueryDeps(
            websocket=ctx.deps.websocket,
            weaviate_url=os.getenv("WEAVIATE_URL", "http://weaviate:8080"),
            output_dir=str(Path(output_clip).parent)
        )
        
        # Run video query
        result = await run_video_query(
            question=question,
            video_id=video_id,
            index_json=index_json,
            collection_name=collection_name,
            deps=query_deps
        )
        
        # Update orchestrator stream
        if result.status == "completed":
            task_stream.status = "completed"
            task_stream.steps.append(
                f"Video query completed: Answer generated with {len(result.evidence_chunks)} evidence chunks"
            )
        else:
            task_stream.status = "failed"
            task_stream.steps.append(f"Video query failed: {result.error}")
        
        if ctx.deps.websocket:
            await ctx.deps.websocket.send_json(task_stream.model_dump())
        
        # Return result as JSON string
        return json.dumps({
            "status": result.status,
            "question": result.question,
            "answer": result.answer,
            "video_id": result.video_id,
            "clip_path": result.clip_path,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "confidence": result.confidence,
            "evidence_count": len(result.evidence_chunks),
            "error": result.error
        }, indent=2)
        
    except Exception as e:
        logfire.error(f"video_query_task failed: {e}")
        return json.dumps({
            "status": "failed",
            "error": str(e)
        }, indent=2)


# Tool registration for orchestrator agent
VIDEO_TOOLS = {
    "video_ingest_task": video_ingest_task,
    "vector_index_task": vector_index_task,
    "keyframe_analysis_task": keyframe_analysis_task,
    "video_query_task": video_query_task
}


# Enhanced system prompt addition for orchestrator
VIDEO_ORCHESTRATOR_PROMPT_EXTENSION = """

[VIDEO UNDERSTANDING CAPABILITIES]

You now have access to specialized video understanding agents through these tools:

1. video_ingest_task(video_url: str, output_dir: str) -> str:
   - Downloads and processes videos from YouTube
   - Extracts transcripts with timestamps
   - Creates structured JSON output
   - Returns video metadata and snippets path

2. vector_index_task(json_path: str, video_id: str, collection_name: str) -> str:
   - Indexes video transcripts in vector database
   - Creates semantic chunks for efficient search
   - Stores in Weaviate collection
   - Returns indexing statistics

3. keyframe_analysis_task(video_path: str, output_path: str, fps: float, k: int) -> str:
   - Extracts representative frames from video
   - Generates visual descriptions using LLM
   - Creates keyframes JSON with timestamps
   - Returns keyframe count and paths

4. video_query_task(question: str, video_id: str, collection_name: str, index_json: str, output_clip: str) -> str:
   - Answers questions about video content
   - Searches vector database for relevant segments
   - Generates video clips containing answers
   - Returns natural language answer with evidence

[VIDEO UNDERSTANDING WORKFLOWS]

For video Q&A tasks:
1. Use video_ingest_task to download and transcribe the video
2. Use vector_index_task to index the transcript
3. Optionally use keyframe_analysis_task for visual content
4. Use video_query_task to answer specific questions

For video analysis tasks:
1. Use video_ingest_task to process the video
2. Use both vector_index_task and keyframe_analysis_task
3. Use video_query_task multiple times for different aspects

Always update the planner after each video task completion.
"""
