"""
Video Ingestion Agent for CortexON (with Human-in-the-Loop)

This agent handles video download, transcript extraction, and segmentation.
It integrates with the existing video understanding pipeline and CortexON's
multi-agent orchestration framework.

UPDATED: Includes human-in-the-loop decision points for quality control
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

from utils.stream_response_format import StreamResponse


@dataclass
class VideoIngestDeps:
    """Dependencies for Video Ingestion Agent"""
    websocket: Optional[WebSocket] = None
    stream_output: Optional[StreamResponse] = None
    download_dir: str = "/data/downloads"
    output_dir: str = "/data/out"
    ask_human: Optional[callable] = None  # Human-in-the-loop callback


class VideoIngestResult(BaseModel):
    """Result from video ingestion process"""
    video_id: str
    video_title: str
    duration: float
    snippets_json: str
    segment_count: int
    transcript_word_count: int
    status: str
    error: Optional[str] = None


# System prompt for the video ingestion agent
VIDEO_INGEST_SYSTEM_PROMPT = """You are a video ingestion specialist agent responsible for downloading videos, 
extracting transcripts, and preparing them for semantic analysis.

Your capabilities:
1. Download videos from YouTube and other sources
2. Extract audio and generate transcripts
3. Segment videos into meaningful chunks
4. Create structured JSON output with timestamps
5. Consult humans when quality or decisions are uncertain

You work as part of a multi-agent system coordinated by an orchestrator agent.
Always provide clear progress updates and handle errors gracefully.

When processing videos:
- Validate URLs before downloading
- Check for existing downloads to avoid duplication
- Generate high-quality transcripts with accurate timestamps
- Create semantic segments based on natural boundaries
- Produce well-structured JSON output for downstream agents
- Ask humans for guidance when transcript quality is questionable
- Confirm with humans before re-processing expensive operations
"""


# Initialize the agent
video_ingest_agent = Agent(
    model=AnthropicModel(os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-7-sonnet-20250219")),
    system_prompt=VIDEO_INGEST_SYSTEM_PROMPT,
    deps_type=VideoIngestDeps,
    result_type=VideoIngestResult
)


@video_ingest_agent.tool
async def download_youtube_video(
    ctx: RunContext[VideoIngestDeps], 
    video_url: str
) -> str:
    """
    Download a YouTube video using yt-dlp
    
    Args:
        ctx: Runtime context with dependencies
        video_url: YouTube video URL
        
    Returns:
        Path to downloaded video file
    """
    try:
        # Update progress
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(f"Downloading video from {video_url}...")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        # Create download directory
        download_dir = Path(ctx.deps.download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract video ID from URL
        video_id = video_url.split("v=")[-1].split("&")[0]
        output_template = str(download_dir / f"{video_id}.%(ext)s")
        
        # Check if video already exists
        existing_files = list(download_dir.glob(f"{video_id}.*"))
        existing_video = next((f for f in existing_files if f.suffix in ['.mp4', '.mkv', '.webm']), None)
        
        if existing_video:
            # HUMAN-IN-THE-LOOP: Ask if we should re-download
            if ctx.deps.ask_human:
                response = await ctx.deps.ask_human(
                    f"Video {video_id} already exists at {existing_video}. "
                    f"Do you want to re-download it? (yes/no)"
                )
                if response.lower() not in ['yes', 'y']:
                    logfire.info(f"Using existing video: {existing_video}")
                    return str(existing_video)
        
        # Download using yt-dlp
        cmd = [
            "yt-dlp",
            "-f", "best",
            "--write-info-json",
            "--write-auto-sub",
            "--sub-lang", "en",
            "-o", output_template,
            video_url
        ]
        
        logfire.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")
        
        # Find downloaded video file
        video_files = list(download_dir.glob(f"{video_id}.*"))
        video_file = next((f for f in video_files if f.suffix in ['.mp4', '.mkv', '.webm']), None)
        
        if not video_file:
            raise FileNotFoundError(f"Downloaded video not found in {download_dir}")
        
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(f"Video downloaded: {video_file.name}")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        return str(video_file)
        
    except Exception as e:
        logfire.error(f"Error downloading video: {e}")
        raise


@video_ingest_agent.tool
async def extract_transcript(
    ctx: RunContext[VideoIngestDeps],
    video_path: str
) -> Dict[str, Any]:
    """
    Extract transcript from video using speech-to-text
    
    Args:
        ctx: Runtime context with dependencies
        video_path: Path to video file
        
    Returns:
        Dictionary with transcript segments and metadata
    """
    try:
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append("Extracting audio and generating transcript...")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        # Use the existing yt_slice_chatgpt.py script
        video_file = Path(video_path)
        video_id = video_file.stem
        output_json = Path(ctx.deps.output_dir) / f"{video_id}_snippets_with_transcripts.json"
        
        # Call the existing video processing script
        cmd = [
            "python3",
            "/app/video_scripts/yt_slice_chatgpt.py",
            video_id,
            ctx.deps.download_dir,
            str(output_json)
        ]
        
        logfire.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Transcript extraction failed: {result.stderr}")
        
        # Load and validate the output
        with open(output_json, 'r') as f:
            transcript_data = json.load(f)
        
        segment_count = len(transcript_data.get('segments', []))
        
        # HUMAN-IN-THE-LOOP: Quality check for transcript
        if segment_count == 0:
            if ctx.deps.ask_human:
                response = await ctx.deps.ask_human(
                    f"Warning: No transcript segments were extracted from video {video_id}. "
                    f"This might indicate an issue with the audio or transcription. "
                    f"Do you want to: (1) Continue anyway, (2) Retry with different settings, or (3) Skip this video? "
                    f"Please respond with 1, 2, or 3."
                )
                if response.strip() == "3":
                    raise ValueError("User chose to skip video due to transcript issues")
                elif response.strip() == "2":
                    # Could implement retry logic here
                    logfire.info("User requested retry - not implemented yet")
        
        # HUMAN-IN-THE-LOOP: Show sample transcript for approval
        if segment_count > 0 and ctx.deps.ask_human:
            sample_text = transcript_data['segments'][0].get('text', '')[:200]
            response = await ctx.deps.ask_human(
                f"Transcript extracted with {segment_count} segments. "
                f"Sample text: '{sample_text}...' "
                f"Does this look correct? (yes/no)"
            )
            if response.lower() in ['no', 'n']:
                logfire.warning("User indicated transcript quality is poor")
                # Could implement alternative transcription method here
        
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(
                f"Transcript extracted: {segment_count} segments"
            )
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        return {
            "json_path": str(output_json),
            "data": transcript_data
        }
        
    except Exception as e:
        logfire.error(f"Error extracting transcript: {e}")
        raise


@video_ingest_agent.tool
async def validate_video_url(
    ctx: RunContext[VideoIngestDeps],
    video_url: str
) -> bool:
    """
    Validate that the video URL is accessible
    
    Args:
        ctx: Runtime context with dependencies
        video_url: Video URL to validate
        
    Returns:
        True if valid, raises exception otherwise
    """
    try:
        # Basic URL validation
        if not video_url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        
        if "youtube.com" not in video_url and "youtu.be" not in video_url:
            # HUMAN-IN-THE-LOOP: Confirm non-YouTube URLs
            if ctx.deps.ask_human:
                response = await ctx.deps.ask_human(
                    f"This appears to be a non-YouTube URL: {video_url}. "
                    f"Currently only YouTube is fully supported. Continue anyway? (yes/no)"
                )
                if response.lower() not in ['yes', 'y']:
                    raise ValueError("User chose not to process non-YouTube URL")
        
        # Check if video exists using yt-dlp
        cmd = ["yt-dlp", "--get-title", video_url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise ValueError(f"Video not accessible: {result.stderr}")
        
        logfire.info(f"Video validated: {result.stdout.strip()}")
        return True
        
    except Exception as e:
        logfire.error(f"URL validation failed: {e}")
        raise


async def run_video_ingestion(
    video_url: str,
    deps: VideoIngestDeps
) -> VideoIngestResult:
    """
    Main entry point for video ingestion workflow
    
    Args:
        video_url: URL of video to process
        deps: Dependencies including websocket and output directories
        
    Returns:
        VideoIngestResult with processing details
    """
    try:
        # Initialize stream output
        if deps.websocket:
            deps.stream_output = StreamResponse(
                agent="video_ingest_agent",
                status="in_progress",
                steps=["Starting video ingestion..."]
            )
            await deps.websocket.send_json(deps.stream_output.model_dump())
        
        # Run the agent
        prompt = f"""Process the following video:
        
URL: {video_url}

Steps:
1. Validate the video URL
2. Download the video (ask human if already exists)
3. Extract and generate transcript (ask human to verify quality)
4. Create structured JSON output

Provide detailed progress updates at each step.
Use human-in-the-loop when quality checks are needed."""

        result = await video_ingest_agent.run(
            user_prompt=prompt,
            deps=deps
        )
        
        # Update final status
        if deps.stream_output and deps.websocket:
            deps.stream_output.status = "completed"
            deps.stream_output.steps.append("Video ingestion completed successfully")
            await deps.websocket.send_json(deps.stream_output.model_dump())
        
        return result.data
        
    except Exception as e:
        logfire.error(f"Video ingestion failed: {e}")
        
        if deps.stream_output and deps.websocket:
            deps.stream_output.status = "failed"
            deps.stream_output.steps.append(f"Error: {str(e)}")
            await deps.websocket.send_json(deps.stream_output.model_dump())
        
        return VideoIngestResult(
            video_id="",
            video_title="",
            duration=0.0,
            snippets_json="",
            segment_count=0,
            transcript_word_count=0,
            status="failed",
            error=str(e)
        )
