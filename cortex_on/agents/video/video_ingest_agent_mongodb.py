"""
Video Ingestion Agent for CortexON (with MongoDB Storage)

This agent handles video download, transcript extraction, and segmentation.
It stores all data in MongoDB instead of JSON files.

FEATURES:
- MongoDB storage integration
- Human-in-the-loop decision points
- Processing job tracking
- Quality metrics recording
"""

import os
import json
import subprocess
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from fastapi import WebSocket
import logfire

from utils.stream_response_format import StreamResponse
from storage.mongodb_storage import (
    MongoDBStorage,
    VideoMetadata,
    TranscriptSegment,
    ProcessingJob,
    QualityMetric
)


@dataclass
class VideoIngestDeps:
    """Dependencies for Video Ingestion Agent"""
    websocket: Optional[WebSocket] = None
    stream_output: Optional[StreamResponse] = None
    storage: Optional[MongoDBStorage] = None  # MongoDB storage
    download_dir: str = "/data/downloads"
    output_dir: str = "/data/out"
    ask_human: Optional[callable] = None


class VideoIngestResult(BaseModel):
    """Result from video ingestion process"""
    video_id: str
    video_title: str
    duration: float
    segment_count: int
    transcript_word_count: int
    status: str
    error: Optional[str] = None


VIDEO_INGEST_SYSTEM_PROMPT = """You are a video ingestion specialist agent responsible for downloading videos, 
extracting transcripts, and storing them in MongoDB for semantic analysis.

Your capabilities:
1. Download videos from YouTube and other sources
2. Extract audio and generate transcripts
3. Segment videos into meaningful chunks
4. Store all data in MongoDB (not JSON files)
5. Track processing jobs and quality metrics
6. Consult humans when quality or decisions are uncertain

You work as part of a multi-agent system coordinated by an orchestrator agent.
Always provide clear progress updates and handle errors gracefully.
"""


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
) -> Dict[str, Any]:
    """
    Download a YouTube video using yt-dlp and store metadata in MongoDB
    """
    try:
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(f"Downloading video from {video_url}...")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        # Extract video ID
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError(f"Could not extract video ID from URL: {video_url}")
        
        # Check if video already exists in MongoDB
        if ctx.deps.storage:
            existing = ctx.deps.storage.get_video(video_id)
            if existing:
                if ctx.deps.ask_human:
                    response = await ctx.deps.ask_human(
                        f"Video {video_id} already exists in database. Re-download? (yes/no)"
                    )
                    if response.lower() not in ['yes', 'y']:
                        logfire.info(f"Using existing video: {video_id}")
                        return {
                            "video_id": video_id,
                            "video_path": existing['files']['video_path'],
                            "metadata": existing
                        }
        
        # Create download directory
        download_dir = Path(ctx.deps.download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        
        output_template = str(download_dir / f"{video_id}.%(ext)s")
        
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
        
        # Find downloaded files
        video_files = list(download_dir.glob(f"{video_id}.*"))
        video_file = next((f for f in video_files if f.suffix in ['.mp4', '.mkv', '.webm']), None)
        info_file = next((f for f in video_files if f.suffix == '.info.json'), None)
        
        if not video_file:
            raise FileNotFoundError(f"Downloaded video not found in {download_dir}")
        
        # Load metadata from info.json
        metadata = {}
        if info_file and info_file.exists():
            with open(info_file, 'r') as f:
                metadata = json.load(f)
        
        # Store in MongoDB
        if ctx.deps.storage:
            video_metadata = VideoMetadata(
                video_id=video_id,
                title=metadata.get('title', 'Unknown Title'),
                description=metadata.get('description', ''),
                duration=metadata.get('duration', 0.0),
                source={
                    "platform": "youtube",
                    "url": video_url,
                    "uploader": metadata.get('uploader', ''),
                    "uploader_id": metadata.get('uploader_id', ''),
                    "upload_date": datetime.fromisoformat(metadata.get('upload_date', '20000101')),
                    "view_count": metadata.get('view_count', 0),
                    "like_count": metadata.get('like_count', 0),
                    "comment_count": metadata.get('comment_count', 0)
                },
                files={
                    "video_path": str(video_file),
                    "video_size": video_file.stat().st_size,
                    "format": video_file.suffix[1:],
                    "resolution": metadata.get('resolution', 'unknown'),
                    "fps": metadata.get('fps', 30)
                },
                processing={
                    "status": "processing",
                    "stages": {
                        "download": {
                            "status": "completed",
                            "started_at": datetime.now(timezone.utc),
                            "completed_at": datetime.now(timezone.utc),
                            "agent": "video_ingest_agent"
                        }
                    },
                    "last_updated": datetime.now(timezone.utc),
                    "error": None
                },
                quality={
                    "transcript_confidence": 0.0,
                    "transcript_word_count": 0,
                    "transcript_language": "en",
                    "audio_quality_score": 0.0,
                    "has_music": False,
                    "has_multiple_speakers": False
                },
                tags=[],
                topics=[],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            try:
                ctx.deps.storage.insert_video(video_metadata)
            except ValueError:
                # Video already exists, update instead
                ctx.deps.storage.update_video_status(
                    video_id=video_id,
                    stage="download",
                    status="completed",
                    completed_at=datetime.now(timezone.utc)
                )
        
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(f"Video downloaded: {video_file.name}")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        return {
            "video_id": video_id,
            "video_path": str(video_file),
            "metadata": metadata
        }
        
    except Exception as e:
        logfire.error(f"Error downloading video: {e}")
        raise


@video_ingest_agent.tool
async def extract_transcript(
    ctx: RunContext[VideoIngestDeps],
    video_path: str,
    video_id: str
) -> Dict[str, Any]:
    """
    Extract transcript and store segments in MongoDB
    """
    try:
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append("Extracting audio and generating transcript...")
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        # Update processing status
        if ctx.deps.storage:
            ctx.deps.storage.update_video_status(
                video_id=video_id,
                stage="transcription",
                status="processing",
                started_at=datetime.now(timezone.utc),
                agent="video_ingest_agent"
            )
        
        # Use the existing yt_slice_chatgpt.py script
        output_json = Path(ctx.deps.output_dir) / f"{video_id}_snippets_with_transcripts.json"
        
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
        
        # Load the output
        with open(output_json, 'r') as f:
            transcript_data = json.load(f)
        
        segment_count = len(transcript_data.get('segments', []))
        
        # HITL: Quality check
        if segment_count == 0 and ctx.deps.ask_human:
            response = await ctx.deps.ask_human(
                f"Warning: No transcript segments extracted. Continue anyway? (yes/no)"
            )
            if response.lower() not in ['yes', 'y']:
                raise ValueError("User cancelled due to transcript issues")
        
        # Store transcript segments in MongoDB
        if ctx.deps.storage and segment_count > 0:
            total_words = 0
            confidence_sum = 0.0
            
            for i, segment in enumerate(transcript_data['segments']):
                # Calculate confidence
                words = segment.get('words', [])
                seg_confidence = sum(w.get('confidence', 0.9) for w in words) / len(words) if words else 0.9
                confidence_sum += seg_confidence
                total_words += len(segment.get('text', '').split())
                
                transcript_segment = TranscriptSegment(
                    video_id=video_id,
                    segment_index=i,
                    start_time=segment['start'],
                    end_time=segment['end'],
                    duration=segment['end'] - segment['start'],
                    text=segment['text'],
                    words=words,
                    confidence=seg_confidence,
                    language="en",
                    speaker_id=segment.get('speaker_id'),
                    chunks=[],  # Will be populated during indexing
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                
                try:
                    ctx.deps.storage.insert_transcript_segment(transcript_segment)
                except ValueError:
                    # Segment already exists, skip
                    pass
            
            # Update video quality metrics
            avg_confidence = confidence_sum / segment_count if segment_count > 0 else 0.0
            
            ctx.deps.storage.videos.update_one(
                {"video_id": video_id},
                {"$set": {
                    "quality.transcript_confidence": avg_confidence,
                    "quality.transcript_word_count": total_words,
                    "updated_at": datetime.now(timezone.utc)
                }}
            )
            
            # Store quality metric
            quality_metric = QualityMetric(
                video_id=video_id,
                metric_type="transcript_quality",
                score=avg_confidence,
                score_details={
                    "word_count": total_words,
                    "segment_count": segment_count,
                    "confidence_avg": avg_confidence
                },
                computed_by="video_ingest_agent",
                computation_method="whisper_confidence_analysis",
                user_feedback=None,
                measured_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc)
            )
            ctx.deps.storage.insert_quality_metric(quality_metric)
            
            # Update processing status
            ctx.deps.storage.update_video_status(
                video_id=video_id,
                stage="transcription",
                status="completed",
                completed_at=datetime.now(timezone.utc),
                segment_count=segment_count
            )
        
        if ctx.deps.stream_output and ctx.deps.websocket:
            ctx.deps.stream_output.steps.append(
                f"Transcript extracted: {segment_count} segments, {total_words} words"
            )
            await ctx.deps.websocket.send_json(ctx.deps.stream_output.model_dump())
        
        return {
            "segment_count": segment_count,
            "word_count": total_words,
            "confidence": avg_confidence if segment_count > 0 else 0.0
        }
        
    except Exception as e:
        logfire.error(f"Error extracting transcript: {e}")
        if ctx.deps.storage:
            ctx.deps.storage.update_video_status(
                video_id=video_id,
                stage="transcription",
                status="failed",
                error=str(e)
            )
        raise


async def run_video_ingestion(
    video_url: str,
    deps: VideoIngestDeps
) -> VideoIngestResult:
    """
    Main entry point for video ingestion workflow with MongoDB storage
    """
    try:
        # Initialize stream output
        if deps.websocket:
            deps.stream_output = StreamResponse(
                agent="video_ingest_agent",
                status="in_progress",
                steps=["Starting video ingestion with MongoDB storage..."]
            )
            await deps.websocket.send_json(deps.stream_output.model_dump())
        
        # Extract video ID for job tracking
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError(f"Invalid YouTube URL: {video_url}")
        
        # Create processing job in MongoDB
        if deps.storage:
            job = ProcessingJob(
                job_id=f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{video_id}",
                job_type="video_ingestion",
                video_id=video_id,
                status="processing",
                priority=5,
                agent={
                    "name": "video_ingest_agent",
                    "instance_id": "agent_001",
                    "version": "1.0.0"
                },
                created_at=datetime.now(timezone.utc),
                started_at=datetime.now(timezone.utc),
                completed_at=None,
                duration_seconds=None,
                result=None,
                error=None,
                retry_count=0,
                max_retries=3,
                hitl_interactions=[],
                depends_on=[],
                blocks=[]
            )
            deps.storage.insert_processing_job(job)
        
        # Run the agent workflow
        prompt = f"""Process the following video:
        
URL: {video_url}

Steps:
1. Download the video and store metadata in MongoDB
2. Extract transcript and store segments in MongoDB
3. Record quality metrics

Provide detailed progress updates at each step."""

        result = await video_ingest_agent.run(
            user_prompt=prompt,
            deps=deps
        )
        
        # Update job status
        if deps.storage:
            deps.storage.update_job_status(
                job_id=job.job_id,
                status="completed",
                completed_at=datetime.now(timezone.utc),
                result=result.data.dict()
            )
            
            # Update video overall status
            deps.storage.videos.update_one(
                {"video_id": video_id},
                {"$set": {
                    "processing.status": "completed",
                    "processing.last_updated": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }}
            )
        
        # Update final status
        if deps.stream_output and deps.websocket:
            deps.stream_output.status = "completed"
            deps.stream_output.steps.append("Video ingestion completed successfully")
            await deps.websocket.send_json(deps.stream_output.model_dump())
        
        return result.data
        
    except Exception as e:
        logfire.error(f"Video ingestion failed: {e}")
        
        if deps.storage and 'job' in locals():
            deps.storage.update_job_status(
                job_id=job.job_id,
                status="failed",
                error=str(e)
            )
        
        if deps.stream_output and deps.websocket:
            deps.stream_output.status = "failed"
            deps.stream_output.steps.append(f"Error: {str(e)}")
            await deps.websocket.send_json(deps.stream_output.model_dump())
        
        return VideoIngestResult(
            video_id="",
            video_title="",
            duration=0.0,
            segment_count=0,
            transcript_word_count=0,
            status="failed",
            error=str(e)
        )
