"""
Orchestrator Video Tools (with MongoDB Storage)

These tools are registered with the CortexON orchestrator agent to enable
video understanding capabilities with MongoDB storage.
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from pydantic_ai import RunContext
import logfire

from storage.mongodb_storage import MongoDBStorage
from agents.video.video_ingest_agent_mongodb import run_video_ingestion, VideoIngestDeps
from agents.video.video_query_agent_mongodb import run_video_query, VideoQueryDeps


# Initialize MongoDB storage (singleton)
_storage_instance: Optional[MongoDBStorage] = None


def get_storage() -> MongoDBStorage:
    """Get or create MongoDB storage instance"""
    global _storage_instance
    if _storage_instance is None:
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://admin:changeme@mongodb:27017/")
        _storage_instance = MongoDBStorage(
            connection_string=mongodb_uri,
            database_name="video_understanding"
        )
        logfire.info("MongoDB storage initialized for orchestrator tools")
    return _storage_instance


async def ingest_video_tool(
    ctx: RunContext,
    video_url: str,
    priority: int = 5
) -> Dict[str, Any]:
    """
    Tool for orchestrator to ingest videos with MongoDB storage
    
    Args:
        ctx: Runtime context
        video_url: YouTube video URL
        priority: Processing priority (1-10)
        
    Returns:
        Ingestion result with video metadata
    """
    try:
        logfire.info(f"Orchestrator: Ingesting video {video_url}")
        
        # Get storage instance
        storage = get_storage()
        
        # Create dependencies
        deps = VideoIngestDeps(
            websocket=ctx.deps.websocket if hasattr(ctx.deps, 'websocket') else None,
            storage=storage,
            download_dir="/data/downloads",
            output_dir="/data/out",
            ask_human=ctx.deps.ask_human if hasattr(ctx.deps, 'ask_human') else None
        )
        
        # Run ingestion
        result = await run_video_ingestion(video_url, deps)
        
        return {
            "status": "success",
            "video_id": result.video_id,
            "title": result.video_title,
            "duration": result.duration,
            "segments": result.segment_count,
            "words": result.transcript_word_count
        }
        
    except Exception as e:
        logfire.error(f"Video ingestion tool failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def query_video_tool(
    ctx: RunContext,
    video_id: str,
    question: str,
    generate_clips: bool = False
) -> Dict[str, Any]:
    """
    Tool for orchestrator to query videos with MongoDB storage
    
    Args:
        ctx: Runtime context
        video_id: Video identifier
        question: Natural language question
        generate_clips: Whether to generate video clips
        
    Returns:
        Query result with answer and evidence
    """
    try:
        logfire.info(f"Orchestrator: Querying video {video_id}: {question}")
        
        # Get storage instance
        storage = get_storage()
        
        # Create dependencies
        deps = VideoQueryDeps(
            websocket=ctx.deps.websocket if hasattr(ctx.deps, 'websocket') else None,
            storage=storage,
            weaviate_url=os.getenv("WEAVIATE_URL", "http://weaviate:8080"),
            output_dir="/data/out",
            ask_human=ctx.deps.ask_human if hasattr(ctx.deps, 'ask_human') else None
        )
        
        # Run query
        result = await run_video_query(
            video_id=video_id,
            question=question,
            user_id=ctx.deps.user_id if hasattr(ctx.deps, 'user_id') else "anonymous",
            session_id=ctx.deps.session_id if hasattr(ctx.deps, 'session_id') else "default",
            generate_clips=generate_clips,
            deps=deps
        )
        
        return {
            "status": "success",
            "query_id": result.query_id,
            "answer": result.answer,
            "confidence": result.confidence,
            "evidence": result.evidence,
            "clips": result.video_clips
        }
        
    except Exception as e:
        logfire.error(f"Video query tool failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def get_video_status_tool(
    ctx: RunContext,
    video_id: str
) -> Dict[str, Any]:
    """
    Tool for orchestrator to check video processing status
    
    Args:
        ctx: Runtime context
        video_id: Video identifier
        
    Returns:
        Video processing status and metadata
    """
    try:
        storage = get_storage()
        
        video = storage.get_video(video_id)
        if not video:
            return {
                "status": "not_found",
                "video_id": video_id
            }
        
        return {
            "status": "success",
            "video_id": video_id,
            "title": video['title'],
            "duration": video['duration'],
            "processing_status": video['processing']['status'],
            "stages": video['processing']['stages'],
            "quality": video['quality'],
            "created_at": video['created_at'].isoformat(),
            "updated_at": video['updated_at'].isoformat()
        }
        
    except Exception as e:
        logfire.error(f"Get video status tool failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def list_videos_tool(
    ctx: RunContext,
    status: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Tool for orchestrator to list videos
    
    Args:
        ctx: Runtime context
        status: Filter by processing status
        limit: Maximum number of results
        
    Returns:
        List of videos with metadata
    """
    try:
        storage = get_storage()
        
        if status:
            videos = storage.find_videos_by_status(status, limit)
        else:
            videos = list(storage.videos.find().sort("created_at", -1).limit(limit))
        
        return {
            "status": "success",
            "count": len(videos),
            "videos": [
                {
                    "video_id": v['video_id'],
                    "title": v['title'],
                    "duration": v['duration'],
                    "processing_status": v['processing']['status'],
                    "quality": v['quality'].get('transcript_confidence', 0.0),
                    "created_at": v['created_at'].isoformat()
                }
                for v in videos
            ]
        }
        
    except Exception as e:
        logfire.error(f"List videos tool failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def search_videos_tool(
    ctx: RunContext,
    query: str,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Tool for orchestrator to search videos by keywords
    
    Args:
        ctx: Runtime context
        query: Search query
        limit: Maximum number of results
        
    Returns:
        Search results with video metadata
    """
    try:
        storage = get_storage()
        
        # Search in MongoDB using text search and tag matching
        results = storage.videos.find({
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}},
                {"tags": {"$in": [query.lower()]}},
                {"topics": {"$in": [query.lower()]}}
            ]
        }).limit(limit)
        
        videos = list(results)
        
        return {
            "status": "success",
            "query": query,
            "count": len(videos),
            "videos": [
                {
                    "video_id": v['video_id'],
                    "title": v['title'],
                    "duration": v['duration'],
                    "processing_status": v['processing']['status'],
                    "quality": v['quality'].get('transcript_confidence', 0.0),
                    "tags": v.get('tags', []),
                    "created_at": v['created_at'].isoformat()
                }
                for v in videos
            ]
        }
        
    except Exception as e:
        logfire.error(f"Search videos tool failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def get_processing_stats_tool(
    ctx: RunContext
) -> Dict[str, Any]:
    """
    Tool for orchestrator to get processing statistics
    
    Args:
        ctx: Runtime context
        
    Returns:
        Processing statistics and metrics
    """
    try:
        storage = get_storage()
        
        stats = storage.get_processing_stats()
        
        # Get average job durations
        avg_ingestion = storage.get_average_job_duration("video_ingestion")
        avg_indexing = storage.get_average_job_duration("vector_indexing")
        
        # Get total counts
        total_videos = storage.videos.count_documents({})
        total_queries = storage.queries.count_documents({})
        
        return {
            "status": "success",
            "processing_stats": stats,
            "average_durations": {
                "ingestion_seconds": avg_ingestion,
                "indexing_seconds": avg_indexing
            },
            "totals": {
                "videos": total_videos,
                "queries": total_queries
            }
        }
        
    except Exception as e:
        logfire.error(f"Get processing stats tool failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


# Tool registration for CortexON orchestrator
VIDEO_TOOLS = [
    {
        "name": "ingest_video",
        "description": "Ingest a video from YouTube URL with MongoDB storage",
        "function": ingest_video_tool,
        "parameters": {
            "video_url": "YouTube video URL",
            "priority": "Processing priority (1-10, default: 5)"
        }
    },
    {
        "name": "query_video",
        "description": "Ask a question about video content with MongoDB storage",
        "function": query_video_tool,
        "parameters": {
            "video_id": "Video identifier",
            "question": "Natural language question",
            "generate_clips": "Whether to generate video clips (default: False)"
        }
    },
    {
        "name": "get_video_status",
        "description": "Get video processing status from MongoDB",
        "function": get_video_status_tool,
        "parameters": {
            "video_id": "Video identifier"
        }
    },
    {
        "name": "list_videos",
        "description": "List videos from MongoDB",
        "function": list_videos_tool,
        "parameters": {
            "status": "Filter by processing status (optional)",
            "limit": "Maximum number of results (default: 20)"
        }
    },
    {
        "name": "search_videos",
        "description": "Search videos by keywords in MongoDB",
        "function": search_videos_tool,
        "parameters": {
            "query": "Search query",
            "limit": "Maximum number of results (default: 10)"
        }
    },
    {
        "name": "get_processing_stats",
        "description": "Get processing statistics from MongoDB",
        "function": get_processing_stats_tool,
        "parameters": {}
    }
]


def register_video_tools(orchestrator_agent):
    """
    Register video tools with the CortexON orchestrator agent
    
    Args:
        orchestrator_agent: The orchestrator agent instance
    """
    for tool in VIDEO_TOOLS:
        orchestrator_agent.tool(
            name=tool["name"],
            description=tool["description"]
        )(tool["function"])
    
    logfire.info(f"Registered {len(VIDEO_TOOLS)} video tools with orchestrator")
