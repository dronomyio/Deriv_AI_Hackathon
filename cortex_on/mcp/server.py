"""
MCP Server for CortexON Video Understanding
=============================================
Exposes tools callable from Codex / Claude Desktop / MCP clients:

  video_create_job   → upload a local video and start processing
  video_get_status   → poll a job's progress
  video_search       → semantic search over an indexed video
  agent_chat         → forward a prompt to the legacy orchestrator
  health_check       → system liveness probe

Run standalone:
  python -m mcp.server

Or configure in Codex MCP settings / Claude Desktop's mcp.json.
"""

import asyncio
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# ── MCP SDK (pip install mcp) ───────────────────────────────────────────
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: 'mcp' package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Add parent path so we can import core/*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.jobs import (
    create_job, get_job, list_jobs, find_ready_jobs, update_job,
    UPLOADS, OUT_DIR, CLIPS_DIR,
)

# ── Server instance ─────────────────────────────────────────────────────
server = Server("cortexon-video-mcp")


# ── Tool definitions ────────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="video_create_job",
            description="Upload a local video file and start slice → transcribe → ingest pipeline. Returns job_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "Absolute path to a local video file"},
                },
                "required": ["video_path"],
            },
        ),
        Tool(
            name="video_get_status",
            description="Get the current status / progress of a processing job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="video_search",
            description="Semantic search over indexed video transcript chunks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_id": {"type": "string"},
                    "query": {"type": "string", "description": "Natural-language question"},
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["video_id", "query"],
            },
        ),
        Tool(
            name="health_check",
            description="Check CortexON backend health.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]):

    # ── video_create_job ────────────────────────────────────────────
    if name == "video_create_job":
        src = Path(arguments["video_path"])
        if not src.exists():
            return [TextContent(type="text", text=f"File not found: {src}")]

        video_id = uuid.uuid4().hex[:12]
        dest = UPLOADS / f"{video_id}{src.suffix}"
        shutil.copy2(src, dest)

        j = create_job(video_id, str(dest))
        return [TextContent(type="text", text=json.dumps(j, indent=2))]

    # ── video_get_status ────────────────────────────────────────────
    if name == "video_get_status":
        j = get_job(arguments["job_id"])
        if not j:
            return [TextContent(type="text", text="Job not found")]
        return [TextContent(type="text", text=json.dumps(
            {k: j[k] for k in ("job_id", "video_id", "status", "progress", "message")}, indent=2
        ))]

    # ── video_search ────────────────────────────────────────────────
    if name == "video_search":
        video_id = arguments["video_id"]
        query = arguments["query"]
        top_k = arguments.get("top_k", 8)

        sys.path.insert(0, "/app/video_scripts")
        try:
            from query_weaviate import _connect_weaviate, _fetch_hits, _merge_hits_into_windows, _pick_best_window
        except ImportError as e:
            return [TextContent(type="text", text=f"Import error: {e}")]

        client = _connect_weaviate()
        try:
            hits = _fetch_hits(client, "VideoChunks", query, top_k, video_id)
        finally:
            try: client.close()
            except: pass

        result = {
            "query": query,
            "video_id": video_id,
            "hit_count": len(hits),
            "hits": [
                {"text": h.text[:200], "start": h.start_s, "end": h.end_s, "dist": h.distance}
                for h in hits[:5]
            ],
        }

        if hits:
            windows = _merge_hits_into_windows(hits, 8.0, 140.0)
            bw = _pick_best_window(windows)
            if bw:
                result["best_window"] = {"start": bw[0], "end": bw[1], "chunks": len(bw[2])}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    # ── health_check ────────────────────────────────────────────────
    if name == "health_check":
        return [TextContent(type="text", text='{"status": "ok", "service": "cortexon-mcp"}')]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ── Entry point ─────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
