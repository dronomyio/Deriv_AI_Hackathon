"""
CortexON Video Understanding — FastAPI Backend
================================================
Versioned REST API under /api/v1:
  POST /api/v1/videos/upload        → upload + kick off processing
  GET  /api/v1/jobs/{job_id}        → poll job status / progress
  GET  /api/v1/jobs                 → list all jobs
  GET  /api/v1/videos/{id}/search   → semantic search over indexed video
  GET  /api/v1/clips/{name}         → stream a generated clip
  GET  /api/v1/videos/{id}/file     → stream the original upload
  GET  /api/v1/health               → health check
Plus legacy /agent/chat + /ws for backwards compat.

Observability: Integrated with Opik for tracing (if enabled via OPIK_ENABLED=true).
"""

import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Observability Setup ──────────────────────────────────────────────────────
# Initialize observability before other imports that might use it
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "video_scripts"))

try:
    from observability import init_opik, metrics, trace_span, evaluate_search_quality
    OBSERVABILITY_AVAILABLE = init_opik()
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    class DummyMetrics:
        def record(self, *args, **kwargs): pass
        def get_stats(self, *args, **kwargs): return {}
    metrics = DummyMetrics()
    from contextlib import contextmanager
    @contextmanager
    def trace_span(name, metadata=None):
        yield type('obj', (object,), {'set_metadata': lambda self, x: None})()
    def evaluate_search_quality(*args, **kwargs):
        return {"passed": True, "issues": []}

# ── Directories ──────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.getenv("DATA_DIR", "/data"))
UPLOADS    = DATA_DIR / "uploads"
OUT_DIR    = DATA_DIR / "out"
CLIPS_DIR  = OUT_DIR / "clips"
for _d in (UPLOADS, OUT_DIR, CLIPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="CortexON Video Understanding API",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store (swap for Mongo in prod) ────────────────────────────
jobs: Dict[str, Dict[str, Any]] = {}


# ── Pydantic schemas ────────────────────────────────────────────────────────
class JobStatus(BaseModel):
    job_id: str
    video_id: str
    status: str        # pending | processing | ready | failed
    progress: float    # 0.0 – 1.0
    message: str
    created_at: str
    updated_at: str

class SearchHit(BaseModel):
    text: str
    start_seconds: float
    end_seconds: float
    distance: Optional[float] = None
    snippet_index: Optional[int] = None
    chunk_index: Optional[int] = None
    source: Optional[str] = "transcript"   # "transcript" or "visual"

class SearchResult(BaseModel):
    query: str
    video_id: str
    hits: List[SearchHit]
    best_window: Optional[Dict[str, Any]] = None
    clip_url: Optional[str] = None
    answer: Optional[str] = None
    visual_hits_count: Optional[int] = 0
    transcript_hits_count: Optional[int] = 0


# ── Helpers ──────────────────────────────────────────────────────────────────
def _job_status(j: Dict) -> JobStatus:
    return JobStatus(**{k: j[k] for k in JobStatus.model_fields})


async def _run(cmd: List[str], cwd: Optional[str] = None):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    out, err = await proc.communicate()
    return subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode,
        stdout=out.decode("utf-8", errors="replace"),
        stderr=err.decode("utf-8", errors="replace"),
    )


# ── Background processing pipeline ──────────────────────────────────────────
async def _process_video(job_id: str, video_path: Path, video_id: str):
    j = jobs[job_id]
    import time
    t0 = time.time()
    try:
        # Step 1 — Slice + Transcribe (this is the slow part)
        j.update(status="processing", progress=0.05,
                 message="Step 1/3: Splitting video into segments with ffmpeg…",
                 updated_at=datetime.now(timezone.utc).isoformat())

        vid_out = OUT_DIR / video_id
        vid_out.mkdir(parents=True, exist_ok=True)

        whisper_model = os.getenv("WHISPER_MODEL", "tiny")
        j.update(progress=0.10,
                 message=f"Step 1/3: Running Whisper ({whisper_model}) transcription… this may take a few minutes on CPU",
                 updated_at=datetime.now(timezone.utc).isoformat())

        r = await _run([
            "python3", "/app/video_scripts/yt_slice_chatgpt.py",
            "--input", str(video_path),
            "--outdir", str(vid_out),
            "--whisper-model", whisper_model,
        ])
        elapsed = int(time.time() - t0)
        if r.returncode != 0:
            raise RuntimeError(f"Slice/transcribe failed ({elapsed}s): {r.stderr[:500]}")

        json_path = vid_out / "snippets_with_transcripts.json"
        if not json_path.exists():
            cands = list(vid_out.rglob("snippets_with_transcripts.json"))
            if not cands:
                raise RuntimeError("Slicer produced no snippets JSON")
            json_path = cands[0]

        j["index_json"] = str(json_path)
        j.update(progress=0.60,
                 message=f"Step 2/3: Transcription done ({elapsed}s). Ingesting into Weaviate…",
                 updated_at=datetime.now(timezone.utc).isoformat())

        # Step 2 — Weaviate ingest (transcript chunks)
        r = await _run([
            "python3", "/app/video_scripts/weaviate_ingest.py",
            "--json", str(json_path),
            "--video-id", video_id,
            "--collection", "VideoChunks",
        ])
        if r.returncode != 0:
            raise RuntimeError(f"Weaviate ingest failed: {r.stderr[:500]}")

        elapsed = int(time.time() - t0)
        j.update(progress=0.70,
                 message=f"Step 3/5: Transcript indexed ({elapsed}s). Extracting keyframes with CLIP…",
                 updated_at=datetime.now(timezone.utc).isoformat())

        # Step 3 — CLIP keyframe extraction per segment
        # The slicer already extracts frames at 1 FPS per segment.
        # We run keyframes_describe.py on each segment's video to get
        # CLIP-selected representative frames + LLM descriptions.
        snippets_data = json.loads(json_path.read_text(encoding="utf-8"))
        segments = snippets_data.get("segments", [])
        all_keyframe_jsons: list[Path] = []

        print(f"[CLIP] Processing {len(segments)} segments for keyframes...")

        for seg in segments:
            seg_idx = seg.get("index", 0)
            seg_video = seg.get("video_path", "")
            seg_start = seg.get("start_seconds", 0.0)
            seg_end = seg.get("end_seconds", 0.0)
            frames_info = seg.get("frames", {})
            frames_dir = frames_info.get("frames_dir", "")

            print(f"[CLIP] Segment {seg_idx}: video={seg_video}, exists={Path(seg_video).exists() if seg_video else False}")

            if not seg_video or not Path(seg_video).exists():
                print(f"[CLIP] Skipping segment {seg_idx}: video path missing or doesn't exist")
                continue

            kf_json = vid_out / f"keyframes_seg_{seg_idx:03d}.json"

            # Build args: prefer existing frames dir if available
            kf_cmd = [
                "python3", "/app/video_scripts/keyframes_describe.py",
                "--out", str(kf_json),
                "--fps", "1",
                "--k", "4",           # 4 representative frames per segment
                "--max-hamming", "6",
                "--clip-id", f"{video_id}_seg{seg_idx}",
                "--t1", str(seg_start),
                "--t2", str(seg_end),
                "--no-llm", "1",      # Skip LLM descriptions, use CLIP embeddings only
            ]

            # Use pre-extracted frames if available, otherwise use the clip
            if frames_dir and Path(frames_dir).exists():
                kf_cmd += ["--frames-dir", frames_dir]
            else:
                kf_cmd += ["--clip", seg_video]

            r = await _run(kf_cmd)
            print(f"[CLIP] Segment {seg_idx} keyframes_describe.py returned: {r.returncode}, output exists: {kf_json.exists()}")
            if r.returncode != 0:
                print(f"[CLIP] Segment {seg_idx} STDERR (last 2000 chars): ...{r.stderr[-2000:]}")
                print(f"[CLIP] Segment {seg_idx} STDOUT: {r.stdout[-500:]}")
            if r.returncode == 0 and kf_json.exists():
                all_keyframe_jsons.append(kf_json)
            # Non-fatal: if CLIP fails for a segment, we continue

        print(f"[CLIP] Total keyframe JSONs produced: {len(all_keyframe_jsons)}")

        elapsed = int(time.time() - t0)
        j.update(progress=0.85,
                 message=f"Step 4/5: CLIP extracted {len(all_keyframe_jsons)} keyframe sets ({elapsed}s). Indexing visual content…",
                 updated_at=datetime.now(timezone.utc).isoformat())

        # Step 4 — Ingest keyframes into Weaviate
        kf_ingested = 0
        for kf_json in all_keyframe_jsons:
            clip_id = kf_json.stem  # e.g. keyframes_seg_000
            r = await _run([
                "python3", "/app/video_scripts/weaviate_ingest_keyframes.py",
                "--json", str(kf_json),
                "--video-id", video_id,
                "--clip-id", clip_id,
                "--collection", "VideoKeyframe",
            ])
            if r.returncode == 0:
                kf_ingested += 1

        elapsed = int(time.time() - t0)
        j.update(status="ready", progress=1.0,
                 message=f"Step 5/5: Done ✓ ({elapsed}s total) — transcript + {kf_ingested} visual keyframe sets indexed",
                 updated_at=datetime.now(timezone.utc).isoformat())

    except Exception as exc:
        elapsed = int(time.time() - t0)
        j.update(status="failed", progress=0.0,
                 message=f"Failed after {elapsed}s: {str(exc)[:400]}",
                 updated_at=datetime.now(timezone.utc).isoformat())


# ═══════════════════════════════════════════════════════════════════════════
# API v1 routes
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "service": "chartseek-video",
        "ts": datetime.now(timezone.utc).isoformat(),
        "observability": {
            "enabled": OBSERVABILITY_AVAILABLE,
            "opik_url": os.getenv("OPIK_URL_OVERRIDE", "http://localhost:8000/api"),
            "project": os.getenv("OPIK_PROJECT_NAME", "chartseek"),
        }
    }

@app.get("/health")
async def health_compat():
    return await health()

@app.get("/api/v1/metrics")
async def get_metrics():
    """Return collected observability metrics."""
    return {
        "clip_text_embed": metrics.get_stats("clip_text_embed_latency_ms"),
        "clip_image_embed": metrics.get_stats("clip_image_embed_latency_ms"),
        "weaviate_search": metrics.get_stats("weaviate_search_latency_ms"),
        "search_e2e": metrics.get_stats("search_e2e_latency_ms"),
    }

@app.get("/api/v1/debug")
async def debug_info():
    """Quick diagnostic: what's in the system right now."""
    import shutil
    return {
        "active_jobs": len(jobs),
        "jobs_summary": [
            {"id": j["job_id"][:8], "video": j["video_id"], "status": j["status"],
             "progress": j["progress"], "msg": j["message"][:100]}
            for j in jobs.values()
        ],
        "disk_uploads": len(list(UPLOADS.glob("*"))) if UPLOADS.exists() else 0,
        "disk_out": len(list(OUT_DIR.glob("*"))) if OUT_DIR.exists() else 0,
        "whisper_model": os.getenv("WHISPER_MODEL", "tiny"),
        "weaviate_url": WEAVIATE_URL,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "whisper": shutil.which("whisper") is not None,
    }


# ── Upload ───────────────────────────────────────────────────────────────────
@app.post("/api/v1/videos/upload", response_model=JobStatus)
async def upload_video(file: UploadFile = File(...)):
    video_id = uuid.uuid4().hex[:12]
    job_id   = uuid.uuid4().hex
    ext      = Path(file.filename or "v.mp4").suffix or ".mp4"
    dest     = UPLOADS / f"{video_id}{ext}"

    with dest.open("wb") as fout:
        shutil.copyfileobj(file.file, fout)

    now = datetime.now(timezone.utc).isoformat()
    jobs[job_id] = dict(
        job_id=job_id, video_id=video_id, status="pending",
        progress=0.0, message="Queued", video_path=str(dest),
        index_json=None, created_at=now, updated_at=now,
    )
    asyncio.create_task(_process_video(job_id, dest, video_id))
    return _job_status(jobs[job_id])


# ── Jobs ─────────────────────────────────────────────────────────────────────
@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return _job_status(jobs[job_id])

@app.get("/api/v1/jobs")
async def list_jobs():
    return [_job_status(j) for j in
            sorted(jobs.values(), key=lambda x: x["created_at"], reverse=True)]


# ── Search ───────────────────────────────────────────────────────────────────
@app.get("/api/v1/videos/{video_id}/search", response_model=SearchResult)
async def search_video(
    video_id: str,
    q: str = Query(..., description="Natural-language query"),
    top_k: int = Query(8, ge=1, le=50),
    collection: str = Query("VideoChunks"),
    include_visual: bool = Query(True, description="Also search CLIP keyframe descriptions"),
):
    search_start = time.time()
    
    with trace_span("chartseek_search", {"video_id": video_id, "query": q, "top_k": top_k}) as span:
        import sys
        sys.path.insert(0, "/app/video_scripts")
        try:
            from query_weaviate import (
                _connect_weaviate, _fetch_hits,
                _merge_hits_into_windows, _pick_best_window,
            )
        except ImportError as e:
            raise HTTPException(500, f"Import error: {e}")

        client = _connect_weaviate()
    try:
        # 1) Transcript hits from VideoChunks
        hits = _fetch_hits(client, collection, q, top_k, video_id)

        # 2) Visual hits from VideoKeyframe (if enabled and collection exists)
        visual_hits_raw = []
        if include_visual:
            try:
                visual_hits_raw = _fetch_hits(client, "VideoKeyframe", q, min(top_k, 6), video_id)
            except Exception:
                pass  # Collection may not exist yet
    finally:
        try: client.close()
        except Exception: pass

    # Build transcript SearchHits
    search_hits = [
        SearchHit(text=h.text, start_seconds=h.start_s, end_seconds=h.end_s,
                  distance=h.distance, snippet_index=h.snippet_index,
                  chunk_index=h.chunk_index, source="transcript")
        for h in hits
    ]
    transcript_count = len(search_hits)

    # Build visual SearchHits — keyframes have absolute_time_s, use ±5s window
    visual_count = 0
    for vh in visual_hits_raw:
        t = vh.start_s  # This is absolute_time_s from the keyframe
        search_hits.append(
            SearchHit(
                text=f"[Visual] {vh.text}",
                start_seconds=max(0, t - 3.0),
                end_seconds=t + 5.0,
                distance=vh.distance,
                snippet_index=None,
                chunk_index=None,
                source="visual",
            )
        )
        visual_count += 1

    # Re-sort all hits by distance (best first)
    search_hits.sort(key=lambda h: (h.distance if h.distance is not None else 9999.0))

    # Merge all hits (both transcript + visual) for windowing
    all_raw_hits = list(hits) + list(visual_hits_raw)

    best_window = None
    clip_url = None

    if all_raw_hits:
        windows = _merge_hits_into_windows(all_raw_hits, 8.0, 140.0)
        bw = _pick_best_window(windows)
        if bw:
            t1, t2, wh = bw
            best_window = {"start_seconds": max(0, t1 - 2), "end_seconds": t2 + 2,
                           "hit_count": len(wh)}

            # best-effort clip generation
            ready = [j for j in jobs.values()
                     if j["video_id"] == video_id and j["status"] == "ready"]
            if ready:
                idx_json = ready[0].get("index_json")
                if idx_json and Path(idx_json).exists():
                    clip_name = f"{video_id}_{int(t1)}_{int(t2)}.mp4"
                    clip_path = CLIPS_DIR / clip_name
                    try:
                        from get_clip import get_clip as make_clip
                        make_clip(Path(idx_json), max(0, t1-2), t2+2, clip_path)
                        clip_url = f"/api/v1/clips/{clip_name}"
                    except Exception:
                        pass

        # Record end-to-end search metrics
        search_latency_ms = (time.time() - search_start) * 1000
        metrics.record("search_e2e_latency_ms", search_latency_ms, {
            "video_id": video_id,
            "visual_hits": str(visual_count),
            "transcript_hits": str(transcript_count)
        })
        
        # Update span with results
        span.set_metadata({
            "visual_hits": visual_count,
            "transcript_hits": transcript_count,
            "total_hits": len(search_hits),
            "latency_ms": search_latency_ms
        })

    return SearchResult(query=q, video_id=video_id, hits=search_hits,
                        best_window=best_window, clip_url=clip_url,
                        visual_hits_count=visual_count,
                        transcript_hits_count=transcript_count)


# ── Direct time-range clip extraction ────────────────────────────────────────
@app.get("/api/v1/videos/{video_id}/clip")
async def extract_clip(
    video_id: str,
    t1: float = Query(..., ge=0, description="Start time in seconds"),
    t2: float = Query(..., ge=0, description="End time in seconds"),
    reencode: bool = Query(False, description="Re-encode for better compatibility (slower)"),
):
    """
    Cut a clip directly by timestamp — no semantic search needed.
    Example: /api/v1/videos/abc123/clip?t1=5&t2=10
    """
    if t2 <= t1:
        raise HTTPException(400, f"t2 ({t2}) must be greater than t1 ({t1})")

    # Find a completed job for this video
    ready = [j for j in jobs.values()
             if j["video_id"] == video_id and j["status"] == "ready"]
    if not ready:
        raise HTTPException(404, f"No processed video found for video_id={video_id}")

    idx_json = ready[0].get("index_json")
    if not idx_json or not Path(idx_json).exists():
        raise HTTPException(404, "Segment index not found — video may need reprocessing")

    clip_name = f"{video_id}_{int(t1)}_{int(t2)}.mp4"
    clip_path = CLIPS_DIR / clip_name

    if not clip_path.exists():
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent / "video_scripts"))
            from get_clip import get_clip as make_clip
            make_clip(Path(idx_json), t1, t2, clip_path, reencode=reencode)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(500, f"Clip generation failed: {e}")

    return {
        "video_id": video_id,
        "t1": t1,
        "t2": t2,
        "duration_seconds": round(t2 - t1, 3),
        "clip_url": f"/api/v1/clips/{clip_name}",
    }


# ── Clip & file serving ─────────────────────────────────────────────────────
@app.get("/api/v1/clips/{clip_name}")
async def serve_clip(clip_name: str):
    p = CLIPS_DIR / clip_name
    if not p.exists():
        raise HTTPException(404, "Clip not found")
    return FileResponse(str(p), media_type="video/mp4", filename=clip_name)

@app.get("/api/v1/videos/{video_id}/file")
async def serve_video(video_id: str):
    m = [j for j in jobs.values() if j["video_id"] == video_id]
    if not m:
        raise HTTPException(404, "Video not found")
    vp = Path(m[0]["video_path"])
    if not vp.exists():
        raise HTTPException(404, "File missing")
    return FileResponse(str(vp), media_type="video/mp4")


# ═══════════════════════════════════════════════════════════════════════════
# Legacy endpoints
# ═══════════════════════════════════════════════════════════════════════════
try:
    from instructor import SystemInstructor

    async def generate_response(task: str, websocket: Optional[WebSocket] = None):
        orchestrator: SystemInstructor = SystemInstructor()
        return await orchestrator.run(task, websocket)

    @app.get("/agent/chat")
    async def agent_chat(task: str) -> List:
        return await generate_response(task)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            await generate_response(data, websocket)
except ImportError:
    pass
