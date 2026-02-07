# ChartSeek Observability & Evaluation Integration

## Overview

This document outlines where and how to integrate **Opik** (Open Source LLM Observability) into the ChartSeek application for comprehensive tracing, monitoring, and evaluation of AI components.

---

## Integration Points

ChartSeek has **4 key AI components** that benefit from observability:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CHARTSEEK OBSERVABILITY INTEGRATION POINTS               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. WHISPER     │    │   2. CLIP       │    │   3. CLIP       │    │  4. WEAVIATE    │
│  Transcription  │    │  Image Embed    │    │  Text Embed     │    │  Vector Search  │
│                 │    │                 │    │                 │    │                 │
│ • Input: audio  │    │ • Input: frame  │    │ • Input: query  │    │ • Input: vector │
│ • Output: text  │    │ • Output: 512d  │    │ • Output: 512d  │    │ • Output: hits  │
│ • Latency       │    │ • Latency       │    │ • Latency       │    │ • Latency       │
│ • Word count    │    │ • Frame count   │    │ • Query length  │    │ • Hit count     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │                      │
        └──────────────────────┴──────────────────────┴──────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │    OPIK TRACES      │
                               │                     │
                               │ • Latency breakdown │
                               │ • Success/failure   │
                               │ • Input/output logs │
                               │ • Cost tracking     │
                               └─────────────────────┘
```

---

## File Locations for Integration

### 1. Whisper Transcription
**File**: `cortex_on/video_scripts/yt_slice_chatgpt.py`

```python
# BEFORE (current code)
def transcribe_segment(audio_path):
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path)
    return result

# AFTER (with Opik tracing)
from opik import track

@track(name="whisper_transcribe", capture_input=True, capture_output=True)
def transcribe_segment(audio_path):
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path)
    return result
```

---

### 2. CLIP Image Embedding (Keyframe Extraction)
**File**: `cortex_on/video_scripts/keyframes_describe.py`

```python
# BEFORE (current code)
def embed_frames(frames):
    inputs = processor(images=frames, return_tensors="pt")
    features = model.get_image_features(**inputs)
    return features

# AFTER (with Opik tracing)
from opik import track

@track(name="clip_embed_images", capture_input=False, capture_output=False)
def embed_frames(frames, metadata=None):
    """
    metadata: {"video_id": "abc123", "frame_count": 4, "segment": 0}
    """
    import opik
    opik.set_span_metadata(metadata or {})
    
    inputs = processor(images=frames, return_tensors="pt")
    features = model.get_image_features(**inputs)
    
    opik.set_span_metadata({"embedding_dim": features.shape[-1]})
    return features
```

---

### 3. CLIP Text Embedding (Search Query)
**File**: `cortex_on/video_scripts/clip_embedding_service.py`

```python
# BEFORE (current code)
def embed_text(self, text: str) -> List[float]:
    inputs = self.tokenizer([text], return_tensors="pt")
    feat = self.model.get_text_features(**inputs)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].tolist()

# AFTER (with Opik tracing)
from opik import track

@track(name="clip_embed_text", capture_input=True)
def embed_text(self, text: str) -> List[float]:
    inputs = self.tokenizer([text], return_tensors="pt")
    feat = self.model.get_text_features(**inputs)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].tolist()
```

---

### 4. Weaviate Vector Search
**File**: `cortex_on/video_scripts/query_weaviate.py`

```python
# BEFORE (current code)
def search_visual(client, query_vector, video_id, top_k):
    collection = client.collections.get("VideoKeyframe")
    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        filters=Filter.by_property("video_id").equal(video_id)
    )
    return results

# AFTER (with Opik tracing)
from opik import track

@track(name="weaviate_visual_search", capture_input=True)
def search_visual(client, query_vector, video_id, top_k):
    import opik
    opik.set_span_metadata({
        "video_id": video_id,
        "top_k": top_k,
        "collection": "VideoKeyframe"
    })
    
    collection = client.collections.get("VideoKeyframe")
    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        filters=Filter.by_property("video_id").equal(video_id)
    )
    
    opik.set_span_metadata({"hits_returned": len(results.objects)})
    return results
```

---

### 5. Main Search Endpoint (Parent Trace)
**File**: `cortex_on/main.py`

```python
# Add parent trace to capture entire search flow
from opik import track

@track(name="chartseek_search", capture_input=True, capture_output=True)
async def search_video(video_id: str, query: str, top_k: int = 10, include_visual: bool = True):
    """
    This creates a parent trace that captures:
    - clip_embed_text (child)
    - weaviate_visual_search (child)
    - weaviate_transcript_search (child)
    - result_merge (child)
    """
    # Existing search logic...
    pass
```

---

## Setup Instructions

### 1. Install Opik

```bash
# In cortex_on container or local environment
pip install opik
```

### 2. Start Opik Server (Docker)

```bash
# Create opik directory in project
mkdir -p opik && cd opik

# docker-compose.yml for Opik
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  opik:
    image: ghcr.io/comet-ml/opik:latest
    ports:
      - "5173:5173"  # Dashboard
      - "8000:8000"  # API
    volumes:
      - opik_data:/data
    environment:
      - OPIK_BACKEND_URL=http://localhost:8000

volumes:
  opik_data:
EOF

docker-compose up -d
```

### 3. Configure Environment Variables

```bash
# Add to cortex_on service in docker-compose.yml
environment:
  - OPIK_URL_OVERRIDE=http://opik:8000/api
  - OPIK_WORKSPACE=default
  - OPIK_PROJECT_NAME=chartseek-traces
```

### 4. Initialize Opik in Application

```python
# Add to cortex_on/main.py at startup
import os
os.environ["OPIK_URL_OVERRIDE"] = os.getenv("OPIK_URL_OVERRIDE", "http://localhost:8000/api")
os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK_WORKSPACE", "default")
os.environ["OPIK_PROJECT_NAME"] = os.getenv("OPIK_PROJECT_NAME", "chartseek")
```

---

## Metrics to Track

### Latency Metrics
| Operation | Target | Alert Threshold |
|-----------|--------|-----------------|
| Whisper transcription | < 1x realtime | > 2x realtime |
| CLIP image embedding | < 100ms/frame | > 500ms/frame |
| CLIP text embedding | < 50ms | > 200ms |
| Weaviate search | < 100ms | > 500ms |
| Total search E2E | < 500ms | > 2s |

### Quality Metrics
| Metric | Description |
|--------|-------------|
| Visual hit rate | % searches with visual matches |
| Transcript hit rate | % searches with transcript matches |
| Average relevance score | Mean (1 - distance) across results |
| Zero-result rate | % searches returning no results |

### Volume Metrics
| Metric | Description |
|--------|-------------|
| Videos processed/day | Upload volume |
| Searches/day | Query volume |
| Frames indexed | Total visual index size |
| Transcript chunks indexed | Total text index size |

---

## Evaluation Framework

### 1. Search Quality Evaluation

```python
# cortex_on/eval/search_eval.py
from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import Equals, Contains

# Define test cases
test_cases = [
    {
        "query": "double bottom pattern",
        "video_id": "trading101",
        "expected_visual_hits": True,
        "expected_min_score": 0.3
    },
    {
        "query": "fibonacci retracement",
        "video_id": "trading101",
        "expected_visual_hits": True,
        "expected_min_score": 0.25
    }
]

def evaluate_search_quality():
    results = []
    for case in test_cases:
        response = search_video(case["video_id"], case["query"])
        
        # Check if visual hits exist
        visual_hits = [h for h in response.hits if h.source == "visual"]
        has_visual = len(visual_hits) > 0
        
        # Check relevance score
        if visual_hits:
            best_score = 1 - visual_hits[0].distance
            score_pass = best_score >= case["expected_min_score"]
        else:
            score_pass = False
        
        results.append({
            "query": case["query"],
            "visual_hits": has_visual,
            "score_pass": score_pass,
            "pass": has_visual == case["expected_visual_hits"] and score_pass
        })
    
    return results
```

### 2. Embedding Quality Evaluation

```python
# Test CLIP embedding consistency
def evaluate_embedding_quality():
    """
    Test that similar queries produce similar embeddings
    """
    similar_pairs = [
        ("double bottom", "W pattern formation"),
        ("bullish engulfing", "green candle engulfing red"),
        ("head and shoulders", "reversal pattern with three peaks")
    ]
    
    for q1, q2 in similar_pairs:
        emb1 = clip_service.embed_text(q1)
        emb2 = clip_service.embed_text(q2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2)
        
        # Similar queries should have > 0.7 similarity
        assert similarity > 0.7, f"Low similarity for {q1} vs {q2}: {similarity}"
```

---

## Dashboard Views

Once Opik is running, access **http://localhost:5173** to see:

1. **Traces View**: All search requests with latency breakdown
2. **Projects View**: Separate projects for different video types
3. **Metrics View**: Aggregated latency and throughput
4. **Evaluation View**: Test case pass/fail rates

---

## Architecture Diagram with Observability

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CHARTSEEK WITH OBSERVABILITY                        │
└─────────────────────────────────────────────────────────────────────────────┘

     User Search: "double bottom pattern"
                    │
                    ▼
     ┌──────────────────────────┐
     │    FastAPI Endpoint      │ ◄─── @track("chartseek_search")
     │    /api/v1/search        │
     └──────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ CLIP Text     │       │ Sentence-     │
│ Embedding     │       │ Transformers  │
│               │       │               │
│ @track(       │       │ @track(       │
│  "clip_text") │       │  "st_embed")  │
└───────────────┘       └───────────────┘
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ Weaviate      │       │ Weaviate      │
│ VideoKeyframe │       │ VideoChunks   │
│               │       │               │
│ @track(       │       │ @track(       │
│  "visual_     │       │  "transcript_ │
│   search")    │       │   search")    │
└───────────────┘       └───────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
     ┌──────────────────────────┐
     │    Merge & Rank          │ ◄─── @track("merge_results")
     └──────────────────────────┘
                    │
                    ▼
     ┌──────────────────────────┐
     │    Return Response       │
     └──────────────────────────┘
                    │
                    ▼
     ┌──────────────────────────┐
     │         OPIK             │
     │                          │
     │  ┌────────────────────┐  │
     │  │ Trace: search_xyz  │  │
     │  │ ├─ clip_text: 45ms │  │
     │  │ ├─ visual: 82ms    │  │
     │  │ ├─ transcript: 31ms│  │
     │  │ └─ merge: 5ms      │  │
     │  │ Total: 163ms       │  │
     │  └────────────────────┘  │
     │                          │
     │  Dashboard: :5173        │
     └──────────────────────────┘
```

---

## Quick Start Integration

### Step 1: Add Opik to requirements.txt

```
# cortex_on/requirements.txt
opik>=0.1.0
```

### Step 2: Create observability wrapper

```python
# cortex_on/video_scripts/observability.py
import os
from functools import wraps
from opik import track

# Configure Opik
os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:8000/api")
os.environ.setdefault("OPIK_WORKSPACE", "default")
os.environ.setdefault("OPIK_PROJECT_NAME", "chartseek")

def trace_whisper(func):
    """Decorator for Whisper transcription"""
    @wraps(func)
    @track(name="whisper_transcribe")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def trace_clip_image(func):
    """Decorator for CLIP image embedding"""
    @wraps(func)
    @track(name="clip_embed_image")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def trace_clip_text(func):
    """Decorator for CLIP text embedding"""
    @wraps(func)
    @track(name="clip_embed_text")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def trace_search(func):
    """Decorator for Weaviate search"""
    @wraps(func)
    @track(name="weaviate_search")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

### Step 3: Apply decorators

```python
# In each file, import and use:
from observability import trace_whisper, trace_clip_image, trace_clip_text, trace_search

@trace_whisper
def transcribe_segment(audio_path):
    # existing code...

@trace_clip_image
def embed_frames(frames):
    # existing code...

@trace_clip_text
def embed_text(self, text):
    # existing code...

@trace_search
def search_visual(client, query_vector, video_id, top_k):
    # existing code...
```

---

## Summary

| Component | File | Decorator |
|-----------|------|-----------|
| Whisper Transcription | `yt_slice_chatgpt.py` | `@trace_whisper` |
| CLIP Image Embedding | `keyframes_describe.py` | `@trace_clip_image` |
| CLIP Text Embedding | `clip_embedding_service.py` | `@trace_clip_text` |
| Weaviate Visual Search | `query_weaviate.py` | `@trace_search` |
| Weaviate Transcript Search | `query_weaviate.py` | `@trace_search` |
| Main Search Endpoint | `main.py` | `@track("chartseek_search")` |

This integration provides complete observability into ChartSeek's AI pipeline, enabling performance monitoring, debugging, and quality evaluation.
