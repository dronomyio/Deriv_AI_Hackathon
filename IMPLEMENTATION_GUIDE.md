# CortexON Video Understanding Implementation Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Integration](#step-by-step-integration)
4. [Agent Implementation Details](#agent-implementation-details)
5. [Testing and Validation](#testing-and-validation)
6. [Deployment Strategies](#deployment-strategies)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Introduction

This guide provides detailed instructions for integrating video understanding capabilities into CortexON's multi-agent orchestration framework. By following this guide, you will enable your CortexON instance to process YouTube videos, extract transcripts, perform semantic search, and answer questions about video content.

### What You'll Build

A complete video understanding system with:

- **Video Ingestion Pipeline**: Download and transcribe videos
- **Vector Search Infrastructure**: Semantic search over transcripts and keyframes
- **Multi-Modal Analysis**: Combined text and visual understanding
- **Interactive Q&A**: Natural language queries with video clip generation

### Architecture Overview

The integration adds four specialized agents to CortexON:

```
┌─────────────────────────────────────────────────────────┐
│                  Orchestrator Agent                     │
│              (Coordinates all agents)                   │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┬──────────────┐
        │                 │                 │              │
        ▼                 ▼                 ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Video      │  │   Vector     │  │  Keyframe    │  │    Video     │
│  Ingestion   │  │  Indexing    │  │  Analysis    │  │    Query     │
│    Agent     │  │    Agent     │  │    Agent     │  │    Agent     │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │              │
        ▼                 ▼                 ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   YouTube    │  │   Weaviate   │  │   Vision     │  │     LLM      │
│   Download   │  │   Vector DB  │  │   Models     │  │  Synthesis   │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: Minimum 50GB free space for video processing

### API Keys and Services

1. **Anthropic API Key**
   - Sign up at https://console.anthropic.com
   - Create an API key with Claude access
   - Model required: `claude-3-7-sonnet-20250219`

2. **OpenAI API Key**
   - Sign up at https://platform.openai.com
   - Create an API key with embeddings access
   - Models required: `text-embedding-3-small`, `gpt-4.1-mini`

3. **Browserbase Account**
   - Sign up at https://browserbase.com
   - Create a project and obtain API key
   - Note your project ID

4. **Google Custom Search**
   - Enable Custom Search API at https://console.cloud.google.com
   - Create credentials and obtain API key
   - Create a Custom Search Engine and note the CX ID

5. **Logfire Token**
   - Sign up at https://logfire.pydantic.dev
   - Create a project and obtain token

### Software Dependencies

The following will be installed via Docker:

- Python 3.11+
- PydanticAI
- Weaviate Python client
- yt-dlp (video download)
- FFmpeg (video processing)
- OpenAI Python client
- FastAPI
- React/TypeScript (frontend)

## Step-by-Step Integration

### Step 1: Clone CortexON Repository

```bash
# Clone the official CortexON repository
git clone https://github.com/TheAgenticAI/CortexON.git
cd CortexON

# Create a backup of the original configuration
cp docker-compose.yaml docker-compose.yaml.backup
```

### Step 2: Add Video Understanding Agents

Create the agents directory structure:

```bash
# Create directories for video agents
mkdir -p cortex_on/agents/video
mkdir -p cortex_on/utils/video
```

Copy the video agent implementations:

```bash
# Copy Video Ingestion Agent
cp /path/to/video_ingest_agent.py cortex_on/agents/video/

# Copy Video Query Agent
cp /path/to/video_query_agent.py cortex_on/agents/video/

# Copy Orchestrator Video Tools
cp /path/to/orchestrator_video_tools.py cortex_on/agents/
```

### Step 3: Update Docker Compose Configuration

Replace the existing `docker-compose.yaml` with the enhanced version:

```bash
# Backup original
mv docker-compose.yaml docker-compose.yaml.original

# Copy new configuration with Weaviate
cp /path/to/docker-compose.yaml .
```

Key additions in the new configuration:

- **Weaviate service** for vector storage
- **Volume mounts** for video data
- **Environment variables** for video processing
- **Service dependencies** to ensure proper startup order

### Step 4: Configure Environment Variables

Create a comprehensive `.env` file:

```bash
cat > .env << 'EOF'
# ============================================
# CortexON Video Understanding Configuration
# ============================================

# Anthropic Configuration
ANTHROPIC_MODEL_NAME=claude-3-7-sonnet-20250219
ANTHROPIC_API_KEY=sk-ant-xxxxx

# OpenAI Configuration
OPENAI_API_KEY=sk-xxxxx
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=512

# Weaviate Configuration
WEAVIATE_URL=http://weaviate:8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_HTTP_PORT=8080

# Browserbase Configuration
BROWSERBASE_API_KEY=xxxxx
BROWSERBASE_PROJECT_ID=xxxxx

# Google Custom Search
GOOGLE_API_KEY=xxxxx
GOOGLE_CX=xxxxx

# Logging
LOGFIRE_TOKEN=xxxxx

# Vault Integration (Optional)
VITE_APP_VA_NAMESPACE=cortexon-video-$(uuidgen)
VA_TOKEN=xxxxx
VA_URL=https://your-vault-url
VA_TTL=24h
VA_TOKEN_REFRESH_SECONDS=43200

# WebSocket
VITE_WEBSOCKET_URL=ws://localhost:8081/ws

# Video Processing Configuration
VIDEO_DOWNLOAD_DIR=/data/downloads
VIDEO_OUTPUT_DIR=/data/out
VIDEO_MAX_DURATION=7200
VIDEO_CACHE_ENABLED=true
VIDEO_CACHE_DIR=/data/cache

# Performance Tuning
MAX_CONCURRENT_DOWNLOADS=3
MAX_CONCURRENT_INDEXING=2
CHUNK_SIZE=350
KEYFRAME_FPS=1.0
KEYFRAME_COUNT=6
EOF
```

### Step 5: Update Orchestrator Agent

Modify `cortex_on/agents/orchestrator_agent.py`:

```python
# Add imports at the top
from agents.orchestrator_video_tools import (
    VIDEO_TOOLS,
    VIDEO_ORCHESTRATOR_PROMPT_EXTENSION,
    video_ingest_task,
    vector_index_task,
    keyframe_analysis_task,
    video_query_task
)

# Update the system prompt (around line 330)
orchestrator_system_prompt = """You are an AI orchestrator that manages a team of agents to solve tasks.
...
[existing prompt content]
...
""" + VIDEO_ORCHESTRATOR_PROMPT_EXTENSION

# Register video tools (add after orchestrator_agent initialization)
@orchestrator_agent.tool
async def video_ingest(ctx: RunContext[orchestrator_deps], video_url: str, output_dir: str = "/data/out") -> str:
    """Ingest a video from URL and prepare it for analysis"""
    return await video_ingest_task(ctx, video_url, output_dir)

@orchestrator_agent.tool
async def vector_index(ctx: RunContext[orchestrator_deps], json_path: str, video_id: str, collection_name: str = "VideoChunks") -> str:
    """Index video transcript in vector database"""
    return await vector_index_task(ctx, json_path, video_id, collection_name)

@orchestrator_agent.tool
async def keyframe_analysis(ctx: RunContext[orchestrator_deps], video_path: str, output_path: str, fps: float = 1.0, k: int = 6) -> str:
    """Extract and analyze keyframes from video"""
    return await keyframe_analysis_task(ctx, video_path, output_path, fps, k)

@orchestrator_agent.tool
async def video_query(ctx: RunContext[orchestrator_deps], question: str, video_id: str, collection_name: str, index_json: str, output_clip: str) -> str:
    """Query video content and generate answer with clip"""
    return await video_query_task(ctx, question, video_id, collection_name, index_json, output_clip)
```

### Step 6: Update Planner Agent Prompts

Modify `cortex_on/agents/planner_agent.py` to include video understanding workflows:

```python
# Add to the planner system prompt
VIDEO_PLANNING_TEMPLATES = """

[VIDEO UNDERSTANDING TASK TEMPLATES]

For "Analyze YouTube video and answer questions":
1. [ ] Ingest video from URL
2. [ ] Index transcript in vector database
3. [ ] Extract and analyze keyframes
4. [ ] Index keyframes in vector database
5. [ ] Ready for interactive queries

For "Compare multiple videos":
1. [ ] Ingest video 1
2. [ ] Ingest video 2
3. [ ] Ingest video 3
4. [ ] Index all transcripts
5. [ ] Query across all videos
6. [ ] Synthesize comparative analysis

For "Find visual content in video":
1. [ ] Ingest video
2. [ ] Extract keyframes with high sampling rate
3. [ ] Generate detailed visual descriptions
4. [ ] Index keyframe descriptions
5. [ ] Query for visual patterns
"""

# Append to existing planner system prompt
planner_system_prompt += VIDEO_PLANNING_TEMPLATES
```

### Step 7: Update Dependencies

Add required Python packages to `cortex_on/requirements.txt`:

```txt
# Existing dependencies
pydantic-ai>=0.0.1
anthropic>=0.40.0
fastapi>=0.115.0
...

# Video understanding dependencies
weaviate-client>=4.9.0
yt-dlp>=2024.12.0
opencv-python>=4.10.0
pillow>=10.4.0
imagehash>=4.3.1
tiktoken>=0.8.0
```

### Step 8: Build Docker Images

```bash
# Build all services
docker-compose build

# This will take 10-15 minutes on first build
# Subsequent builds will be faster due to caching
```

### Step 9: Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# Monitor logs
docker-compose logs -f
```

Wait for all services to be healthy:

```bash
# Check service status
docker-compose ps

# Should show all services as "Up" and healthy
```

### Step 10: Verify Installation

```bash
# Test Weaviate
curl http://localhost:8080/v1/meta

# Test CortexON backend
curl http://localhost:8081/docs

# Test frontend
open http://localhost:3000

# Test video ingestion (from inside container)
docker-compose exec cortex_on python3 -c "
from agents.video.video_ingest_agent import validate_video_url
print('Video agent loaded successfully')
"
```

## Agent Implementation Details

### Video Ingestion Agent Architecture

The Video Ingestion Agent follows this workflow:

```python
1. validate_video_url(url)
   ├─ Check URL format
   ├─ Verify video accessibility
   └─ Extract video metadata

2. download_youtube_video(url)
   ├─ Use yt-dlp for download
   ├─ Save to VIDEO_DOWNLOAD_DIR
   └─ Return video file path

3. extract_transcript(video_path)
   ├─ Extract audio track
   ├─ Generate transcript with timestamps
   ├─ Segment into semantic chunks
   └─ Create snippets JSON

4. Return VideoIngestResult
   ├─ video_id
   ├─ snippets_json path
   ├─ segment_count
   └─ metadata
```

**Key Implementation Points:**

- Uses `yt-dlp` for robust video downloading
- Supports auto-generated and manual subtitles
- Creates semantic segments based on natural boundaries
- Streams progress updates via WebSocket

### Vector Indexing Agent Architecture

The Vector Indexing Agent implements semantic chunking:

```python
1. Load transcript JSON
   └─ Parse segments and timestamps

2. Semantic chunking
   ├─ Split into paragraphs
   ├─ Further split into sentences
   ├─ Pack sentences into token-budget chunks
   └─ Preserve timestamp boundaries

3. Generate embeddings
   ├─ Use OpenAI text-embedding-3-small
   ├─ Batch process for efficiency
   └─ Handle rate limits

4. Store in Weaviate
   ├─ Create/update collection
   ├─ Insert chunk objects with metadata
   └─ Build vector index

5. Return indexing statistics
```

**Key Implementation Points:**

- Token-aware chunking (default 350 tokens)
- Preserves temporal alignment
- Supports custom embedding dimensions
- Handles large videos efficiently

### Keyframe Analysis Agent Architecture

The Keyframe Analysis Agent uses perceptual hashing:

```python
1. Extract frames at specified FPS
   ├─ Use OpenCV for frame extraction
   └─ Save frames to temporary directory

2. Compute perceptual hashes
   ├─ Use imagehash library
   ├─ Calculate average hash for each frame
   └─ Measure Hamming distances

3. Select diverse keyframes
   ├─ Start with first frame
   ├─ Add frames with max Hamming distance
   ├─ Continue until k keyframes selected
   └─ Ensure temporal distribution

4. Generate visual descriptions
   ├─ Use GPT-4.1-mini with vision
   ├─ Describe each keyframe in detail
   └─ Include temporal context

5. Create keyframes JSON
   ├─ Frame paths
   ├─ Descriptions
   ├─ Timestamps
   └─ Perceptual hashes
```

**Key Implementation Points:**

- Perceptual hashing ensures visual diversity
- Configurable FPS and keyframe count
- Vision-language model for descriptions
- Temporal alignment with transcript

### Video Query Agent Architecture

The Video Query Agent orchestrates retrieval and synthesis:

```python
1. Parse natural language question
   └─ Extract key concepts and entities

2. Search vector collections
   ├─ Query transcript collection
   ├─ Query keyframe collection (if available)
   └─ Retrieve top-K results with distances

3. Merge temporal windows
   ├─ Sort results by timestamp
   ├─ Merge adjacent chunks (gap < threshold)
   ├─ Limit window duration
   └─ Score windows by relevance

4. Generate video clip
   ├─ Select best window
   ├─ Add padding (default 2s)
   ├─ Call get_clip.py
   └─ Save to output directory

5. Synthesize answer
   ├─ Format evidence with timestamps
   ├─ Call LLM with strict instructions
   ├─ Enforce evidence-based reasoning
   └─ Include timestamp citations

6. Return VideoQueryResult
   ├─ Natural language answer
   ├─ Video clip path
   ├─ Confidence score
   └─ Evidence chunks
```

**Key Implementation Points:**

- Multi-modal retrieval (text + visual)
- Intelligent window merging
- Evidence-based answer synthesis
- Fallback for unanswerable questions

## Testing and Validation

### Unit Tests

Create `cortex_on/tests/agents/test_video_ingest_agent.py`:

```python
import pytest
from agents.video.video_ingest_agent import (
    video_ingest_agent,
    validate_video_url,
    VideoIngestDeps
)

@pytest.mark.asyncio
async def test_validate_video_url():
    """Test URL validation"""
    deps = VideoIngestDeps()
    
    # Valid YouTube URL
    result = await validate_video_url(deps, "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert result == True
    
    # Invalid URL
    with pytest.raises(ValueError):
        await validate_video_url(deps, "not_a_url")

@pytest.mark.asyncio
async def test_video_ingestion_workflow():
    """Test complete ingestion workflow"""
    # This would use a test video
    # Implementation depends on test infrastructure
    pass
```

### Integration Tests

Create `cortex_on/tests/integration/test_video_workflow.py`:

```python
import pytest
from agents.orchestrator_agent import orchestrator_agent
from agents.orchestrator_video_tools import video_ingest_task

@pytest.mark.asyncio
async def test_end_to_end_video_qa():
    """Test complete video Q&A workflow"""
    # 1. Ingest test video
    # 2. Index transcript
    # 3. Query for specific information
    # 4. Validate answer and clip generation
    pass
```

### Manual Testing

Test the system with a real video:

```bash
# Enter the container
docker-compose exec cortex_on bash

# Test video ingestion
python3 -c "
import asyncio
from agents.video.video_ingest_agent import run_video_ingestion, VideoIngestDeps

async def test():
    deps = VideoIngestDeps()
    result = await run_video_ingestion(
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        deps
    )
    print(f'Status: {result.status}')
    print(f'Segments: {result.segment_count}')

asyncio.run(test())
"
```

## Deployment Strategies

### Development Deployment

For local development:

```bash
# Use docker-compose with hot reload
docker-compose up

# Make changes to code
# Container automatically reloads
```

### Production Deployment

For production environments:

```yaml
# docker-compose.prod.yaml
version: '3.8'

services:
  weaviate:
    image: semitechnologies/weaviate:latest
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    restart: unless-stopped

  cortex_on:
    build:
      context: .
      dockerfile: Dockerfile.prod
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    restart: unless-stopped
    environment:
      - LOG_LEVEL=INFO
      - ENABLE_METRICS=true
```

### Kubernetes Deployment

For Kubernetes:

```yaml
# k8s/weaviate-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: weaviate
spec:
  replicas: 1
  selector:
    matchLabels:
      app: weaviate
  template:
    metadata:
      labels:
        app: weaviate
    spec:
      containers:
      - name: weaviate
        image: semitechnologies/weaviate:latest
        ports:
        - containerPort: 8080
        - containerPort: 50051
        env:
        - name: PERSISTENCE_DATA_PATH
          value: /var/lib/weaviate
        volumeMounts:
        - name: weaviate-data
          mountPath: /var/lib/weaviate
      volumes:
      - name: weaviate-data
        persistentVolumeClaim:
          claimName: weaviate-pvc
```

## Best Practices

### Performance Optimization

1. **Batch Processing**: Process multiple videos in parallel
2. **Caching**: Cache downloaded videos and embeddings
3. **Connection Pooling**: Reuse Weaviate connections
4. **Async Operations**: Use async/await throughout

### Error Handling

1. **Graceful Degradation**: Continue with partial results
2. **Retry Logic**: Retry failed operations with exponential backoff
3. **User Feedback**: Provide clear error messages
4. **Logging**: Log all errors with context

### Security

1. **API Key Management**: Use environment variables, never commit keys
2. **Input Validation**: Validate all user inputs
3. **Rate Limiting**: Implement rate limits for API calls
4. **Access Control**: Restrict access to sensitive endpoints

### Monitoring

1. **Logfire Integration**: Use Logfire for observability
2. **Metrics Collection**: Track processing times, error rates
3. **Health Checks**: Implement health check endpoints
4. **Alerting**: Set up alerts for failures

## Troubleshooting

### Common Issues

**Issue: Weaviate connection refused**

```bash
# Check Weaviate is running
docker-compose ps weaviate

# Check Weaviate logs
docker-compose logs weaviate

# Restart Weaviate
docker-compose restart weaviate
```

**Issue: Video download fails**

```bash
# Check yt-dlp version
docker-compose exec cortex_on yt-dlp --version

# Update yt-dlp
docker-compose exec cortex_on pip install --upgrade yt-dlp

# Test download manually
docker-compose exec cortex_on yt-dlp -f best "VIDEO_URL"
```

**Issue: Out of memory during processing**

```bash
# Increase Docker memory limit
# Edit Docker Desktop settings or docker-compose.yaml

# Reduce concurrent operations
export MAX_CONCURRENT_DOWNLOADS=1
export MAX_CONCURRENT_INDEXING=1
```

**Issue: Slow query performance**

```bash
# Check Weaviate index status
curl http://localhost:8080/v1/schema

# Rebuild index
docker-compose exec cortex_on python3 -c "
import weaviate
client = weaviate.connect_to_local()
# Rebuild logic here
"

# Increase embedding dimensions
export OPENAI_EMBEDDING_DIMENSIONS=1536
```

## Conclusion

You now have a fully integrated video understanding system within CortexON. The multi-agent architecture enables sophisticated video analysis workflows with semantic search, keyframe analysis, and natural language Q&A capabilities.

For additional support, refer to:
- CortexON Documentation: https://cortexon.ai/docs
- GitHub Issues: https://github.com/TheAgenticAI/CortexON/issues
- Community Discussions: https://github.com/TheAgenticAI/CortexON/discussions
