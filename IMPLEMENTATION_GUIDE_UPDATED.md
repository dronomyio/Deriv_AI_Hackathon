# CortexON Video Understanding Implementation Guide (Updated)

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

## Prerequisites

(Prerequisites remain the same as the original guide)

## Step-by-Step Integration

### Step 1: Clone CortexON Repository

```bash
# Clone the official CortexON repository
git clone https://github.com/TheAgenticAI/CortexON.git
cd CortexON
```

### Step 2: Integrate Video Understanding Scripts

This step integrates the core Python scripts from the original `video_understanding` application into the CortexON project. These scripts will be called by the specialized video agents.

First, create a directory to house these scripts within the `cortex_on` service:

```bash
# Create a directory for the video processing scripts
mkdir -p cortex_on/video_scripts
```

Next, copy the necessary Python scripts from your `video_understanding-main-4/app` directory into this new directory. The required scripts are:

- `yt_slice_chatgpt.py`
- `weaviate_ingest.py`
- `weaviate_ingest_keyframes.py`
- `keyframes_describe.py`
- `get_clip.py`
- `query_weaviate.py`

```bash
# Copy the scripts from your original project
cp /path/to/your/video_understanding-main-4/app/yt_slice_chatgpt.py cortex_on/video_scripts/
cp /path/to/your/video_understanding-main-4/app/weaviate_ingest.py cortex_on/video_scripts/
cp /path/to/your/video_understanding-main-4/app/weaviate_ingest_keyframes.py cortex_on/video_scripts/
cp /path/to/your/video_understanding-main-4/app/keyframes_describe.py cortex_on/video_scripts/
cp /path/to/your/video_understanding-main-4/app/get_clip.py cortex_on/video_scripts/
cp /path/to/your/video_understanding-main-4/app/query_weaviate.py cortex_on/video_scripts/
```

**Important**: The agent code in `orchestrator_video_tools.py` and the video agents themselves must be updated to call these scripts from the new path, for example: `subprocess.run(["python3", "/app/video_scripts/yt_slice_chatgpt.py", ...])` instead of `subprocess.run(["python3", "/app/yt_slice_chatgpt.py", ...])`.

### Step 3: Create an Updated Dockerfile for CortexON

The `cortex_on` service needs to have all the dependencies for both the CortexON framework and the `video_understanding` scripts. Create a new Dockerfile named `Dockerfile.cortexon_video` in the root of the CortexON project.

```bash
cat > Dockerfile.cortexon_video << 'EOF'
# Use the official CortexON base image or a suitable Python image
FROM python:3.11-slim

# Install system dependencies required for video processing (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies for both CortexON and video understanding
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install video understanding specific dependencies
RUN pip install --no-cache-dir yt-dlp openai-whisper torch torchvision weaviate-client tiktoken pillow ImageHash scikit-learn transformers

# Copy the CortexON application code
COPY ./cortex_on /app

# The entrypoint should start the CortexON application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081", "--reload"]
EOF
```

### Step 4: Update Docker Compose Configuration

Modify the `docker-compose.yaml` to use the new Dockerfile for the `cortex_on` service.

```yaml
services:
  cortex_on:
    build:
      context: .
      dockerfile: Dockerfile.cortexon_video # Use the new Dockerfile
    # ... rest of the service definition
```

### Step 5: Add Video Understanding Agents

(This step remains the same as the original guide - it involves creating the agent files and placing them in the correct directories)

### Step 6: Configure Environment Variables

(This step remains the same as the original guide)

### Step 7: Update Orchestrator Agent

(This step remains the same as the original guide)

### Step 8: Update Planner Agent Prompts

(This step remains the same as the original guide)

### Step 9: Build and Start Services

```bash
# Build all services using the updated configuration
docker-compose build

# Start all services
docker-compose up -d
```

### Step 10: Verify Installation

(This step remains the same as the original guide)

## Agent Implementation Details

(This section remains the same as the original guide)

## Testing and Validation

(This section remains the same as the original guide)

## Deployment Strategies

(This section remains the same as the original guide)

## Best Practices

(This section remains the same as the original guide)

## Troubleshooting

(This section remains the same as the original guide)
