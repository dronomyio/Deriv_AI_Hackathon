# CortexON Video Understanding Integration

This project integrates video understanding capabilities with TheAgentic's CortexON multi-agent orchestration framework. It enables specialized AI agents to collaborate on complex video analysis tasks including transcript extraction, keyframe analysis, semantic search, and question answering.

## Overview

The integration extends CortexON's existing multi-agent system with specialized video processing agents that work alongside the web, coder, and file agents. The orchestrator agent coordinates these specialized agents to handle end-to-end video understanding workflows.

## Architecture

### Specialized Agents

1. **Video Ingestion Agent**: Downloads videos, extracts transcripts, and creates structured JSON output
2. **Vector Indexing Agent**: Indexes transcripts and keyframes in Weaviate vector database
3. **Keyframe Analysis Agent**: Extracts representative frames and generates visual descriptions
4. **Video Query Agent**: Answers questions using semantic search and evidence-based reasoning

### Orchestration Flow

```
User Request
    ↓
Orchestrator Agent (plans and coordinates)
    ↓
├── Video Ingestion Agent → Downloads & Transcribes
├── Vector Indexing Agent → Creates Searchable Index
├── Keyframe Analysis Agent → Extracts Visual Content
└── Video Query Agent → Answers Questions
    ↓
Results with Video Clips
```

## Prerequisites

- Docker and Docker Compose
- Anthropic API key (for Claude)
- OpenAI API key (for embeddings and vision models)
- Browserbase account (for web agent)
- Google Custom Search API credentials
- Logfire token (for observability)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/TheAgenticAI/CortexON.git
cd CortexON
```

### 2. Copy Video Understanding Integration

```bash
# Copy the integration files into the CortexON project
cp -r cortexon_video_integration/agents/* cortex_on/agents/
cp -r cortexon_video_integration/utils/* cortex_on/utils/
cp cortexon_video_integration/docker-compose.yaml docker-compose.yaml
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Anthropic Configuration
ANTHROPIC_MODEL_NAME=claude-3-7-sonnet-20250219
ANTHROPIC_API_KEY=your_anthropic_api_key

# OpenAI Configuration (for embeddings and video analysis)
OPENAI_API_KEY=your_openai_api_key

# Weaviate Configuration (automatically configured in Docker)
WEAVIATE_URL=http://weaviate:8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_HTTP_PORT=8080

# Browserbase Configuration
BROWSERBASE_API_KEY=your_browserbase_api_key
BROWSERBASE_PROJECT_ID=your_browserbase_project_id

# Google Custom Search
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CX=your_google_cx_id

# Logging
LOGFIRE_TOKEN=your_logfire_token

# Vault Integration (Optional)
VITE_APP_VA_NAMESPACE=your_unique_namespace_id
VA_TOKEN=your_vault_token
VA_URL=your_vault_url
VA_TTL=24h
VA_TOKEN_REFRESH_SECONDS=43200

# WebSocket
VITE_WEBSOCKET_URL=ws://localhost:8081/ws

# Video Processing
VIDEO_DOWNLOAD_DIR=/data/downloads
VIDEO_OUTPUT_DIR=/data/out
VIDEO_MAX_DURATION=7200
```

### 4. Update Orchestrator Agent

Modify `cortex_on/agents/orchestrator_agent.py` to register video tools:

```python
# Add at the top of the file
from agents.orchestrator_video_tools import (
    VIDEO_TOOLS,
    VIDEO_ORCHESTRATOR_PROMPT_EXTENSION
)

# Update the system prompt
orchestrator_system_prompt = """You are an AI orchestrator that manages a team of agents to solve tasks.
...
""" + VIDEO_ORCHESTRATOR_PROMPT_EXTENSION

# Register video tools with the orchestrator agent
for tool_name, tool_func in VIDEO_TOOLS.items():
    orchestrator_agent.tool(tool_func)
```

### 5. Build and Start Services

```bash
# Build Docker images
docker-compose build

# Start all services
docker-compose up -d

# Check service health
docker-compose ps
```

### 6. Verify Installation

```bash
# Check Weaviate is running
curl http://localhost:8080/v1/meta

# Check CortexON backend
curl http://localhost:8081/docs

# Check frontend
open http://localhost:3000
```

## Usage

### Example 1: Single Video Q&A

**User Request:**
```
Analyze this YouTube video and answer: What do they say about BMNR and Tom Lee?
https://www.youtube.com/watch?v=VIDEO_ID
```

**Orchestrator Workflow:**
1. Plans the task with 5 steps
2. Delegates to Video Ingestion Agent → Downloads and transcribes
3. Delegates to Vector Indexing Agent → Indexes transcript
4. Delegates to Keyframe Analysis Agent → Extracts visual content
5. Delegates to Video Query Agent → Searches for "BMNR" and "Tom Lee"
6. Returns answer with supporting video clips

### Example 2: Multi-Video Comparison

**User Request:**
```
Compare what these 3 videos say about Bitcoin ETFs:
- https://www.youtube.com/watch?v=VIDEO_ID_1
- https://www.youtube.com/watch?v=VIDEO_ID_2
- https://www.youtube.com/watch?v=VIDEO_ID_3
```

**Orchestrator Workflow:**
1. Plans parallel ingestion of all videos
2. Processes each video independently
3. Queries all collections for "Bitcoin ETFs"
4. Synthesizes comparative analysis
5. Returns summary with clips from each video

### Example 3: Visual Search

**User Request:**
```
Find all moments in this video where charts or graphs are shown:
https://www.youtube.com/watch?v=VIDEO_ID
```

**Orchestrator Workflow:**
1. Plans visual-focused analysis
2. Ingests video and extracts keyframes
3. Generates detailed visual descriptions
4. Searches keyframe collection for "charts graphs"
5. Returns clips for each matching moment

## API Reference

### Video Ingestion Tool

```python
video_ingest_task(
    video_url: str,
    output_dir: str = "/data/out"
) -> str
```

Downloads and processes a video, returning metadata and transcript JSON path.

### Vector Indexing Tool

```python
vector_index_task(
    json_path: str,
    video_id: str,
    collection_name: str = "VideoChunks"
) -> str
```

Indexes video transcript in Weaviate for semantic search.

### Keyframe Analysis Tool

```python
keyframe_analysis_task(
    video_path: str,
    output_path: str,
    fps: float = 1.0,
    k: int = 6
) -> str
```

Extracts and analyzes representative frames from video.

### Video Query Tool

```python
video_query_task(
    question: str,
    video_id: str,
    collection_name: str,
    index_json: str,
    output_clip: str
) -> str
```

Answers questions about video content with evidence-based clips.

## Architecture Details

### Data Flow

```
Raw Video → Video Ingestion → Snippets + Transcripts JSON
                                    ↓
                          Vector Indexing Agent
                                    ↓
                        Weaviate Transcript Collection
                                    
Video Snippets → Keyframe Analysis → Keyframes + Descriptions JSON
                                    ↓
                          Vector Indexing Agent
                                    ↓
                        Weaviate Keyframe Collection

Natural Language Query → Video Query Agent
                              ↓
                    Vector Retrieval + Merging
                              ↓
                    Clip Generation + LLM Synthesis
                              ↓
                    Answer + Supporting Clips
```

### Service Topology

- **Frontend** (React/TypeScript) - Port 3000
- **CortexON Backend** (FastAPI) - Port 8081
- **Agentic Browser** (Browserbase) - Port 8000
- **Weaviate Vector DB** - Ports 8080 (HTTP), 50051 (gRPC)

All services communicate through Docker's internal network.

## Configuration

### Video Processing Settings

```bash
# Maximum video duration (seconds)
VIDEO_MAX_DURATION=7200

# Download directory
VIDEO_DOWNLOAD_DIR=/data/downloads

# Output directory
VIDEO_OUTPUT_DIR=/data/out
```

### Weaviate Settings

```bash
# Weaviate URL (internal Docker network)
WEAVIATE_URL=http://weaviate:8080

# Default collection names
TRANSCRIPT_COLLECTION=VideoChunks
KEYFRAME_COLLECTION=VideoKeyframes
```

### Embedding Settings

```bash
# OpenAI embedding model
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Embedding dimensions
EMBEDDING_DIMENSIONS=512

# Maximum tokens per chunk
MAX_CHUNK_TOKENS=350
```

## Troubleshooting

### Weaviate Connection Issues

```bash
# Check Weaviate health
curl http://localhost:8080/v1/.well-known/ready

# View Weaviate logs
docker-compose logs weaviate

# Restart Weaviate
docker-compose restart weaviate
```

### Video Download Failures

```bash
# Check yt-dlp installation
docker-compose exec cortex_on yt-dlp --version

# View download logs
docker-compose logs cortex_on | grep "video_ingest"

# Clear download cache
docker-compose exec cortex_on rm -rf /data/downloads/*
```

### Vector Indexing Errors

```bash
# Check OpenAI API key
docker-compose exec cortex_on env | grep OPENAI_API_KEY

# View indexing logs
docker-compose logs cortex_on | grep "vector_index"

# Delete and recreate collection
docker-compose exec cortex_on python3 -c "
import weaviate
client = weaviate.connect_to_local()
client.collections.delete('VideoChunks')
client.close()
"
```

## Performance Optimization

### Parallel Processing

For multiple videos, the orchestrator can process them in parallel:

```python
# The orchestrator automatically detects independent tasks
# and executes them concurrently
```

### GPU Acceleration

For faster keyframe analysis, use GPU-enabled Docker images:

```yaml
services:
  cortex_on:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Caching

Enable caching for frequently accessed videos:

```bash
# Set cache directory
VIDEO_CACHE_DIR=/data/cache

# Enable caching in environment
VIDEO_CACHE_ENABLED=true
```

## Development

### Adding New Video Agents

1. Create agent file in `cortex_on/agents/`
2. Implement using PydanticAI framework
3. Register tools in `orchestrator_video_tools.py`
4. Update orchestrator system prompt

### Testing

```bash
# Run unit tests
docker-compose exec cortex_on pytest tests/

# Run integration tests
docker-compose exec cortex_on pytest tests/integration/

# Test specific agent
docker-compose exec cortex_on pytest tests/agents/test_video_ingest_agent.py
```

### Debugging

```bash
# Enable debug logging
docker-compose exec cortex_on export LOG_LEVEL=DEBUG

# View agent traces in Logfire
open https://logfire.pydantic.dev

# Inspect Weaviate collections
docker-compose exec cortex_on python3 -c "
import weaviate
client = weaviate.connect_to_local()
collections = client.collections.list_all()
print(collections)
client.close()
"
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Submit a pull request

## License

This integration is licensed under the CortexON Open Source License Agreement.

## Support

For issues and questions:

- GitHub Issues: https://github.com/TheAgenticAI/CortexON/issues
- Documentation: https://cortexon.ai/docs
- Community: https://github.com/TheAgenticAI/CortexON/discussions

## Acknowledgments

This integration builds upon:

- **CortexON** by TheAgentic - Multi-agent orchestration framework
- **Weaviate** - Vector database for semantic search
- **PydanticAI** - Agent framework with type safety
- **yt-dlp** - Video download utility
- **OpenAI** - Embeddings and vision models

## Roadmap

Future enhancements:

- [ ] Real-time video stream processing
- [ ] Multi-language transcript support
- [ ] Advanced multi-modal fusion
- [ ] Collaborative filtering for query optimization
- [ ] Video summarization agent
- [ ] Subtitle translation agent
- [ ] Audio analysis agent
