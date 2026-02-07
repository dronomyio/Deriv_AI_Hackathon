# MCP Server — CortexON Video Understanding

## What it does

Exposes CortexON's video analysis pipeline as MCP tools so you can drive
the system from **Codex app**, **Claude Desktop**, or any MCP client:

| Tool               | Description                                     |
|--------------------|-------------------------------------------------|
| `video_create_job` | Submit a local video → starts slice/transcribe  |
| `video_get_status` | Poll processing progress                        |
| `video_search`     | Semantic search over indexed video chunks        |
| `health_check`     | Liveness probe                                  |

## Running standalone

```bash
cd cortex_on
python -m mcp.server
```

## Codex MCP settings

Add this to your Codex project's MCP config (Settings → MCP Servers):

```json
{
  "cortexon-video": {
    "command": "python",
    "args": ["-m", "mcp.server"],
    "cwd": "/path/to/cortex_on"
  }
}
```

## Claude Desktop MCP config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or the equivalent on your OS:

```json
{
  "mcpServers": {
    "cortexon-video": {
      "command": "python",
      "args": ["-m", "mcp.server"],
      "cwd": "/path/to/cortex_on",
      "env": {
        "WEAVIATE_URL": "http://localhost:8084",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## Docker Compose (production)

```yaml
cortexon_mcp:
  build:
    context: ./cortex_on
    dockerfile: Dockerfile.cortexon_video
  command: ["python", "-m", "mcp.server"]
  depends_on:
    - mongodb
    - weaviate
  volumes:
    - ./cortex_on:/app
  env_file:
    - .env
  networks:
    - cortex_network
```

## Prerequisites

```bash
pip install mcp
```
