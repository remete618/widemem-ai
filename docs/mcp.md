# MCP Server

widemem ships with an MCP (Model Context Protocol) server so you can plug it directly into Claude Desktop, Cursor, or any MCP-compatible client. Memory as a tool: add, search, delete, and count memories without writing a single line of glue code.

## Install

```bash
pip install widemem-ai[mcp]
```

## Run it

```bash
python -m widemem.mcp_server
```

This starts a stdio-based MCP server.

## Claude Desktop config

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "widemem": {
      "command": "python",
      "args": ["-m", "widemem.mcp_server"],
      "env": {
        "WIDEMEM_LLM_PROVIDER": "ollama",
        "WIDEMEM_LLM_MODEL": "llama3.2",
        "WIDEMEM_EMBEDDING_PROVIDER": "sentence-transformers"
      }
    }
  }
}
```

## Available tools

| Tool | Description |
|---|---|
| `widemem_add` | Add memories (extracts facts, resolves conflicts) |
| `widemem_search` | Semantic search over memories |
| `widemem_delete` | Delete a memory by ID |
| `widemem_count` | Count stored memories |
| `widemem_health` | Health check |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `WIDEMEM_DATA_PATH` | `~/.widemem/data` | Storage directory |
| `WIDEMEM_LLM_PROVIDER` | `ollama` | LLM provider (`openai`, `anthropic`, `ollama`) |
| `WIDEMEM_LLM_MODEL` | `llama3.2` | LLM model name |
| `WIDEMEM_LLM_BASE_URL` | `http://localhost:11434` | LLM API base URL |
| `WIDEMEM_EMBEDDING_PROVIDER` | `sentence-transformers` | Embedding provider |
| `WIDEMEM_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `WIDEMEM_API_KEY` | (unset) | Optional shared key for the optional REST server |
