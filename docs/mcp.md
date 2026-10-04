# MCP Server

widemem ships with an MCP (Model Context Protocol) server so you can plug it directly into Claude Desktop, Cursor, or any MCP-compatible client. Memory as a tool: add, search, delete, and count memories without writing a single line of glue code.

## Install

```bash
pip install "widemem-ai[mcp,local]"
ollama pull llama3.1:8b
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
        "WIDEMEM_LLM_MODEL": "llama3.1:8b",
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
| `widemem_pin` | Pin a critical fact at elevated importance (9.0) |
| `widemem_export` | Export stored memories as JSON, optionally per user |
| `widemem_health` | Health check |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `WIDEMEM_DATA_PATH` | `~/.widemem/data` | Storage directory |
| `WIDEMEM_LLM_PROVIDER` | `ollama` | LLM provider (`openai`, `anthropic`, `ollama`) |
| `WIDEMEM_LLM_MODEL` | (per provider) | LLM model name. Unset, each provider uses its default: `ollama` `llama3.1:8b`, `openai` `gpt-4o-mini`, `anthropic` `claude-haiku-4-5-20251001` |
| `WIDEMEM_LLM_BASE_URL` | (unset) | Base URL for the OpenAI and Ollama providers; Anthropic ignores it. Unset, OpenAI uses its SDK default (honours `OPENAI_BASE_URL`) and Ollama uses `http://localhost:11434` |
| `WIDEMEM_EMBEDDING_PROVIDER` | `sentence-transformers` | Embedding provider |
| `WIDEMEM_API_KEY` | (unset) | Optional shared key for the optional REST server |
