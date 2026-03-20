"""widemem MCP server — exposes memory operations as MCP tools over stdio."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    VectorStoreConfig,
)

_memory: Optional[WideMemory] = None


def _build_config() -> MemoryConfig:
    data_path = os.environ.get("WIDEMEM_DATA_PATH", "~/.widemem/data")
    data_path = str(Path(data_path).expanduser())

    llm_provider = os.environ.get("WIDEMEM_LLM_PROVIDER", "ollama")
    llm_model = os.environ.get("WIDEMEM_LLM_MODEL", "llama3.2")
    llm_base_url = os.environ.get("WIDEMEM_LLM_BASE_URL", "http://localhost:11434")
    embedding_provider = os.environ.get("WIDEMEM_EMBEDDING_PROVIDER", "sentence-transformers")

    llm_cfg = LLMConfig(provider=llm_provider, model=llm_model, base_url=llm_base_url)

    embedding_kwargs: dict = {"provider": embedding_provider}
    if embedding_provider == "sentence-transformers":
        embedding_kwargs["model"] = "all-MiniLM-L6-v2"
        embedding_kwargs["dimensions"] = 384
    emb_cfg = EmbeddingConfig(**embedding_kwargs)

    vs_cfg = VectorStoreConfig(provider="faiss", path=os.path.join(data_path, "faiss"))

    return MemoryConfig(
        llm=llm_cfg,
        embedding=emb_cfg,
        vector_store=vs_cfg,
        history_db_path=os.path.join(data_path, "history.db"),
    )


def _get_memory() -> WideMemory:
    global _memory
    if _memory is None:
        _memory = WideMemory(config=_build_config())
    return _memory


server = Server("widemem")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="widemem_add",
            description=(
                "Add a memory. The text is processed by the LLM to extract facts, "
                "resolve conflicts with existing memories, and store them."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text containing facts to remember (e.g. 'I live in San Francisco and work as an engineer')",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier to scope memories",
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="widemem_search",
            description=(
                "Search memories by semantic similarity. Returns the most relevant "
                "memories ranked by similarity, importance, and recency."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier to scope the search",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 100)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="widemem_delete",
            description="Delete a memory by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The UUID of the memory to delete",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        types.Tool(
            name="widemem_count",
            description="Count stored memories, optionally filtered by user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier to filter the count",
                    },
                },
            },
        ),
        types.Tool(
            name="widemem_pin",
            description=(
                "Pin a critical fact with elevated importance (9.0). Use when the user "
                "explicitly asks to remember something, corrects a forgotten fact, or "
                "for YMYL (health, financial, legal) information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The fact to pin as a high-importance memory",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier to scope memories",
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="widemem_export",
            description="Export all stored memories as a JSON array, optionally filtered by user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Optional user identifier to filter the export",
                    },
                },
            },
        ),
        types.Tool(
            name="widemem_health",
            description="Health check — verify the widemem server is running and responsive.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "widemem_add":
        return await _handle_add(arguments)
    elif name == "widemem_search":
        return await _handle_search(arguments)
    elif name == "widemem_delete":
        return await _handle_delete(arguments)
    elif name == "widemem_count":
        return await _handle_count(arguments)
    elif name == "widemem_pin":
        return await _handle_pin(arguments)
    elif name == "widemem_export":
        return await _handle_export(arguments)
    elif name == "widemem_health":
        return await _handle_health(arguments)
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_add(arguments: dict) -> list[types.TextContent]:
    text = arguments.get("text", "")
    user_id = arguments.get("user_id")
    if not text:
        return [types.TextContent(type="text", text=json.dumps({"error": "text is required"}))]
    try:
        mem = _get_memory()
        result = mem.add(text=text, user_id=user_id)
        response = {
            "added": len(result.memories),
            "memories": [
                {"id": m.id, "content": m.content, "importance": m.importance}
                for m in result.memories
            ],
        }
        if result.has_clarifications:
            response["clarifications"] = [
                {"question": c.question, "existing": c.existing_content, "new": c.new_fact}
                for c in result.clarifications
            ]
        return [types.TextContent(type="text", text=json.dumps(response, default=str))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_search(arguments: dict) -> list[types.TextContent]:
    query = arguments.get("query", "")
    user_id = arguments.get("user_id")
    top_k = min(arguments.get("top_k", 5), 100)
    if not query:
        return [types.TextContent(type="text", text=json.dumps({"error": "query is required"}))]
    try:
        mem = _get_memory()
        results = mem.search(query=query, user_id=user_id, top_k=top_k)
        response = {
            "count": len(results),
            "confidence": results.confidence.value if hasattr(results, "confidence") else "unknown",
            "memories": [
                {
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "importance": r.memory.importance,
                    "score": r.final_score if r.final_score else r.similarity_score,
                }
                for r in results
            ],
        }
        return [types.TextContent(type="text", text=json.dumps(response, default=str))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_delete(arguments: dict) -> list[types.TextContent]:
    memory_id = arguments.get("memory_id", "")
    if not memory_id:
        return [types.TextContent(type="text", text=json.dumps({"error": "memory_id is required"}))]
    try:
        mem = _get_memory()
        mem.delete(memory_id)
        return [types.TextContent(type="text", text=json.dumps({"deleted": memory_id}))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_count(arguments: dict) -> list[types.TextContent]:
    user_id = arguments.get("user_id")
    try:
        mem = _get_memory()
        count = mem.count(user_id=user_id)
        return [types.TextContent(type="text", text=json.dumps({"count": count}))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_pin(arguments: dict) -> list[types.TextContent]:
    text = arguments.get("text", "")
    user_id = arguments.get("user_id")
    if not text:
        return [types.TextContent(type="text", text=json.dumps({"error": "text is required"}))]
    try:
        mem = _get_memory()
        result = mem.pin(text=text, user_id=user_id)
        response = {
            "pinned": len(result.memories),
            "memories": [
                {"id": m.id, "content": m.content, "importance": m.importance}
                for m in result.memories
            ],
        }
        return [types.TextContent(type="text", text=json.dumps(response, default=str))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_export(arguments: dict) -> list[types.TextContent]:
    user_id = arguments.get("user_id")
    try:
        mem = _get_memory()
        data = mem.export_json(user_id=user_id)
        return [types.TextContent(type="text", text=data)]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_health(arguments: dict) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps({"status": "ok"}))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
