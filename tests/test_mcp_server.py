"""Tests for the MCP server tool handlers: top_k clamping and error masking."""

from __future__ import annotations

import json

import widemem.mcp_server as mcp_server


class _CaptureMemory:
    """Fake WideMemory that records the top_k it was called with."""

    def __init__(self):
        self.last_top_k = None

    def search(self, query, user_id, top_k):
        self.last_top_k = top_k
        return []


class _RaisingMemory:
    def search(self, query, user_id, top_k):
        raise RuntimeError("connect failed: postgresql://user:secret@10.0.0.5/db /home/app/.widemem")


async def test_search_clamps_negative_top_k(monkeypatch):
    fake = _CaptureMemory()
    monkeypatch.setattr(mcp_server, "_get_memory", lambda: fake)
    await mcp_server._handle_search({"query": "hi", "top_k": -5})
    assert fake.last_top_k == 1


async def test_search_clamps_upper_bound(monkeypatch):
    fake = _CaptureMemory()
    monkeypatch.setattr(mcp_server, "_get_memory", lambda: fake)
    await mcp_server._handle_search({"query": "hi", "top_k": 10_000})
    assert fake.last_top_k == 100


async def test_search_zero_top_k_clamped_to_one(monkeypatch):
    fake = _CaptureMemory()
    monkeypatch.setattr(mcp_server, "_get_memory", lambda: fake)
    await mcp_server._handle_search({"query": "hi", "top_k": 0})
    assert fake.last_top_k == 1


async def test_search_masks_internal_error(monkeypatch):
    monkeypatch.setattr(mcp_server, "_get_memory", lambda: _RaisingMemory())
    result = await mcp_server._handle_search({"query": "hi"})
    payload = json.loads(result[0].text)
    assert payload == {"error": "internal error"}
    # The real exception text (paths / connection strings) must not leak.
    assert "secret" not in result[0].text
    assert "10.0.0.5" not in result[0].text
    assert "/home/app" not in result[0].text
