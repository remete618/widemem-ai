"""Tests for the MCP server: tool registration, dispatch, clamping and error masking."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

# widemem.mcp_server imports the `mcp` package at module level, so skip cleanly
# when the [mcp] extra is absent. CI installs it, so these run there.
pytest.importorskip("mcp")

import widemem.mcp_server as mcp_server  # noqa: E402


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


# ---------------------------------------------------------------------------
# Tool registration and dispatch
#
# mcp 2.x removed the `@server.list_tools()` / `@server.call_tool()` decorators
# the 1.x server was built on, so the module did not import at all under 2.x.
# These pin the replacement wiring.
# ---------------------------------------------------------------------------
async def test_list_tools_returns_every_registered_tool():
    result = await mcp_server._on_list_tools(None, None)
    assert [t.name for t in result.tools] == [t.name for t in mcp_server.TOOLS]
    assert result.tools, "a server advertising no tools is useless"


def test_every_tool_declares_an_object_schema():
    for tool in mcp_server.TOOLS:
        assert tool.input_schema.get("type") == "object", (
            f"{tool.name} does not declare an object input schema"
        )
        assert tool.description, f"{tool.name} has no description for the model to read"


async def test_every_advertised_tool_dispatches_somewhere(monkeypatch):
    """A tool in TOOLS with no branch in _dispatch would advertise a dead name.

    The dispatcher's fallback is the only thing that reports an unknown tool,
    so reaching it from a name we advertise is the failure being caught here.
    """
    monkeypatch.setattr(mcp_server, "_get_memory", lambda: _CaptureMemory())
    orphans = []
    for tool in mcp_server.TOOLS:
        content = await mcp_server._dispatch(tool.name, {})
        if content and content[0].text.startswith("Unknown tool:"):
            orphans.append(tool.name)
    assert not orphans, f"advertised but not dispatched: {orphans}"


async def test_unknown_tool_names_the_tool():
    content = await mcp_server._dispatch("widemem_nope", {})
    assert content[0].text == "Unknown tool: widemem_nope"


async def test_call_tool_passes_dispatch_output_through_untouched(monkeypatch):
    """The 2.x handler wraps in CallToolResult; it must not reshape the payload.

    Compared against `_dispatch` rather than a hardcoded body, so this stays
    honest if a handler's response shape changes.
    """
    monkeypatch.setattr(mcp_server, "_get_memory", lambda: _CaptureMemory())
    args = {"query": "hi"}
    direct = await mcp_server._dispatch("widemem_search", args)

    result = await mcp_server._on_call_tool(None, SimpleNamespace(name="widemem_search", arguments=args))

    assert [c.text for c in result.content] == [c.text for c in direct]
    assert json.loads(result.content[0].text)["memories"] == []


async def test_call_tool_tolerates_absent_arguments():
    """`arguments` is optional in the protocol; None must not become a crash."""
    params = SimpleNamespace(name="widemem_health", arguments=None)

    result = await mcp_server._on_call_tool(None, params)

    assert json.loads(result.content[0].text) == {"status": "ok"}


def test_the_server_actually_has_the_handlers_registered():
    """Defining the handlers is not the same as wiring them to the server.

    Every other test in this file calls `_on_list_tools` / `_on_call_tool`
    directly, so dropping them from the `Server(...)` construction leaves the
    suite green and the server answering nothing. Checked against the server
    object rather than the source.
    """
    for method in ("tools/list", "tools/call"):
        assert mcp_server.server.get_request_handler(method) is not None, (
            f"{method} has no handler; the server would reject the request"
        )


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for var in ("WIDEMEM_LLM_PROVIDER", "WIDEMEM_LLM_MODEL", "WIDEMEM_LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WIDEMEM_DATA_PATH", str(tmp_path))
    return monkeypatch


def test_default_openai_config_does_not_point_at_ollama(clean_env):
    cfg = mcp_server._build_config()
    assert cfg.llm.provider == "openai"
    assert cfg.llm.base_url is None


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_hosted_providers_get_no_base_url_by_default(clean_env, provider):
    clean_env.setenv("WIDEMEM_LLM_PROVIDER", provider)
    assert mcp_server._build_config().llm.base_url is None


def test_ollama_leaves_the_host_to_the_provider(clean_env):
    clean_env.setenv("WIDEMEM_LLM_PROVIDER", "ollama")
    assert mcp_server._build_config().llm.base_url is None


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_explicit_base_url_wins(clean_env, provider):
    clean_env.setenv("WIDEMEM_LLM_PROVIDER", provider)
    clean_env.setenv("WIDEMEM_LLM_BASE_URL", "http://gateway.internal:8080/v1")
    assert mcp_server._build_config().llm.base_url == "http://gateway.internal:8080/v1"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_base_url_is_treated_as_unset(clean_env, blank):
    clean_env.setenv("WIDEMEM_LLM_BASE_URL", blank)
    assert mcp_server._build_config().llm.base_url is None
