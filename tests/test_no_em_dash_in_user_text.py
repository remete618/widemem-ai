"""User-facing text (response templates, uncertainty and frustration messages,
MCP tool descriptions) uses plain punctuation, never an em dash."""
from __future__ import annotations

import inspect

import pytest

import widemem.retrieval.responses as responses
import widemem.retrieval.uncertainty as uncertainty

EM_DASH = "—"


def _template_strings():
    for name, value in vars(responses).items():
        if name.isupper() and isinstance(value, (list, tuple)):
            yield from (s for s in value if isinstance(s, str))


def test_response_templates_have_no_em_dash():
    strings = list(_template_strings())
    assert strings, "no templates found; update this test"
    assert not [s for s in strings if EM_DASH in s]


def test_uncertainty_messages_have_no_em_dash():
    assert EM_DASH not in inspect.getsource(uncertainty)


def test_mcp_tool_descriptions_have_no_em_dash():
    mcp_server = pytest.importorskip("widemem.mcp_server")
    descriptions = [t.description or "" for t in mcp_server.TOOLS]
    assert descriptions
    assert not [d for d in descriptions if EM_DASH in d]
