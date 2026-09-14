"""One contract every `BaseLLM` implementation must satisfy.

Each provider is driven through its real SDK with the HTTP transport mocked,
so the SDK validates the call the way it would in production. A `MagicMock`
client accepts any keyword and would pass whatever the provider sent, which is
exactly how anthropic 1.0 removing `temperature` reached a green CI run.

Provider-specific behaviour lives in `test_llm_providers.py`. What is pinned
here is only what all three owe a caller of `generate()` / `generate_json()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import patch

import pytest

from tests._sdk_transport import http_module_for, mocked_client
from widemem.core.exceptions import ProviderError
from widemem.core.types import LLMConfig


@dataclass(frozen=True)
class ProviderCase:
    """How to stand one provider up against a mocked transport."""

    name: str
    build: Callable[[Callable[..., Any], float], Any]
    reply: Callable[[Any, str], Any]
    empty_reply: Callable[[Any], Any]
    wire_model: Callable[[dict], str]

    def __str__(self) -> str:  # keeps parametrize ids readable
        return self.name


# --- openai ----------------------------------------------------------------
def _openai_body(hx, content):
    return hx.Response(200, json={
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def _build_openai(handler, temperature):
    import openai

    from widemem.providers.llm.openai import OpenAILLM

    client = mocked_client(openai.OpenAI, handler)
    with patch("widemem.providers.llm.openai.OpenAI", return_value=client):
        return OpenAILLM(LLMConfig(model="gpt-4o-mini", api_key="sk-test",
                                   temperature=temperature, max_tokens=64))


# --- anthropic -------------------------------------------------------------
def _anthropic_body(hx, content):
    return hx.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
        "content": content, "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })


def _build_anthropic(handler, temperature):
    import anthropic

    from widemem.providers.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM(LLMConfig(model="claude-sonnet-4-5", api_key="sk-test",
                                 temperature=temperature, max_tokens=64))
    llm.client = mocked_client(anthropic.Anthropic, handler)
    return llm


# --- ollama ----------------------------------------------------------------
def _ollama_body(hx, content):
    return hx.Response(200, json={
        "model": "llama3", "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True, "done_reason": "stop",
    })


def _build_ollama(handler, temperature):
    from widemem.providers.llm.ollama import OllamaLLM

    llm = OllamaLLM(LLMConfig(provider="ollama", model="llama3",
                              temperature=temperature, max_tokens=64))
    # The ollama client builds its own transport rather than taking one, so
    # the mock is installed on the client it already made.
    hx = http_module_for(llm.client)
    llm.client._client = hx.Client(
        base_url="http://localhost:11434",
        transport=hx.MockTransport(lambda req: handler(hx, req)),
    )
    return llm


CASES = [
    pytest.param(
        ProviderCase("openai", _build_openai,
                     lambda hx, text: _openai_body(hx, text),
                     lambda hx: _openai_body(hx, None),
                     lambda body: body["model"]),
        id="openai",
    ),
    pytest.param(
        ProviderCase("anthropic", _build_anthropic,
                     lambda hx, text: _anthropic_body(hx, [{"type": "text", "text": text}]),
                     lambda hx: _anthropic_body(hx, []),
                     lambda body: body["model"]),
        id="anthropic",
    ),
    pytest.param(
        ProviderCase("ollama", _build_ollama,
                     lambda hx, text: _ollama_body(hx, text),
                     lambda hx: _ollama_body(hx, ""),
                     lambda body: body["model"]),
        id="ollama",
    ),
]


@pytest.fixture(params=CASES)
def case(request):
    name = request.param.name
    if name == "anthropic":
        pytest.importorskip("anthropic", reason="anthropic extra not installed")
    if name == "ollama":
        pytest.importorskip("ollama", reason="ollama extra not installed")
    return request.param


def _llm(case, text, temperature=0.4, capture=None):
    def handler(hx, req):
        if capture is not None:
            capture.update(json.loads(req.content))
        return case.reply(hx, text)

    return case.build(handler, temperature)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------
def test_generate_returns_the_model_text(case):
    assert _llm(case, "hello there").generate("hi") == "hello there"


def test_a_configured_temperature_does_not_break_the_call(case):
    """anthropic 1.0 removed `temperature` and the provider kept sending it.

    Every affected call raised `TypeError`, which the retry layer treated as
    transient. Each provider must accept a non-default temperature without
    sending its SDK something the SDK rejects.
    """
    assert _llm(case, "ok", temperature=0.7).generate("hi") == "ok"


def test_the_configured_model_reaches_the_wire(case):
    sent: dict = {}
    llm = _llm(case, "ok", capture=sent)
    llm.generate("hi")
    assert case.wire_model(sent) == llm.config.model


def test_the_prompt_reaches_the_wire_intact(case):
    """Unicode must survive encoding; a mangled prompt is a silent quality bug."""
    sent: dict = {}
    prompt = "wer wohnt in Wien? 好的 \U0001f600"
    _llm(case, "ok", capture=sent).generate(prompt)
    assert prompt in json.dumps(sent, ensure_ascii=False)


@pytest.mark.parametrize(
    "wrapped",
    [
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        '  {"a": 1}  ',
        # A doubled fence: both peels have to run, which is why the helper
        # uses two ifs rather than if/elif.
        '```json```{"a": 1}```',
    ],
)
def test_generate_json_parses_through_fences_and_whitespace(case, wrapped):
    assert _llm(case, wrapped).generate_json("hi") == {"a": 1}


def test_generate_json_rejects_text_that_is_not_json(case):
    with pytest.raises(ProviderError):
        _llm(case, "sorry, I cannot do that").generate_json("hi")


def test_an_empty_model_response_raises_provider_error(case):
    """Not a bare exception, and not an empty string handed to the caller."""

    def handler(hx, req):
        return case.empty_reply(hx)

    llm = case.build(handler, 0.4)
    with pytest.raises(ProviderError):
        llm.generate("hi")


def test_every_concrete_provider_is_covered():
    """A provider added to the package without a case here would drift unseen."""
    import importlib
    import pkgutil

    import widemem.providers.llm as package
    from widemem.providers.llm.base import BaseLLM

    concrete = set()
    for info in pkgutil.iter_modules(package.__path__):
        if info.name == "base":
            continue
        try:
            module = importlib.import_module(f"{package.__name__}.{info.name}")
        except Exception:  # an optional SDK that is not installed
            concrete.add(info.name)
            continue
        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, BaseLLM) and attr is not BaseLLM:
                if attr.__module__ == module.__name__:
                    concrete.add(info.name)

    covered = {c.values[0].name for c in CASES}
    assert concrete == covered, (
        f"providers with no contract case: {sorted(concrete - covered)}; "
        f"cases with no provider: {sorted(covered - concrete)}"
    )
