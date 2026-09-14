"""LLM provider calls must match the signature of the SDK actually installed.

These tests mock the HTTP transport, not the client. A `MagicMock` client
accepts any keyword, so a mock-the-client test passes while the real SDK
raises `TypeError` on an unknown parameter. That is how anthropic 1.0
removing `temperature` reached a green CI run: no test constructed a real
client, and the extra was not installed.

Where a specific SDK generation is the subject, a hand-written fake with an
explicit signature stands in, so the assertion does not read from the same
source the provider reads.
"""

from __future__ import annotations

import importlib
import json
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from widemem.core.exceptions import ProviderError
from widemem.core.types import LLMConfig
from widemem.providers.llm.anthropic import AnthropicLLM
from widemem.providers.llm.openai import OpenAILLM

try:
    import anthropic
except ImportError:  # the extra is optional; only the tests below need it
    anthropic = None

# Not `importorskip`: that skips the whole module, taking the OpenAI tests and
# the benchmark guard with it. Losing tests quietly is the failure this file
# exists to catch.
needs_anthropic = pytest.mark.skipif(anthropic is None, reason="anthropic extra not installed")


def _http_module(client):
    """The http library the installed SDK is built on.

    anthropic >= 1 and openai >= 2 moved from `httpx` to `httpx2`, and each
    rejects the other package's client. Read the answer off the SDK's own
    transport class rather than guessing.
    """
    for base in type(client._client).__mro__:
        root = base.__module__.split(".")[0]
        if root in ("httpx", "httpx2"):
            return importlib.import_module(root)
    raise AssertionError(f"no httpx flavour found in {type(client._client).__mro__}")


def _mocked(factory, handler):
    """An SDK client whose requests are served by `handler`."""
    hx = _http_module(factory(api_key="sk-test"))
    client = factory(
        api_key="sk-test",
        http_client=hx.Client(transport=hx.MockTransport(lambda req: handler(hx, req))),
    )
    return client


def _config(**kw):
    return LLMConfig(model=kw.pop("model", "claude-sonnet-4-5"), api_key="sk-test", **kw)


# ---------------------------------------------------------------------------
# Anthropic, against the installed SDK
# ---------------------------------------------------------------------------
def _anthropic_reply(hx, req, text='{"ok": true}', content=None):
    body = [{"type": "text", "text": text}] if content is None else content
    return hx.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
        "content": body, "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })


@needs_anthropic
def test_anthropic_generate_survives_a_nonzero_temperature():
    """The regression: anthropic 1.0 dropped `temperature` from `messages.create`.

    The provider passed it whenever it was above zero, so every call raised
    `TypeError`, which `BaseLLM._retry` retried twice with backoff before
    reporting a retry failure. Default temperature is 0, so only configs that
    set one were affected.
    """
    sent = {}

    def handler(hx, req):
        sent.update(json.loads(req.content))
        return _anthropic_reply(hx, req, text="hello")

    llm = AnthropicLLM(_config(temperature=0.7, max_tokens=64))
    llm.client = _mocked(anthropic.Anthropic, handler)

    assert llm.generate("hi", system="sys") == "hello"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["system"] == "sys"


@needs_anthropic
def test_anthropic_empty_content_raises_rather_than_indexing():
    def handler(hx, req):
        return _anthropic_reply(hx, req, content=[])

    llm = AnthropicLLM(_config(temperature=0.5))
    llm.client = _mocked(anthropic.Anthropic, handler)

    with pytest.raises(ProviderError, match="empty response"):
        llm.generate("hi")


@needs_anthropic
@pytest.mark.parametrize("wrapped", ['```json\n{"a": 1}\n```', '```\n{"a": 1}\n```', '{"a": 1}'])
def test_anthropic_generate_json_strips_code_fences(wrapped):
    def handler(hx, req):
        return _anthropic_reply(hx, req, text=wrapped)

    llm = AnthropicLLM(_config(temperature=0.5))
    llm.client = _mocked(anthropic.Anthropic, handler)

    assert llm.generate_json("hi") == {"a": 1}


@needs_anthropic
def test_anthropic_generate_json_rejects_malformed_json():
    def handler(hx, req):
        return _anthropic_reply(hx, req, text="not json at all")

    llm = AnthropicLLM(_config(temperature=0.5))
    llm.client = _mocked(anthropic.Anthropic, handler)

    with pytest.raises(ProviderError, match="invalid JSON"):
        llm.generate_json("hi")


# ---------------------------------------------------------------------------
# Anthropic, against hand-written signatures
# ---------------------------------------------------------------------------
class _FakeMessages:
    """A `messages` namespace whose `create` has an explicit signature.

    The signature is written here rather than taken from an installed SDK,
    so these tests pin both branches on any machine.
    """

    def __init__(self, *, takes_temperature: bool, reply_text: str = "ok") -> None:
        self.calls: list[dict] = []
        reply = SimpleNamespace(content=[SimpleNamespace(text=reply_text)])

        if takes_temperature:
            def create(*, model, messages, max_tokens, system=None, temperature=None):
                self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens,
                                   "system": system, "temperature": temperature})
                return reply
        else:
            def create(*, model, messages, max_tokens, system=None):
                self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens,
                                   "system": system})
                return reply

        self.create = create


def _llm_with_fake(*, takes_temperature, temperature=0.7, reply_text="ok"):
    """Build a provider on a fake client, returning warnings raised while building."""
    messages = _FakeMessages(takes_temperature=takes_temperature, reply_text=reply_text)
    fake = SimpleNamespace(messages=messages)
    with patch.object(anthropic, "Anthropic", return_value=fake):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            llm = AnthropicLLM(_config(temperature=temperature))
    return llm, messages, list(caught)


@needs_anthropic
def test_temperature_is_sent_when_the_sdk_signature_takes_it():
    llm, messages, caught = _llm_with_fake(takes_temperature=True)

    assert llm.generate("hi") == "ok"
    assert messages.calls[0]["temperature"] == 0.7
    assert not caught, "nothing to warn about when the SDK takes the parameter"


@needs_anthropic
def test_temperature_is_dropped_when_the_sdk_signature_omits_it():
    llm, messages, caught = _llm_with_fake(takes_temperature=False)

    assert llm.generate("hi") == "ok"
    assert "temperature" not in messages.calls[0]
    assert len(caught) == 1, "dropping a configured value silently is worse than the crash"
    assert "temperature" in str(caught[0].message)


@needs_anthropic
def test_the_dropped_temperature_warning_does_not_repeat_per_call():
    """Warned once when the client is built; silent for the rest of its life.

    A warning on every call would bury a long extraction run in its own
    output, and the condition cannot change between calls.
    """
    llm, messages, at_construction = _llm_with_fake(takes_temperature=False, reply_text='{"a": 1}')
    assert len(at_construction) == 1

    with warnings.catch_warnings(record=True) as during_calls:
        warnings.simplefilter("always")
        llm.generate("one")
        llm.generate("two")
        llm.generate_json("give me json")

    assert len(messages.calls) == 3
    assert not during_calls


@needs_anthropic
@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_zero_or_negative_temperature_is_never_sent(temperature):
    """Zero means "leave the default alone", so the parameter stays off the call."""
    llm, messages, caught = _llm_with_fake(takes_temperature=True, temperature=temperature)

    assert llm.generate("hi") == "ok"
    assert messages.calls[0]["temperature"] is None
    assert not caught


@needs_anthropic
def test_a_client_taking_kwargs_still_receives_temperature():
    """Test doubles and forwarding clients take `**kwargs`; trust them."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text="ok")])

    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    with patch.object(anthropic, "Anthropic", return_value=fake):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            llm = AnthropicLLM(_config(temperature=0.4))

    assert llm.generate("hi") == "ok"
    assert calls[0]["temperature"] == 0.4
    assert not caught


# ---------------------------------------------------------------------------
# OpenAI, against the installed SDK
# ---------------------------------------------------------------------------
def _openai_reply(hx, content):
    return hx.Response(200, json={
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def _openai_llm(handler, **cfg):
    import openai

    client = _mocked(openai.OpenAI, handler)
    with patch("widemem.providers.llm.openai.OpenAI", return_value=client):
        return OpenAILLM(LLMConfig(model="gpt-4o-mini", api_key="sk-test", **cfg))


def test_openai_generate_reaches_the_wire_with_its_sampling_parameters():
    sent = {}

    def handler(hx, req):
        sent.update(json.loads(req.content))
        return _openai_reply(hx, "hello")

    llm = _openai_llm(handler, temperature=0.3, max_tokens=64)

    assert llm.generate("hi", system="sys") == "hello"
    assert sent["temperature"] == 0.3
    assert sent["max_tokens"] == 64
    assert sent["messages"][0] == {"role": "system", "content": "sys"}


def test_openai_generate_json_asks_for_a_json_object():
    sent = {}

    def handler(hx, req):
        sent.update(json.loads(req.content))
        return _openai_reply(hx, '{"a": 1}')

    llm = _openai_llm(handler, temperature=0.0)

    assert llm.generate_json("hi") == {"a": 1}
    assert sent["response_format"] == {"type": "json_object"}


def test_openai_null_content_raises_rather_than_returning_none():
    llm = _openai_llm(lambda hx, req: _openai_reply(hx, None))

    with pytest.raises(ProviderError, match="empty response"):
        llm.generate("hi")


def test_openai_invalid_json_raises():
    llm = _openai_llm(lambda hx, req: _openai_reply(hx, "definitely not json"))

    with pytest.raises(ProviderError, match="invalid JSON"):
        llm.generate_json("hi")


# ---------------------------------------------------------------------------
# The benchmark harnesses must not name an http library
# ---------------------------------------------------------------------------
def test_benchmark_harnesses_do_not_import_an_http_library():
    """`import httpx` in a harness is a dependency the project never declared.

    Three harnesses built an `httpx.Client` to hand the OpenAI client a
    timeout. openai >= 2 is built on `httpx2` and rejects an `httpx` client,
    and a fresh install no longer brings `httpx` at all, so the import
    failed before the mismatch could. `OpenAI(timeout=...)` takes a float and
    works on every generation.
    """
    root = Path(__file__).resolve().parent.parent
    offenders = [
        path.name
        for path in sorted((root / "benchmark").glob("*.py"))
        if re.search(r"^\s*(import|from)\s+httpx2?\b", path.read_text(encoding="utf-8"), re.M)
    ]
    assert not offenders, (
        f"{offenders} import an http library directly. The installed SDK owns that "
        "choice: openai < 2 uses httpx, openai >= 2 uses httpx2, and neither accepts "
        "the other's client. Pass a plain timeout instead."
    )
