"""Tests for the fail-closed auth policy in widemem.server.

The HTTP server's auth dependency disables itself when WIDEMEM_API_KEY is
unset (zero-config local dev). That is only safe on a loopback bind.
_enforce_auth_policy() must refuse to start when the server is bound to a
non-local interface without an API key, so a public deployment cannot
silently expose /search and /add.
"""

from __future__ import annotations

import pytest

# widemem.server imports fastapi at module level. Skip cleanly if the
# [server] extra is not installed (local dev without it). CI installs
# [server] so this security policy is actually exercised there.
pytest.importorskip("fastapi")

from widemem.server import _enforce_auth_policy  # noqa: E402


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("WIDEMEM_HOST", raising=False)
    monkeypatch.delenv("WIDEMEM_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Must FAIL CLOSED: public bind, no key
# ---------------------------------------------------------------------------
def test_public_bind_no_key_raises(monkeypatch):
    monkeypatch.setenv("WIDEMEM_HOST", "0.0.0.0")
    with pytest.raises(RuntimeError, match="Refusing to start"):
        _enforce_auth_policy()


def test_specific_public_ip_no_key_raises(monkeypatch):
    monkeypatch.setenv("WIDEMEM_HOST", "10.0.0.5")
    with pytest.raises(RuntimeError, match="WIDEMEM_API_KEY is not set"):
        _enforce_auth_policy()


# ---------------------------------------------------------------------------
# Must ALLOW: public bind WITH key
# ---------------------------------------------------------------------------
def test_public_bind_with_key_ok(monkeypatch):
    monkeypatch.setenv("WIDEMEM_HOST", "0.0.0.0")
    monkeypatch.setenv("WIDEMEM_API_KEY", "a-strong-secret")
    _enforce_auth_policy()  # must not raise


# ---------------------------------------------------------------------------
# Must ALLOW: local bind, no key (zero-config dev preserved)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_local_bind_no_key_ok(monkeypatch, host):
    monkeypatch.setenv("WIDEMEM_HOST", host)
    _enforce_auth_policy()  # must not raise (warning only)


def test_default_host_no_key_ok(monkeypatch):
    """No WIDEMEM_HOST set at all defaults to 127.0.0.1 -> local -> allowed."""
    _enforce_auth_policy()  # must not raise


# ---------------------------------------------------------------------------
# Must ALLOW: local bind WITH key
# ---------------------------------------------------------------------------
def test_local_bind_with_key_ok(monkeypatch):
    monkeypatch.setenv("WIDEMEM_HOST", "127.0.0.1")
    monkeypatch.setenv("WIDEMEM_API_KEY", "secret")
    _enforce_auth_policy()  # must not raise


# ---------------------------------------------------------------------------
# Warning emitted on local bind without key (dev visibility)
# ---------------------------------------------------------------------------
def test_local_no_key_warns(monkeypatch, caplog):
    monkeypatch.setenv("WIDEMEM_HOST", "127.0.0.1")
    with caplog.at_level("WARNING", logger="widemem.server"):
        _enforce_auth_policy()
    assert any(
        "authentication is disabled" in r.message for r in caplog.records
    )


def test_public_bind_with_key_no_warning(monkeypatch, caplog):
    monkeypatch.setenv("WIDEMEM_HOST", "0.0.0.0")
    monkeypatch.setenv("WIDEMEM_API_KEY", "secret")
    with caplog.at_level("WARNING", logger="widemem.server"):
        _enforce_auth_policy()
    assert not any(
        "authentication is disabled" in r.message for r in caplog.records
    )
