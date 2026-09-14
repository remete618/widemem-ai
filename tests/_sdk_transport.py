"""Helpers for driving a provider SDK through a mocked HTTP transport.

Mocking the client is not good enough. A `MagicMock` accepts any keyword, so
such a test passes while the real SDK raises `TypeError` on a parameter it
does not take. That is how anthropic 1.0 removing `temperature` reached a
green CI run. Mocking one layer lower puts the installed SDK back in the
path, where it validates every call.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


def http_module_for(client: Any):
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


def mocked_client(factory: Callable[..., Any], handler: Callable[..., Any], **kwargs: Any):
    """An SDK client whose requests are served by `handler(http_module, request)`."""
    hx = http_module_for(factory(api_key="sk-test", **kwargs))
    return factory(
        api_key="sk-test",
        http_client=hx.Client(transport=hx.MockTransport(lambda req: handler(hx, req))),
        **kwargs,
    )
