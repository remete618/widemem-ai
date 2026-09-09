from __future__ import annotations

import inspect
import json
import warnings

from widemem.core.exceptions import ProviderError
from widemem.core.types import LLMConfig
from widemem.providers.llm.base import BaseLLM


class AnthropicLLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ProviderError('Install anthropic: pip install "widemem-ai[anthropic]"')
        self.client = Anthropic(api_key=config.api_key.get_secret_value() if config.api_key else None)
        if config.temperature > 0 and not self._accepts("temperature"):
            import anthropic

            warnings.warn(
                f"anthropic {anthropic.__version__} does not accept a temperature, so the "
                f"configured value ({config.temperature}) is ignored. Set "
                "LLMConfig.temperature to 0 to silence this, or pin anthropic<1 to keep "
                "temperature control.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _accepts(self, name: str) -> bool:
        """Whether the installed SDK's `messages.create` takes this parameter.

        anthropic 1.0 removed `temperature`, `top_p` and `top_k`: the models
        it targets ignore them. Passing one raises `TypeError` before a
        request is built, and `BaseLLM._retry` treats that as transient, so
        the wrong kwarg costs three attempts and two backoff sleeps before
        surfacing as a retry failure rather than a signature error.

        A client that takes `**kwargs` (a test double, or a future SDK that
        forwards unknown parameters) is trusted to accept the name.
        """
        try:
            params = inspect.signature(self.client.messages.create).parameters
        except (TypeError, ValueError):
            return True
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return True
        return name in params

    def _sampling_kwargs(self) -> dict:
        """Sampling parameters the installed SDK will accept.

        Decided per call rather than cached, so replacing `self.client` is
        honoured. Silent by design: the warning for a dropped value fires
        once, when the client is built, rather than on every call of a long
        extraction run.
        """
        if self.config.temperature <= 0 or not self._accepts("temperature"):
            return {}
        return {"temperature": self.config.temperature}

    def _generate(self, prompt: str, system: str | None = None) -> str:
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        kwargs.update(self._sampling_kwargs())

        response = self.client.messages.create(**kwargs)
        if not response.content:
            raise ProviderError("Anthropic returned empty response")
        return response.content[0].text

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        json_prompt = prompt + "\n\nRespond with valid JSON only."
        text = self._generate(json_prompt, system=system)

        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ProviderError(f"Anthropic returned invalid JSON: {e}") from e
