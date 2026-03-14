from __future__ import annotations

import json

from widemem.core.exceptions import ProviderError
from widemem.core.types import LLMConfig
from widemem.providers.llm.base import BaseLLM


class AnthropicLLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ProviderError("Install anthropic: pip install widemem[anthropic]")
        self.client = Anthropic(api_key=config.api_key.get_secret_value() if config.api_key else None)

    def _generate(self, prompt: str, system: str | None = None) -> str:
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if self.config.temperature > 0:
            kwargs["temperature"] = self.config.temperature

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
