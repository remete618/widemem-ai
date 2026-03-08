from __future__ import annotations

import json

from widemem.core.exceptions import ProviderError
from widemem.core.types import LLMConfig
from widemem.providers.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            from ollama import Client
        except ImportError:
            raise ProviderError("Install ollama: pip install widemem[ollama]")
        self.client = Client(host=config.base_url or "http://localhost:11434")

    def _generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.config.model,
            messages=messages,
            options={"temperature": self.config.temperature},
        )
        content = response.get("message", {}).get("content", "")
        if not content:
            raise ProviderError("Ollama returned empty response")
        return content

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        json_system = (system or "") + "\n\nYou must respond with valid JSON only. No other text."
        text = self._generate(prompt, system=json_system.strip())

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
            raise ProviderError(f"Ollama returned invalid JSON: {e}") from e
