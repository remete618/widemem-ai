from __future__ import annotations

import json

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params.response_format_json_object import ResponseFormatJSONObject

from widemem.core.exceptions import ProviderError
from widemem.core.types import LLMConfig
from widemem.providers.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key.get_secret_value() if config.api_key else None,
            base_url=config.base_url,
        )

    def _generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ProviderError("LLM returned empty response")
        return content

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response_format: ResponseFormatJSONObject = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format=response_format,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ProviderError("LLM returned empty response")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ProviderError(f"LLM returned invalid JSON: {e}") from e
