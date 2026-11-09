"""Provider-agnostic LLM client.

Every feature in the app (summaries, query translation, grounded chat) needs the
same thing from a model: "given a system prompt and a user message, return text".
This module hides the four provider SDKs behind that single interface so the
rest of the codebase never branches on provider names.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Effort = Literal["low", "medium", "high"]

DEFAULT_MAX_TOKENS = 16_000
AZURE_API_VERSION = "2024-10-21"

# Anthropic models that support server-side refusal fallbacks ("default" routing).
_ANTHROPIC_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


class LLMError(RuntimeError):
    """Raised when a provider call fails or returns no usable text."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    tokens_used: int = 0


class LLMClient(ABC):
    """Minimal interface every provider adapter implements."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_output: bool = False,
        effort: Effort = "medium",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResult:
        """Run one system+user turn and return the model's text."""


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str, *, azure_endpoint: str | None = None):
        super().__init__(model)
        if azure_endpoint:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                api_key=api_key,
                api_version=AZURE_API_VERSION,
                azure_endpoint=azure_endpoint,
            )
        else:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)

    def complete(self, system, user, *, json_output=False, effort="medium", max_tokens=DEFAULT_MAX_TOKENS):
        kwargs = {}
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=max_tokens,
            **kwargs,
        )
        text = (response.choices[0].message.content or "").strip()
        tokens = response.usage.total_tokens if response.usage else 0
        return _require_text(LLMResult(text, tokens), self.model)


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        super().__init__(model)
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)

    def complete(self, system, user, *, json_output=False, effort="medium", max_tokens=DEFAULT_MAX_TOKENS):
        if json_output:
            system += "\n\nRespond with a single JSON object and nothing else."
        kwargs = {}
        if self.model in _ANTHROPIC_FALLBACK_MODELS:
            # If a safety classifier declines, the API retries on a fallback model.
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"{self.model} declined the request")

        # Current models may return thinking blocks before the answer,
        # so never assume content[0] is the text.
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return _require_text(LLMResult(text, tokens), self.model)


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        super().__init__(model)
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def complete(self, system, user, *, json_output=False, effort="medium", max_tokens=DEFAULT_MAX_TOKENS):
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_output else None,
        )
        response = self._client.models.generate_content(model=self.model, contents=user, config=config)
        tokens = (response.usage_metadata.total_token_count or 0) if response.usage_metadata else 0
        return _require_text(LLMResult((response.text or "").strip(), tokens), self.model)


def create_llm_client(
    provider: str,
    api_key: str,
    model: str,
    azure_endpoint: str | None = None,
) -> LLMClient:
    """Build the adapter for ``provider`` (openai, anthropic, google, azure)."""
    if provider == "openai":
        return OpenAIClient(api_key, model)
    if provider == "azure":
        if not azure_endpoint:
            raise ValueError("Azure endpoint required for Azure provider")
        return OpenAIClient(api_key, model, azure_endpoint=azure_endpoint)
    if provider == "anthropic":
        return AnthropicClient(api_key, model)
    if provider == "google":
        return GeminiClient(api_key, model)
    raise ValueError(f"Unknown provider: {provider}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating markdown fences and preamble."""
    candidates = [text]
    if match := _FENCE.search(text):
        candidates.insert(0, match.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMError("Model did not return a valid JSON object")


def _require_text(result: LLMResult, model: str) -> LLMResult:
    if not result.text:
        raise LLMError(f"{model} returned an empty response")
    return result
