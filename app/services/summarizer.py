"""AI summarization service supporting multiple providers."""

import json
from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SummaryResult:
    """Result of summarization."""
    summary: str
    key_points: list[str]
    topics: list[str]
    tokens_used: int


class BaseSummarizer(ABC):
    """Base class for summarizers."""

    @abstractmethod
    def summarize(self, text: str, language: str = "en") -> SummaryResult:
        """Generate summary, key points, and topics."""
        pass


class OpenAISummarizer(BaseSummarizer):
    """OpenAI-based summarizer."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def summarize(self, text: str, language: str = "en") -> SummaryResult:
        system_prompt = """You are a content analyst. Analyze the provided transcript and return a JSON object with:
1. "summary": A concise 2-3 paragraph summary of the main content
2. "key_points": An array of 5-7 key takeaways as bullet points
3. "topics": An array of 3-5 main topics/themes discussed

Respond ONLY with valid JSON, no markdown or other formatting."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this transcript:\n\n{text}"},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        tokens = response.usage.total_tokens if response.usage else 0

        return SummaryResult(
            summary=result.get("summary", ""),
            key_points=result.get("key_points", []),
            topics=result.get("topics", []),
            tokens_used=tokens,
        )


class AnthropicSummarizer(BaseSummarizer):
    """Anthropic Claude-based summarizer."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def summarize(self, text: str, language: str = "en") -> SummaryResult:
        system_prompt = """You are a content analyst. Analyze the provided transcript and return a JSON object with:
1. "summary": A concise 2-3 paragraph summary of the main content
2. "key_points": An array of 5-7 key takeaways as bullet points
3. "topics": An array of 3-5 main topics/themes discussed

Respond ONLY with valid JSON, no markdown or other formatting."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Analyze this transcript:\n\n{text}"},
            ],
        )

        # Parse the response
        content = response.content[0].text
        # Try to extract JSON from potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        tokens = response.usage.input_tokens + response.usage.output_tokens

        return SummaryResult(
            summary=result.get("summary", ""),
            key_points=result.get("key_points", []),
            topics=result.get("topics", []),
            tokens_used=tokens,
        )


class GeminiSummarizer(BaseSummarizer):
    """Google Gemini-based summarizer."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def summarize(self, text: str, language: str = "en") -> SummaryResult:
        prompt = """Analyze the following transcript and return a JSON object with:
1. "summary": A concise 2-3 paragraph summary of the main content
2. "key_points": An array of 5-7 key takeaways as bullet points
3. "topics": An array of 3-5 main topics/themes discussed

Respond ONLY with valid JSON, no markdown or other formatting.

Transcript:
""" + text

        response = self.model.generate_content(prompt)

        # Parse the response
        content = response.text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())

        return SummaryResult(
            summary=result.get("summary", ""),
            key_points=result.get("key_points", []),
            topics=result.get("topics", []),
            tokens_used=0,  # Gemini doesn't easily expose token count
        )


class AzureSummarizer(BaseSummarizer):
    """Azure OpenAI-based summarizer."""

    def __init__(self, api_key: str, endpoint: str, model: str = "gpt-4o"):
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint,
        )
        self.model = model

    def summarize(self, text: str, language: str = "en") -> SummaryResult:
        system_prompt = """You are a content analyst. Analyze the provided transcript and return a JSON object with:
1. "summary": A concise 2-3 paragraph summary of the main content
2. "key_points": An array of 5-7 key takeaways as bullet points
3. "topics": An array of 3-5 main topics/themes discussed

Respond ONLY with valid JSON, no markdown or other formatting."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this transcript:\n\n{text}"},
            ],
            temperature=0.3,
        )

        result = json.loads(response.choices[0].message.content)
        tokens = response.usage.total_tokens if response.usage else 0

        return SummaryResult(
            summary=result.get("summary", ""),
            key_points=result.get("key_points", []),
            topics=result.get("topics", []),
            tokens_used=tokens,
        )


class Summarizer:
    """Factory for creating summarizers based on provider."""

    @staticmethod
    def create(
        provider: str,
        api_key: str,
        model: str,
        azure_endpoint: Optional[str] = None,
    ) -> BaseSummarizer:
        """
        Create a summarizer for the specified provider.

        Args:
            provider: Provider name (openai, anthropic, google, azure)
            api_key: API key
            model: Model name
            azure_endpoint: Azure endpoint (required for azure provider)

        Returns:
            Summarizer instance
        """
        if provider == "openai":
            return OpenAISummarizer(api_key, model)
        elif provider == "anthropic":
            return AnthropicSummarizer(api_key, model)
        elif provider == "google":
            return GeminiSummarizer(api_key, model)
        elif provider == "azure":
            if not azure_endpoint:
                raise ValueError("Azure endpoint required for Azure provider")
            return AzureSummarizer(api_key, azure_endpoint, model)
        else:
            raise ValueError(f"Unknown provider: {provider}")
