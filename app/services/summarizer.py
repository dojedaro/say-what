"""Transcript summarization (summary, key points, topics)."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.llm import LLMClient, LLMError, create_llm_client, parse_json_object

SUMMARY_SYSTEM_PROMPT = """You are a content analyst. Analyze the provided transcript and return a JSON object with:
1. "summary": A concise 2-3 paragraph summary of the main content
2. "key_points": An array of 5-7 key takeaways
3. "topics": An array of 3-5 main topics/themes discussed

Respond ONLY with valid JSON, no markdown or other formatting."""


@dataclass
class SummaryResult:
    summary: str
    key_points: list[str]
    topics: list[str]
    tokens_used: int


class Summarizer:
    """Generates a structured summary of a transcript with any supported LLM."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @classmethod
    def create(
        cls,
        provider: str,
        api_key: str,
        model: str,
        azure_endpoint: str | None = None,
    ) -> Summarizer:
        return cls(create_llm_client(provider, api_key, model, azure_endpoint))

    def summarize(self, text: str) -> SummaryResult:
        result = self.llm.complete(
            SUMMARY_SYSTEM_PROMPT,
            f"Analyze this transcript:\n\n{text}",
            json_output=True,
        )
        data = parse_json_object(result.text)
        if not isinstance(data.get("summary"), str):
            raise LLMError("Summary response is missing the 'summary' field")

        return SummaryResult(
            summary=data["summary"],
            key_points=_as_str_list(data.get("key_points")),
            topics=_as_str_list(data.get("topics")),
            tokens_used=result.tokens_used,
        )


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
