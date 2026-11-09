from types import SimpleNamespace

import pytest

from app.services.llm import AnthropicClient, LLMError, create_llm_client, parse_json_object
from app.services.summarizer import Summarizer
from tests.conftest import FakeLLM


class TestParseJsonObject:
    def test_plain_json(self):
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert parse_json_object('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_preamble(self):
        assert parse_json_object('Sure! {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}

    @pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", "{broken"])
    def test_rejects_non_objects(self, text):
        with pytest.raises(LLMError):
            parse_json_object(text)


class TestFactory:
    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client("nope", "key", "model")

    def test_azure_requires_endpoint(self):
        with pytest.raises(ValueError, match="Azure endpoint"):
            create_llm_client("azure", "key", "model")


def _anthropic_with_response(model: str, response) -> tuple[AnthropicClient, dict]:
    client = AnthropicClient("sk-ant-test", model)
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return response

    client._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    return client, captured


def _message(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class TestAnthropicClient:
    def test_skips_thinking_blocks(self):
        # Current Claude models can return a thinking block before the answer.
        response = _message(
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="The answer."),
        )
        client, _ = _anthropic_with_response("claude-haiku-5-5", response)
        result = client.complete("sys", "question")
        assert result.text == "The answer."
        assert result.tokens_used == 15

    def test_enables_refusal_fallback_on_supported_models(self):
        client, captured = _anthropic_with_response(
            "claude-sonnet-5-5", _message(SimpleNamespace(type="text", text="ok"))
        )
        client.complete("sys", "q", effort="low")
        assert captured["fallbacks"] == "default"
        assert captured["output_config"] == {"effort": "low"}
        assert "temperature" not in captured

    def test_no_fallback_on_haiku(self):
        client, captured = _anthropic_with_response(
            "claude-haiku-5-5", _message(SimpleNamespace(type="text", text="ok"))
        )
        client.complete("sys", "q")
        assert "fallbacks" not in captured

    def test_refusal_raises(self):
        client, _ = _anthropic_with_response("claude-opus-5-5", _message(stop_reason="refusal"))
        with pytest.raises(LLMError, match="declined"):
            client.complete("sys", "q")

    def test_empty_text_raises(self):
        client, _ = _anthropic_with_response("claude-opus-5-5", _message(SimpleNamespace(type="thinking", thinking="")))
        with pytest.raises(LLMError, match="empty"):
            client.complete("sys", "q")


class TestSummarizer:
    def test_parses_structured_summary(self):
        llm = FakeLLM('```json\n{"summary": "S", "key_points": ["a", "", "b"], "topics": "not-a-list"}\n```')
        result = Summarizer(llm).summarize("transcript text")

        assert result.summary == "S"
        assert result.key_points == ["a", "b"]
        assert result.topics == []
        assert result.tokens_used == 42
        assert llm.calls[0]["json_output"] is True
        assert "transcript text" in llm.calls[0]["user"]

    def test_missing_summary_raises(self):
        with pytest.raises(LLMError):
            Summarizer(FakeLLM('{"key_points": []}')).summarize("t")
