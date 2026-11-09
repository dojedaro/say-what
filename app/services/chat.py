"""Grounded RAG chat over a single video's transcript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.services.llm import LLMClient

_GROUNDING_RULES = {
    "en": """You are a helpful assistant answering questions about video content.

CRITICAL RULES:
1. ONLY use information from the provided transcript chunks to answer
2. If the answer is not in the chunks, say "This isn't covered in the video."
3. Always cite which part of the video your answer comes from
4. Be concise and direct
5. Never make up information or speculate beyond what's in the transcript

Respond in English.""",
    "es": """Eres un asistente útil que responde preguntas sobre contenido de video.

REGLAS CRÍTICAS:
1. SOLO usa información de los fragmentos de transcripción proporcionados
2. Si la respuesta no está en los fragmentos, di "Esto no se menciona en el video."
3. Siempre cita de qué parte del video proviene tu respuesta
4. Sé conciso y directo
5. Nunca inventes información o especules más allá de lo que está en la transcripción

Responde en español.""",
    "ko": """당신은 비디오 콘텐츠에 대한 질문에 답변하는 유용한 도우미입니다.

중요한 규칙:
1. 제공된 자막 청크의 정보만 사용하세요
2. 답변이 청크에 없으면 "이 내용은 비디오에서 다루지 않습니다."라고 말하세요
3. 항상 비디오의 어느 부분에서 답변이 나왔는지 인용하세요
4. 간결하고 직접적으로 답변하세요
5. 자막에 없는 정보를 만들거나 추측하지 마세요

한국어로 답변하세요.""",
}

_NO_CONTEXT_MESSAGES = {
    "en": "I couldn't find relevant information in this video to answer your question.",
    "es": "No pude encontrar información relevante en este video para responder tu pregunta.",
    "ko": "이 질문에 답하기 위한 관련 정보를 비디오에서 찾을 수 없습니다.",
}

_TRANSLATE_PROMPT = "Translate the user's text to English. Output only the translation, nothing else."


class Retriever(Protocol):
    def search(self, content_id: int, query: str, n_results: int, min_score: float) -> list[dict]: ...


@dataclass
class ChatResponse:
    message: str
    source_chunks: list[dict]
    confidence: float
    language: str


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def build_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as numbered, timestamped context for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        start = chunk.get("start_time")
        time_info = f" [at {format_timestamp(start)}]" if start is not None else ""
        parts.append(f"[Chunk {i}]{time_info}: {chunk['text']}")
    return "\n\n".join(parts)


class ChatService:
    """Answers questions using only retrieved transcript chunks.

    Retrieval runs over English embeddings, so non-English questions are translated
    first; the answer is then written in the user's chosen language.
    """

    def __init__(self, retriever: Retriever, llm: LLMClient):
        self.retriever = retriever
        self.llm = llm

    def chat(
        self,
        content_id: int,
        query: str,
        language: str,
        n_chunks: int = 5,
        min_similarity: float = 0.3,
    ) -> ChatResponse:
        search_query = self._to_english(query, language)
        chunks = self.retriever.search(
            content_id=content_id,
            query=search_query,
            n_results=n_chunks,
            min_score=min_similarity,
        )

        if not chunks:
            # Refuse rather than let the model answer from its own knowledge.
            return ChatResponse(
                message=_NO_CONTEXT_MESSAGES.get(language, _NO_CONTEXT_MESSAGES["en"]),
                source_chunks=[],
                confidence=0.0,
                language=language,
            )

        system_prompt = _GROUNDING_RULES.get(language, _GROUNDING_RULES["en"])
        user_message = (
            "Based on the following transcript chunks, answer the question.\n\n"
            f"TRANSCRIPT CHUNKS:\n{build_context(chunks)}\n\n"
            f"QUESTION: {query}"
        )
        answer = self.llm.complete(system_prompt, user_message, effort="medium")

        return ChatResponse(
            message=answer.text,
            source_chunks=chunks,
            confidence=sum(c.get("similarity", 0) for c in chunks) / len(chunks),
            language=language,
        )

    def _to_english(self, query: str, language: str) -> str:
        if language == "en":
            return query
        return self.llm.complete(_TRANSLATE_PROMPT, query, effort="low", max_tokens=1_000).text
