"""RAG-powered chat service with hallucination prevention."""

import json
from typing import Optional
from dataclasses import dataclass

from app.services.vector_store import VectorStore


@dataclass
class ChatResponse:
    """Response from chat service."""
    message: str
    source_chunks: list[dict]
    confidence: float
    language: str


class ChatService:
    """RAG-powered chat with grounded responses."""

    # System prompts for each language
    SYSTEM_PROMPTS = {
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

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize chat service.

        Args:
            vector_store: VectorStore instance (creates new one if not provided)
        """
        self.vector_store = vector_store or VectorStore()

    def _translate_query_to_english(
        self,
        query: str,
        language: str,
        provider: str,
        api_key: str,
        model: str,
        azure_endpoint: Optional[str] = None,
    ) -> str:
        """Translate query to English for better retrieval."""
        if language == "en":
            return query

        # Use the appropriate provider to translate
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Translate the following to English. Only output the translation, nothing else."},
                    {"role": "user", "content": query},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()

        elif provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": f"Translate the following to English. Only output the translation, nothing else.\n\n{query}"},
                ],
            )
            return response.content[0].text.strip()

        elif provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(
                f"Translate the following to English. Only output the translation, nothing else.\n\n{query}"
            )
            return response.text.strip()

        elif provider == "azure":
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=azure_endpoint,
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Translate the following to English. Only output the translation, nothing else."},
                    {"role": "user", "content": query},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()

        return query

    def _generate_response(
        self,
        query: str,
        chunks: list[dict],
        language: str,
        provider: str,
        api_key: str,
        model: str,
        azure_endpoint: Optional[str] = None,
    ) -> str:
        """Generate grounded response using retrieved chunks."""
        system_prompt = self.SYSTEM_PROMPTS.get(language, self.SYSTEM_PROMPTS["en"])

        # Format chunks for context
        context_parts = []
        for i, chunk in enumerate(chunks):
            time_info = ""
            if chunk.get("start_time"):
                minutes = int(chunk["start_time"] // 60)
                seconds = int(chunk["start_time"] % 60)
                time_info = f" [at {minutes}:{seconds:02d}]"

            context_parts.append(f"[Chunk {i+1}]{time_info}: {chunk['text']}")

        context = "\n\n".join(context_parts)

        user_message = f"""Based on the following transcript chunks, answer the question.

TRANSCRIPT CHUNKS:
{context}

QUESTION: {query}"""

        # Generate response using appropriate provider
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()

        elif provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message},
                ],
            )
            return response.content[0].text.strip()

        elif provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(
                f"{system_prompt}\n\n{user_message}"
            )
            return response.text.strip()

        elif provider == "azure":
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=azure_endpoint,
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()

        raise ValueError(f"Unknown provider: {provider}")

    def chat(
        self,
        content_id: int,
        query: str,
        language: str,
        provider: str,
        api_key: str,
        model: str,
        azure_endpoint: Optional[str] = None,
        n_chunks: int = 5,
        min_similarity: float = 0.3,
    ) -> ChatResponse:
        """
        Process a chat query with RAG.

        Args:
            content_id: Content ID to chat about
            query: User's question
            language: Response language (en, es, ko)
            provider: API provider
            api_key: API key
            model: Model name
            azure_endpoint: Azure endpoint (if using Azure)
            n_chunks: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold

        Returns:
            ChatResponse with message and sources
        """
        # Translate query to English for better retrieval
        english_query = self._translate_query_to_english(
            query, language, provider, api_key, model, azure_endpoint
        )

        # Retrieve relevant chunks
        chunks = self.vector_store.search(
            content_id=content_id,
            query=english_query,
            n_results=n_chunks,
            min_score=min_similarity,
        )

        if not chunks:
            # No relevant chunks found
            no_info_messages = {
                "en": "I couldn't find relevant information in this video to answer your question.",
                "es": "No pude encontrar información relevante en este video para responder tu pregunta.",
                "ko": "이 질문에 답하기 위한 관련 정보를 비디오에서 찾을 수 없습니다.",
            }
            return ChatResponse(
                message=no_info_messages.get(language, no_info_messages["en"]),
                source_chunks=[],
                confidence=0.0,
                language=language,
            )

        # Calculate average confidence from chunk similarities
        avg_confidence = sum(c.get("similarity", 0) for c in chunks) / len(chunks)

        # Generate grounded response
        response_text = self._generate_response(
            query=query,
            chunks=chunks,
            language=language,
            provider=provider,
            api_key=api_key,
            model=model,
            azure_endpoint=azure_endpoint,
        )

        return ChatResponse(
            message=response_text,
            source_chunks=chunks,
            confidence=avg_confidence,
            language=language,
        )
