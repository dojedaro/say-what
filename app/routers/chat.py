"""Chat API endpoints for RAG-powered conversations."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import DEFAULT_MODELS, server_llm_config
from app.database import get_db
from app.models import ChatMessage, Content
from app.schemas import ChatMessageResponse, ChatRequest, ChatResponse, ChunkResponse
from app.services.extractor import timestamp_url
from app.services.llm import LLMClient, LLMError, create_llm_client

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def resolve_llm(content: Content, chat_request: ChatRequest) -> LLMClient:
    """Server key for demo content when configured; otherwise the caller's own key."""
    if content.is_demo and (server := server_llm_config()):
        return create_llm_client(server["provider"], server["api_key"], server["model"])

    if not chat_request.api_key:
        raise HTTPException(status_code=400, detail="API key required. Please provide your API key.")
    provider = chat_request.provider or "openai"
    model = chat_request.model or DEFAULT_MODELS[provider]
    try:
        return create_llm_client(provider, chat_request.api_key, model, chat_request.azure_endpoint)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def get_retriever():
    from app.services.vector_store import VectorStore

    return VectorStore()


@router.post("", response_model=ChatResponse)
def chat(chat_request: ChatRequest, db: Session = Depends(get_db), retriever=Depends(get_retriever)):
    """Answer a question about one video using only its transcript.

    Sync on purpose: FastAPI runs it in a threadpool, so slow LLM calls
    don't block other requests on the event loop.
    """
    from app.services.chat import ChatService

    content = db.get(Content, chat_request.content_id)
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    if content.status != "completed":
        raise HTTPException(status_code=400, detail="Content is not fully processed")

    llm = resolve_llm(content, chat_request)

    db.add(
        ChatMessage(
            content_id=content.id,
            session_id=chat_request.session_id,
            role="user",
            message=chat_request.message,
            language=chat_request.language,
        )
    )
    db.commit()

    try:
        result = ChatService(retriever, llm).chat(
            content_id=content.id,
            query=chat_request.message,
            language=chat_request.language,
        )
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.exception("Chat failed for content %s", content.id)
        raise HTTPException(
            status_code=502, detail="The model provider returned an error. Check your API key and model."
        ) from e

    source_chunk_ids = [c.get("chunk_index", 0) for c in result.source_chunks]
    assistant_message = ChatMessage(
        content_id=content.id,
        session_id=chat_request.session_id,
        role="assistant",
        message=result.message,
        language=result.language,
        source_chunks=json.dumps(source_chunk_ids),
        confidence_score=result.confidence,
    )
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    sources = [
        ChunkResponse(
            id=0,  # vector hits are not tied to a DB row id
            chunk_index=chunk.get("chunk_index", 0),
            text=chunk.get("text", ""),
            text_english=chunk.get("text_embedded"),
            token_count=chunk.get("token_count"),
            start_time=chunk.get("start_time"),
            end_time=chunk.get("end_time"),
            timestamp_link=timestamp_url(content.platform, content.video_id, chunk.get("start_time")),
        )
        for chunk in result.source_chunks
    ]

    return ChatResponse(
        message=ChatMessageResponse(
            id=assistant_message.id,
            role="assistant",
            message=result.message,
            language=result.language,
            source_chunks=source_chunk_ids,
            confidence_score=result.confidence,
            created_at=assistant_message.created_at,
        ),
        sources=sources,
    )


@router.get("/history/{content_id}/{session_id}", response_model=list[ChatMessageResponse])
def get_chat_history(content_id: int, session_id: str, db: Session = Depends(get_db)):
    """Chat history for a content item and session."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.content_id == content_id, ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    return [
        ChatMessageResponse(
            id=msg.id,
            role=msg.role,
            message=msg.message,
            language=msg.language,
            source_chunks=json.loads(msg.source_chunks) if msg.source_chunks else None,
            confidence_score=msg.confidence_score,
            created_at=msg.created_at,
        )
        for msg in messages
    ]


@router.delete("/history/{content_id}/{session_id}")
def clear_chat_history(content_id: int, session_id: str, db: Session = Depends(get_db)):
    """Clear chat history for a session."""
    db.query(ChatMessage).filter(
        ChatMessage.content_id == content_id,
        ChatMessage.session_id == session_id,
    ).delete()
    db.commit()
    return {"status": "ok", "message": "Chat history cleared"}
