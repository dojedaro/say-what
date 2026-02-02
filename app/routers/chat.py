"""Chat API endpoints for RAG-powered conversations."""

import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Content, ChatMessage
from app.schemas import ChatRequest, ChatMessageResponse, ChatResponse, ChunkResponse
from app.config import get_settings
from app.services.chat import ChatService
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/chat", tags=["chat"])
settings = get_settings()


# Default API settings for demo content (uses server-side keys if available)
def get_demo_api_settings():
    """Get API settings for demo content."""
    if settings.openai_api_key:
        return {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": settings.openai_api_key,
            "azure_endpoint": None,
        }
    elif settings.anthropic_api_key:
        return {
            "provider": "anthropic",
            "model": "claude-3-5-haiku-20241022",
            "api_key": settings.anthropic_api_key,
            "azure_endpoint": None,
        }
    return None


@router.post("", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Send a message and get a RAG-powered response.

    For demo content, uses server API keys if available.
    For user content, requires user's API key.
    """
    # Get content
    content = db.query(Content).filter(Content.id == chat_request.content_id).first()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    if content.status != "completed":
        raise HTTPException(status_code=400, detail="Content is not fully processed")

    # Determine API settings
    if content.is_demo:
        # Try server-side keys for demo content
        demo_settings = get_demo_api_settings()
        if demo_settings:
            provider = demo_settings["provider"]
            model = demo_settings["model"]
            api_key = demo_settings["api_key"]
            azure_endpoint = demo_settings["azure_endpoint"]
        elif chat_request.api_key:
            # Fall back to user's key
            provider = chat_request.provider
            model = chat_request.model
            api_key = chat_request.api_key
            azure_endpoint = None
        else:
            raise HTTPException(
                status_code=400,
                detail="No API key available. Please provide your API key.",
            )
    else:
        # User content requires their API key
        if not chat_request.api_key:
            raise HTTPException(status_code=400, detail="API key required for user content")
        provider = chat_request.provider
        model = chat_request.model
        api_key = chat_request.api_key
        azure_endpoint = None

    # Save user message
    user_message = ChatMessage(
        content_id=content.id,
        session_id=chat_request.session_id,
        role="user",
        message=chat_request.message,
        language=chat_request.language,
    )
    db.add(user_message)
    db.commit()

    # Generate response
    chat_service = ChatService()

    try:
        result = chat_service.chat(
            content_id=content.id,
            query=chat_request.message,
            language=chat_request.language,
            provider=provider,
            api_key=api_key,
            model=model,
            azure_endpoint=azure_endpoint,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

    # Save assistant message
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

    # Build response
    sources = []
    for chunk in result.source_chunks:
        source = ChunkResponse(
            id=0,  # We don't have the DB ID here
            chunk_index=chunk.get("chunk_index", 0),
            text=chunk.get("text", ""),
            text_english=chunk.get("text_embedded"),
            token_count=chunk.get("token_count"),
            start_time=chunk.get("start_time"),
            end_time=chunk.get("end_time"),
        )

        # Add timestamp link for YouTube
        if content.platform == "youtube" and chunk.get("start_time"):
            source.timestamp_link = f"https://www.youtube.com/watch?v={content.video_id}&t={int(chunk['start_time'])}s"

        sources.append(source)

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
async def get_chat_history(
    content_id: int,
    session_id: str,
    db: Session = Depends(get_db),
):
    """Get chat history for a content item and session."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.content_id == content_id, ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )

    result = []
    for msg in messages:
        response = ChatMessageResponse(
            id=msg.id,
            role=msg.role,
            message=msg.message,
            language=msg.language,
            confidence_score=msg.confidence_score,
            created_at=msg.created_at,
        )

        if msg.source_chunks:
            response.source_chunks = json.loads(msg.source_chunks)

        result.append(response)

    return result


@router.delete("/history/{content_id}/{session_id}")
async def clear_chat_history(
    content_id: int,
    session_id: str,
    db: Session = Depends(get_db),
):
    """Clear chat history for a session."""
    db.query(ChatMessage).filter(
        ChatMessage.content_id == content_id,
        ChatMessage.session_id == session_id,
    ).delete()
    db.commit()

    return {"status": "ok", "message": "Chat history cleared"}
