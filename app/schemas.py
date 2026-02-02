"""Pydantic schemas for request/response validation."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


# ============== Request Schemas ==============

class ExtractionRequest(BaseModel):
    """Request schema for content extraction."""
    url: str = Field(..., description="URL of the video to extract")
    provider: str = Field(..., description="API provider: openai, anthropic, google, azure")
    model: str = Field(..., description="Model to use for summarization")
    api_key: str = Field(..., description="User's API key")
    azure_endpoint: Optional[str] = Field(None, description="Azure endpoint (required for Azure)")


class ChatRequest(BaseModel):
    """Request schema for RAG chat."""
    content_id: int = Field(..., description="Content ID to chat about")
    message: str = Field(..., description="User's message")
    language: str = Field(default="en", description="Response language: en, es, ko")
    session_id: str = Field(..., description="Session ID for chat history")

    # Optional: user's API key for non-demo content
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None


# ============== Response Schemas ==============

class ChunkResponse(BaseModel):
    """Response schema for a single chunk."""
    id: int
    chunk_index: int
    text: str
    text_english: Optional[str]
    token_count: Optional[int]
    start_time: Optional[float]
    end_time: Optional[float]

    # Computed fields
    timestamp_link: Optional[str] = None

    class Config:
        from_attributes = True


class ContentResponse(BaseModel):
    """Response schema for processed content."""
    id: int
    url: str
    video_id: Optional[str]
    platform: str
    status: str
    is_demo: bool

    # Metadata
    title: Optional[str]
    author: Optional[str]
    duration: Optional[int]
    views: Optional[int]
    thumbnail_url: Optional[str]

    # Language
    detected_language: Optional[str]

    # Transcripts
    original_transcript: Optional[str]
    translated_transcript: Optional[str]

    # AI Content
    summary: Optional[str]
    key_points: Optional[list[str]] = None
    topics: Optional[list[str]] = None

    # Token counts
    transcript_tokens: Optional[int]
    summary_tokens: Optional[int]

    # Timestamps
    created_at: datetime
    updated_at: datetime

    # Error info
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class ContentListItem(BaseModel):
    """Simplified content item for list views."""
    id: int
    url: str
    video_id: Optional[str]
    platform: str
    status: str
    is_demo: bool
    title: Optional[str]
    author: Optional[str]
    thumbnail_url: Optional[str]
    duration: Optional[int]
    detected_language: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ChatMessageResponse(BaseModel):
    """Response schema for a chat message."""
    id: int
    role: str
    message: str
    language: Optional[str]
    source_chunks: Optional[list[int]] = None
    confidence_score: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    message: ChatMessageResponse
    sources: list[ChunkResponse] = []


class ProcessingStatus(BaseModel):
    """Response for processing status check."""
    id: int
    status: str
    progress: Optional[str] = None
    error: Optional[str] = None


class APIProviderInfo(BaseModel):
    """Information about an API provider."""
    id: str
    name: str
    models: dict[str, str]
    key_placeholder: str
    requires_endpoint: bool = False


class ExportResponse(BaseModel):
    """Response for content export."""
    format: str
    filename: str
    content: str
