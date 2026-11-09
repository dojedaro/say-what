"""Pydantic schemas for request/response validation."""

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Provider = Literal["openai", "anthropic", "google", "azure"]
Language = Literal["en", "es", "ko"]


# ============== Request Schemas ==============


class ExtractionRequest(BaseModel):
    """Request schema for content extraction."""

    url: str = Field(..., max_length=500, description="URL of the video to extract")
    provider: Provider = Field(..., description="LLM provider used for summarization")
    model: str = Field(..., min_length=1, max_length=100, description="Model to use for summarization")
    api_key: str = Field(..., description="User's API key (used for this request only, never stored)")
    azure_endpoint: str | None = Field(None, description="Azure endpoint (required for Azure)")


class ChatRequest(BaseModel):
    """Request schema for RAG chat."""

    content_id: int
    message: str = Field(..., min_length=1, max_length=2000)
    language: Language = "en"
    session_id: str = Field(..., max_length=100)

    # User's own credentials; required unless the server has a key for demo content.
    provider: Provider | None = None
    model: str | None = Field(None, max_length=100)
    api_key: str | None = None
    azure_endpoint: str | None = None


# ============== Response Schemas ==============


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    chunk_index: int
    text: str
    text_english: str | None
    token_count: int | None
    start_time: float | None
    end_time: float | None
    timestamp_link: str | None = None


class ContentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    video_id: str | None
    platform: str
    status: str
    is_demo: bool

    title: str | None
    author: str | None
    duration: int | None
    views: int | None
    thumbnail_url: str | None

    detected_language: str | None

    original_transcript: str | None
    translated_transcript: str | None

    summary: str | None
    key_points: list[str] | None = None
    topics: list[str] | None = None

    transcript_tokens: int | None
    summary_tokens: int | None

    created_at: datetime
    updated_at: datetime

    error_message: str | None = None

    @field_validator("key_points", "topics", mode="before")
    @classmethod
    def _decode_json_list(cls, value):
        # Stored as JSON text in SQLite; decode transparently.
        return json.loads(value) if isinstance(value, str) else value


class ContentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    video_id: str | None
    platform: str
    status: str
    is_demo: bool
    title: str | None
    author: str | None
    thumbnail_url: str | None
    duration: int | None
    detected_language: str | None
    created_at: datetime


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    message: str
    language: str | None
    source_chunks: list[int] | None = None
    confidence_score: float | None
    created_at: datetime


class ChatResponse(BaseModel):
    message: ChatMessageResponse
    sources: list[ChunkResponse] = []


class ProcessingStatus(BaseModel):
    id: int
    status: str
    progress: str | None = None
    error: str | None = None


class APIProviderInfo(BaseModel):
    id: str
    name: str
    models: dict[str, str]
    key_placeholder: str
    requires_endpoint: bool = False


class ExportResponse(BaseModel):
    format: str
    filename: str
    content: str
