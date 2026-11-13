"""REST API endpoints for content extraction and management."""

import json
import logging
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.config import API_PROVIDERS, get_settings
from app.database import get_db, get_db_context
from app.models import Chunk, Content
from app.schemas import (
    APIProviderInfo,
    ChunkResponse,
    ContentListItem,
    ContentResponse,
    ExportResponse,
    ExtractionRequest,
    ProcessingStatus,
)
from app.services.extractor import VideoExtractor, timestamp_url

router = APIRouter(prefix="/api", tags=["api"])
logger = logging.getLogger(__name__)

PROGRESS_MESSAGES = {
    "pending": "Waiting to start...",
    "extracting": "Extracting video metadata...",
    "transcribing": "Transcribing audio...",
    "summarizing": "Generating AI summary...",
    "indexing": "Building search index...",
    "completed": "Processing complete!",
    "failed": "Processing failed",
}


def get_content_or_404(db: Session, content_id: int) -> Content:
    content = db.get(Content, content_id)
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    return content


@router.get("/providers", response_model=list[APIProviderInfo])
def get_providers():
    """Supported LLM providers and their models."""
    return [
        APIProviderInfo(
            id=provider_id,
            name=info["name"],
            models=info["models"],
            key_placeholder=info["key_placeholder"],
            requires_endpoint=info.get("requires_endpoint", False),
        )
        for provider_id, info in API_PROVIDERS.items()
    ]


@router.post("/extract", response_model=ProcessingStatus)
def extract_content(
    request: Request,
    extraction: ExtractionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Queue a video URL for processing with the caller's own API key.

    Previously failed URLs are re-processed; limited per session to deter abuse.
    """
    if not extraction.api_key.strip():
        raise HTTPException(status_code=400, detail="API key required")
    if extraction.provider == "azure" and not extraction.azure_endpoint:
        raise HTTPException(status_code=400, detail="Azure endpoint required for Azure provider")

    existing = db.query(Content).filter(Content.url == extraction.url).first()
    if existing and existing.status != "failed":
        return ProcessingStatus(
            id=existing.id,
            status=existing.status,
            progress="Already processed" if existing.status == "completed" else "Processing",
        )

    settings = get_settings()
    extraction_count = request.session.get("extraction_count", 0)
    if extraction_count >= settings.max_extractions_per_session:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {settings.max_extractions_per_session} extractions per session.",
        )

    try:
        platform, video_id = VideoExtractor.detect_platform(extraction.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if existing:  # retry a failed extraction in place
        content = existing
        content.status, content.error_message = "pending", None
    else:
        content = Content(url=extraction.url, video_id=video_id, platform=platform, status="pending")
        db.add(content)
    db.commit()
    db.refresh(content)

    request.session["extraction_count"] = extraction_count + 1
    background_tasks.add_task(
        process_content,
        content.id,
        extraction.provider,
        extraction.model,
        extraction.api_key,
        extraction.azure_endpoint,
    )
    return ProcessingStatus(id=content.id, status="pending", progress="Starting extraction...")


def process_content(
    content_id: int,
    provider: str,
    model: str,
    api_key: str,
    azure_endpoint: str | None = None,
) -> None:
    """Background task. Deliberately sync: Starlette runs it in a worker thread, so the
    blocking download/transcription/LLM work never stalls the event loop."""
    from app.services.pipeline import ContentPipeline
    from app.services.summarizer import Summarizer

    with get_db_context() as db:
        content = db.get(Content, content_id)
        if not content:
            return
        try:
            summarizer = Summarizer.create(provider, api_key, model, azure_endpoint)
            ContentPipeline().run(db, content, summarizer)
        except Exception as exc:
            # The pipeline already logged and recorded the failure on the row.
            if content.status != "failed":
                logger.exception("Could not start processing for content %s", content_id)
                content.status, content.error_message = "failed", str(exc)


@router.get("/content/{content_id}", response_model=ContentResponse)
def get_content(content_id: int, db: Session = Depends(get_db)):
    """Processed content by ID."""
    return ContentResponse.model_validate(get_content_or_404(db, content_id))


@router.get("/content/{content_id}/status", response_model=ProcessingStatus)
def get_content_status(content_id: int, db: Session = Depends(get_db)):
    """Processing status, polled by the UI."""
    content = get_content_or_404(db, content_id)
    return ProcessingStatus(
        id=content.id,
        status=content.status,
        progress=PROGRESS_MESSAGES.get(content.status, "Processing..."),
        error=content.error_message if content.status == "failed" else None,
    )


def _chunks_for(db: Session, content_id: int) -> list[Chunk]:
    return db.query(Chunk).filter(Chunk.content_id == content_id).order_by(Chunk.chunk_index).all()


@router.get("/content/{content_id}/chunks", response_model=list[ChunkResponse])
def get_content_chunks(content_id: int, db: Session = Depends(get_db)):
    """RAG-ready chunks for content, with deep links where available."""
    content = get_content_or_404(db, content_id)
    result = []
    for chunk in _chunks_for(db, content_id):
        response = ChunkResponse.model_validate(chunk)
        response.timestamp_link = timestamp_url(content.platform, content.video_id, chunk.start_time)
        result.append(response)
    return result


@router.get("/content/{content_id}/export", response_model=ExportResponse)
def export_content(
    content_id: int,
    format: Literal["json", "markdown"] = "json",
    db: Session = Depends(get_db),
):
    """Export content as JSON (including RAG-ready chunks) or Markdown."""
    content = get_content_or_404(db, content_id)
    key_points = json.loads(content.key_points) if content.key_points else []
    if format == "json":
        return _export_json(content, key_points, _chunks_for(db, content_id))
    return _export_markdown(content, key_points)


def _export_json(content: Content, key_points: list[str], chunks: list[Chunk]) -> ExportResponse:
    export_data = {
        "metadata": {
            "title": content.title,
            "author": content.author,
            "platform": content.platform,
            "url": content.url,
            "duration": content.duration,
            "language": content.detected_language,
        },
        "transcript": {
            "original": content.original_transcript,
            "translated": content.translated_transcript,
        },
        "summary": content.summary,
        "key_points": key_points,
        "topics": json.loads(content.topics) if content.topics else [],
        "chunks": [
            {
                "index": c.chunk_index,
                "text": c.text,
                "text_english": c.text_english,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "token_count": c.token_count,
            }
            for c in chunks
        ],
    }
    return ExportResponse(
        format="json",
        filename=f"{content.video_id}_export.json",
        content=json.dumps(export_data, indent=2, ensure_ascii=False),
    )


def _export_markdown(content: Content, key_points: list[str]) -> ExportResponse:
    md_lines = [
        f"# {content.title}",
        "",
        f"**Author:** {content.author}",
        f"**Platform:** {content.platform}",
        f"**URL:** {content.url}",
        f"**Language:** {content.detected_language}",
        "",
        "## Summary",
        "",
        content.summary or "No summary available.",
        "",
        "## Key Points",
        "",
        *(f"- {point}" for point in key_points),
        "",
        "## Transcript",
        "",
        content.translated_transcript or content.original_transcript or "No transcript available.",
        "",
    ]
    if content.detected_language != "en" and content.original_transcript:
        md_lines += [f"## Original Transcript ({content.detected_language})", "", content.original_transcript]
    return ExportResponse(format="markdown", filename=f"{content.video_id}_export.md", content="\n".join(md_lines))


@router.get("/contents", response_model=list[ContentListItem])
def list_contents(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    demo_only: bool = False,
    db: Session = Depends(get_db),
):
    """List processed content, newest first."""
    query = db.query(Content)
    if demo_only:
        query = query.filter(Content.is_demo.is_(True))
    contents = query.order_by(Content.created_at.desc()).offset(skip).limit(limit).all()
    return [ContentListItem.model_validate(c) for c in contents]
