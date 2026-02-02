"""REST API endpoints for content extraction and management."""

import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Content, Chunk
from app.schemas import (
    ExtractionRequest,
    ContentResponse,
    ContentListItem,
    ChunkResponse,
    ProcessingStatus,
    APIProviderInfo,
    ExportResponse,
)
from app.config import get_settings, API_PROVIDERS
from app.services.extractor import VideoExtractor
from app.services.transcriber import Transcriber
from app.services.summarizer import Summarizer
from app.services.chunker import Chunker
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api", tags=["api"])
settings = get_settings()


def get_session_extraction_count(request: Request) -> int:
    """Get the number of extractions in the current session."""
    return request.session.get("extraction_count", 0)


def increment_session_extraction_count(request: Request):
    """Increment the extraction count for the current session."""
    current = request.session.get("extraction_count", 0)
    request.session["extraction_count"] = current + 1


@router.get("/providers", response_model=list[APIProviderInfo])
async def get_providers():
    """Get list of supported API providers and their models."""
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
async def extract_content(
    request: Request,
    extraction: ExtractionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Submit a URL for content extraction.

    Requires user's API key for processing.
    Limited to 5 extractions per session.
    """
    # Check rate limit
    extraction_count = get_session_extraction_count(request)
    if extraction_count >= settings.max_extractions_per_session:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {settings.max_extractions_per_session} extractions per session.",
        )

    # Check if URL already processed
    existing = db.query(Content).filter(Content.url == extraction.url).first()
    if existing:
        return ProcessingStatus(
            id=existing.id,
            status=existing.status,
            progress="Already processed" if existing.status == "completed" else "Processing",
        )

    # Validate API key provided
    if not extraction.api_key:
        raise HTTPException(status_code=400, detail="API key required")

    # Create initial content record
    try:
        extractor = VideoExtractor()
        platform, video_id = extractor.detect_platform(extraction.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    content = Content(
        url=extraction.url,
        video_id=video_id,
        platform=platform,
        status="pending",
        is_demo=False,
    )
    db.add(content)
    db.commit()
    db.refresh(content)

    # Increment extraction count
    increment_session_extraction_count(request)

    # Start background processing
    background_tasks.add_task(
        process_content,
        content.id,
        extraction.provider,
        extraction.model,
        extraction.api_key,
        extraction.azure_endpoint,
    )

    return ProcessingStatus(
        id=content.id,
        status="pending",
        progress="Starting extraction...",
    )


async def process_content(
    content_id: int,
    provider: str,
    model: str,
    api_key: str,
    azure_endpoint: Optional[str] = None,
):
    """Background task to process content."""
    from app.database import get_db_context

    with get_db_context() as db:
        content = db.query(Content).filter(Content.id == content_id).first()
        if not content:
            return

        try:
            # Update status
            content.status = "extracting"
            db.commit()

            # Extract metadata
            extractor = VideoExtractor()
            metadata = extractor.extract_metadata(content.url)

            content.title = metadata.title
            content.author = metadata.author
            content.duration = metadata.duration
            content.views = metadata.views
            content.thumbnail_url = metadata.thumbnail_url
            db.commit()

            # Transcribe
            content.status = "transcribing"
            db.commit()

            transcriber = Transcriber()
            audio_path = None

            # For non-YouTube or if YouTube transcript fails, download audio
            if metadata.platform != "youtube":
                audio_path = extractor.download_audio(content.url)

            try:
                result = transcriber.transcribe(
                    url=content.url,
                    video_id=metadata.video_id,
                    platform=metadata.platform,
                    audio_path=audio_path,
                )
            finally:
                if audio_path:
                    extractor.cleanup_audio(audio_path)

            content.original_transcript = result.original_text
            content.translated_transcript = result.translated_text
            content.transcript_with_timestamps = transcriber.segments_to_json(result.segments)
            content.detected_language = result.detected_language
            db.commit()

            # Summarize
            content.status = "summarizing"
            db.commit()

            # Use English transcript for summarization
            text_for_summary = result.translated_text or result.original_text

            summarizer = Summarizer.create(provider, api_key, model, azure_endpoint)
            summary_result = summarizer.summarize(text_for_summary)

            content.summary = summary_result.summary
            content.key_points = json.dumps(summary_result.key_points)
            content.topics = json.dumps(summary_result.topics)
            content.summary_tokens = summary_result.tokens_used
            db.commit()

            # Chunk for RAG
            chunker = Chunker()
            timestamps = json.loads(content.transcript_with_timestamps) if content.transcript_with_timestamps else None

            chunks = chunker.chunk_text(
                text=result.original_text,
                text_english=result.translated_text,
                timestamps=timestamps,
            )

            content.transcript_tokens = sum(c.token_count for c in chunks)

            # Save chunks to database
            for chunk in chunks:
                db_chunk = Chunk(
                    content_id=content.id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                    text_english=chunk.text_english,
                    token_count=chunk.token_count,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                )
                db.add(db_chunk)

            db.commit()

            # Add to vector store
            vector_store = VectorStore()
            chunk_dicts = [
                {
                    "index": c.index,
                    "text": c.text,
                    "text_english": c.text_english,
                    "token_count": c.token_count,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                }
                for c in chunks
            ]
            vector_store.add_chunks(content.id, chunk_dicts)

            # Done!
            content.status = "completed"
            db.commit()

        except Exception as e:
            content.status = "failed"
            content.error_message = str(e)
            db.commit()
            raise


@router.get("/content/{content_id}", response_model=ContentResponse)
async def get_content(content_id: int, db: Session = Depends(get_db)):
    """Get processed content by ID."""
    content = db.query(Content).filter(Content.id == content_id).first()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    response = ContentResponse.model_validate(content)

    # Parse JSON fields
    if content.key_points:
        response.key_points = json.loads(content.key_points)
    if content.topics:
        response.topics = json.loads(content.topics)

    return response


@router.get("/content/{content_id}/status", response_model=ProcessingStatus)
async def get_content_status(content_id: int, db: Session = Depends(get_db)):
    """Get processing status for content."""
    content = db.query(Content).filter(Content.id == content_id).first()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    progress_messages = {
        "pending": "Waiting to start...",
        "extracting": "Extracting video metadata...",
        "transcribing": "Transcribing audio...",
        "summarizing": "Generating AI summary...",
        "completed": "Processing complete!",
        "failed": "Processing failed",
    }

    return ProcessingStatus(
        id=content.id,
        status=content.status,
        progress=progress_messages.get(content.status, "Processing..."),
        error=content.error_message if content.status == "failed" else None,
    )


@router.get("/content/{content_id}/chunks", response_model=list[ChunkResponse])
async def get_content_chunks(content_id: int, db: Session = Depends(get_db)):
    """Get RAG-ready chunks for content."""
    content = db.query(Content).filter(Content.id == content_id).first()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    chunks = db.query(Chunk).filter(Chunk.content_id == content_id).order_by(Chunk.chunk_index).all()

    result = []
    for chunk in chunks:
        response = ChunkResponse.model_validate(chunk)

        # Add timestamp link for YouTube
        if content.platform == "youtube" and chunk.start_time:
            response.timestamp_link = f"https://www.youtube.com/watch?v={content.video_id}&t={int(chunk.start_time)}s"

        result.append(response)

    return result


@router.get("/content/{content_id}/export")
async def export_content(
    content_id: int,
    format: str = "json",
    db: Session = Depends(get_db),
):
    """Export content in various formats."""
    content = db.query(Content).filter(Content.id == content_id).first()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")

    chunks = db.query(Chunk).filter(Chunk.content_id == content_id).order_by(Chunk.chunk_index).all()

    if format == "json":
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
            "key_points": json.loads(content.key_points) if content.key_points else [],
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

    elif format == "markdown":
        md_lines = [
            f"# {content.title}",
            f"",
            f"**Author:** {content.author}",
            f"**Platform:** {content.platform}",
            f"**URL:** {content.url}",
            f"**Language:** {content.detected_language}",
            f"",
            f"## Summary",
            f"",
            content.summary or "No summary available.",
            f"",
            f"## Key Points",
            f"",
        ]

        key_points = json.loads(content.key_points) if content.key_points else []
        for point in key_points:
            md_lines.append(f"- {point}")

        md_lines.extend([
            f"",
            f"## Transcript",
            f"",
            content.translated_transcript or content.original_transcript or "No transcript available.",
            f"",
        ])

        if content.detected_language != "en" and content.original_transcript:
            md_lines.extend([
                f"## Original Transcript ({content.detected_language})",
                f"",
                content.original_transcript,
            ])

        return ExportResponse(
            format="markdown",
            filename=f"{content.video_id}_export.md",
            content="\n".join(md_lines),
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@router.get("/contents", response_model=list[ContentListItem])
async def list_contents(
    skip: int = 0,
    limit: int = 20,
    demo_only: bool = False,
    db: Session = Depends(get_db),
):
    """List all processed content."""
    query = db.query(Content)

    if demo_only:
        query = query.filter(Content.is_demo == True)

    contents = query.order_by(Content.created_at.desc()).offset(skip).limit(limit).all()

    return [ContentListItem.model_validate(c) for c in contents]
