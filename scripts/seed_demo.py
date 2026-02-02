#!/usr/bin/env python3
"""Seed the database with pre-processed demo content."""

import sys
import os
import json

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings, DEMO_VIDEOS
from app.database import init_db, get_db_context
from app.models import Content, Chunk
from app.services.extractor import VideoExtractor
from app.services.transcriber import Transcriber
from app.services.summarizer import Summarizer
from app.services.chunker import Chunker
from app.services.vector_store import VectorStore

settings = get_settings()


def get_api_config():
    """Get API configuration from environment."""
    if settings.openai_api_key:
        return {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": settings.openai_api_key,
        }
    elif settings.anthropic_api_key:
        return {
            "provider": "anthropic",
            "model": "claude-3-5-haiku-20241022",
            "api_key": settings.anthropic_api_key,
        }
    else:
        raise ValueError("No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")


def seed_demo_content():
    """Seed all demo videos into the database."""
    print("🎬 Say What? Demo Seeder")
    print("=" * 50)

    # Initialize database
    init_db()

    # Get API config
    try:
        api_config = get_api_config()
        print(f"✅ Using {api_config['provider']} with {api_config['model']}")
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Initialize services
    extractor = VideoExtractor()
    transcriber = Transcriber()
    chunker = Chunker()
    vector_store = VectorStore()

    with get_db_context() as db:
        for i, video in enumerate(DEMO_VIDEOS, 1):
            print(f"\n[{i}/{len(DEMO_VIDEOS)}] Processing: {video['title']}")
            print(f"    URL: {video['url']}")

            # Check if already exists
            existing = db.query(Content).filter(Content.url == video["url"]).first()
            if existing:
                print(f"    ⏭️  Already exists (ID: {existing.id})")
                continue

            try:
                # Extract metadata
                print("    📥 Extracting metadata...")
                metadata = extractor.extract_metadata(video["url"])

                # Create content record
                content = Content(
                    url=video["url"],
                    video_id=metadata.video_id,
                    platform=metadata.platform,
                    status="processing",
                    is_demo=True,
                    title=metadata.title,
                    author=metadata.author,
                    duration=metadata.duration,
                    views=metadata.views,
                    thumbnail_url=metadata.thumbnail_url,
                )
                db.add(content)
                db.commit()
                db.refresh(content)

                # Transcribe
                print("    📝 Transcribing...")
                audio_path = None

                if metadata.platform != "youtube":
                    audio_path = extractor.download_audio(video["url"])

                try:
                    result = transcriber.transcribe(
                        url=video["url"],
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

                print(f"    🌐 Detected language: {result.detected_language}")

                # Summarize
                print("    🤖 Generating summary...")
                text_for_summary = result.translated_text or result.original_text

                summarizer = Summarizer.create(
                    api_config["provider"],
                    api_config["api_key"],
                    api_config["model"],
                )
                summary_result = summarizer.summarize(text_for_summary)

                content.summary = summary_result.summary
                content.key_points = json.dumps(summary_result.key_points)
                content.topics = json.dumps(summary_result.topics)
                content.summary_tokens = summary_result.tokens_used
                db.commit()

                # Chunk for RAG
                print("    📦 Creating chunks...")
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
                print("    🔍 Building vector index...")
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

                # Mark as completed
                content.status = "completed"
                db.commit()

                print(f"    ✅ Done! (ID: {content.id}, {len(chunks)} chunks)")

            except Exception as e:
                print(f"    ❌ Error: {e}")
                db.rollback()
                continue

    print("\n" + "=" * 50)
    print("🎉 Demo seeding complete!")


if __name__ == "__main__":
    seed_demo_content()
