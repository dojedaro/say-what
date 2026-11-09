"""End-to-end processing pipeline: extract -> transcribe -> summarize -> chunk -> index.

Shared by the API background task and the demo seeding script so there is
exactly one implementation of how a video becomes searchable content.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from sqlalchemy.orm import Session

from app.models import Chunk, Content
from app.services.chunker import Chunker
from app.services.extractor import VideoExtractor
from app.services.summarizer import Summarizer
from app.services.transcriber import Transcriber
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ContentPipeline:
    def __init__(
        self,
        extractor: VideoExtractor | None = None,
        transcriber: Transcriber | None = None,
        chunker: Chunker | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.extractor = extractor or VideoExtractor()
        self.transcriber = transcriber or Transcriber()
        self.chunker = chunker or Chunker()
        self.vector_store = vector_store or VectorStore()

    def run(self, db: Session, content: Content, summarizer: Summarizer) -> None:
        """Process ``content`` in place, committing status after each stage.

        Safe to re-run on a previously failed item: old chunks and embeddings are
        replaced. On error the content is marked ``failed`` and the error re-raised.
        """
        try:
            self._run(db, content, summarizer)
        except Exception as exc:
            logger.exception("Processing failed for content %s (%s)", content.id, content.url)
            db.rollback()
            content.status = "failed"
            content.error_message = str(exc)
            db.commit()
            raise

    def _run(self, db: Session, content: Content, summarizer: Summarizer) -> None:
        self._set_status(db, content, "extracting")
        metadata = self.extractor.extract_metadata(content.url)
        content.title = metadata.title
        content.author = metadata.author
        content.duration = metadata.duration
        content.views = metadata.views
        content.thumbnail_url = metadata.thumbnail_url
        content.video_id = metadata.video_id
        content.platform = metadata.platform

        self._set_status(db, content, "transcribing")
        # YouTube usually has captions; other platforms need the audio for Whisper.
        audio_path = self.extractor.download_audio(content.url) if metadata.platform != "youtube" else None
        try:
            transcript = self.transcriber.transcribe(
                url=content.url,
                video_id=metadata.video_id,
                platform=metadata.platform,
                audio_path=audio_path,
            )
        finally:
            if audio_path:
                self.extractor.cleanup_audio(audio_path)

        content.original_transcript = transcript.original_text
        content.translated_transcript = transcript.translated_text
        content.transcript_with_timestamps = self.transcriber.segments_to_json(transcript.segments)
        content.detected_language = transcript.detected_language

        self._set_status(db, content, "summarizing")
        summary = summarizer.summarize(transcript.translated_text or transcript.original_text)
        content.summary = summary.summary
        content.key_points = json.dumps(summary.key_points, ensure_ascii=False)
        content.topics = json.dumps(summary.topics, ensure_ascii=False)
        content.summary_tokens = summary.tokens_used

        self._set_status(db, content, "indexing")
        chunks = self.chunker.chunk_text(
            text=transcript.original_text,
            text_english=transcript.translated_text,
            timestamps=[asdict(s) for s in transcript.segments] or None,
        )
        content.transcript_tokens = sum(c.token_count for c in chunks)

        db.query(Chunk).filter(Chunk.content_id == content.id).delete()
        db.add_all(
            Chunk(
                content_id=content.id,
                chunk_index=c.index,
                text=c.text,
                text_english=c.text_english,
                token_count=c.token_count,
                start_time=c.start_time,
                end_time=c.end_time,
            )
            for c in chunks
        )
        self.vector_store.delete_collection(content.id)
        self.vector_store.add_chunks(content.id, [asdict(c) for c in chunks])

        content.error_message = None
        self._set_status(db, content, "completed")

    @staticmethod
    def _set_status(db: Session, content: Content, status: str) -> None:
        content.status = status
        db.commit()
