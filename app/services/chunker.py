"""Content chunking service for RAG."""

import json
from typing import Optional
from dataclasses import dataclass
import tiktoken

from app.config import get_settings

settings = get_settings()


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    index: int
    text: str
    text_english: Optional[str]
    token_count: int
    start_time: Optional[float]
    end_time: Optional[float]


class Chunker:
    """Chunk text for RAG applications."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target tokens per chunk
            chunk_overlap: Overlapping tokens between chunks
            encoding_name: Tiktoken encoding name
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def chunk_text(
        self,
        text: str,
        text_english: Optional[str] = None,
        timestamps: Optional[list[dict]] = None,
    ) -> list[TextChunk]:
        """
        Chunk text into RAG-ready pieces.

        Args:
            text: Original text to chunk
            text_english: English translation (optional)
            timestamps: List of {start, end, text} segments (optional)

        Returns:
            List of TextChunk objects
        """
        if timestamps:
            return self._chunk_with_timestamps(text, text_english, timestamps)
        else:
            return self._chunk_plain_text(text, text_english)

    def _chunk_plain_text(
        self, text: str, text_english: Optional[str] = None
    ) -> list[TextChunk]:
        """Chunk text without timestamp information."""
        tokens = self.encoding.encode(text)
        english_tokens = self.encoding.encode(text_english) if text_english else None

        chunks = []
        start_idx = 0
        chunk_index = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))

            # Get chunk text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)

            # Get corresponding English chunk if available
            chunk_english = None
            if english_tokens:
                eng_end_idx = min(start_idx + self.chunk_size, len(english_tokens))
                chunk_eng_tokens = english_tokens[start_idx:eng_end_idx]
                chunk_english = self.encoding.decode(chunk_eng_tokens)

            chunks.append(
                TextChunk(
                    index=chunk_index,
                    text=chunk_text,
                    text_english=chunk_english,
                    token_count=len(chunk_tokens),
                    start_time=None,
                    end_time=None,
                )
            )

            chunk_index += 1
            start_idx = end_idx - self.chunk_overlap
            if start_idx >= end_idx:
                break

        return chunks

    def _chunk_with_timestamps(
        self,
        text: str,
        text_english: Optional[str],
        timestamps: list[dict],
    ) -> list[TextChunk]:
        """Chunk text using timestamp segments for natural boundaries."""
        chunks = []
        current_chunk_text = ""
        current_chunk_english = ""
        current_start_time = None
        current_end_time = None
        chunk_index = 0

        # If we have English text, try to align it with timestamps
        # This is approximate - we'll just chunk based on original timestamps
        english_segments = None
        if text_english and len(timestamps) > 0:
            # Simple approach: split English text proportionally
            words = text_english.split()
            words_per_segment = max(1, len(words) // len(timestamps))
            english_segments = []
            for i in range(len(timestamps)):
                start_word = i * words_per_segment
                end_word = min((i + 1) * words_per_segment, len(words))
                english_segments.append(" ".join(words[start_word:end_word]))

        for i, seg in enumerate(timestamps):
            seg_text = seg.get("text", "")
            seg_english = english_segments[i] if english_segments and i < len(english_segments) else None

            if current_start_time is None:
                current_start_time = seg.get("start", 0)

            current_chunk_text += " " + seg_text
            if seg_english:
                current_chunk_english += " " + seg_english
            current_end_time = seg.get("end", seg.get("start", 0))

            # Check if we've reached chunk size
            current_tokens = self.count_tokens(current_chunk_text)
            if current_tokens >= self.chunk_size:
                chunks.append(
                    TextChunk(
                        index=chunk_index,
                        text=current_chunk_text.strip(),
                        text_english=current_chunk_english.strip() if current_chunk_english else None,
                        token_count=current_tokens,
                        start_time=current_start_time,
                        end_time=current_end_time,
                    )
                )

                chunk_index += 1
                current_chunk_text = ""
                current_chunk_english = ""
                current_start_time = None
                current_end_time = None

        # Don't forget the last chunk
        if current_chunk_text.strip():
            chunks.append(
                TextChunk(
                    index=chunk_index,
                    text=current_chunk_text.strip(),
                    text_english=current_chunk_english.strip() if current_chunk_english else None,
                    token_count=self.count_tokens(current_chunk_text),
                    start_time=current_start_time,
                    end_time=current_end_time,
                )
            )

        return chunks

    def chunks_to_json(self, chunks: list[TextChunk]) -> str:
        """Convert chunks to JSON string."""
        return json.dumps(
            [
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
        )
