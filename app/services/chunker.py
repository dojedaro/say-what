"""Token-aware transcript chunking for retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from app.config import get_settings


@dataclass
class TextChunk:
    """A retrievable piece of transcript, optionally with its English translation."""

    index: int
    text: str
    text_english: str | None
    token_count: int
    start_time: float | None
    end_time: float | None


class Chunker:
    """Split transcripts into overlapping, token-bounded chunks.

    When timestamped segments are available they are used as natural boundaries,
    so each chunk maps back to a playable moment in the video.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding_name: str = "cl100k_base",
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk_text(
        self,
        text: str,
        text_english: str | None = None,
        timestamps: list[dict] | None = None,
    ) -> list[TextChunk]:
        """Chunk ``text``; uses ``timestamps`` ({start, end, text} segments) when given."""
        if timestamps:
            return self._chunk_with_timestamps(text_english, timestamps)
        return self._chunk_plain_text(text, text_english)

    def _chunk_plain_text(self, text: str, text_english: str | None) -> list[TextChunk]:
        tokens = self.encoding.encode(text)
        english_tokens = self.encoding.encode(text_english) if text_english else None
        step = self.chunk_size - self.chunk_overlap

        chunks: list[TextChunk] = []
        for start in range(0, len(tokens), step):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            chunk_english = None
            if english_tokens:
                # A translation has a different token count than the original,
                # so map the window proportionally instead of by raw index.
                ratio = len(english_tokens) / len(tokens)
                eng_start, eng_end = round(start * ratio), round(end * ratio)
                chunk_english = self.encoding.decode(english_tokens[eng_start:eng_end])

            chunks.append(
                TextChunk(
                    index=len(chunks),
                    text=self.encoding.decode(chunk_tokens),
                    text_english=chunk_english,
                    token_count=len(chunk_tokens),
                    start_time=None,
                    end_time=None,
                )
            )
            if end == len(tokens):
                break
        return chunks

    def _chunk_with_timestamps(
        self,
        text_english: str | None,
        timestamps: list[dict],
    ) -> list[TextChunk]:
        english_segments = _split_proportionally(text_english, len(timestamps)) if text_english else None

        chunks: list[TextChunk] = []
        window: list[int] = []  # indices into ``timestamps`` for the chunk being built
        window_tokens = 0
        last_emitted = -1  # index of the last segment already included in a chunk

        def flush() -> None:
            nonlocal last_emitted
            last_emitted = window[-1]
            first, last = timestamps[window[0]], timestamps[window[-1]]
            text = " ".join(timestamps[i].get("text", "").strip() for i in window).strip()
            english = " ".join(english_segments[i] for i in window).strip() if english_segments else None
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    text=text,
                    text_english=english or None,
                    token_count=self.count_tokens(text),
                    start_time=first.get("start", 0),
                    end_time=last.get("end", last.get("start", 0)),
                )
            )

        for i, segment in enumerate(timestamps):
            window.append(i)
            window_tokens += self.count_tokens(segment.get("text", ""))
            if window_tokens >= self.chunk_size:
                flush()
                window, window_tokens = self._overlap_tail(window, timestamps)

        # Emit the remainder unless it is only overlap already covered by the previous chunk.
        if window and window[-1] > last_emitted:
            flush()
        return chunks

    def _overlap_tail(self, window: list[int], timestamps: list[dict]) -> tuple[list[int], int]:
        """Return the trailing segments of ``window`` that fit within ``chunk_overlap`` tokens."""
        tail: list[int] = []
        tokens = 0
        for i in reversed(window[1:]):  # never carry the whole window forward
            seg_tokens = self.count_tokens(timestamps[i].get("text", ""))
            if tokens + seg_tokens > self.chunk_overlap:
                break
            tail.insert(0, i)
            tokens += seg_tokens
        return tail, tokens


def _split_proportionally(text: str, parts: int) -> list[str]:
    """Split ``text`` into ``parts`` word groups of near-equal size (approximate alignment)."""
    words = text.split()
    return [" ".join(words[len(words) * i // parts : len(words) * (i + 1) // parts]) for i in range(parts)]
