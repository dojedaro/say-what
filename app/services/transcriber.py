"""Transcription service using YouTube API and Whisper."""

import json
import os
from typing import Optional
from dataclasses import dataclass
from langdetect import detect

from app.config import get_settings

settings = get_settings()


@dataclass
class TranscriptSegment:
    """A segment of transcript with timestamp."""
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Result of transcription."""
    original_text: str
    translated_text: Optional[str]  # English translation if not already English
    segments: list[TranscriptSegment]
    detected_language: str


class Transcriber:
    """Transcription service supporting YouTube captions and Whisper."""

    def __init__(self):
        """Initialize transcriber."""
        self.whisper_model = None  # Lazy load

    def _load_whisper(self):
        """Lazy load Whisper model."""
        if self.whisper_model is None:
            import whisper
            self.whisper_model = whisper.load_model(settings.whisper_model)

    def get_youtube_transcript(
        self, video_id: str, preferred_language: str = "en"
    ) -> Optional[TranscriptionResult]:
        """
        Try to get YouTube's native transcript.

        Args:
            video_id: YouTube video ID
            preferred_language: Preferred language code

        Returns:
            TranscriptionResult if available, None otherwise
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            # Try to get transcript in any available language
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            transcript = None
            detected_lang = preferred_language

            # First, try to get manually created transcript
            try:
                transcript = transcript_list.find_manually_created_transcript(
                    [preferred_language, "en", "es", "ko"]
                )
                detected_lang = transcript.language_code
            except Exception:
                pass

            # Fall back to auto-generated
            if transcript is None:
                try:
                    transcript = transcript_list.find_generated_transcript(
                        [preferred_language, "en", "es", "ko"]
                    )
                    detected_lang = transcript.language_code
                except Exception:
                    # Get any available transcript
                    for t in transcript_list:
                        transcript = t
                        detected_lang = t.language_code
                        break

            if transcript is None:
                return None

            # Fetch the transcript data
            transcript_data = transcript.fetch()

            segments = [
                TranscriptSegment(
                    start=item["start"],
                    end=item["start"] + item.get("duration", 0),
                    text=item["text"],
                )
                for item in transcript_data
            ]

            original_text = " ".join(seg.text for seg in segments)

            # Get English translation if not already English
            translated_text = None
            if detected_lang != "en":
                try:
                    translated = transcript.translate("en").fetch()
                    translated_text = " ".join(item["text"] for item in translated)
                except Exception:
                    # Translation not available
                    pass

            return TranscriptionResult(
                original_text=original_text,
                translated_text=translated_text,
                segments=segments,
                detected_language=detected_lang,
            )

        except Exception as e:
            print(f"YouTube transcript not available: {e}")
            return None

    def transcribe_with_whisper(
        self, audio_path: str, translate_to_english: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio using Whisper.

        Args:
            audio_path: Path to audio file
            translate_to_english: Whether to also translate to English

        Returns:
            TranscriptionResult
        """
        self._load_whisper()

        # First, transcribe in original language
        result = self.whisper_model.transcribe(audio_path, task="transcribe")

        detected_lang = result.get("language", "en")

        segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            )
            for seg in result.get("segments", [])
        ]

        original_text = result.get("text", "").strip()

        # Translate to English if needed and requested
        translated_text = None
        if translate_to_english and detected_lang != "en":
            translate_result = self.whisper_model.transcribe(
                audio_path, task="translate"
            )
            translated_text = translate_result.get("text", "").strip()

        return TranscriptionResult(
            original_text=original_text,
            translated_text=translated_text,
            segments=segments,
            detected_language=detected_lang,
        )

    def transcribe(
        self,
        url: str,
        video_id: str,
        platform: str,
        audio_path: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Main transcription method - tries YouTube first, falls back to Whisper.

        Args:
            url: Video URL
            video_id: Video ID
            platform: Platform name
            audio_path: Optional path to downloaded audio

        Returns:
            TranscriptionResult
        """
        # Try YouTube transcript first (only for YouTube)
        if platform == "youtube":
            result = self.get_youtube_transcript(video_id)
            if result:
                return result

        # Fall back to Whisper
        if audio_path and os.path.exists(audio_path):
            return self.transcribe_with_whisper(audio_path)

        raise ValueError("No transcript available and no audio file provided")

    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            lang = detect(text)
            # Map to our supported languages
            if lang in ["en", "es", "ko"]:
                return lang
            return "en"  # Default to English
        except Exception:
            return "en"

    def segments_to_json(self, segments: list[TranscriptSegment]) -> str:
        """Convert segments to JSON string."""
        return json.dumps(
            [
                {"start": seg.start, "end": seg.end, "text": seg.text}
                for seg in segments
            ]
        )
