"""Video extraction service using yt-dlp."""

import re
import os
import json
from typing import Optional
from dataclasses import dataclass
import yt_dlp

from app.config import get_settings

settings = get_settings()


@dataclass
class VideoMetadata:
    """Extracted video metadata."""
    video_id: str
    platform: str
    title: str
    author: str
    duration: int  # seconds
    views: Optional[int]
    thumbnail_url: str
    url: str


class VideoExtractor:
    """Extract video metadata and audio using yt-dlp."""

    PLATFORM_PATTERNS = {
        "youtube": [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        ],
        "tiktok": [
            r"tiktok\.com/@[\w.-]+/video/(\d+)",
            r"tiktok\.com/t/(\w+)",
        ],
        "instagram": [
            r"instagram\.com/(?:p|reel)/([a-zA-Z0-9_-]+)",
        ],
    }

    def __init__(self):
        """Initialize extractor with temp directory."""
        self.temp_dir = settings.temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def detect_platform(self, url: str) -> tuple[str, str]:
        """
        Detect platform and extract video ID from URL.

        Returns:
            Tuple of (platform, video_id)

        Raises:
            ValueError: If URL doesn't match any supported platform
        """
        for platform, patterns in self.PLATFORM_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return platform, match.group(1)

        raise ValueError(f"Unsupported URL format: {url}")

    def extract_metadata(self, url: str) -> VideoMetadata:
        """
        Extract video metadata without downloading.

        Args:
            url: Video URL

        Returns:
            VideoMetadata object
        """
        platform, video_id = self.detect_platform(url)

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "skip_download": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        return VideoMetadata(
            video_id=video_id,
            platform=platform,
            title=info.get("title", "Unknown Title"),
            author=info.get("uploader", info.get("channel", "Unknown")),
            duration=info.get("duration", 0),
            views=info.get("view_count"),
            thumbnail_url=info.get("thumbnail", ""),
            url=url,
        )

    def download_audio(self, url: str) -> str:
        """
        Download audio from video for transcription.

        Args:
            url: Video URL

        Returns:
            Path to downloaded audio file
        """
        platform, video_id = self.detect_platform(url)
        output_path = os.path.join(self.temp_dir, f"{video_id}.%(ext)s")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "outtmpl": output_path,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Return the path to the MP3 file
        return os.path.join(self.temp_dir, f"{video_id}.mp3")

    def cleanup_audio(self, audio_path: str):
        """Remove temporary audio file."""
        if os.path.exists(audio_path):
            os.remove(audio_path)

    def get_youtube_timestamp_url(self, video_id: str, seconds: float) -> str:
        """Generate YouTube URL with timestamp."""
        return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"
