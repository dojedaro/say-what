"""Configuration settings for Say What? platform."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App settings
    app_name: str = "Say What? · ¿Qué dijiste? · 뭐라고요?"
    debug: bool = False
    secret_key: str = "change-this-in-production"

    # Database
    database_url: str = "sqlite:///./say_what.db"

    # API Keys (for seeding demo content - optional)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Whisper settings
    whisper_model: str = "base"  # tiny, base, small, medium, large

    # Chunking settings
    chunk_size: int = 500  # tokens per chunk
    chunk_overlap: int = 50  # overlapping tokens between chunks

    # Rate limiting
    max_extractions_per_session: int = 5

    # Temp directory for audio files
    temp_dir: str = "./temp"

    # Supported languages
    supported_languages: list = ["en", "es", "ko"]
    language_names: dict = {
        "en": "English",
        "es": "Español",
        "ko": "한국어"
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# API Provider configurations
API_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "models": {
            "gpt-4o": "GPT-4o (Recommended)",
            "gpt-4o-mini": "GPT-4o Mini (Faster & Cheaper)",
        },
        "key_prefix": "sk-",
        "key_placeholder": "sk-...",
    },
    "anthropic": {
        "name": "Anthropic",
        "models": {
            "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Recommended)",
            "claude-3-5-haiku-20241022": "Claude 3.5 Haiku (Faster & Cheaper)",
            "claude-3-opus-20240229": "Claude 3 Opus (Most Capable)",
        },
        "key_prefix": "sk-ant-",
        "key_placeholder": "sk-ant-...",
    },
    "google": {
        "name": "Google Gemini",
        "models": {
            "gemini-1.5-pro": "Gemini 1.5 Pro (Recommended)",
            "gemini-1.5-flash": "Gemini 1.5 Flash (Faster & Cheaper)",
        },
        "key_prefix": "AI",
        "key_placeholder": "AI...",
    },
    "azure": {
        "name": "Microsoft Copilot (Azure)",
        "models": {
            "gpt-4o": "GPT-4o via Azure",
        },
        "key_prefix": "",
        "key_placeholder": "Your Azure API key",
        "requires_endpoint": True,
    },
}

# Demo videos configuration
DEMO_VIDEOS = [
    {
        "id": "demo-en-dario",
        "url": "https://www.youtube.com/watch?v=a3TTFErF3FY",
        "language": "en",
        "title": "Dario Amodei Interview",
        "description": "Anthropic CEO discussing AI and the future",
        "category": "Tech Interview",
    },
    {
        "id": "demo-es-checo",
        "url": "https://www.youtube.com/watch?v=b1mdi_jF_14",
        "language": "es",
        "title": "Checo Pérez Entrevista",
        "description": "Formula 1 driver interview in Spanish",
        "category": "Sports Interview",
    },
    {
        "id": "demo-ko-news",
        "url": "https://www.youtube.com/watch?v=SivMtK5ysOE",
        "language": "ko",
        "title": "한국 뉴스 리포트",
        "description": "Korean news report",
        "category": "News Report",
    },
]
