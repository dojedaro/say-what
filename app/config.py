"""Configuration settings for Say What? platform."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
INSECURE_DEFAULT_SECRET = "change-this-in-production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App settings
    app_name: str = "Say What? · ¿Qué dijiste? · 뭐라고요?"
    debug: bool = False
    secret_key: str = INSECURE_DEFAULT_SECRET

    # Storage
    database_url: str = "sqlite:///./say_what.db"
    chroma_dir: str = "./chroma_data"
    temp_dir: str = "./temp"

    # Server-side API keys (used for demo content and seeding - optional)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # Whisper settings (optional dependency, see requirements-whisper.txt)
    whisper_model: str = "base"  # tiny, base, small, medium, large

    # Chunking settings
    chunk_size: int = 500  # tokens per chunk
    chunk_overlap: int = 50  # overlapping tokens between chunks

    # Rate limiting
    max_extractions_per_session: int = 5

    # Supported languages
    supported_languages: list[str] = ["en", "es", "ko"]
    language_names: dict[str, str] = {"en": "English", "es": "Español", "ko": "한국어"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Provider catalog: the first model listed is the default for that provider.
# Model IDs were checked against each provider's SDK/docs in Nov 2026; update here only.
API_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "models": {
            "gpt-6.1-sol": "GPT-6.1 Sol",
            "gpt-6.1-terra": "GPT-6.1 Terra",
            "gpt-6-luna": "GPT-6 Luna",
        },
        "key_placeholder": "sk-...",
    },
    "anthropic": {
        "name": "Anthropic",
        "models": {
            "claude-sonnet-5-5": "Claude Sonnet 5.5 (Recommended)",
            "claude-opus-5-5": "Claude Opus 5.5 (Most Capable)",
            "claude-haiku-5-5": "Claude Haiku 5.5 (Fastest & Cheapest)",
        },
        "key_placeholder": "sk-ant-...",
    },
    "google": {
        "name": "Google Gemini",
        "models": {
            "gemini-3.5-flash": "Gemini 3.5 Flash",
            "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite (Faster & Cheaper)",
        },
        "key_placeholder": "AI...",
    },
    "azure": {
        "name": "Microsoft Azure OpenAI",
        "models": {
            "gpt-6.1-sol": "GPT-6.1 Sol (your deployment must use this name)",
        },
        "key_placeholder": "Your Azure API key",
        "requires_endpoint": True,
    },
}

DEFAULT_MODELS = {provider: next(iter(info["models"])) for provider, info in API_PROVIDERS.items()}

# Cheaper models for server-paid demo traffic (seeding + chat on pre-loaded videos).
SERVER_DEMO_MODELS = {
    "openai": DEFAULT_MODELS["openai"],
    "anthropic": "claude-haiku-5-5",
    "google": "gemini-3.5-flash-lite",
}


def server_llm_config(settings: Settings | None = None) -> dict | None:
    """Provider/model/key for server-paid calls, from whichever key is configured."""
    settings = settings or get_settings()
    for provider, key in (
        ("openai", settings.openai_api_key),
        ("anthropic", settings.anthropic_api_key),
        ("google", settings.google_api_key),
    ):
        if key:
            return {"provider": provider, "model": SERVER_DEMO_MODELS[provider], "api_key": key}
    return None


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
