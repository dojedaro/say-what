"""Shared fixtures. The app is pointed at a throwaway SQLite DB before it is imported."""

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="say-what-tests-"))
os.environ.update(
    DATABASE_URL=f"sqlite:///{(_TMP / 'test.db').as_posix()}",
    CHROMA_DIR=str(_TMP / "chroma"),
    TEMP_DIR=str(_TMP / "temp"),
    SECRET_KEY="test-secret",
    OPENAI_API_KEY="",
    ANTHROPIC_API_KEY="",
    GOOGLE_API_KEY="",
)


class FakeLLM:
    """Records prompts and returns canned replies in order."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.model = "fake-model"

    def complete(self, system, user, *, json_output=False, effort="medium", max_tokens=16_000):
        from app.services.llm import LLMResult

        self.calls.append({"system": system, "user": user, "json_output": json_output, "effort": effort})
        return LLMResult(self.replies.pop(0), tokens_used=42)


class FakeRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.queries: list[str] = []

    def search(self, content_id, query, n_results=5, min_score=0.3):
        self.queries.append(query)
        return [c for c in self.chunks if c.get("similarity", 1) >= min_score][:n_results]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.database import Base, engine
    from app.main import app

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db():
    from app.database import SessionLocal

    session = SessionLocal()
    yield session
    session.close()
