"""HTTP-level tests. External work (downloads, LLMs, embeddings) is stubbed out."""

import json

import pytest

from app.models import Chunk, Content
from app.routers import api as api_module
from app.routers import chat as chat_module
from tests.conftest import FakeLLM, FakeRetriever

EXTRACT_BODY = {
    "url": "https://www.youtube.com/watch?v=a3TTFErF3FY",
    "provider": "anthropic",
    "model": "claude-sonnet-5-5",
    "api_key": "sk-ant-test",
}


@pytest.fixture
def queued(monkeypatch):
    """Capture background jobs instead of running the real pipeline."""
    jobs = []
    monkeypatch.setattr(api_module, "process_content", lambda *args: jobs.append(args))
    return jobs


def make_completed_content(db, **overrides) -> Content:
    content = Content(
        url="https://www.youtube.com/watch?v=xxxxxxxxxxx",
        video_id="xxxxxxxxxxx",
        platform="youtube",
        status="completed",
        title="Demo",
        author="Someone",
        detected_language="es",
        original_transcript="hola",
        translated_transcript="hello",
        summary="A summary.",
        key_points=json.dumps(["point one"]),
        topics=json.dumps(["topic"]),
        **overrides,
    )
    db.add(content)
    db.commit()
    db.add(Chunk(content_id=content.id, chunk_index=0, text="hola", start_time=0.0, end_time=5.0))
    db.commit()
    return content


def test_health(client):
    assert client.get("/health").json()["status"] == "healthy"


def test_providers_lists_current_models(client):
    providers = {p["id"]: p for p in client.get("/api/providers").json()}
    assert set(providers) == {"openai", "anthropic", "google", "azure"}
    assert "claude-sonnet-5-5" in providers["anthropic"]["models"]
    assert providers["azure"]["requires_endpoint"] is True


class TestExtract:
    def test_queues_processing(self, client, queued):
        res = client.post("/api/extract", json=EXTRACT_BODY)
        assert res.status_code == 200
        assert res.json()["status"] == "pending"
        assert queued[0][1:] == ("anthropic", "claude-sonnet-5-5", "sk-ant-test", None)

    @pytest.mark.parametrize(
        "patch, status",
        [
            ({"api_key": "  "}, 400),
            ({"provider": "azure"}, 400),  # missing endpoint
            ({"url": "https://vimeo.com/1"}, 400),
            ({"provider": "mistral"}, 422),
        ],
    )
    def test_validation(self, client, queued, patch, status):
        assert client.post("/api/extract", json=EXTRACT_BODY | patch).status_code == status
        assert queued == []

    def test_duplicate_url_is_not_reprocessed(self, client, queued):
        first = client.post("/api/extract", json=EXTRACT_BODY).json()
        second = client.post("/api/extract", json=EXTRACT_BODY).json()
        assert second["id"] == first["id"]
        assert len(queued) == 1

    def test_failed_url_can_be_retried(self, client, queued, db):
        content_id = client.post("/api/extract", json=EXTRACT_BODY).json()["id"]
        content = db.get(Content, content_id)
        content.status, content.error_message = "failed", "boom"
        db.commit()

        res = client.post("/api/extract", json=EXTRACT_BODY).json()
        assert res == {"id": content_id, "status": "pending", "progress": "Starting extraction...", "error": None}
        assert len(queued) == 2

    def test_rate_limited_per_session(self, client, queued):
        for i in range(5):
            body = EXTRACT_BODY | {"url": f"https://youtu.be/{'abcdefghij' + str(i)}"}
            assert client.post("/api/extract", json=body).status_code == 200
        res = client.post("/api/extract", json=EXTRACT_BODY | {"url": "https://youtu.be/zzzzzzzzzzz"})
        assert res.status_code == 429


class TestContent:
    def test_get_status_and_chunks(self, client, db):
        content = make_completed_content(db)
        assert client.get(f"/api/content/{content.id}").json()["key_points"] == ["point one"]
        assert client.get(f"/api/content/{content.id}/status").json()["progress"] == "Processing complete!"

        chunks = client.get(f"/api/content/{content.id}/chunks").json()
        assert chunks[0]["timestamp_link"].endswith("&t=0s")  # 0s is a valid timestamp

    def test_missing_content_is_404(self, client):
        assert client.get("/api/content/999").status_code == 404
        assert client.get("/api/content/999/export").status_code == 404

    @pytest.mark.parametrize("fmt, needle", [("json", '"summary": "A summary."'), ("markdown", "- point one")])
    def test_export(self, client, db, fmt, needle):
        content = make_completed_content(db)
        res = client.get(f"/api/content/{content.id}/export", params={"format": fmt})
        assert res.status_code == 200
        assert needle in res.json()["content"]

    def test_export_rejects_unknown_format(self, client, db):
        content = make_completed_content(db)
        assert client.get(f"/api/content/{content.id}/export", params={"format": "pdf"}).status_code == 422


class TestChat:
    @pytest.fixture
    def fakes(self, client, monkeypatch):
        llm = FakeLLM("It says hello.")
        retriever = FakeRetriever([{"chunk_index": 0, "text": "hola", "start_time": 0.0, "similarity": 0.8}])
        monkeypatch.setattr(chat_module, "create_llm_client", lambda *a, **k: llm)
        client.app.dependency_overrides[chat_module.get_retriever] = lambda: retriever
        yield llm
        client.app.dependency_overrides.clear()

    def test_chat_round_trip_and_history(self, client, db, fakes):
        content = make_completed_content(db)
        body = {"content_id": content.id, "message": "What is said?", "session_id": "s1", "api_key": "k"}

        res = client.post("/api/chat", json=body)
        assert res.status_code == 200
        data = res.json()
        assert data["message"]["message"] == "It says hello."
        assert data["sources"][0]["timestamp_link"].endswith("&t=0s")

        history = client.get(f"/api/chat/history/{content.id}/s1").json()
        assert [m["role"] for m in history] == ["user", "assistant"]

        client.delete(f"/api/chat/history/{content.id}/s1")
        assert client.get(f"/api/chat/history/{content.id}/s1").json() == []

    def test_user_content_requires_key(self, client, db, fakes):
        content = make_completed_content(db)
        body = {"content_id": content.id, "message": "q", "session_id": "s1"}
        assert client.post("/api/chat", json=body).status_code == 400

    def test_unprocessed_content_rejected(self, client, db, fakes):
        content = make_completed_content(db)
        content.status = "transcribing"
        db.commit()
        body = {"content_id": content.id, "message": "q", "session_id": "s1", "api_key": "k"}
        assert client.post("/api/chat", json=body).status_code == 400


def test_web_pages_render(client, db):
    content = make_completed_content(db)
    assert client.get("/").status_code == 200
    assert client.get("/about").status_code == 200
    page = client.get(f"/content/{content.id}")
    assert page.status_code == 200
    assert "claude-sonnet-5-5" in page.text  # model defaults injected from config
    assert client.get("/content/999").status_code == 404
