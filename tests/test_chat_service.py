from app.services.chat import ChatService, build_context, format_timestamp
from tests.conftest import FakeLLM, FakeRetriever

CHUNKS = [
    {"chunk_index": 0, "text": "Intro about AI safety.", "start_time": 0.0, "similarity": 0.9},
    {"chunk_index": 3, "text": "Discussion of scaling.", "start_time": 125.0, "similarity": 0.7},
]


def test_format_timestamp():
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(125.9) == "2:05"


def test_build_context_includes_zero_and_missing_timestamps():
    context = build_context([*CHUNKS, {"text": "No time."}])
    assert "[Chunk 1] [at 0:00]: Intro about AI safety." in context
    assert "[Chunk 2] [at 2:05]: Discussion of scaling." in context
    assert "[Chunk 3]: No time." in context


def test_english_question_skips_translation():
    llm = FakeLLM("Grounded answer.")
    retriever = FakeRetriever(CHUNKS)
    response = ChatService(retriever, llm).chat(1, "What about scaling?", "en")

    assert retriever.queries == ["What about scaling?"]
    assert len(llm.calls) == 1
    assert response.message == "Grounded answer."
    assert response.confidence == 0.8
    assert response.source_chunks == CHUNKS


def test_non_english_question_is_translated_for_retrieval_but_answered_in_language():
    llm = FakeLLM("What about scaling?", "Respuesta fundamentada.")
    retriever = FakeRetriever(CHUNKS)
    response = ChatService(retriever, llm).chat(1, "¿Qué dicen sobre el escalado?", "es")

    assert retriever.queries == ["What about scaling?"]
    translate_call, answer_call = llm.calls
    assert translate_call["effort"] == "low"
    assert "español" in answer_call["system"]
    assert "¿Qué dicen sobre el escalado?" in answer_call["user"]  # original question goes to the model
    assert response.language == "es"


def test_no_relevant_chunks_refuses_without_calling_model():
    llm = FakeLLM("Unrelated?")  # only the retrieval translation runs
    response = ChatService(FakeRetriever([]), llm).chat(1, "관련 없는 질문?", "ko")

    assert len(llm.calls) == 1  # no answer generated without context
    assert response.confidence == 0.0
    assert response.source_chunks == []
    assert "비디오" in response.message
