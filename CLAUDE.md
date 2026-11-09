# Say What? · ¿Qué dijiste? · 뭐라고요?

A multilingual content intelligence platform with RAG-powered chat.

## Project Overview

This is a portfolio demo showcasing:
- Multi-platform video content extraction (YouTube, TikTok, Instagram)
- Multilingual transcription and translation (English, Spanish, Korean)
- AI-powered summarization with key points extraction
- RAG-based chat with hallucination prevention
- Clean, responsive UI with dark/light mode

## Architecture

See README.md "Project structure". Key rules:
- All model calls go through `app/services/llm.py` (`create_llm_client`); never import a provider SDK elsewhere.
- Video processing lives only in `app/services/pipeline.py` (used by the API and `scripts/seed_demo.py`).
- Provider/model IDs live only in `API_PROVIDERS` in `app/config.py`.
- Blocking work (yt-dlp, Whisper, LLM calls) must run in sync `def` endpoints/tasks, not `async def`.

## Key Commands

```bash
pip install -r requirements-dev.txt     # + requirements-whisper.txt for non-YouTube videos
uvicorn app.main:app --reload
pytest                                  # offline; external services are faked
ruff check . && ruff format .
```

## Demo Videos (Pre-loaded)

1. **English**: Dario Amodei interview - https://youtube.com/watch?v=a3TTFErF3FY
2. **Spanish**: Checo Pérez interview - https://youtube.com/watch?v=b1mdi_jF_14
3. **Korean**: Korean news report - https://youtube.com/watch?v=SivMtK5ysOE

## API Providers Supported

OpenAI, Anthropic (Claude Sonnet/Opus/Haiku 5.5), Google Gemini, Azure OpenAI. See `API_PROVIDERS` in `app/config.py`.

## Design Decisions

- **Hallucination prevention**: Strict grounding, source citations, confidence thresholds
- **Multilingual chat**: User selects language, queries translated for retrieval, responses in selected language
- **Pre-loaded demos**: No API key needed to explore; user provides key for their own videos
- **Rate limiting**: 5 extractions per session to prevent abuse
