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

```
app/
├── main.py              # FastAPI entry point
├── config.py            # Environment settings
├── database.py          # SQLAlchemy setup
├── models.py            # Database models (Content, Chunk)
├── schemas.py           # Pydantic schemas
├── routers/
│   ├── api.py           # REST API endpoints
│   ├── web.py           # Web UI routes
│   └── chat.py          # RAG chat endpoints
├── services/
│   ├── extractor.py     # yt-dlp video extraction
│   ├── transcriber.py   # Whisper + YouTube transcripts
│   ├── summarizer.py    # Multi-provider AI summarization
│   ├── chunker.py       # RAG-ready chunking with tiktoken
│   ├── vector_store.py  # ChromaDB embeddings
│   └── chat.py          # Grounded RAG chat
└── templates/           # Jinja2 + Tailwind + HTMX
```

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload

# Run with specific host/port
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Demo Videos (Pre-loaded)

1. **English**: Dario Amodei interview - https://youtube.com/watch?v=a3TTFErF3FY
2. **Spanish**: Checo Pérez interview - https://youtube.com/watch?v=b1mdi_jF_14
3. **Korean**: Korean news report - https://youtube.com/watch?v=SivMtK5ysOE

## API Providers Supported

- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus)
- Google Gemini (Gemini 1.5 Pro, Gemini 1.5 Flash)
- Microsoft Azure OpenAI (Copilot)

## Design Decisions

- **Hallucination prevention**: Strict grounding, source citations, confidence thresholds
- **Multilingual chat**: User selects language, queries translated for retrieval, responses in selected language
- **Pre-loaded demos**: No API key needed to explore; user provides key for their own videos
- **Rate limiting**: 5 extractions per session to prevent abuse
