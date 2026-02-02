# Say What? · ¿Qué dijiste? · 뭐라고요?

A multilingual content intelligence platform with RAG-powered chat.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 What is this?

**Say What?** extracts transcripts from YouTube videos, translates them across languages (English, Spanish, Korean), generates AI-powered summaries, and enables RAG-powered conversations with the content.

### Features

- 🌍 **Multilingual** - Transcription and translation between EN/ES/KO
- 🤖 **AI Summaries** - Intelligent summaries and key points extraction
- 💬 **RAG Chat** - Ask questions about video content with cited sources
- 🛡️ **Grounded Responses** - Strict hallucination prevention
- 📦 **Export Ready** - Download as JSON or Markdown

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg (for audio processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/say-what.git
cd say-what

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your API keys (optional - only needed for seeding demo content)

# Initialize database and run
uvicorn app.main:app --reload
```

Visit http://localhost:8000 to see the app.

### Seed Demo Content (Optional)

To pre-load the 3 demo videos:

```bash
python scripts/seed_demo.py
```

This requires an API key in your `.env` file.

## 🏗️ Architecture

```
app/
├── main.py              # FastAPI entry point
├── config.py            # Environment settings
├── database.py          # SQLAlchemy setup
├── models.py            # Database models
├── schemas.py           # Pydantic schemas
├── routers/
│   ├── api.py           # REST API endpoints
│   ├── chat.py          # RAG chat endpoints
│   └── web.py           # Web UI routes
├── services/
│   ├── extractor.py     # yt-dlp video extraction
│   ├── transcriber.py   # Whisper + YouTube transcripts
│   ├── summarizer.py    # Multi-provider summarization
│   ├── chunker.py       # RAG-ready chunking
│   ├── vector_store.py  # ChromaDB embeddings
│   └── chat.py          # Grounded RAG chat
└── templates/           # Jinja2 + Tailwind + Alpine.js
```

## 🔑 API Providers Supported

| Provider | Models |
|----------|--------|
| OpenAI | GPT-4o, GPT-4o Mini |
| Anthropic | Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus |
| Google | Gemini 1.5 Pro, Gemini 1.5 Flash |
| Azure | GPT-4o via Azure OpenAI |

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/extract` | Submit URL for processing |
| GET | `/api/content/{id}` | Get processed content |
| GET | `/api/content/{id}/chunks` | Get RAG-ready chunks |
| GET | `/api/content/{id}/export` | Export as JSON/MD |
| GET | `/api/contents` | List all content |
| POST | `/api/chat` | Send chat message |
| GET | `/api/chat/history/{id}/{session}` | Get chat history |

## 🎬 Demo Videos

The platform includes 3 pre-loaded demo videos showcasing multilingual capabilities:

1. **English** - Dario Amodei (Anthropic CEO) interview
2. **Spanish** - Checo Pérez (F1 driver) interview
3. **Korean** - Korean news report

## 🚢 Deployment

### Render

1. Connect your GitHub repository
2. Create a new Web Service
3. Set environment variables in Render dashboard
4. Deploy!

The `render.yaml` file is included for easy deployment.

### Docker

```bash
docker build -t say-what .
docker run -p 8000:8000 --env-file .env say-what
```

## 🛠️ Tech Stack

- **Backend**: FastAPI, SQLAlchemy, Python 3.11+
- **Frontend**: Jinja2, Tailwind CSS, Alpine.js, HTMX
- **AI**: OpenAI, Anthropic, Google Gemini APIs
- **Vector Store**: ChromaDB
- **Transcription**: yt-dlp, OpenAI Whisper, YouTube Transcript API

## 📝 Environment Variables

```env
# Required for demo seeding (optional for visitors)
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...

# App settings
DEBUG=false
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///./say_what.db
```

## 📄 License

MIT License - feel free to use this for your own portfolio!

---

Built with ❤️ as a portfolio demonstration of full-stack AI development.
