FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# FFmpeg is needed by yt-dlp/Whisper to extract audio from non-YouTube videos
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so code changes don't invalidate this layer.
# Includes Whisper (PyTorch); build with --build-arg WITH_WHISPER=false for a ~1 GB smaller image
# that only supports videos with YouTube captions.
ARG WITH_WHISPER=true
COPY requirements.txt requirements-whisper.txt ./
RUN if [ "$WITH_WHISPER" = "true" ]; then pip install -r requirements-whisper.txt; else pip install -r requirements.txt; fi

COPY app ./app
COPY static ./static
COPY scripts ./scripts

RUN useradd --create-home appuser && mkdir -p data temp chroma_data && chown -R appuser /app
USER appuser

EXPOSE 8000

# python:3.11-slim has no curl, so probe with the standard library
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
