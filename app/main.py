"""FastAPI application entry point for Say What? platform."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.config import BASE_DIR, INSECURE_DEFAULT_SECRET, get_settings
from app.database import init_db
from app.routers import api_router, chat_router, web_router

settings = get_settings()

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.secret_key == INSECURE_DEFAULT_SECRET and not settings.debug:
        logger.warning("SECRET_KEY is the insecure default; session cookies can be forged. Set SECRET_KEY.")
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    description="Multi-language content intelligence with RAG-powered chat",
    version="1.1.0",
    lifespan=lifespan,
)

# Signed cookie session: holds the anonymous session id and per-session extraction count.
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, max_age=3600)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

app.include_router(api_router)
app.include_router(chat_router)
app.include_router(web_router)


@app.get("/health")
def health_check():
    return {"status": "healthy", "app": settings.app_name}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
