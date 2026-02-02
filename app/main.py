"""FastAPI application entry point for Say What? platform."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.config import get_settings
from app.database import init_db
from app.routers import api_router, web_router, chat_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    init_db()
    yield
    # Shutdown (cleanup if needed)


app = FastAPI(
    title=settings.app_name,
    description="Multi-language content intelligence with RAG-powered chat",
    version="1.0.0",
    lifespan=lifespan,
)

# Add session middleware for rate limiting and user tracking
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    max_age=3600,  # 1 hour
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(api_router)
app.include_router(chat_router)
app.include_router(web_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "app": settings.app_name}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
