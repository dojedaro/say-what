"""Routers package for Say What? platform."""

from app.routers.api import router as api_router
from app.routers.web import router as web_router
from app.routers.chat import router as chat_router

__all__ = ["api_router", "web_router", "chat_router"]
