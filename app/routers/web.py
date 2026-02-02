"""Web UI routes using Jinja2 templates."""

import json
import uuid
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Content
from app.config import get_settings, API_PROVIDERS, DEMO_VIDEOS

router = APIRouter(tags=["web"])
settings = get_settings()

templates = Jinja2Templates(directory="app/templates")


def get_or_create_session_id(request: Request) -> str:
    """Get or create a session ID for the user."""
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session["session_id"] = session_id
    return session_id


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    """Home page with demo videos and extraction form."""
    # Get demo content
    demo_contents = db.query(Content).filter(Content.is_demo == True).all()

    # Get user's recent extractions
    session_id = get_or_create_session_id(request)
    extraction_count = request.session.get("extraction_count", 0)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "demo_contents": demo_contents,
            "demo_videos": DEMO_VIDEOS,
            "providers": API_PROVIDERS,
            "extraction_count": extraction_count,
            "max_extractions": settings.max_extractions_per_session,
            "languages": settings.language_names,
        },
    )


@router.get("/content/{content_id}", response_class=HTMLResponse)
async def content_detail(
    request: Request,
    content_id: int,
    db: Session = Depends(get_db),
):
    """Content detail page with transcript, summary, and chat."""
    content = db.query(Content).filter(Content.id == content_id).first()

    if not content:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "app_name": settings.app_name,
                "error": "Content not found",
            },
            status_code=404,
        )

    # Parse JSON fields
    key_points = json.loads(content.key_points) if content.key_points else []
    topics = json.loads(content.topics) if content.topics else []

    # Get session ID for chat
    session_id = get_or_create_session_id(request)

    # Check if API key is needed for chat
    needs_api_key = not content.is_demo or (
        not settings.openai_api_key and not settings.anthropic_api_key
    )

    return templates.TemplateResponse(
        "content.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "content": content,
            "key_points": key_points,
            "topics": topics,
            "session_id": session_id,
            "needs_api_key": needs_api_key,
            "providers": API_PROVIDERS,
            "languages": settings.language_names,
        },
    )


@router.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """About page explaining the project."""
    return templates.TemplateResponse(
        "about.html",
        {
            "request": request,
            "app_name": settings.app_name,
        },
    )
