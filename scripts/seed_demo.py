#!/usr/bin/env python3
"""Seed the database with the pre-processed demo videos.

Usage: python scripts/seed_demo.py   (needs OPENAI_API_KEY, ANTHROPIC_API_KEY or GOOGLE_API_KEY in .env)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEMO_VIDEOS, server_llm_config  # noqa: E402
from app.database import get_db_context, init_db  # noqa: E402
from app.models import Content  # noqa: E402
from app.services.extractor import VideoExtractor  # noqa: E402
from app.services.pipeline import ContentPipeline  # noqa: E402
from app.services.summarizer import Summarizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("seed_demo")


def seed_demo_content() -> int:
    """Process every demo video not already stored. Returns the number of failures."""
    config = server_llm_config()
    if config is None:
        logger.error("No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY or GOOGLE_API_KEY in .env")
        return 1
    logger.info("Using %s / %s", config["provider"], config["model"])

    init_db()
    summarizer = Summarizer.create(config["provider"], config["api_key"], config["model"])
    pipeline = ContentPipeline()
    failures = 0

    with get_db_context() as db:
        for i, video in enumerate(DEMO_VIDEOS, 1):
            logger.info("[%d/%d] %s (%s)", i, len(DEMO_VIDEOS), video["title"], video["url"])

            content = db.query(Content).filter(Content.url == video["url"]).first()
            if content and content.status == "completed":
                logger.info("    already processed (id=%s), skipping", content.id)
                continue
            if content is None:
                platform, video_id = VideoExtractor.detect_platform(video["url"])
                content = Content(url=video["url"], platform=platform, video_id=video_id, is_demo=True)
                db.add(content)
                db.commit()

            try:
                pipeline.run(db, content, summarizer)
                logger.info("    done (id=%s)", content.id)
            except Exception:
                failures += 1  # already logged and recorded by the pipeline

    logger.info("Demo seeding complete with %d failure(s)", failures)
    return failures


if __name__ == "__main__":
    sys.exit(1 if seed_demo_content() else 0)
