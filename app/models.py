"""SQLAlchemy database models."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, Boolean
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Content(Base):
    """Main content model for extracted videos."""

    __tablename__ = "contents"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(500), unique=True, nullable=False, index=True)
    video_id = Column(String(100), index=True)
    platform = Column(String(50), nullable=False)  # youtube, tiktok, instagram
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    is_demo = Column(Boolean, default=False)

    # Metadata
    title = Column(String(500))
    author = Column(String(200))
    duration = Column(Integer)  # seconds
    views = Column(Integer)
    thumbnail_url = Column(String(500))

    # Language
    detected_language = Column(String(10))  # en, es, ko

    # Transcripts
    original_transcript = Column(Text)  # In source language
    translated_transcript = Column(Text)  # English translation (null if already English)
    transcript_with_timestamps = Column(Text)  # JSON: [{start, end, text}, ...]

    # AI Generated Content
    summary = Column(Text)
    key_points = Column(Text)  # JSON array
    topics = Column(Text)  # JSON array

    # Token counts
    transcript_tokens = Column(Integer)
    summary_tokens = Column(Integer)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Error tracking
    error_message = Column(Text)

    # Relationships
    chunks = relationship("Chunk", back_populates="content", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="content", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Content(id={self.id}, title='{self.title[:30] if self.title else 'N/A'}...')>"


class Chunk(Base):
    """RAG-ready content chunks for vector search."""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(Integer, ForeignKey("contents.id"), nullable=False)

    # Chunk data
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    text_english = Column(Text)  # English version for embedding
    token_count = Column(Integer)

    # Timestamps from video
    start_time = Column(Float)  # seconds from start
    end_time = Column(Float)

    # Embedding stored in ChromaDB, reference ID here
    embedding_id = Column(String(100))

    # Relationships
    content = relationship("Content", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk(id={self.id}, content_id={self.content_id}, index={self.chunk_index})>"


class ChatMessage(Base):
    """Chat history for RAG conversations."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(Integer, ForeignKey("contents.id"), nullable=False)
    session_id = Column(String(100), nullable=False, index=True)

    # Message
    role = Column(String(20), nullable=False)  # user, assistant
    message = Column(Text, nullable=False)
    language = Column(String(10))  # en, es, ko

    # For assistant messages - source chunks used
    source_chunks = Column(Text)  # JSON array of chunk IDs
    confidence_score = Column(Float)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    content = relationship("Content", back_populates="chat_messages")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role='{self.role}')>"
