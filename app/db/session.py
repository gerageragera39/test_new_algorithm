"""Database engine and session factory configuration.

Creates the SQLAlchemy engine from application settings and exposes a session
generator suitable for use with FastAPI dependency injection.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and close it when the caller is done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
