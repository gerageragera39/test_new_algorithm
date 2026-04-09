"""FastAPI dependency providers for configuration and database sessions.

Wraps core singletons as thin dependency functions so they can be injected
into route handlers via FastAPI's Depends mechanism.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db


def get_settings_dep() -> Settings:
    """Return the cached application settings instance."""
    return get_settings()


def get_db_dep() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session and ensure it is closed after use."""
    yield from get_db()
