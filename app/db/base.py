"""SQLAlchemy declarative base and shared mixins.

Provides the ORM base class and a reusable timestamp mixin that automatically
tracks row creation and update times.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""

    type_annotation_map: dict[Any, Any] = {}


class TimestampMixin:
    """Mixin that adds created_at and updated_at timestamp columns to a model."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
