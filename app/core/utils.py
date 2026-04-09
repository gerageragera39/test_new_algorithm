"""General-purpose utility functions for text processing, hashing, and file I/O.

Provides helpers for URL stripping, whitespace normalization, noise detection,
Cyrillic character checks, and JSON serialization used throughout the application.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LINK_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_ONLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def utcnow() -> datetime:
    """Return the current UTC-aware datetime."""
    return datetime.now(UTC)


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Strip URLs and collapse whitespace in the given text."""
    text = _LINK_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def looks_like_noise(text: str) -> bool:
    """Return True if the text is empty, punctuation-only, or too short to be meaningful."""
    if not text:
        return True
    if _ONLY_PUNCT_RE.match(text):
        return True
    without_spaces = text.replace(" ", "")
    return len(without_spaces) <= 3


def contains_cyrillic(text: str) -> bool:
    """Return True if the text contains at least one Cyrillic character."""
    return bool(_CYRILLIC_RE.search(text))


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Serialize payload as pretty-printed JSON and write it to the given path, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
