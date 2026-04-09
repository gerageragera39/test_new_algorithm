"""Detect comments that directly address the video author.

Builds regex patterns from the configured author name (supporting Russian
morphological forms) and common second-person pronouns.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.db.models import Comment

# Second-person pronouns and forms indicating direct address
_DIRECT_ADDRESS_RE = re.compile(
    r"\b(вы|вас|вам|ваш|ваша|ваше|ваши|вашим|вашей|вашего|вашу|вами)\b",
    re.IGNORECASE,
)

# "ты"-forms
_TY_ADDRESS_RE = re.compile(
    r"\b(ты|тебя|тебе|тобой|тобою|твой|твоя|твоё|твоего|твоей|твоему|твою|твоим|твоих)\b",
    re.IGNORECASE,
)


def _build_name_pattern(author_name: str) -> re.Pattern[str] | None:
    """Build a regex that matches the author's name in various Russian forms."""
    name = author_name.strip()
    if not name:
        return None

    parts = name.split()
    variants: list[str] = []

    # Add the full name as-is
    if len(parts) >= 2:
        variants.append(re.escape(name))

    for part in parts:
        if len(part) < 2:
            continue
        stem = part.rstrip("аеёийоуыьюя")
        if len(stem) < 2:
            stem = part[: max(2, len(part) - 1)]
        # Match stem + optional Russian suffix
        variants.append(re.escape(stem) + r"\w*")

    if not variants:
        return None

    pattern = r"\b(?:" + "|".join(variants) + r")\b"
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


def classify_author_appeals(
    comments: list[Comment],
    author_name: str,
) -> list[int]:
    """Return IDs of comments that address the author directly."""
    name_re = _build_name_pattern(author_name)
    result: list[int] = []

    for comment in comments:
        text = (comment.text_raw or "").strip()
        if not text:
            continue

        has_name = name_re is not None and bool(name_re.search(text))
        has_direct = bool(_DIRECT_ADDRESS_RE.search(text))
        has_ty = bool(_TY_ADDRESS_RE.search(text))

        # Author appeal: mentions name, or uses "вы"/"ты" forms
        if has_name or has_direct or has_ty:
            result.append(comment.id)

    return result
