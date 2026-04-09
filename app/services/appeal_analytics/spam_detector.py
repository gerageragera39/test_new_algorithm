"""Detect spam comments: authors who posted identical messages.

An author is classified as spam if they posted at least two comments with
exactly the same text (matched by text_hash).  Only authors with exact
duplicates are flagged — near-duplicates are not enough to avoid false
positives (this list may be used for automated banning).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.db.models import Comment


def classify_spam(
    comments: list[Comment],
    *,
    similarity_threshold: float = 0.85,
) -> list[int]:
    """Return IDs of all comments by authors who posted identical messages.

    Groups comments by author_name.  For each author with >1 comment, checks
    whether at least two comments have the same text_hash (exact duplicate).
    If yes, ALL comments by that author are flagged as spam.

    The similarity_threshold parameter is kept for API compatibility but
    is no longer used — only exact duplicates are considered.
    """
    _ = similarity_threshold

    by_author: dict[str, list[Comment]] = {}
    for comment in comments:
        name = (comment.author_name or "").strip()
        if not name:
            continue
        by_author.setdefault(name, []).append(comment)

    spam_ids: list[int] = []
    for _author, author_comments in by_author.items():
        if len(author_comments) < 2:
            continue

        hash_counts = Counter(c.text_hash for c in author_comments)
        has_exact_duplicate = any(count >= 2 for count in hash_counts.values())

        if has_exact_duplicate:
            spam_ids.extend(c.id for c in author_comments)

    return spam_ids
