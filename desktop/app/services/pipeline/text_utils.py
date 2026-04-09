"""Text utility functions extracted from the monolithic pipeline module.

Provides helpers for text normalisation, clipping, keyword extraction,
cluster-title validation, and duplicate detection.  Every function is
a standalone callable (no class dependency) so it can be imported from
both ``runner.py`` and other pipeline sub-modules.
"""

from __future__ import annotations

import re

from app.db.models import Comment
from app.schemas.domain import ProcessedComment

# ---------------------------------------------------------------------------
# Compiled regular expressions
# ---------------------------------------------------------------------------

_CLUSTER_TOKEN_RE = re.compile(r"\w{4,}", re.UNICODE)
_MATCH_TEXT_RE = re.compile(r"[\W_]+", re.UNICODE)
_TITLE_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_TITLE_BAD_TOKEN_RE = re.compile(
    r"\b("
    r"разное|прочее|misc|other|комментарии|обсуждение|разбор|новости|видео|политика|"
    r"спасибо|дело|меня|ваши|есть|просто|вообще"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Sentinel constants
# ---------------------------------------------------------------------------

_UNDETERMINED_POSITION_KEY = "undetermined"
_UNDETERMINED_POSITION_TITLE = "Неопределенные"
_UNCERTAIN_TOPIC_LABEL = "Разное / Неопределенные комментарии"

# ---------------------------------------------------------------------------
# Stopword set used by several token-extraction helpers
# ---------------------------------------------------------------------------

_CLUSTER_STOPWORDS: set[str] = {
    "это",
    "как",
    "что",
    "для",
    "или",
    "если",
    "про",
    "все",
    "сегодня",
    "завтра",
    "когда",
    "очень",
    "тоже",
    "почему",
    "просто",
    "чтобы",
    "который",
    "которые",
    "потому",
    "будет",
    "нужно",
    "можно",
    "канал",
    "видео",
    "news",
    "video",
    "comments",
    "того",
    "только",
    "этого",
    "этом",
    "этой",
    "такой",
    "такая",
    "такие",
    "именно",
}

# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def comment_text_for_output(comment: ProcessedComment | Comment) -> str:
    """Return clean single-line text suitable for display or LLM input."""
    raw = (
        getattr(comment, "text_raw", None) or getattr(comment, "text_normalized", "") or ""
    ).strip()
    return " ".join(raw.split())


def clip_text_for_llm(text: str, *, max_chars: int) -> str:
    """Clip *text* to at most *max_chars* characters, appending ``...`` when truncated."""
    cleaned = " ".join(str(text or "").split()).strip()
    if max_chars <= 0 or not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    clipped = cleaned[: max_chars - 3].rstrip(" ,;:-")
    return f"{clipped}..."


def normalize_text_for_matching(text: str) -> str:
    """Lowercase and collapse whitespace for fuzzy matching."""
    return " ".join(str(text or "").lower().split())


def compact_text_for_matching(text: str) -> str:
    """Remove all non-word characters for compact comparison."""
    return _MATCH_TEXT_RE.sub("", text)


def is_question_comment_text(text: str) -> bool:
    """Return ``True`` when *text* contains a question-mark character."""
    normalized = " ".join((text or "").split())
    return "?" in normalized or "\u00bf" in normalized or "\uff1f" in normalized


def text_token_set(text: str, stopwords: set[str] | None = None) -> set[str]:
    """Extract the set of 4+ character non-stopword tokens from *text*.

    Parameters
    ----------
    text:
        Raw or normalised text.
    stopwords:
        Optional stopword set.  Falls back to :data:`_CLUSTER_STOPWORDS`.
    """
    if stopwords is None:
        stopwords = _CLUSTER_STOPWORDS
    return {
        token
        for token in _CLUSTER_TOKEN_RE.findall((text or "").lower())
        if token not in stopwords and len(token) >= 4
    }


def sanitize_cluster_title(
    title: str,
    comments: list[ProcessedComment],
) -> str:
    """Validate and clean a cluster title.

    Returns an empty string when the title is deemed too generic,
    malformed, or unsupported by the cluster's comments.
    """
    cleaned = " ".join(str(title or "").split()).strip(" -:;,.")
    if not cleaned:
        return ""
    if "/" in cleaned or "\\" in cleaned:
        return ""
    tokens = [token.lower() for token in _TITLE_TOKEN_RE.findall(cleaned)]
    if len(tokens) < 3 or len(tokens) > 10:
        return ""
    if any(token in {"misc", "other", "разное", "прочее"} for token in tokens):
        return ""
    generic_tokens = {
        "comment",
        "comments",
        "topic",
        "topics",
        "discussion",
        "комментарии",
        "обсуждение",
        "разбор",
        "новости",
        "видео",
        "политика",
        "спасибо",
        "дело",
        "меня",
        "ваши",
        "есть",
        "просто",
        "вообще",
    }
    # Also filter short connector/function words that don't carry meaning.
    _function_words = {
        "и",
        "а",
        "но",
        "или",
        "что",
        "как",
        "это",
        "для",
        "все",
        "они",
        "он",
        "она",
        "мы",
        "вы",
        "то",
        "не",
        "да",
        "нет",
        "уже",
        "еще",
        "его",
        "ее",
        "их",
        "мне",
        "нам",
        "вам",
        "был",
        "была",
        "было",
        "были",
        "быть",
        "при",
        "без",
        "под",
        "над",
        "между",
    }
    non_generic = [
        token for token in tokens if token not in generic_tokens and token not in _function_words
    ]
    if len(non_generic) < 2:
        return ""
    if _TITLE_BAD_TOKEN_RE.search(cleaned) and len(non_generic) < 3:
        return ""

    # Reject titles with zero lexical support from cluster comments.
    joined_comments = " ".join((comment.text_normalized or "").lower() for comment in comments[:80])
    support_hits = sum(1 for token in non_generic[:4] if token in joined_comments)
    if support_hits <= 0 and len(non_generic) >= 2:
        return ""
    return cleaned


def extract_salient_comment_keywords(
    comments: list[str],
    *,
    max_keywords: int = 5,
) -> list[str]:
    """Extract the most salient keywords from *comments* using TF-IDF heuristics.

    Parameters
    ----------
    comments:
        Plain-text comment strings.
    max_keywords:
        Maximum number of keywords to return.
    """
    token_tf: dict[str, int] = {}
    token_df: dict[str, int] = {}
    docs = 0
    low_value_tokens = {
        "более",
        "менее",
        "теперь",
        "просто",
        "только",
        "его",
        "ее",
        "её",
        "их",
        "они",
        "она",
        "оно",
        "their",
        "about",
        "very",
        "just",
    }
    for text in comments[:24]:
        normalized = " ".join((text or "").split()).strip().lower()
        if not normalized:
            continue
        docs += 1
        seen: set[str] = set()
        for token in _CLUSTER_TOKEN_RE.findall(normalized):
            if token in _CLUSTER_STOPWORDS or token in low_value_tokens:
                continue
            if token.isdigit():
                continue
            token_tf[token] = token_tf.get(token, 0) + 1
            seen.add(token)
        for token in seen:
            token_df[token] = token_df.get(token, 0) + 1
    if docs <= 0:
        return []
    min_tf = 2 if docs >= 6 else 1
    scored: list[tuple[float, str]] = []
    for token, tf in token_tf.items():
        if tf < min_tf:
            continue
        df_ratio = token_df.get(token, 1) / float(max(1, docs))
        if df_ratio > 0.72:
            continue
        scored.append((tf * (1.6 - df_ratio), token))
    scored.sort(key=lambda item: item[0], reverse=True)
    result: list[str] = []
    for _, token in scored:
        if token in result:
            continue
        result.append(token)
        if len(result) >= max(1, max_keywords):
            break
    return result


def is_duplicate_text_signature(
    *,
    key: str,
    tokens: set[str],
    seen_keys: set[str],
    seen_token_sets: list[set[str]],
    threshold: float,
) -> bool:
    """Return ``True`` when *key* or its token set is a near-duplicate of an already-seen entry.

    Parameters
    ----------
    key:
        Compact text key (e.g. from :func:`compact_text_for_matching`).
    tokens:
        Token set for the text.
    seen_keys:
        Already-processed compact keys.
    seen_token_sets:
        Already-processed token sets.
    threshold:
        Jaccard similarity threshold for considering two token sets duplicates.
    """
    if key in seen_keys:
        return True
    if not tokens:
        return False
    for candidate_tokens in seen_token_sets:
        if token_jaccard(tokens, candidate_tokens) >= threshold:
            return True
    return False


def topic_label_tokens(label: str, stopwords: set[str] | None = None) -> set[str]:
    """Extract 4+ character non-stopword tokens from a topic label.

    Parameters
    ----------
    label:
        The topic label text.
    stopwords:
        Optional stopword set.  Falls back to :data:`_CLUSTER_STOPWORDS`.
    """
    if stopwords is None:
        stopwords = _CLUSTER_STOPWORDS
    return {
        token
        for token in _CLUSTER_TOKEN_RE.findall((label or "").lower())
        if token not in stopwords and len(token) >= 4
    }


def token_jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(set_a.intersection(set_b)) / len(union)


def position_title_single_claim_passed(title: str) -> bool:
    """Return ``True`` when *title* expresses a single focused claim.

    Rejects titles that contain multiple claims separated by ``/``,
    ``\\``, ``;``, ``|``, or excessive commas/colons.
    """
    cleaned = " ".join(str(title or "").split()).strip().lower()
    if not cleaned:
        return False
    if any(marker in cleaned for marker in ("/", "\\", ";", "|")):
        return False
    if cleaned.count(":") > 1 or cleaned.count(",") > 1:
        return False
    if ":" in cleaned and re.search(r"\b(и|или|and|or|but)\b", cleaned):
        return False
    return not re.search(r"\b(и|или|and|or)\b.*\b(и|или|and|or)\b", cleaned)


def is_detailed_description(description: str) -> bool:
    """Return ``True`` when *description* is long, multi-sentence, and semantically rich.

    Checks that the text has at least 96 characters, at least 2 sentence-ending
    marks, and at least 8 distinct semantic tokens.
    """
    text = " ".join((description or "").split()).strip()
    if len(text) < 96:
        return False
    sentence_count = sum(text.count(mark) for mark in ".!?")
    if sentence_count < 2:
        return False
    semantic_tokens = {
        token
        for token in _CLUSTER_TOKEN_RE.findall(text.lower())
        if token not in _CLUSTER_STOPWORDS
    }
    return len(semantic_tokens) >= 8


def description_comment_support_score(description: str, comments: list[str]) -> float:
    """Compute a lexical support score for *description* against *comments*.

    The score reflects how many description tokens appear in the first
    12 comments.  Returns a normalised float.
    """
    if not description or not comments:
        return 0.0
    description_tokens = {
        token
        for token in _CLUSTER_TOKEN_RE.findall(description.lower())
        if token not in _CLUSTER_STOPWORDS
    }
    if not description_tokens:
        return 0.0
    comment_tokens: set[str] = set()
    for text in comments[:12]:
        comment_tokens.update(
            token
            for token in _CLUSTER_TOKEN_RE.findall(text.lower())
            if token not in _CLUSTER_STOPWORDS
        )
    if not comment_tokens:
        return 0.0
    hits = len(description_tokens.intersection(comment_tokens))
    norm = max(1, min(10, len(description_tokens)))
    return hits / norm


def build_comment_grounded_description(
    comments: list[str],
    sentiment: str,
) -> str:
    """Generate a short grounded description from *comments* and overall *sentiment*.

    Builds a summary lead from extracted keywords, appends a tone
    assessment, and optionally notes audience questions.

    Parameters
    ----------
    comments:
        Plain-text comment strings.
    sentiment:
        One of ``"negative"``, ``"positive"``, or any other value (treated as mixed).
    """
    if not comments:
        return ""
    question_count = 0
    for text in comments[:16]:
        if is_question_comment_text(text):
            question_count += 1
    keywords = extract_salient_comment_keywords(comments, max_keywords=5)
    if keywords:
        lead = f"В комментариях по теме чаще обсуждают: {', '.join(keywords)}."
    else:
        lead = "В комментариях по теме обсуждают детали события и его последствия."
    if sentiment == "negative":
        tone = "Преобладают критические оценки и спорные трактовки."
    elif sentiment == "positive":
        tone = "Преобладают поддержка и одобрительные оценки."
    else:
        tone = "Тон обсуждения смешанный: мнения разделены."
    question_tail = (
        " Отдельно заметен блок вопросов аудитории по этой теме." if question_count else ""
    )
    return f"{lead} {tone}{question_tail}".strip()
