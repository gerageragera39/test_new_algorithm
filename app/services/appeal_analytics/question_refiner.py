"""Second-pass LLM enrichment for question candidates in Appeal Analytics.

After the unified classifier identifies question comments, this module
sends them through a structured second prompt that:
  - Assigns a topic label (geopolitics, economy, AI, etc.)
  - Scores the question quality (1-10)
  - Produces a Russian-language short summary
  - Classifies the question type (analytical depth vs. attack vs. one-liner)

The refined score becomes the primary sort key for constructive_question items.

STRICT scoring rules applied in post-processing:
  - topic must be one of the allowed values (fallback: "Other")
  - question_type must be one of the allowed values (fallback: "clarification_needed")
  - short: Russian, ≤160 chars, no emoji, no curly quotes; real questions end with "?"
  - attack_ragebait / meme_one_liner → score capped at 3
  - depth_score < 6 → score capped at 6
  - depth_score ≥ 7 only valid for real question types
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from app.core.exceptions import BudgetExceededError, ExternalServiceError

if TYPE_CHECKING:
    from app.db.models import Comment
    from app.services.labeling import LLMProvider

logger = logging.getLogger(__name__)

_BATCH_SIZE = 20

_ALLOWED_TOPICS = {
    "Ukraine_Russia",
    "Israel_Palestine_Iran",
    "USA",
    "Economy",
    "AI",
    "Philosophy",
    "Life",
    "Other",
}

_ALLOWED_QUESTION_TYPES = {
    "myth_claim",
    "fact_check",
    "analysis_why_how",
    "prediction_what_next",
    "strategy_advice",
    "clarification_needed",
    "attack_ragebait",
    "meme_one_liner",
}

# Real question types get short ending with "?"
_REAL_QUESTION_TYPES = {
    "myth_claim",
    "fact_check",
    "analysis_why_how",
    "prediction_what_next",
    "strategy_advice",
    "clarification_needed",
}

# Low-value types: score capped at 3
_LOW_VALUE_QUESTION_TYPES = {"attack_ragebait", "meme_one_liner"}

# Remove emoji characters from short summaries
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f9ff"
    "\U00002600-\U000027bf"
    "\U0001fa00-\U0001faff"
    "]+",
    flags=re.UNICODE,
)
# Remove curly / typographic quotes
_CURLY_QUOTES_RE = re.compile(r'[«»„"\u201c\u201d\u2018\u2019\u00ab\u00bb]')

RequestLLMJsonFn = Any


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _build_refiner_prompt(numbered_comments: list[tuple[int, str]]) -> str:
    lines = [f"{num}. {text}" for num, text in numbered_comments]
    comments_text = "\n".join(lines)

    return (
        "You classify YouTube questions from a political Russian-language channel.\n"
        "Return STRICT JSON only. No markdown. No extra text.\n\n"
        "For EACH question below provide:\n"
        "  topic — exactly one of:\n"
        "    Ukraine_Russia, Israel_Palestine_Iran, USA, Economy, AI, Philosophy, Life, Other\n"
        "  score — integer 1-10:\n"
        "    1-3  = ragebait / meme / trolling / offensive\n"
        "    4-6  = shallow / vague / needs clarification\n"
        "    7-10 = substantive, specific, worth a long structured answer\n"
        "  short — 5-15 word summary IN RUSSIAN:\n"
        "    - Must be a single sentence\n"
        "    - Must be in Russian (no English words)\n"
        "    - No emoji, no quotation marks\n"
        "    - If question_type is attack_ragebait or meme_one_liner: must NOT end with '?'\n"
        "    - For all other types: must end with '?'\n"
        "    - Max 160 characters\n"
        "  depth_score — integer 0-10:\n"
        "    0-5 = shallow or rhetorical\n"
        "    6   = moderate analytical depth\n"
        "    7-10 = deep analytical question requiring structured research\n"
        "    NOTE: depth_score >=7 only if question_type is one of:\n"
        "      analysis_why_how, prediction_what_next, fact_check, myth_claim, strategy_advice\n"
        "  question_type — exactly one of:\n"
        "    myth_claim, fact_check, analysis_why_how, prediction_what_next,\n"
        "    strategy_advice, clarification_needed, attack_ragebait, meme_one_liner\n\n"
        "SCORING RULES (apply strictly):\n"
        "  attack_ragebait or meme_one_liner → score MUST be 1-3\n"
        "  depth_score 0-5 → score MUST be ≤6\n"
        "  depth_score 6 → score MAY be up to 7\n"
        "  depth_score 7-10 → score MAY be 7-10\n\n"
        "Response format:\n"
        '{"results": {\n'
        '  "1": {"topic": "Ukraine_Russia", "score": 8, "short": "Почему переговоры зашли в тупик?", '
        '"depth_score": 7, "question_type": "analysis_why_how"},\n'
        '  "2": {"topic": "Economy", "score": 3, "short": "Такой вопрос бессмысленен", '
        '"depth_score": 1, "question_type": "attack_ragebait"}\n'
        "}}\n\n"
        f"Questions:\n{comments_text}"
    )


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _sanitize_short(text: str | None, question_type: str) -> str | None:
    """Normalize the short summary to comply with strict rules.

    - Strip emoji and curly quotes.
    - Truncate to 160 characters.
    - Real question types → ensure ends with '?'.
    - Low-value types → strip trailing '?' to avoid confusing UI filters.
    - Returns None if the result is empty after sanitisation.
    """
    if not text:
        return None

    cleaned = _EMOJI_RE.sub("", str(text))
    cleaned = _CURLY_QUOTES_RE.sub("", cleaned)
    cleaned = cleaned.strip(" \t\n\"'")
    # Collapse multiple spaces
    cleaned = re.sub(r"  +", " ", cleaned)

    if not cleaned:
        return None

    # Truncate to 160 chars (hard limit)
    if len(cleaned) > 160:
        cleaned = cleaned[:157].rstrip(" ,;") + "…"

    if question_type in _REAL_QUESTION_TYPES:
        if not cleaned.endswith("?"):
            # Add "?" only if it doesn't already end with punctuation
            cleaned = cleaned.rstrip(".!") + "?"
    elif question_type in _LOW_VALUE_QUESTION_TYPES:
        # Ragebait/meme should NOT end with "?" — strip it
        cleaned = cleaned.rstrip("?").rstrip(" ")

    return cleaned or None


def _apply_strict_score_rules(score: int, depth_score: int, question_type: str) -> int:
    """Apply scoring rules based on question_type and depth_score.
    
    Improved from v1: rules are less restrictive to avoid losing high-quality questions.
    - Low-value types capped at 3
    - depth_score 4-5 → score up to 7 (was 6)
    - depth_score 6+ → score can reach 8-10 for analytical questions
    - Real questions with depth 7+ get minimum score of 7
    """
    # Low-value types (attack/meme) strictly capped at 3
    if question_type in _LOW_VALUE_QUESTION_TYPES:
        return min(score, 3)
    
    # For clarification_needed: cap depth >= 7 to avoid inflation
    if depth_score >= 7 and question_type == "clarification_needed":
        score = min(score, 6)
    
    # Depth-based score guidance (softer than v1):
    # depth 0-3: shallow questions, cap at 5
    # depth 4-5: moderate depth, cap at 7 (improved from 6)
    # depth 6: good depth, cap at 8
    # depth 7+: high analytical depth, allow full range 7-10
    if depth_score <= 3:
        score = min(score, 5)
    elif depth_score <= 5:
        score = min(score, 7)  # Was 6, now 7 to avoid losing good questions
    elif depth_score == 6:
        score = min(score, 8)  # Was 7, now 8
    
    # For high-depth analytical questions, ensure minimum quality score
    if depth_score >= 7 and question_type in {
        "analysis_why_how", "prediction_what_next", "fact_check",
        "myth_claim", "strategy_advice"
    }:
        score = max(score, 7)  # Ensure analytical questions get proper recognition
    
    return max(1, min(10, score))


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_refiner_response(
    raw: dict | list | None,
    valid_nums: set[int],
) -> dict[int, dict[str, Any]]:
    """Parse refiner LLM response into {comment_number: enriched_fields}.

    All fields are strictly validated and normalised:
    - topic/question_type fallback to safe defaults.
    - score/depth_score clamped to valid ranges.
    - short sanitised (emoji stripped, length capped, "?" rule enforced).
    - Strict score clamping applied via _apply_strict_score_rules().
    """
    if not isinstance(raw, dict):
        return {}

    results = raw.get("results")
    if not isinstance(results, dict):
        results = raw

    parsed: dict[int, dict[str, Any]] = {}
    for key, value in results.items():
        try:
            num = int(key)
        except (ValueError, TypeError):
            continue
        if num not in valid_nums:
            continue
        if not isinstance(value, dict):
            continue

        # --- topic ---
        topic = str(value.get("topic") or "Other").strip()
        if topic not in _ALLOWED_TOPICS:
            topic = "Other"

        # --- score ---
        raw_score = value.get("score")
        try:
            score = max(1, min(10, int(raw_score)))
        except (ValueError, TypeError):
            score = 5

        # --- depth_score ---
        raw_depth = value.get("depth_score")
        try:
            depth_score = max(0, min(10, int(raw_depth)))
        except (ValueError, TypeError):
            depth_score = 0

        # --- question_type ---
        question_type = str(value.get("question_type") or "clarification_needed").strip()
        if question_type not in _ALLOWED_QUESTION_TYPES:
            question_type = "clarification_needed"

        # --- Apply STRICT score rules ---
        score = _apply_strict_score_rules(score, depth_score, question_type)

        # Re-clamp depth_score if low-value type
        if question_type in _LOW_VALUE_QUESTION_TYPES:
            depth_score = min(depth_score, 3)

        # --- short ---
        raw_short = value.get("short")
        short = _sanitize_short(raw_short, question_type)

        parsed[num] = {
            "topic": topic,
            "score": score,
            "short": short,
            "depth_score": depth_score,
            "question_type": question_type,
        }

    return parsed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def refine_questions(
    comments: list[Comment],
    llm_provider: LLMProvider,
    request_llm_json: RequestLLMJsonFn,
) -> dict[int, dict[str, Any]]:
    """Run second-pass LLM enrichment on question candidates.

    Returns a dict mapping comment_id -> {topic, score, short, depth_score, question_type}.
    Comments that fail enrichment are omitted (callers keep original scores).
    Failures are non-fatal and logged at WARNING level.
    """
    from app.services.labeling import NoLLMFallbackProvider

    if isinstance(llm_provider, NoLLMFallbackProvider):
        return {}

    result: dict[int, dict[str, Any]] = {}

    for batch_start in range(0, len(comments), _BATCH_SIZE):
        batch = comments[batch_start : batch_start + _BATCH_SIZE]
        numbered: list[tuple[int, str]] = []
        batch_id_map: dict[int, int] = {}  # num -> comment_id

        for i, comment in enumerate(batch, start=1):
            text = (comment.text_raw or "").strip()
            numbered.append((i, text))
            batch_id_map[i] = comment.id

        prompt = _build_refiner_prompt(numbered)

        try:
            data = request_llm_json(
                llm_provider,
                prompt,
                task="question_refiner",
                estimated_out_tokens=max(150, len(batch) * 35),
                max_output_tokens=max(400, len(batch) * 60),
                system_prompt=(
                    "You are a strict question quality classifier for a YouTube analytics system. "
                    "Return STRICT JSON only. No markdown, no explanation, no extra keys. "
                    "Follow all scoring and short-text rules exactly as specified."
                ),
            )
        except (BudgetExceededError, ExternalServiceError) as exc:
            logger.warning("Question refiner LLM call failed: %s", exc)
            continue
        except Exception as exc:
            logger.warning("Unexpected question refiner error: %s", exc)
            continue

        valid_nums = set(batch_id_map.keys())
        parsed = _parse_refiner_response(data, valid_nums)

        for num, fields in parsed.items():
            cid = batch_id_map.get(num)
            if cid is not None:
                result[cid] = fields
                logger.debug(
                    "Refiner: comment %d → topic=%s score=%d depth=%d type=%s short=%r",
                    cid,
                    fields.get("topic"),
                    fields.get("score", 0),
                    fields.get("depth_score", 0),
                    fields.get("question_type"),
                    (fields.get("short") or "")[:60],
                )

    logger.info(
        "Question refiner enriched %d/%d question candidates",
        len(result),
        len(comments),
    )
    return result
