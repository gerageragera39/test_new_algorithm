"""LLM-based moderation decision parsing utilities.

Provides data structures and helpers for converting raw LLM moderation
payloads into structured ModerationDecision objects with validated actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast


@dataclass
class ModerationDecision:
    """Represents a moderation verdict for a single comment.

    Holds the action to take (keep, flag, or drop), a machine-readable
    reason code, and a confidence score between 0 and 1.
    """

    action: Literal["keep", "flag", "drop"]
    reason_code: str
    score: float


def decision_from_payload(
    payload: dict[str, Any] | None,
    *,
    fallback_action: Literal["keep", "flag", "drop"] = "flag",
    fallback_reason: str = "llm_fallback",
) -> ModerationDecision:
    """Parse an LLM moderation response payload into a ModerationDecision.

    Args:
        payload: Raw dictionary returned by the LLM, may be None.
        fallback_action: Action to use when the payload action is invalid.
        fallback_reason: Reason code to use when the payload reason is empty.

    Returns:
        A validated ModerationDecision with clamped score in [0, 1].
    """
    data = payload or {}
    action_raw = str(data.get("action", "")).strip().lower()
    action: Literal["keep", "flag", "drop"] = fallback_action
    if action_raw in {"keep", "flag", "drop"}:
        action = cast(Literal["keep", "flag", "drop"], action_raw)

    reason = str(data.get("reason_code", "")).strip().lower()
    if not reason:
        reason = fallback_reason

    score_raw = data.get("score", 0.5)
    try:
        score = float(score_raw)
    except Exception:
        score = 0.5
    score = max(0.0, min(1.0, score))

    return ModerationDecision(action=action, reason_code=reason, score=score)
