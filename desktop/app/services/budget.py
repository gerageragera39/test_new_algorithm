"""OpenAI API budget enforcement and usage tracking.

Provides daily and per-run spending limits, token cost estimation for
embedding and chat models, and persistent usage recording via the database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import BudgetExceededError
from app.db.models import BudgetUsage

EMBEDDING_PRICING_PER_TOKEN = {
    "text-embedding-3-small": 0.02 / 1_000_000,
    "text-embedding-3-large": 0.13 / 1_000_000,
}

CHAT_PRICING_PER_TOKEN = {
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "cached_input": 0.075 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "gpt-4o": {
        "input": 2.50 / 1_000_000,
        "cached_input": 1.25 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-5-mini": {
        "input": 0.25 / 1_000_000,
        "cached_input": 0.025 / 1_000_000,
        "output": 2.00 / 1_000_000,
    },
    "gpt-5.4": {
        "input": 2.50 / 1_000_000,
        "cached_input": 0.25 / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
    "gpt-5.4-mini": {
        "input": 0.75 / 1_000_000,
        "cached_input": 0.075 / 1_000_000,
        "output": 4.50 / 1_000_000,
    },
    "gpt-5.4-nano": {
        "input": 0.20 / 1_000_000,
        "cached_input": 0.02 / 1_000_000,
        "output": 1.25 / 1_000_000,
    },
    "gpt-5.4-pro": {
        "input": 30.00 / 1_000_000,
        "output": 180.00 / 1_000_000,
    },
    "gpt-5.2": {
        "input": 1.75 / 1_000_000,
        "cached_input": 0.175 / 1_000_000,
        "output": 14.00 / 1_000_000,
    },
}


def _resolve_chat_pricing(model: str) -> dict[str, float]:
    normalized = str(model or "").strip().lower()
    if normalized in CHAT_PRICING_PER_TOKEN:
        return CHAT_PRICING_PER_TOKEN[normalized]
    if normalized.startswith("gpt-5.4-pro"):
        return CHAT_PRICING_PER_TOKEN["gpt-5.4-pro"]
    if normalized.startswith("gpt-5.4-mini"):
        return CHAT_PRICING_PER_TOKEN["gpt-5.4-mini"]
    if normalized.startswith("gpt-5.4-nano"):
        return CHAT_PRICING_PER_TOKEN["gpt-5.4-nano"]
    if normalized.startswith("gpt-5.4"):
        return CHAT_PRICING_PER_TOKEN["gpt-5.4"]
    if normalized.startswith("gpt-5.2"):
        return CHAT_PRICING_PER_TOKEN["gpt-5.2"]
    if normalized.startswith("gpt-5-mini"):
        return CHAT_PRICING_PER_TOKEN["gpt-5-mini"]
    if normalized.startswith("gpt-4o-mini"):
        return CHAT_PRICING_PER_TOKEN["gpt-4o-mini"]
    if normalized.startswith("gpt-4o"):
        return CHAT_PRICING_PER_TOKEN["gpt-4o"]
    return CHAT_PRICING_PER_TOKEN["gpt-4o-mini"]


@dataclass
class BudgetSnapshot:
    usage_date: date
    spent_usd: float
    tokens_used: int
    entries: list[dict[str, Any]]


class BudgetGovernor:
    """Enforces daily and per-run budget limits for OpenAI API usage.

    Tracks token consumption and estimated USD costs, raises BudgetExceededError
    when configured thresholds would be exceeded, and persists usage records.
    """

    def __init__(self, settings: Settings, db: Session) -> None:
        self.settings = settings
        self.db = db
        self.logger = logging.getLogger(self.__class__.__name__)
        self.run_budget_usd = max(0.0, float(settings.openai_max_usd_per_run))
        self._run_spent_usd = 0.0
        self._run_tokens = 0

    def _today(self) -> date:
        return datetime.now(UTC).date()

    def estimate_tokens(self, texts: list[str]) -> int:
        return max(1, int(sum(len(text) for text in texts) / 4))

    def estimate_tokens_upper_bound(self, texts: list[str], *, overhead_tokens: int = 0) -> int:
        # Hard upper bound for token count: tokenization is byte-based, so tokens do not exceed UTF-8 byte count.
        byte_count = int(sum(len((text or "").encode("utf-8")) for text in texts))
        return max(1, byte_count + max(0, int(overhead_tokens)))

    def estimate_embedding_cost(self, model: str, tokens: int) -> float:
        unit = EMBEDDING_PRICING_PER_TOKEN.get(
            model, EMBEDDING_PRICING_PER_TOKEN["text-embedding-3-small"]
        )
        return tokens * unit

    def estimate_chat_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_input_tokens: int = 0,
    ) -> float:
        pricing = _resolve_chat_pricing(model)
        cached_unit = pricing.get("cached_input", pricing["input"])
        cached = max(0, min(int(cached_input_tokens), int(input_tokens)))
        uncached = max(0, int(input_tokens) - cached)
        return (
            uncached * pricing["input"]
            + cached * cached_unit
            + int(output_tokens) * pricing["output"]
        )

    def get_snapshot(self, usage_day: date | None = None) -> BudgetSnapshot:
        """Retrieve a snapshot of budget usage for the given day.

        Args:
            usage_day: Date to query; defaults to today (UTC).

        Returns:
            BudgetSnapshot with spending totals and remaining allowances.
        """
        usage_day = usage_day or self._today()
        stmt = (
            select(BudgetUsage)
            .where(BudgetUsage.usage_date == usage_day)
            .order_by(BudgetUsage.created_at.desc())
        )
        rows = list(self.db.scalars(stmt))
        spent = float(sum(float(row.estimated_cost_usd) for row in rows))
        tokens = int(sum(row.tokens_input + row.tokens_output for row in rows))
        return BudgetSnapshot(
            usage_date=usage_day,
            spent_usd=spent,
            tokens_used=tokens,
            entries=[
                {
                    "provider": row.provider,
                    "model": row.model,
                    "tokens_input": row.tokens_input,
                    "tokens_output": row.tokens_output,
                    "estimated_cost_usd": float(row.estimated_cost_usd),
                    "request_count": row.request_count,
                    "task": str((row.meta_json or {}).get("task", "")),
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ],
        )

    def assert_can_spend(self, estimated_cost: float, estimated_tokens: int) -> None:
        """Enforce the configured per-run budget when it is enabled."""
        _ = estimated_tokens
        if self.run_budget_usd <= 0:
            return
        projected_spend = self._run_spent_usd + max(0.0, float(estimated_cost))
        if projected_spend > self.run_budget_usd:
            raise BudgetExceededError(
                f"OpenAI per-run budget exceeded: ${projected_spend:.6f} > ${self.run_budget_usd:.6f}."
            )

    def record_usage(
        self,
        *,
        model: str,
        provider: str,
        tokens_input: int,
        tokens_output: int,
        estimated_cost_usd: float,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Persist an API usage record to the database and update run totals.

        Args:
            model: Model name used for the API call.
            provider: Provider identifier (e.g. "openai_chat").
            tokens_input: Number of input tokens consumed.
            tokens_output: Number of output tokens consumed.
            estimated_cost_usd: Estimated cost of the call in USD.
            meta: Optional metadata dict to store alongside the record.
        """
        usage = BudgetUsage(
            usage_date=self._today(),
            provider=provider,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            estimated_cost_usd=Decimal(f"{estimated_cost_usd:.6f}"),
            request_count=1,
            meta_json=meta or {},
        )
        self.db.add(usage)
        self.db.flush()
        self._run_spent_usd += float(estimated_cost_usd)
        self._run_tokens += int(tokens_input) + int(tokens_output)

    def get_spent_usd(self, usage_day: date | None = None) -> float:
        usage_day = usage_day or self._today()
        stmt = select(func.coalesce(func.sum(BudgetUsage.estimated_cost_usd), 0)).where(
            BudgetUsage.usage_date == usage_day
        )
        return float(self.db.scalar(stmt) or 0.0)
