"""Report-building and author-stance analysis helpers.

Extracts the disagreement detection fallback and the optional LLM-based
briefing polish step into a cohesive ``ReportBuilder`` class.

All regex patterns used for classification are imported from
``.constants`` to keep a single source of truth.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from app.core.config import Settings
from app.core.exceptions import BudgetExceededError, ExternalServiceError
from app.schemas.domain import DailyBriefing, ProcessedComment
from app.services.budget import BudgetGovernor
from app.services.labeling import OpenAIChatProvider
from app.services.openai_compat import (
    build_completion_token_kwargs,
    build_temperature_kwargs,
    extract_cached_input_tokens,
)

from .constants import (
    _AUTHOR_REFERENCE_RE,
    _DISAGREEMENT_RE,
    _OFFENSIVE_DISAGREEMENT_RE,
)


class ReportBuilder:
    """Encapsulates disagreement detection fallback and briefing polish logic.

    The class groups together the methods concerned with:

    * Determining whether a comment references the video author.
    * Filtering offensive disagreement comments.
    * Extracting disagreement comments via regex (fallback for LLM-based detection).
    * Optionally polishing a daily briefing via an LLM call.
    """

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    # ------------------------------------------------------------------
    # Author reference & disagreement helpers
    # ------------------------------------------------------------------

    def has_author_reference(self, text: str) -> bool:
        """Check whether *text* contains a reference to the video author."""
        normalized = f" {text.lower()} "
        author = (self.settings.author_name or "").strip().lower()
        if author:
            # Check each word of the author name (3+ chars) as a substring
            for token in author.split():
                if len(token) >= 3 and token in normalized:
                    return True
        if any(token in normalized for token in (" вы ", " вас ", " вам ", " ваш ")):
            return True
        return bool(_AUTHOR_REFERENCE_RE.search(normalized))

    def is_offensive_disagreement_text(self, text: str) -> bool:
        """Return ``True`` if *text* contains offensive language."""
        normalized = " ".join((text or "").split()).lower()
        if not normalized:
            return True
        return bool(_OFFENSIVE_DISAGREEMENT_RE.search(normalized))

    def extract_author_disagreement_comments(self, comments: list[ProcessedComment]) -> list[str]:
        """Regex-based fallback: return non-offensive disagreement comments directed at the author.

        Used when LLM-based position-level disagreement detection did not
        produce results (e.g. in free mode without LLM).
        """
        if not comments:
            return []
        items: list[str] = []
        for comment in comments:
            text = " ".join((comment.text_raw or comment.text_normalized).split())
            if not text:
                continue
            if self.is_offensive_disagreement_text(text):
                continue
            if not self.has_author_reference(text):
                continue
            if not _DISAGREEMENT_RE.search(text.lower()):
                continue
            items.append(text)
        return items

    # ------------------------------------------------------------------
    # Briefing polish
    # ------------------------------------------------------------------

    def try_polish_briefing(
        self,
        briefing: DailyBriefing,
        budget: BudgetGovernor,
        *,
        openai_provider: OpenAIChatProvider | None = None,
    ) -> DailyBriefing:
        """Optionally polish *briefing* via an LLM call.

        When an ``openai_provider`` is supplied the request is routed
        through the provider's ``request_json`` method (which handles
        budget tracking internally).  Otherwise a direct ``OpenAI``
        client call is made with explicit budget checks and usage
        recording.

        If the API key is not configured, or the budget is exhausted,
        the original *briefing* is returned unchanged.

        Args:
            briefing: The daily briefing to polish.
            budget: Budget governor for cost tracking.
            openai_provider: Optional high-level chat provider.

        Returns:
            The (possibly updated) ``DailyBriefing`` instance.
        """
        if not self.settings.openai_api_key:
            return briefing

        # Build a lightweight payload to stay within context limits.
        # The LLM only needs topic structure and summaries, not full
        # comment lists, embeddings, or metadata.
        lightweight_topics = [
            {
                "label": topic.label,
                "description": topic.description,
                "sentiment": topic.sentiment,
                "size_count": topic.size_count,
                "share_pct": topic.share_pct,
                "is_emerging": topic.is_emerging,
                "positions": [
                    {"title": pos.title, "summary": pos.summary, "pct": pos.pct}
                    for pos in topic.positions
                ],
            }
            for topic in briefing.top_topics
        ]
        payload = {
            "video_id": briefing.video_id,
            "video_title": briefing.video_title,
            "executive_summary": briefing.executive_summary,
            "top_topics": lightweight_topics,
            "actions_for_tomorrow": briefing.actions_for_tomorrow,
            "misunderstandings_and_controversies": briefing.misunderstandings_and_controversies,
            "audience_requests_and_questions": briefing.audience_requests_and_questions,
            "risks_and_toxicity": briefing.risks_and_toxicity,
            "trend_vs_previous": briefing.trend_vs_previous,
        }
        previous_categories = briefing.metadata.get("previous_report_categories", [])
        previous_categories_text = (
            "\n".join(
                f"- {item}"
                for item in previous_categories
                if isinstance(item, str) and item.strip()
            )
            or "- (нет данных)"
        )

        prompt = (
            "Polish this briefing for a Russian-speaking YouTube news host.\n"
            "Use only the provided structured data, with no new facts.\n"
            "Return JSON with key: executive_summary.\n"
            "Keep executive_summary concise, practical, and directly tied to the input data.\n"
            "Output language must be Russian.\n"
            f"Previous report categories:\n{previous_categories_text}\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        estimated_out = 280
        max_output_tokens = min(self.settings.openai_max_output_tokens, 500)
        system_prompt = "You are an editor of concise, practical briefings."

        data: dict[str, Any] = {}

        if openai_provider is not None:
            try:
                data = openai_provider.request_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    task="briefing_polish",
                    temperature=0.2,
                    estimated_out_tokens=estimated_out,
                    max_output_tokens=max_output_tokens,
                )
            except BudgetExceededError:
                return briefing
            except ExternalServiceError:
                return briefing
        else:
            if self.settings.openai_hard_budget_enforced:
                estimated_in = budget.estimate_tokens_upper_bound(
                    [system_prompt, prompt],
                    overhead_tokens=64,
                )
                budgeted_out = max_output_tokens
            else:
                estimated_in = budget.estimate_tokens([prompt])
                budgeted_out = estimated_out

            estimated_cost = budget.estimate_chat_cost(
                self.settings.openai_chat_model, estimated_in, budgeted_out
            )
            try:
                budget.assert_can_spend(estimated_cost, estimated_in + budgeted_out)
            except BudgetExceededError:
                return briefing

            client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
            response = client.chat.completions.create(
                model=self.settings.openai_chat_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **build_temperature_kwargs(self.settings.openai_chat_model, 0.2),
                **build_completion_token_kwargs(self.settings.openai_chat_model, max_output_tokens),
            )
            content = response.choices[0].message.content or "{}"
            try:
                data = json.loads(content)
            except Exception:
                return briefing

            prompt_tokens = response.usage.prompt_tokens if response.usage else estimated_in
            completion_tokens = (
                response.usage.completion_tokens if response.usage else estimated_out
            )
            cached_input_tokens = extract_cached_input_tokens(response.usage)
            final_cost = budget.estimate_chat_cost(
                self.settings.openai_chat_model,
                prompt_tokens,
                completion_tokens,
                cached_input_tokens=cached_input_tokens,
            )
            budget.record_usage(
                model=self.settings.openai_chat_model,
                provider="openai_chat",
                tokens_input=prompt_tokens,
                tokens_output=completion_tokens,
                estimated_cost_usd=final_cost,
                meta={
                    "task": "briefing_polish",
                    "cached_input_tokens": cached_input_tokens,
                },
            )

        briefing.executive_summary = str(data.get("executive_summary", briefing.executive_summary))
        return briefing
