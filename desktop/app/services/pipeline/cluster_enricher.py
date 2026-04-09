"""Cluster enrichment logic extracted from the monolithic pipeline module.

Provides the :class:`ClusterEnricher` class that encapsulates labeling,
postprocessing, merging, deduplication, and ranking of comment clusters.

Text-level helpers are delegated to :mod:`.text_utils` and compiled
regular-expression constants come from :mod:`.constants`.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.exceptions import BudgetExceededError, ExternalServiceError
from app.schemas.domain import (
    ClusterDraft,
    EpisodeContext,
    ProcessedComment,
    TopicPosition,
    TopicSummary,
)
from app.services.embeddings import EmbeddingService
from app.services.labeling import (
    ClusterContext,
    LLMProvider,
)

from .text_utils import (
    _CLUSTER_STOPWORDS,
    _CLUSTER_TOKEN_RE,
    _UNCERTAIN_TOPIC_LABEL,
    build_comment_grounded_description,
    comment_text_for_output,
    compact_text_for_matching,
    description_comment_support_score,
    extract_salient_comment_keywords,
    is_detailed_description,
    is_duplicate_text_signature,
    is_question_comment_text,
    normalize_text_for_matching,
    sanitize_cluster_title,
    text_token_set,
    token_jaccard,
    topic_label_tokens,
)

# ---------------------------------------------------------------------------
# Type aliases for injectable callables
# ---------------------------------------------------------------------------

#: Callable that sends a prompt to an LLM provider and returns parsed JSON.
#: Signature: ``(provider, prompt, *, task, estimated_out_tokens,
#: max_output_tokens, **kw) -> dict | None``
RequestLLMJsonFn = Callable[..., dict[str, Any] | None]

#: Callable that builds position groups for a single cluster.
#: Signature mirrors ``DailyRunService._build_positions_for_cluster``.
BuildPositionsFn = Callable[..., tuple[list[TopicPosition], bool]]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    vec_a = np.array(a, dtype=np.float32)
    vec_b = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


class ClusterEnricher:
    """Cluster labeling, postprocessing, deduplication and ranking.

    Encapsulates the enrichment pipeline that transforms raw
    :class:`ClusterDraft` objects into fully labelled
    :class:`TopicSummary` instances.  Methods that require LLM access
    receive a ``request_llm_json`` callable instead of depending on
    the full pipeline service.

    Parameters
    ----------
    settings:
        Application-level configuration.
    logger:
        Logger instance used for progress and diagnostic messages.
    """

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings: Settings = settings
        self.logger: logging.Logger = logger

    # ================================================================
    # Labeling orchestration
    # ================================================================

    def label_topics(
        self,
        clusters: list[ClusterDraft],
        comments: list[ProcessedComment],
        vectors: list[list[float]],
        embedding_service: EmbeddingService,
        llm_provider: LLMProvider,
        previous_topics: list[TopicSummary],
        episode_context: EpisodeContext,
        episode_topic_matches: dict[str, str],
        *,
        request_llm_json: RequestLLMJsonFn,
        build_positions_fn: BuildPositionsFn,
    ) -> tuple[list[TopicSummary], dict[str, Any]]:
        """Label every cluster and return enriched topics with diagnostics.

        This is the main orchestrator that iterates over *clusters*, obtains
        LLM labels, resolves titles and quotes, builds positions, and
        assembles :class:`TopicSummary` objects.

        Parameters
        ----------
        clusters:
            Cluster drafts produced by the clustering stage.
        comments:
            Full list of processed comments.
        vectors:
            Embedding vectors aligned with *comments*.
        embedding_service:
            Embedding service (currently unused, kept for interface compat).
        llm_provider:
            LLM provider used for cluster analysis.
        previous_topics:
            Topics from a previous run (used for label continuity).
        episode_context:
            Episode context derived from video transcript.
        episode_topic_matches:
            Mapping from cluster key to matched episode topic title.
        request_llm_json:
            Callable that sends a prompt to an LLM and returns parsed JSON.
        build_positions_fn:
            Callable that builds position groups for a single cluster.
        """
        _ = embedding_service
        previous_labels = [topic.label for topic in previous_topics[:6]]
        episode_prompt_topics = episode_context.topic_lines
        topics: list[TopicSummary] = []
        seen_labels: set[str] = set()
        llm_disabled = False
        diagnostics: dict[str, Any] = {
            "fallback_topic_title_count": 0,
            "fallback_position_title_count": 0,
            "llm_cluster_fail_count": 0,
            "llm_title_fail_count": 0,
            "llm_position_fail_count": 0,
            "llm_disabled_reason": "",
            "llm_disable_events": [],
        }
        total_clusters = max(1, len(clusters))
        self.logger.info("Starting cluster labeling: %s clusters.", len(clusters))
        for idx, cluster in enumerate(clusters, start=1):
            self.logger.info(
                "[label %s/%s] cluster=%s members=%s reps=%s",
                idx,
                total_clusters,
                cluster.cluster_key,
                len(cluster.member_indices),
                len(cluster.representative_indices),
            )
            coherence = self._estimate_cluster_coherence(cluster, comments, vectors)
            representative = [
                comments[comment_idx]
                for comment_idx in cluster.representative_indices
                if 0 <= comment_idx < len(comments)
            ]
            all_comments = [
                comments[comment_idx]
                for comment_idx in cluster.member_indices
                if 0 <= comment_idx < len(comments)
            ]
            if not all_comments:
                continue
            matched_episode_topic = episode_topic_matches.get(cluster.cluster_key)
            source = "episode_topic" if matched_episode_topic else "comment_topic"
            ctx = ClusterContext(
                cluster=cluster,
                representative_comments=representative,
                all_comments=all_comments,
                episode_topics=episode_prompt_topics,
                matched_episode_topic=matched_episode_topic,
                previous_topic_labels=previous_labels,
            )
            if llm_disabled:
                # LLM was disabled mid-run — mark remaining clusters as errors.
                self.logger.warning(
                    "Skipping cluster %s: LLM disabled (%s).",
                    cluster.cluster_key,
                    diagnostics.get("llm_disabled_reason", "unknown"),
                )
                topics.append(self._error_topic(cluster, coherence, source, idx))
                continue
            try:
                label = llm_provider.analyze_cluster(ctx)
            except (BudgetExceededError, ExternalServiceError) as exc:
                diagnostics["llm_cluster_fail_count"] += 1
                message = str(exc).lower()
                reason = ""
                if isinstance(exc, BudgetExceededError):
                    reason = "budget_exceeded"
                elif "429" in message or "rate-limit" in message or "rate limit" in message:
                    reason = "rate_limited"
                else:
                    reason = "external_service_error"
                if reason in {"budget_exceeded", "rate_limited"}:
                    llm_disabled = True
                self._mark_llm_disabled(diagnostics, reason=reason)
                self.logger.error(
                    "Cluster labeling FAILED for %s via %s: %s",
                    cluster.cluster_key,
                    llm_provider.provider_name,
                    exc,
                )
                topics.append(self._error_topic(cluster, coherence, source, idx))
                continue
            except Exception as exc:
                diagnostics["llm_cluster_fail_count"] += 1
                self.logger.error(
                    "Unexpected cluster labeling FAILED for %s via %s: %s",
                    cluster.cluster_key,
                    llm_provider.provider_name,
                    exc,
                )
                topics.append(self._error_topic(cluster, coherence, source, idx))
                continue

            if not label.label:
                self.logger.error(
                    "LLM returned empty label for cluster %s — skipping.",
                    cluster.cluster_key,
                )
                topics.append(self._error_topic(cluster, coherence, source, idx))
                continue

            resolved_label = self._dedupe_label(label.label, all_comments, seen_labels)
            resolved_quotes, question_comments, ordered_comments = self._resolve_cluster_quotes(
                cluster=cluster,
                comments=comments,
                vectors=vectors,
                raw_quotes=label.representative_quotes,
            )
            positions, llm_disabled = build_positions_fn(
                llm_provider=llm_provider,
                cluster=cluster,
                comments=comments,
                vectors=vectors,
                llm_disabled=llm_disabled,
                cluster_title=resolved_label,
                diagnostics=diagnostics,
            )
            resolved_description = self._resolve_topic_description(
                description=label.description,
                fallback_description="",
                ordered_comments=ordered_comments,
                sentiment=label.sentiment,
            )
            topics.append(
                TopicSummary(
                    cluster_key=cluster.cluster_key,
                    label=resolved_label,
                    description=resolved_description,
                    author_actions=label.author_actions,
                    sentiment=label.sentiment,
                    emotion_tags=label.emotion_tags,
                    intent_distribution=label.intent_distribution,
                    representative_quotes=resolved_quotes,
                    question_comments=question_comments,
                    positions=positions,
                    size_count=cluster.size_count,
                    share_pct=cluster.share_pct,
                    weighted_share=cluster.weighted_share,
                    is_emerging=cluster.is_emerging,
                    source=source,
                    coherence_score=coherence,
                    centroid=cluster.centroid,
                    assignment_confidence=cluster.assignment_confidence,
                    ambiguous_share_pct=cluster.ambiguous_share_pct,
                    soft_assignment_notes=self._build_soft_assignment_notes(cluster),
                )
            )
            self.logger.info(
                "[label %s/%s] done cluster=%s label=%s sentiment=%s",
                idx,
                total_clusters,
                cluster.cluster_key,
                resolved_label,
                label.sentiment,
            )
        topics.sort(key=lambda item: item.weighted_share, reverse=True)
        return topics, diagnostics

    @staticmethod
    def _error_topic(
        cluster: ClusterDraft, coherence: float, source: str, idx: int
    ) -> TopicSummary:
        """Create a placeholder topic that signals an LLM error in the report."""
        return TopicSummary(
            cluster_key=cluster.cluster_key,
            label=f"[Ошибка] Кластер #{idx}",
            description="Не удалось проанализировать этот кластер — ошибка LLM. Подробности в логах сервера.",
            author_actions=[],
            sentiment="neutral",
            emotion_tags=[],
            intent_distribution={},
            representative_quotes=[],
            question_comments=[],
            positions=[],
            size_count=cluster.size_count,
            share_pct=cluster.share_pct,
            weighted_share=cluster.weighted_share,
            is_emerging=cluster.is_emerging,
            source=source,
            coherence_score=coherence,
            centroid=cluster.centroid,
            assignment_confidence=cluster.assignment_confidence,
            ambiguous_share_pct=cluster.ambiguous_share_pct,
            soft_assignment_notes=[],
        )

    # ================================================================
    # Post-position cluster naming
    # ================================================================

    def generate_cluster_names_from_positions(
        self,
        topics: list[TopicSummary],
        *,
        llm_provider: LLMProvider,
        request_llm_json: RequestLLMJsonFn,
        llm_disabled: bool = False,
        clusters: list[ClusterDraft] | None = None,
        comments: list[ProcessedComment] | None = None,
        vectors: list[list[float]] | None = None,
    ) -> list[TopicSummary]:
        """Generate cluster labels and descriptions from position titles via LLM.

        For each topic that has non-undetermined positions, sends a prompt
        to the LLM with position titles and their sizes. The LLM returns
        a cluster name and a description explaining what concerns the
        audience and how strongly each aspect is expressed.

        Topics with fewer than 2 positions get their label validated and
        potentially regenerated via ``_resolve_cluster_title``.
        """
        if llm_disabled:
            self.logger.info("Skipping post-position cluster naming: LLM disabled.")
            return topics

        cluster_by_key = {c.cluster_key: c for c in (clusters or [])}
        updated: list[TopicSummary] = []
        for topic in topics:
            positions = [p for p in topic.positions if not p.is_undetermined]
            if len(positions) < 2:
                # For topics without enough positions, try to validate/regenerate the label
                # via _resolve_cluster_title if cluster data is available.
                cluster = cluster_by_key.get(topic.cluster_key)
                if cluster is not None and comments is not None and vectors is not None:
                    title, llm_disabled, _used_fallback = self._resolve_cluster_title(
                        llm_provider=llm_provider,
                        cluster=cluster,
                        comments=comments,
                        vectors=vectors,
                        candidate_title=topic.label,
                        fallback_title=topic.label,
                        llm_disabled=llm_disabled,
                        request_llm_json=request_llm_json,
                    )
                    if title != topic.label:
                        topic = topic.model_copy(update={"label": title})
                        self.logger.info(
                            "Resolved cluster title for %s (few positions): label=%s",
                            topic.cluster_key,
                            topic.label,
                        )
                updated.append(topic)
                continue

            if not self._should_regenerate_from_positions(topic):
                updated.append(topic)
                continue

            position_lines = []
            for pos in positions:
                position_lines.append(
                    f'- "{pos.title}" ({pos.count} комментариев, {pos.pct:.1f}% кластера)'
                )
            positions_text = "\n".join(position_lines)

            prompt = (
                "Ты генерируешь название и описание тематического кластера комментариев YouTube-канала.\n"
                "На основании подкатегорий (позиций аудитории), создай:\n"
                "1. Короткое название кластера (3-8 слов) — обобщающее все подкатегории.\n"
                "2. Описание кластера (2-5 предложений) — опиши что беспокоит аудиторию "
                "в этом кластере и насколько сильно каждый аспект выражен "
                "(ориентируйся на размер подкатегорий: чем больше комментариев, тем сильнее беспокойство).\n\n"
                f"Всего комментариев в кластере: {topic.size_count}\n\n"
                f"Подкатегории:\n{positions_text}\n\n"
                "Верни строго JSON:\n"
                "{\n"
                '  "label": "название кластера",\n'
                '  "description": "описание кластера"\n'
                "}"
            )
            system_prompt = (
                "Ты аналитик комментариев YouTube. "
                "Создавай точные, информативные названия и описания кластеров на русском языке."
            )

            try:
                data = request_llm_json(
                    llm_provider,
                    prompt,
                    task="cluster_naming_from_positions",
                    estimated_out_tokens=300,
                    max_output_tokens=500,
                    system_prompt=system_prompt,
                )
            except (BudgetExceededError, ExternalServiceError) as exc:
                self.logger.warning(
                    "Post-position cluster naming failed for %s: %s",
                    topic.cluster_key,
                    exc,
                )
                updated.append(topic)
                continue

            new_label = str(data.get("label", "") or "").strip()
            new_description = str(data.get("description", "") or "").strip()

            updates: dict[str, Any] = {}
            if new_label and len(new_label.split()) >= 2:
                updates["label"] = new_label
            if new_description and len(new_description) >= 20:
                updates["description"] = new_description

            if updates:
                topic = topic.model_copy(update=updates)
                self.logger.info(
                    "Post-position naming for %s: label=%s",
                    topic.cluster_key,
                    topic.label,
                )

            updated.append(topic)

        return updated

    @staticmethod
    def _should_regenerate_from_positions(topic: TopicSummary) -> bool:
        """Return True when an extra cluster-renaming LLM call is justified.

        The initial cluster-labeling pass already produces a topic label and
        grounded description. Re-running a second naming pass for every topic is
        expensive and often redundant, so we only spend an extra call when the
        current topic metadata still looks weak or generic.
        """
        label = " ".join((topic.label or "").split()).strip()
        description = " ".join((topic.description or "").split()).strip()
        normalized = label.lower()

        if not label or len(label.split()) < 2:
            return True
        if normalized in {"", _UNCERTAIN_TOPIC_LABEL.lower()}:
            return True
        if normalized.startswith("[ошибка]"):
            return True
        if any(
            marker in normalized
            for marker in (
                "новая тема",
                "тема комментариев",
                "комментарии",
                "обсуждение",
                "позиция аудитории",
            )
        ):
            return True
        return len(topic.positions) >= 3 and not is_detailed_description(description)

    def _resolve_cluster_title(
        self,
        *,
        llm_provider: LLMProvider,
        cluster: ClusterDraft,
        comments: list[ProcessedComment],
        vectors: list[list[float]],
        candidate_title: str,
        fallback_title: str,
        llm_disabled: bool,
        diagnostics: dict[str, Any] | None = None,
        request_llm_json: RequestLLMJsonFn,
    ) -> tuple[str, bool, bool]:
        """Validate or regenerate a cluster title via LLM.

        Returns a tuple of ``(title, llm_disabled, used_fallback)``.
        """
        candidate_clean = sanitize_cluster_title(candidate_title, comments)
        if candidate_clean:
            return candidate_clean, llm_disabled, False

        if llm_disabled:
            return candidate_title or "[Ошибка] Название не определено", llm_disabled, True

        error_title = candidate_title or "[Ошибка] Название не определено"
        sample = self._build_cluster_title_sample(
            cluster=cluster, comments=comments, vectors=vectors
        )
        if not sample:
            return error_title, llm_disabled, True
        payload = [
            {
                "id": f"c{idx + 1:04d}",
                "weight": round(float(comment.weight), 3),
                "text": comment_text_for_output(comment),
            }
            for idx, comment in enumerate(sample)
            if comment_text_for_output(comment)
        ]
        if not payload:
            return error_title, llm_disabled, True

        retries_left = max(0, int(self.settings.position_title_retry_count))
        retry_reason: str | None = None
        for attempt in range(retries_left + 1):
            try:
                prompt = self._build_cluster_title_prompt(
                    comments_payload=payload,
                    fallback_title=fallback_title,
                    retry_reason=retry_reason,
                )
                data = request_llm_json(
                    llm_provider,
                    prompt,
                    task="cluster_title_naming",
                    estimated_out_tokens=140,
                    max_output_tokens=min(self.settings.openai_max_output_tokens, 240),
                )
            except (BudgetExceededError, ExternalServiceError) as exc:
                message = str(exc).lower()
                if (
                    isinstance(exc, BudgetExceededError)
                    or "429" in message
                    or "rate-limit" in message
                    or "rate limit" in message
                ):
                    llm_disabled = True
                    if diagnostics is not None:
                        self._mark_llm_disabled(
                            diagnostics,
                            reason=(
                                "budget_exceeded"
                                if isinstance(exc, BudgetExceededError)
                                else "rate_limited"
                            ),
                        )
                elif diagnostics is not None:
                    self._mark_llm_disabled(diagnostics, reason="external_service_error")
                if diagnostics is not None:
                    diagnostics["llm_title_fail_count"] = (
                        diagnostics.get("llm_title_fail_count", 0) + 1
                    )
                self.logger.error(
                    "Cluster title naming failed for %s via %s: %s",
                    cluster.cluster_key,
                    llm_provider.provider_name,
                    exc,
                )
                return error_title, llm_disabled, True

            candidate = (
                sanitize_cluster_title(str(data.get("title", "") or ""), comments) if data else ""
            )
            if candidate:
                return candidate, llm_disabled, False
            if attempt < retries_left:
                retry_reason = (
                    "title violated naming rules; regenerate concise title with strict constraints"
                )
        return error_title, llm_disabled, True

    def _build_cluster_title_prompt(
        self,
        *,
        comments_payload: list[dict[str, Any]],
        fallback_title: str,
        retry_reason: str | None,
    ) -> str:
        """Build the LLM prompt for cluster title generation."""
        retry_note = f"Retry reason: {retry_reason}\n" if retry_reason else ""
        return (
            "You name one semantic cluster of YouTube comments.\n"
            "Return strict JSON only.\n"
            "Output language: Russian.\n"
            "JSON schema:\n"
            "{\n"
            '  "title": "short clean title"\n'
            "}\n"
            "Requirements:\n"
            "- title must be 3-10 words.\n"
            "- no slash '/'.\n"
            "- no generic labels, no random phrase fragments.\n"
            "- forbidden words: спасибо, дело, меня, ваши, есть, просто, вообще, видео, комментарии.\n"
            f"- If unsure, align title near: {fallback_title}\n"
            f"{retry_note}"
            f"Comments sample JSON ({len(comments_payload)}):\n"
            f"{json.dumps(comments_payload, ensure_ascii=False)}"
        )

    def _build_cluster_title_sample(
        self,
        *,
        cluster: ClusterDraft,
        comments: list[ProcessedComment],
        vectors: list[list[float]],
    ) -> list[ProcessedComment]:
        """Select a representative sample of comments for title generation."""
        ranked_indices = self._rank_cluster_member_indices(cluster, comments, vectors)
        if not ranked_indices:
            return []
        target = self._adaptive_representative_count(len(cluster.member_indices))
        selected_indices = ranked_indices[:target]
        return [comments[idx] for idx in selected_indices if 0 <= idx < len(comments)]

    def _adaptive_representative_count(self, member_count: int) -> int:
        """Return a sqrt-based sample size clamped between 8 and 30."""
        if member_count <= 0:
            return 1
        target = int(round(float(np.sqrt(member_count))))
        target = min(30, max(8, target))
        return min(member_count, target)

    def _resolve_cluster_quotes(
        self,
        *,
        cluster: ClusterDraft,
        comments: list[ProcessedComment],
        vectors: list[list[float]],
        raw_quotes: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Rank and deduplicate quotes for a cluster.

        Returns ``(main_comments, question_comments, ordered_comments)``.
        """
        ranked_indices = self._rank_cluster_member_indices(cluster, comments, vectors)
        if not ranked_indices:
            return [], [], []

        matched_indices: list[int] = []
        seen_indices: set[int] = set()
        for quote in raw_quotes:
            matched_idx = self._match_quote_to_cluster_comment(
                quote=quote,
                ranked_indices=ranked_indices,
                comments=comments,
            )
            if matched_idx is None or matched_idx in seen_indices:
                continue
            seen_indices.add(matched_idx)
            matched_indices.append(matched_idx)

        ordered_indices = matched_indices + [
            idx for idx in ranked_indices if idx not in seen_indices
        ]
        main_comments: list[str] = []
        question_comments: list[str] = []
        ordered_comments: list[str] = []
        for member_idx in ordered_indices:
            quote_text = comment_text_for_output(comments[member_idx])
            if not quote_text:
                continue
            ordered_comments.append(quote_text)
            if is_question_comment_text(quote_text):
                question_comments.append(quote_text)
                continue
            main_comments.append(quote_text)

        if not main_comments:
            main_comments = [
                text for text in ordered_comments if not is_question_comment_text(text)
            ]
        return main_comments, question_comments, ordered_comments

    def _cluster_quote_relevance_threshold(self, scores: list[float]) -> float:
        """Compute an adaptive relevance threshold from a list of similarity scores."""
        if not scores:
            return 0.0
        ordered = sorted(float(score) for score in scores)
        count = len(ordered)
        if count >= 80:
            quantile = 0.65
            baseline = 0.52
        elif count >= 40:
            quantile = 0.55
            baseline = 0.50
        elif count >= 20:
            quantile = 0.45
            baseline = 0.47
        else:
            quantile = 0.30
            baseline = 0.42
        idx = max(0, min(count - 1, int(round((count - 1) * quantile))))
        return max(baseline, ordered[idx])

    # ================================================================
    # Topic postprocessing
    # ================================================================

    def _postprocess_labeled_topics(
        self,
        *,
        clusters: list[ClusterDraft],
        topics: list[TopicSummary],
        comments: list[ProcessedComment],
        vectors: list[list[float]],
        episode_context: EpisodeContext,
        allow_harmonization: bool = True,
    ) -> tuple[list[ClusterDraft], list[TopicSummary]]:
        """Merge, harmonize, deduplicate, and collapse uncertain topics.

        Returns the final ``(clusters, topics)`` pair after all
        postprocessing passes.
        """
        if not topics:
            return sorted(clusters, key=lambda item: item.weighted_share, reverse=True), topics

        cluster_by_key = {
            cluster.cluster_key: cluster.model_copy(deep=True) for cluster in clusters
        }
        topic_by_key = {
            topic.cluster_key: topic.model_copy(deep=True)
            for topic in topics
            if topic.cluster_key in cluster_by_key
        }
        if not topic_by_key:
            return sorted(clusters, key=lambda item: item.weighted_share, reverse=True), []

        matrix = np.array(vectors, dtype=np.float32)
        weights = np.array([comment.weight for comment in comments], dtype=np.float32)
        total_count = len(comments)
        total_weight = float(max(weights.sum(), 1e-6))

        if self.settings.topic_merge_enabled and len(topic_by_key) >= 2:
            merge_limit = max(2, len(topic_by_key) * 3)
            merge_round = 0
            merged_any = True
            while merged_any and merge_round < merge_limit:
                merged_any = False
                merge_round += 1
                ordered_keys = sorted(
                    topic_by_key.keys(),
                    key=lambda key: topic_by_key[key].weighted_share,
                    reverse=True,
                )
                for left_idx, left_key in enumerate(ordered_keys):
                    if left_key not in topic_by_key:
                        continue
                    left_topic = topic_by_key[left_key]
                    left_cluster = cluster_by_key.get(left_key)
                    if left_cluster is None:
                        continue
                    for right_key in ordered_keys[left_idx + 1 :]:
                        if right_key not in topic_by_key:
                            continue
                        right_topic = topic_by_key[right_key]
                        right_cluster = cluster_by_key.get(right_key)
                        if right_cluster is None:
                            continue
                        if not self._should_merge_topics(
                            left_topic=left_topic,
                            right_topic=right_topic,
                            left_cluster=left_cluster,
                            right_cluster=right_cluster,
                        ):
                            continue
                        primary_key, secondary_key = self._select_merge_primary_key(
                            left_topic=left_topic,
                            right_topic=right_topic,
                        )
                        merged_cluster, merged_topic = self._merge_topic_pair(
                            primary_key=primary_key,
                            secondary_key=secondary_key,
                            cluster_by_key=cluster_by_key,
                            topic_by_key=topic_by_key,
                            comments=comments,
                            vectors=vectors,
                            matrix=matrix,
                            weights=weights,
                            total_count=total_count,
                            total_weight=total_weight,
                        )
                        cluster_by_key[primary_key] = merged_cluster
                        topic_by_key[primary_key] = merged_topic
                        cluster_by_key.pop(secondary_key, None)
                        topic_by_key.pop(secondary_key, None)
                        self.logger.info(
                            "Merged duplicate topics %s + %s -> %s.",
                            left_topic.label,
                            right_topic.label,
                            merged_topic.label,
                        )
                        merged_any = True
                        break
                    if merged_any:
                        break

        merged_topics = sorted(
            topic_by_key.values(), key=lambda item: item.weighted_share, reverse=True
        )
        if allow_harmonization:
            merged_topics = self._harmonize_comment_topics_with_episode_context(
                merged_topics, episode_context
            )
        else:
            self.logger.info(
                "Topic harmonization skipped (episode source=%s).",
                episode_context.source,
            )
        merged_clusters = sorted(
            [
                cluster_by_key[topic.cluster_key]
                for topic in merged_topics
                if topic.cluster_key in cluster_by_key
            ],
            key=lambda item: item.weighted_share,
            reverse=True,
        )
        merged_topics = self._dedupe_topic_comments_across_topics(
            merged_topics,
            merged_clusters,
            comments,
            vectors,
        )
        merged_topics = self._collapse_uncertain_topics(
            merged_topics,
            merged_clusters,
            comments,
            vectors,
        )
        topic_keys = {topic.cluster_key for topic in merged_topics}
        merged_clusters = [
            cluster for cluster in merged_clusters if cluster.cluster_key in topic_keys
        ]
        merged_clusters.sort(key=lambda item: item.weighted_share, reverse=True)
        return merged_clusters, merged_topics

    def _should_merge_topics(
        self,
        *,
        left_topic: TopicSummary,
        right_topic: TopicSummary,
        left_cluster: ClusterDraft,
        right_cluster: ClusterDraft,
    ) -> bool:
        """Decide whether two topics should be merged based on centroid and label similarity."""
        if left_cluster.centroid is None or right_cluster.centroid is None:
            return False

        centroid_similarity = _cosine_similarity(left_cluster.centroid, right_cluster.centroid)
        label_left = topic_label_tokens(left_topic.label)
        label_right = topic_label_tokens(right_topic.label)
        label_jaccard = token_jaccard(label_left, label_right)
        desc_jaccard = token_jaccard(
            topic_label_tokens(left_topic.description),
            topic_label_tokens(right_topic.description),
        )

        subset_min = max(1, int(self.settings.topic_merge_label_subset_min_tokens))
        subset_relation = (
            len(label_left) >= subset_min
            and len(label_right) >= subset_min
            and (label_left.issubset(label_right) or label_right.issubset(label_left))
        )
        left_label_norm = left_topic.label.lower().strip()
        right_label_norm = right_topic.label.lower().strip()
        contains_relation = (
            left_label_norm in right_label_norm or right_label_norm in left_label_norm
        )

        similarity_threshold = max(0.90, float(self.settings.topic_merge_similarity_threshold))
        label_threshold = max(0.34, float(self.settings.topic_merge_label_jaccard_min))
        min_weighted_share = min(
            float(left_topic.weighted_share), float(right_topic.weighted_share)
        )
        both_large_topics = min_weighted_share >= 8.0

        # Do not merge two substantial topics if lexical overlap is weak.
        if both_large_topics and label_jaccard < 0.42:
            return False

        strict_similarity = min(0.98, similarity_threshold + 0.04)
        strict_label = max(label_threshold, 0.42)
        if centroid_similarity >= strict_similarity and label_jaccard >= strict_label:
            return True
        if (
            subset_relation
            and label_jaccard >= 0.34
            and centroid_similarity >= similarity_threshold
        ):
            return True
        if (
            contains_relation
            and label_jaccard >= 0.38
            and centroid_similarity >= similarity_threshold
        ):
            return True
        if label_jaccard >= 0.62 and centroid_similarity >= 0.90:
            return True
        return desc_jaccard >= 0.45 and label_jaccard >= 0.38 and centroid_similarity >= 0.92

    def _select_merge_primary_key(
        self,
        *,
        left_topic: TopicSummary,
        right_topic: TopicSummary,
    ) -> tuple[str, str]:
        """Select which topic key becomes the primary during a merge."""
        if left_topic.source == "episode_topic" and right_topic.source != "episode_topic":
            return left_topic.cluster_key, right_topic.cluster_key
        if right_topic.source == "episode_topic" and left_topic.source != "episode_topic":
            return right_topic.cluster_key, left_topic.cluster_key
        if left_topic.weighted_share > right_topic.weighted_share:
            return left_topic.cluster_key, right_topic.cluster_key
        if right_topic.weighted_share > left_topic.weighted_share:
            return right_topic.cluster_key, left_topic.cluster_key
        if left_topic.size_count >= right_topic.size_count:
            return left_topic.cluster_key, right_topic.cluster_key
        return right_topic.cluster_key, left_topic.cluster_key

    def _merge_topic_pair(
        self,
        *,
        primary_key: str,
        secondary_key: str,
        cluster_by_key: dict[str, ClusterDraft],
        topic_by_key: dict[str, TopicSummary],
        comments: list[ProcessedComment],
        vectors: list[list[float]],
        matrix: np.ndarray,
        weights: np.ndarray,
        total_count: int,
        total_weight: float,
    ) -> tuple[ClusterDraft, TopicSummary]:
        """Merge two topics and their clusters into one."""
        primary_cluster = cluster_by_key[primary_key]
        secondary_cluster = cluster_by_key[secondary_key]
        primary_topic = topic_by_key[primary_key]
        secondary_topic = topic_by_key[secondary_key]

        merged_members = sorted(
            set(primary_cluster.member_indices).union(secondary_cluster.member_indices)
        )
        merged_cluster = self._rebuild_cluster_from_members(
            cluster_key=primary_key,
            members=merged_members,
            matrix=matrix,
            weights=weights,
            total_count=total_count,
            total_weight=total_weight,
        )
        raw_quotes = (
            primary_topic.representative_quotes
            + secondary_topic.representative_quotes
            + primary_topic.question_comments
            + secondary_topic.question_comments
        )
        merged_quotes, merged_questions, ordered_comments = self._resolve_cluster_quotes(
            cluster=merged_cluster,
            comments=comments,
            vectors=vectors,
            raw_quotes=raw_quotes,
        )
        sentiment = self._merge_topic_sentiment(primary_topic, secondary_topic)
        description = self._resolve_topic_description(
            description=primary_topic.description,
            fallback_description=secondary_topic.description,
            ordered_comments=ordered_comments,
            sentiment=sentiment,
        )
        label = self._pick_topic_label(primary_topic.label, secondary_topic.label)
        source = (
            "episode_topic"
            if (
                primary_topic.source == "episode_topic" or secondary_topic.source == "episode_topic"
            )
            else "comment_topic"
        )
        merged_topic = TopicSummary(
            cluster_key=primary_key,
            label=label,
            description=description,
            author_actions=self._merge_actions(primary_topic, secondary_topic),
            sentiment=sentiment,
            emotion_tags=self._merge_emotion_tags(primary_topic, secondary_topic),
            intent_distribution=self._merge_intents(primary_topic, secondary_topic),
            representative_quotes=merged_quotes,
            question_comments=merged_questions,
            size_count=merged_cluster.size_count,
            share_pct=merged_cluster.share_pct,
            weighted_share=merged_cluster.weighted_share,
            is_emerging=merged_cluster.is_emerging,
            source=source,
            coherence_score=self._estimate_cluster_coherence(merged_cluster, comments, vectors),
            centroid=merged_cluster.centroid,
            assignment_confidence=round(
                (
                    primary_topic.assignment_confidence * max(1, primary_topic.size_count)
                    + secondary_topic.assignment_confidence * max(1, secondary_topic.size_count)
                )
                / max(1, primary_topic.size_count + secondary_topic.size_count),
                4,
            ),
            ambiguous_share_pct=round(
                (
                    primary_topic.ambiguous_share_pct * max(1, primary_topic.size_count)
                    + secondary_topic.ambiguous_share_pct * max(1, secondary_topic.size_count)
                )
                / max(1, primary_topic.size_count + secondary_topic.size_count),
                2,
            ),
            soft_assignment_notes=list(
                dict.fromkeys(
                    [
                        *primary_topic.soft_assignment_notes,
                        *secondary_topic.soft_assignment_notes,
                    ]
                )
            )[: self.settings.cluster_assignment_note_limit],
        )
        return merged_cluster, merged_topic

    def _build_soft_assignment_notes(self, cluster: ClusterDraft) -> list[str]:
        notes: list[str] = []
        if cluster.assignment_confidence > 0:
            notes.append(
                f"Средняя уверенность распределения комментариев: {cluster.assignment_confidence:.2f}"
            )
        if cluster.ambiguous_member_count > 0:
            notes.append(
                f"Пограничных комментариев: {cluster.ambiguous_member_count}/{cluster.size_count}"
            )
        return notes[: self.settings.cluster_assignment_note_limit]

    def _pick_topic_label(self, primary_label: str, secondary_label: str) -> str:
        """Choose the better label from two candidates."""
        primary = " ".join((primary_label or "").split())
        secondary = " ".join((secondary_label or "").split())
        if not primary:
            return secondary
        if not secondary:
            return primary
        primary_tokens = topic_label_tokens(primary)
        secondary_tokens = topic_label_tokens(secondary)
        if token_jaccard(primary_tokens, secondary_tokens) >= 0.6 and len(secondary) < len(primary):
            return secondary
        return primary

    def _merge_topic_sentiment(
        self, primary_topic: TopicSummary, secondary_topic: TopicSummary
    ) -> str:
        """Weight-average two topic sentiments into one label."""
        score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        primary_weight = max(0.01, float(primary_topic.weighted_share))
        secondary_weight = max(0.01, float(secondary_topic.weighted_share))
        weighted_score = (
            score_map.get(primary_topic.sentiment, 0.0) * primary_weight
            + score_map.get(secondary_topic.sentiment, 0.0) * secondary_weight
        )
        avg_score = weighted_score / (primary_weight + secondary_weight)
        if avg_score >= 0.15:
            return "positive"
        if avg_score <= -0.15:
            return "negative"
        return "neutral"

    def _merge_intents(
        self, primary_topic: TopicSummary, secondary_topic: TopicSummary
    ) -> dict[str, int]:
        """Sum intent distributions from two topics."""
        merged: dict[str, int] = {}
        keys = set(primary_topic.intent_distribution).union(secondary_topic.intent_distribution)
        for key in keys:
            merged[key] = int(primary_topic.intent_distribution.get(key, 0) or 0) + int(
                secondary_topic.intent_distribution.get(key, 0) or 0
            )
        return merged

    def _merge_emotion_tags(
        self, primary_topic: TopicSummary, secondary_topic: TopicSummary
    ) -> list[str]:
        """Combine emotion tags from two topics, weighted by share."""
        counter: Counter[str] = Counter()
        for tag in primary_topic.emotion_tags:
            cleaned = str(tag).strip().lower()
            if cleaned:
                counter[cleaned] += max(1, int(round(primary_topic.weighted_share)))
        for tag in secondary_topic.emotion_tags:
            cleaned = str(tag).strip().lower()
            if cleaned:
                counter[cleaned] += max(1, int(round(secondary_topic.weighted_share)))
        return [tag for tag, _ in counter.most_common(2)]

    def _merge_actions(
        self, primary_topic: TopicSummary, secondary_topic: TopicSummary
    ) -> list[str]:
        """Deduplicate and merge author action recommendations."""
        seen: set[str] = set()
        actions: list[str] = []
        for action in primary_topic.author_actions + secondary_topic.author_actions:
            cleaned = " ".join(str(action).split()).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            actions.append(cleaned)
        return actions

    def _harmonize_comment_topics_with_episode_context(
        self,
        topics: list[TopicSummary],
        episode_context: EpisodeContext,
    ) -> list[TopicSummary]:
        """Re-label comment-sourced topics that closely match episode topics."""
        if not topics or not episode_context.topics:
            return topics

        episode_titles: list[tuple[str, set[str]]] = []
        for topic in episode_context.topics:
            title = " ".join((topic.title or "").split()).strip()
            if not title:
                continue
            title_tokens = topic_label_tokens(title)
            if not title_tokens:
                continue
            episode_titles.append((title, title_tokens))
        if not episode_titles:
            return topics

        updated: list[TopicSummary] = []
        for topic in topics:
            if topic.source == "episode_topic":
                updated.append(topic)
                continue
            topic_tokens = topic_label_tokens(f"{topic.label} {topic.description}")
            if not topic_tokens:
                updated.append(topic)
                continue

            best_title = ""
            best_jaccard = 0.0
            best_cover = 0.0
            for title, title_tokens in episode_titles:
                jaccard = token_jaccard(topic_tokens, title_tokens)
                cover = len(topic_tokens.intersection(title_tokens)) / max(1, len(title_tokens))
                if max(jaccard, cover) > max(best_jaccard, best_cover):
                    best_title = title
                    best_jaccard = jaccard
                    best_cover = cover

            if best_title and (best_jaccard >= 0.55 or best_cover >= 0.72):
                updated.append(
                    topic.model_copy(
                        update={
                            "source": "episode_topic",
                            "label": self._pick_topic_label(best_title, topic.label),
                        }
                    )
                )
                continue
            updated.append(topic)
        return updated

    def _collapse_uncertain_topics(
        self,
        topics: list[TopicSummary],
        clusters: list[ClusterDraft],
        comments: list[ProcessedComment],
        vectors: list[list[float]],
    ) -> list[TopicSummary]:
        """Collapse low-quality or near-duplicate topics into one uncertain bucket."""
        if not topics:
            return topics

        cluster_by_key = {cluster.cluster_key: cluster for cluster in clusters}
        kept: list[TopicSummary] = []
        kept_label_tokens: list[set[str]] = []
        kept_labels: set[str] = set()
        uncertain: list[TopicSummary] = []
        min_comments = max(0, int(self.settings.topic_uncertain_min_comments))
        support_min = float(self.settings.topic_uncertain_support_min)

        for topic in sorted(topics, key=lambda item: item.weighted_share, reverse=True):
            topic_comments = topic.representative_quotes + topic.question_comments
            comment_count = len(topic_comments)
            support_score = description_comment_support_score(topic.description, topic_comments)
            label_tokens = topic_label_tokens(topic.label)
            cluster = cluster_by_key.get(topic.cluster_key)
            near_duplicate = any(
                token_jaccard(label_tokens, existing_tokens) >= 0.5
                for existing_tokens in kept_label_tokens
                if label_tokens and existing_tokens
            )
            normalized_label = topic.label.strip().lower()
            duplicate_label = normalized_label in kept_labels
            if cluster is not None and self._skip_low_coherence_cluster(
                cluster, float(topic.coherence_score)
            ):
                uncertain.append(topic)
                continue
            if comment_count < min_comments:
                uncertain.append(topic)
                continue
            if duplicate_label and comment_count <= max(3, min_comments + 2):
                uncertain.append(topic)
                continue
            if (
                near_duplicate
                and topic.source == "comment_topic"
                and comment_count <= max(2, min_comments + 1)
            ):
                uncertain.append(topic)
                continue
            if near_duplicate and support_score < support_min and comment_count <= 2:
                uncertain.append(topic)
                continue
            kept.append(topic)
            kept_label_tokens.append(label_tokens)
            if normalized_label:
                kept_labels.add(normalized_label)

        if not uncertain:
            return sorted(kept, key=lambda item: item.weighted_share, reverse=True)
        primary = sorted(uncertain, key=lambda item: item.weighted_share)[0]
        cluster = cluster_by_key.get(primary.cluster_key)
        main_quotes = list(primary.representative_quotes)
        question_quotes = list(primary.question_comments)
        if cluster is not None and len(main_quotes) + len(question_quotes) < max(1, min_comments):
            fallback_main, fallback_questions, _ = self._resolve_cluster_quotes(
                cluster=cluster,
                comments=comments,
                vectors=vectors,
                raw_quotes=[],
            )
            if not main_quotes:
                main_quotes = fallback_main
            if not question_quotes:
                question_quotes = fallback_questions
        uncertain_topic = primary.model_copy(
            update={
                "label": _UNCERTAIN_TOPIC_LABEL,
                "description": (
                    "Комментарии этой категории не формируют устойчивую самостоятельную тему "
                    "или заметно пересекаются с другими темами. Используйте этот блок как дополнительный контекст."
                ),
                "source": "comment_topic",
                "sentiment": "neutral",
                "emotion_tags": [],
                "representative_quotes": main_quotes,
                "question_comments": question_quotes,
                "is_emerging": True,
            }
        )
        merged = kept + [uncertain_topic]
        merged.sort(key=lambda item: item.weighted_share, reverse=True)
        return merged

    def _resolve_topic_description(
        self,
        *,
        description: str,
        fallback_description: str,
        ordered_comments: list[str],
        sentiment: str,
    ) -> str:
        """Validate, extend, or rebuild a topic description with comment support."""
        cleaned = " ".join((description or "").split()).strip()
        fallback_cleaned = " ".join((fallback_description or "").split()).strip()
        if not cleaned:
            cleaned = fallback_cleaned
        if 0 < len(cleaned) < 80:
            built = build_comment_grounded_description(ordered_comments, sentiment)
            appendix = built or fallback_cleaned
            if appendix and appendix.lower() not in cleaned.lower():
                cleaned = f"{cleaned.rstrip('.')} {appendix}".strip()
        elif not cleaned:
            cleaned = build_comment_grounded_description(ordered_comments, sentiment)
        support = description_comment_support_score(cleaned, ordered_comments)
        if (
            support < 0.12
            and not is_detailed_description(cleaned)
            and fallback_cleaned
            and fallback_cleaned.lower() not in cleaned.lower()
        ):
            cleaned = (
                f"{cleaned.rstrip('.')} {fallback_cleaned}".strip() if cleaned else fallback_cleaned
            )
        if len(cleaned) > 520:
            cleaned = cleaned[:520].rstrip()
        return cleaned

    # ================================================================
    # Deduplication
    # ================================================================

    def _dedupe_label(
        self,
        label: str,
        cluster_comments: list[ProcessedComment],
        seen_labels: set[str],
    ) -> str:
        """Ensure *label* is unique within *seen_labels*, appending a disambiguator if needed."""
        base = " ".join(label.split()) or "Смешанная тема"
        base_key = base.lower()
        if base_key not in seen_labels:
            seen_labels.add(base_key)
            return base

        token_freq: dict[str, int] = {}
        for comment in cluster_comments[:16]:
            for token in re.findall(r"\w{4,}", comment.text_normalized.lower()):
                if token in {
                    "коммент",
                    "комментарии",
                    "видео",
                    "канал",
                    "news",
                    "video",
                    "comments",
                }:
                    continue
                token_freq[token] = token_freq.get(token, 0) + 1

        for token, _ in sorted(token_freq.items(), key=lambda pair: pair[1], reverse=True):
            candidate = f"{base}: {token}"
            candidate_key = candidate.lower()
            if candidate_key in seen_labels:
                continue
            seen_labels.add(candidate_key)
            return candidate

        suffix = 2
        while True:
            candidate = f"{base} ({suffix})"
            candidate_key = candidate.lower()
            if candidate_key not in seen_labels:
                seen_labels.add(candidate_key)
                return candidate
            suffix += 1

    def _dedupe_topic_comments_across_topics(
        self,
        topics: list[TopicSummary],
        clusters: list[ClusterDraft],
        comments: list[ProcessedComment],
        vectors: list[list[float]],
    ) -> list[TopicSummary]:
        """Remove duplicate comment texts across topics, preserving highest-weight copies."""
        if not topics:
            return topics
        if not self.settings.topic_cross_dedupe_enabled:
            return sorted(topics, key=lambda item: item.weighted_share, reverse=True)

        cluster_by_key = {cluster.cluster_key: cluster for cluster in clusters}
        ordered_topics = sorted(topics, key=lambda item: item.weighted_share, reverse=True)
        global_seen_keys: set[str] = set()
        global_seen_token_sets: list[set[str]] = []
        global_threshold = max(
            0.75,
            min(0.99, float(self.settings.topic_cross_dedupe_similarity_threshold)),
        )

        for topic in ordered_topics:
            cluster = cluster_by_key.get(topic.cluster_key)
            if cluster is None:
                continue
            ranked_indices = self._rank_cluster_member_indices(cluster, comments, vectors)
            if not ranked_indices:
                topic.representative_quotes = []
                topic.question_comments = []
                continue

            local_seen_keys: set[str] = set()
            local_seen_token_sets: list[set[str]] = []
            main_comments: list[str] = []
            question_comments: list[str] = []

            for member_idx in ranked_indices:
                quote_text = comment_text_for_output(comments[member_idx])
                if not quote_text:
                    continue
                key = quote_text.lower()
                token_set = text_token_set(quote_text)
                if is_duplicate_text_signature(
                    key=key,
                    tokens=token_set,
                    seen_keys=local_seen_keys,
                    seen_token_sets=local_seen_token_sets,
                    threshold=0.9,
                ):
                    continue
                if is_duplicate_text_signature(
                    key=key,
                    tokens=token_set,
                    seen_keys=global_seen_keys,
                    seen_token_sets=global_seen_token_sets,
                    threshold=global_threshold,
                ):
                    continue

                local_seen_keys.add(key)
                if token_set:
                    local_seen_token_sets.append(token_set)
                global_seen_keys.add(key)
                if token_set:
                    global_seen_token_sets.append(token_set)

                if is_question_comment_text(quote_text):
                    question_comments.append(quote_text)
                    continue
                main_comments.append(quote_text)

            if not main_comments:
                fallback_seen: set[str] = {text.lower() for text in local_seen_keys}
                for member_idx in ranked_indices:
                    quote_text = comment_text_for_output(comments[member_idx])
                    if not quote_text or is_question_comment_text(quote_text):
                        continue
                    key = quote_text.lower()
                    if key in fallback_seen:
                        continue
                    fallback_seen.add(key)
                    main_comments.append(quote_text)

            topic.representative_quotes = main_comments
            topic.question_comments = question_comments

        return ordered_topics

    # ================================================================
    # Cluster ranking and matching
    # ================================================================

    def _rank_cluster_member_indices(
        self,
        cluster: ClusterDraft,
        comments: list[ProcessedComment],
        vectors: list[list[float]],
    ) -> list[int]:
        """Rank cluster members by relevance score (similarity + weight + engagement)."""
        if not cluster.member_indices:
            return []
        representative_set = set(cluster.representative_indices)
        scored: list[tuple[int, float]] = []
        for member_idx in cluster.member_indices:
            if member_idx < 0 or member_idx >= len(comments):
                continue
            comment = comments[member_idx]
            vector = vectors[member_idx] if member_idx < len(vectors) else None
            relevance = self._cluster_member_relevance_score(
                centroid=cluster.centroid,
                vector=vector,
                weight=float(comment.weight),
                like_count=int(comment.like_count),
                reply_count=int(comment.reply_count),
            )
            if member_idx in representative_set:
                relevance += 0.05
            scored.append((member_idx, relevance))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [member_idx for member_idx, _ in scored]

    def _match_quote_to_cluster_comment(
        self,
        *,
        quote: str,
        ranked_indices: list[int],
        comments: list[ProcessedComment],
    ) -> int | None:
        """Find the cluster comment that best matches a raw quote string."""
        quote_norm = normalize_text_for_matching(quote)
        if not quote_norm:
            return None
        quote_compact = compact_text_for_matching(quote_norm)
        prepared: list[tuple[int, str, str]] = []
        for member_idx in ranked_indices:
            comment_norm = normalize_text_for_matching(
                comment_text_for_output(comments[member_idx])
            )
            if not comment_norm:
                continue
            prepared.append((member_idx, comment_norm, compact_text_for_matching(comment_norm)))

        for member_idx, comment_norm, _ in prepared:
            if quote_norm == comment_norm:
                return member_idx

        if len(quote_norm) >= 26:
            for member_idx, comment_norm, _ in prepared:
                if quote_norm in comment_norm:
                    return member_idx

        if len(quote_compact) >= 22:
            for member_idx, _, comment_compact in prepared:
                if quote_compact in comment_compact:
                    return member_idx
        return None

    def _cluster_member_relevance_score(
        self,
        *,
        centroid: list[float] | None,
        vector: list[float] | None,
        weight: float,
        like_count: int,
        reply_count: int,
    ) -> float:
        """Compute a composite relevance score for a cluster member."""
        similarity = 0.0
        if centroid is not None and vector is not None:
            similarity = max(0.0, _cosine_similarity(centroid, vector))
        weight_limit = max(1.0, float(self.settings.comment_weight_max))
        weight_score = max(0.0, min(weight, weight_limit)) / weight_limit
        likes_score = min(max(like_count, 0), 60) / 60.0
        replies_score = min(max(reply_count, 0), 30) / 30.0
        engagement_score = min(1.0, (likes_score + replies_score) / 2.0)
        score = similarity * 0.75 + weight_score * 0.20 + engagement_score * 0.05
        return max(0.0, min(1.0, score))

    # ================================================================
    # Cluster infrastructure
    # ================================================================

    def _rebuild_cluster_from_members(
        self,
        *,
        cluster_key: str,
        members: list[int],
        matrix: np.ndarray,
        weights: np.ndarray,
        total_count: int,
        total_weight: float,
    ) -> ClusterDraft:
        """Rebuild a :class:`ClusterDraft` from a merged member list."""
        centroid = matrix[members].mean(axis=0)
        representatives = self._select_representatives(members, matrix, centroid, weights)
        size_count = len(members)
        share_pct = round(size_count / max(1, total_count) * 100, 2)
        weighted_share = round(float(weights[members].sum() / max(total_weight, 1e-6) * 100), 2)
        return ClusterDraft(
            cluster_key=cluster_key,
            member_indices=members,
            representative_indices=representatives,
            centroid=centroid.tolist(),
            size_count=size_count,
            share_pct=share_pct,
            weighted_share=weighted_share,
            is_emerging=size_count < self.settings.cluster_min_size,
            assignment_confidence=0.0,
            ambiguous_member_count=0,
            ambiguous_share_pct=0.0,
        )

    def _select_representatives(
        self,
        members: list[int],
        matrix: np.ndarray,
        centroid: np.ndarray,
        weights: np.ndarray,
    ) -> list[int]:
        """Pick the most representative member indices by similarity + weight."""
        scored: list[tuple[int, float]] = []
        centroid_norm = float(np.linalg.norm(centroid))
        for idx in members:
            vec = matrix[idx]
            denom = float(np.linalg.norm(vec) * centroid_norm)
            sim = float(np.dot(vec, centroid) / denom) if denom else 0.0
            scored.append((idx, float(weights[idx]) + sim))
        scored.sort(key=lambda item: item[1], reverse=True)
        limit = self._adaptive_representative_count(len(scored))
        return [idx for idx, _ in scored[:limit]]

    def _build_cluster_keywords(
        self,
        cluster: ClusterDraft,
        comments: list[ProcessedComment],
        limit: int = 8,
    ) -> set[str]:
        """Extract the top *limit* keywords from a cluster's highest-weight members."""
        if not cluster.member_indices:
            return set()
        ranked_members = sorted(
            cluster.member_indices,
            key=lambda idx: comments[idx].weight,
            reverse=True,
        )[: max(16, limit * 6)]
        token_freq: dict[str, int] = {}
        for idx in ranked_members:
            text = comments[idx].text_normalized.lower()
            for token in _CLUSTER_TOKEN_RE.findall(text):
                if token in _CLUSTER_STOPWORDS:
                    continue
                token_freq[token] = token_freq.get(token, 0) + 1
        tokens = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
        return {token for token, _ in tokens[:limit]}

    def _cluster_keyword_jaccard(self, left: set[str], right: set[str]) -> float:
        """Compute Jaccard similarity between two keyword sets."""
        if not left or not right:
            return 0.0
        union = left.union(right)
        if not union:
            return 0.0
        return len(left.intersection(right)) / len(union)

    def _merge_similar_clusters(
        self,
        clusters: list[ClusterDraft],
        comments: list[ProcessedComment],
        vectors: list[list[float]],
    ) -> list[ClusterDraft]:
        """Iteratively merge clusters whose centroids and keywords are similar."""
        if not self.settings.cluster_merge_enabled or len(clusters) < 2:
            return sorted(clusters, key=lambda item: item.weighted_share, reverse=True)

        matrix = np.array(vectors, dtype=np.float32)
        weights = np.array([comment.weight for comment in comments], dtype=np.float32)
        total_count = len(comments)
        total_weight = float(max(weights.sum(), 1e-6))
        working = [cluster.model_copy(deep=True) for cluster in clusters]
        similarity_threshold = float(self.settings.cluster_merge_similarity_threshold)
        overlap_threshold = float(self.settings.cluster_merge_keyword_jaccard_min)
        max_rounds = max(1, int(self.settings.cluster_merge_max_rounds))

        for _ in range(max_rounds):
            working.sort(key=lambda item: item.weighted_share, reverse=True)
            merged_any = False
            i = 0
            while i < len(working):
                base = working[i]
                base_keywords = self._build_cluster_keywords(base, comments)
                j = i + 1
                while j < len(working):
                    candidate = working[j]
                    if base.centroid is None or candidate.centroid is None:
                        j += 1
                        continue
                    sim = _cosine_similarity(base.centroid, candidate.centroid)
                    if sim < similarity_threshold:
                        j += 1
                        continue
                    overlap = self._cluster_keyword_jaccard(
                        base_keywords,
                        self._build_cluster_keywords(candidate, comments),
                    )
                    if overlap < overlap_threshold:
                        j += 1
                        continue

                    merged_members = sorted(
                        set(base.member_indices).union(candidate.member_indices)
                    )
                    merged = self._rebuild_cluster_from_members(
                        cluster_key=base.cluster_key,
                        members=merged_members,
                        matrix=matrix,
                        weights=weights,
                        total_count=total_count,
                        total_weight=total_weight,
                    )
                    merged.assignment_confidence = round(
                        (
                            base.assignment_confidence * max(1, base.size_count)
                            + candidate.assignment_confidence * max(1, candidate.size_count)
                        )
                        / max(1, base.size_count + candidate.size_count),
                        4,
                    )
                    merged.ambiguous_member_count = (
                        base.ambiguous_member_count + candidate.ambiguous_member_count
                    )
                    merged.ambiguous_share_pct = round(
                        merged.ambiguous_member_count / max(1, merged.size_count) * 100,
                        2,
                    )
                    self.logger.info(
                        "Merged similar clusters %s + %s (sim=%.3f overlap=%.3f -> size=%s).",
                        base.cluster_key,
                        candidate.cluster_key,
                        sim,
                        overlap,
                        merged.size_count,
                    )
                    working[i] = merged
                    working.pop(j)
                    base = merged
                    base_keywords = self._build_cluster_keywords(base, comments)
                    merged_any = True
                i += 1
            if not merged_any:
                break

        working.sort(key=lambda item: item.weighted_share, reverse=True)
        return working

    # ================================================================
    # Internal helpers
    # ================================================================

    def _estimate_cluster_coherence(
        self,
        cluster: ClusterDraft,
        comments: list[ProcessedComment],
        vectors: list[list[float]],
    ) -> float:
        """Compute a weighted-average cosine coherence for a cluster."""
        if not cluster.member_indices or cluster.centroid is None:
            return 0.0
        centroid = np.array(cluster.centroid, dtype=np.float32)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm == 0:
            return 0.0
        score = 0.0
        weight_sum = 0.0
        for idx in cluster.member_indices:
            vec = np.array(vectors[idx], dtype=np.float32)
            denom = float(np.linalg.norm(vec) * centroid_norm)
            sim = float(np.dot(vec, centroid) / denom) if denom else 0.0
            weight = max(0.1, float(comments[idx].weight))
            score += sim * weight
            weight_sum += weight
        if weight_sum <= 0:
            return 0.0
        return max(0.0, min(1.0, score / weight_sum))

    def _skip_low_coherence_cluster(self, cluster: ClusterDraft, coherence: float) -> bool:
        """Return ``True`` when a small low-coherence cluster should be collapsed."""
        if cluster.weighted_share >= self.settings.topic_min_weighted_share_for_low_coherence:
            return False
        if cluster.size_count >= max(20, self.settings.cluster_min_size * 3):
            return False
        return coherence < self.settings.topic_coherence_min

    def _fallback_cluster_title(self, comments: list[ProcessedComment]) -> str:
        """Generate a fallback cluster title from salient keywords."""
        texts = [comment_text_for_output(comment) for comment in comments[:80]]
        keywords = extract_salient_comment_keywords(texts, max_keywords=4)
        if len(keywords) >= 3:
            candidate = f"{keywords[0]} {keywords[1]} {keywords[2]}"
        elif len(keywords) == 2:
            candidate = f"{keywords[0]} и {keywords[1]}"
        elif len(keywords) == 1:
            candidate = f"{keywords[0]} и позиция аудитории"
        else:
            candidate = "Позиция аудитории по теме"
        sanitized = sanitize_cluster_title(candidate, comments)
        if sanitized:
            return sanitized
        return "Позиция аудитории по теме"

    def _mark_llm_disabled(self, diagnostics: dict[str, Any], *, reason: str) -> None:
        """Record an LLM-disable event in the diagnostics dictionary."""
        normalized = str(reason or "").strip().lower()
        if not normalized:
            return
        events = diagnostics.setdefault("llm_disable_events", [])
        if isinstance(events, list):
            events.append(normalized)
        if not diagnostics.get("llm_disabled_reason"):
            diagnostics["llm_disabled_reason"] = normalized
