"""Quality metrics extraction and evaluation for the analysis pipeline.

Provides the :class:`QualityMetrics` helper that encapsulates position-level
quality scoring, watchdog evaluation for degraded runs, cluster diagnostics
payload construction, and coherence estimation.  All methods were originally
part of ``DailyRunService`` and have been extracted here to improve separation
of concerns.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.utils import utcnow
from app.db.models import Run
from app.schemas.domain import (
    ClusterDraft,
    ProcessedComment,
    TopicSummary,
    VideoMeta,
)


class QualityMetrics:
    """Compute and evaluate quality metrics for the analysis pipeline.

    Centralises calculations that assess the reliability of clustering and
    labeling results, including position-level metrics, coherence estimation,
    and diagnostic payload generation.  Intended to be instantiated once per
    pipeline run and reused across stages.

    Args:
        settings: Application configuration object.
        logger: Optional logger instance.  When *None*, a logger named after
            the class is created automatically.
    """

    def __init__(self, settings: Settings, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pct(numerator: int | float, denominator: int | float) -> float:
        """Return *numerator / denominator * 100* as a percentage.

        Args:
            numerator: The dividend value.
            denominator: The divisor value.

        Returns:
            The percentage as a float, or ``0.0`` when *denominator* is
            zero or negative.
        """
        denom = float(denominator)
        if denom <= 0:
            return 0.0
        return float(numerator) / denom * 100.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def position_quality_metrics(self, topics: list[TopicSummary]) -> dict[str, float]:
        """Aggregate position-level quality metrics across all topics.

        Iterates over every position in the supplied topics and computes:

        * **named_positions_count** -- number of non-undetermined positions.
        * **undetermined_comment_share** -- percentage of comments that belong
          to an *undetermined* position.
        * **single_claim_pass_rate** -- percentage of named positions that
          passed the single-claim quality check.

        Args:
            topics: List of topic summaries whose positions will be inspected.

        Returns:
            A dictionary with the three metric keys listed above, each mapped
            to a ``float`` value.
        """
        total_comments = 0
        undetermined_comments = 0
        named_positions_count = 0
        single_claim_passed_count = 0
        for topic in topics:
            for position in topic.positions:
                count = max(0, int(position.count))
                total_comments += count
                if position.is_undetermined:
                    undetermined_comments += count
                    continue
                named_positions_count += 1
                if position.single_claim_passed:
                    single_claim_passed_count += 1
        return {
            "named_positions_count": float(named_positions_count),
            "undetermined_comment_share": self._pct(undetermined_comments, total_comments),
            "single_claim_pass_rate": self._pct(single_claim_passed_count, named_positions_count),
        }

    def evaluate_quality_watchdog(
        self,
        *,
        undetermined_comment_share: float,
        fallback_title_rate: float,
    ) -> tuple[bool, list[str]]:
        """Decide whether a run should be flagged as degraded.

        Compares the supplied metric values against configurable thresholds
        stored in ``self.settings`` and collects human-readable reasons
        for each violation.

        Args:
            undetermined_comment_share: Percentage of comments classified as
                *undetermined* across all topics.
            fallback_title_rate: Percentage of topics/positions that received
                a fallback (non-LLM-generated) title.

        Returns:
            A two-element tuple *(is_degraded, reasons)* where
            *is_degraded* is ``True`` when at least one threshold was
            exceeded, and *reasons* lists the specific violations.
        """
        reasons: list[str] = []
        undetermined_threshold = float(self.settings.quality_watchdog_undetermined_share_pct)
        fallback_threshold = float(self.settings.quality_watchdog_fallback_title_rate_pct)
        if undetermined_comment_share > undetermined_threshold:
            reasons.append(
                f"undetermined_comment_share>{undetermined_threshold:.1f}% ({undetermined_comment_share:.2f}%)"
            )
        if fallback_title_rate > fallback_threshold:
            reasons.append(
                f"fallback_title_rate>{fallback_threshold:.1f}% ({fallback_title_rate:.2f}%)"
            )
        return bool(reasons), reasons

    def build_cluster_diagnostics_payload(
        self,
        *,
        run: Run,
        video: VideoMeta,
        topics: list[TopicSummary],
        clusters: list[ClusterDraft],
        labeling_diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble a comprehensive diagnostics payload for a pipeline run.

        The payload captures per-cluster quality indicators together with
        run-level metrics and labeling diagnostics so that downstream
        consumers (dashboards, alerting) can evaluate run health.

        Args:
            run: The ORM ``Run`` instance for the current pipeline execution.
            video: Metadata for the video being analysed.
            topics: Final list of topic summaries produced by the pipeline.
            clusters: Draft cluster objects before labeling enrichment.
            labeling_diagnostics: Dictionary of labeling-stage counters and
                events (fallback counts, LLM failure counts, etc.).

        Returns:
            A nested dictionary suitable for JSON serialisation that contains
            run identification, aggregate metrics, labeling diagnostics, and
            a ``clusters`` list with per-cluster detail rows.
        """
        run_meta: dict[str, Any] = run.meta_json if isinstance(run.meta_json, dict) else {}
        topic_by_key = {topic.cluster_key: topic for topic in topics}
        cluster_rows: list[dict[str, Any]] = []
        for cluster in sorted(clusters, key=lambda item: item.weighted_share, reverse=True):
            topic = topic_by_key.get(cluster.cluster_key)
            if topic is None:
                continue
            named_positions = [
                position for position in topic.positions if not position.is_undetermined
            ]
            undetermined_comments = sum(
                max(0, int(position.count))
                for position in topic.positions
                if position.is_undetermined
            )
            total_position_comments = sum(
                max(0, int(position.count)) for position in topic.positions
            )
            single_claim_passed = sum(
                1 for position in named_positions if position.single_claim_passed
            )
            cluster_rows.append(
                {
                    "cluster_key": cluster.cluster_key,
                    "label": topic.label,
                    "source": topic.source,
                    "is_emerging": bool(topic.is_emerging),
                    "size_count": int(topic.size_count),
                    "share_pct": float(topic.share_pct),
                    "weighted_share": float(topic.weighted_share),
                    "coherence_score": float(topic.coherence_score),
                    "assignment_confidence": float(topic.assignment_confidence),
                    "ambiguous_share_pct": float(topic.ambiguous_share_pct),
                    "positions_count": len(named_positions),
                    "undetermined_comments": undetermined_comments,
                    "undetermined_share": round(
                        self._pct(undetermined_comments, total_position_comments), 2
                    ),
                    "single_claim_pass_rate": round(
                        self._pct(single_claim_passed, len(named_positions)), 2
                    ),
                }
            )
        return {
            "run_id": run.id,
            "video_id": video.youtube_video_id,
            "mode": run.mode,
            "generated_at": utcnow().isoformat(),
            "context_reliability": run_meta.get("context_reliability"),
            "metrics": {
                "cluster_noise_ratio": float(run_meta.get("cluster_noise_ratio", 0.0) or 0.0),
                "emerging_cluster_count": int(run_meta.get("emerging_cluster_count", 0) or 0),
                "postprocess_merge_count": int(run_meta.get("postprocess_merge_count", 0) or 0),
                "postprocess_uncertain_collapsed": int(
                    run_meta.get("postprocess_uncertain_collapsed", 0) or 0
                ),
                "fallback_topic_title_count": int(
                    run_meta.get("fallback_topic_title_count", 0) or 0
                ),
                "fallback_position_title_count": int(
                    run_meta.get("fallback_position_title_count", 0) or 0
                ),
                "fallback_topic_title_rate": float(
                    run_meta.get("fallback_topic_title_rate", 0.0) or 0.0
                ),
                "fallback_position_title_rate": float(
                    run_meta.get("fallback_position_title_rate", 0.0) or 0.0
                ),
                "fallback_title_rate": float(run_meta.get("fallback_title_rate", 0.0) or 0.0),
                "llm_cluster_fail_count": int(run_meta.get("llm_cluster_fail_count", 0) or 0),
                "llm_title_fail_count": int(run_meta.get("llm_title_fail_count", 0) or 0),
                "llm_position_fail_count": int(run_meta.get("llm_position_fail_count", 0) or 0),
                "llm_disabled_reason": str(run_meta.get("llm_disabled_reason", "") or ""),
                "llm_disable_events": list(run_meta.get("llm_disable_events", []) or []),
                "undetermined_comment_share": float(
                    run_meta.get("undetermined_comment_share", 0.0) or 0.0
                ),
                "openai_base_url_host": str(run_meta.get("openai_base_url_host", "") or ""),
                "openai_endpoint_mode": str(run_meta.get("openai_endpoint_mode", "") or ""),
                "openai_calls_total": int(run_meta.get("openai_calls_total", 0) or 0),
                "openai_calls_moderation": int(run_meta.get("openai_calls_moderation", 0) or 0),
                "openai_calls_cluster_labeling": int(
                    run_meta.get("openai_calls_cluster_labeling", 0) or 0
                ),
                "openai_calls_cluster_title": int(
                    run_meta.get("openai_calls_cluster_title", 0) or 0
                ),
                "openai_calls_position_naming": int(
                    run_meta.get("openai_calls_position_naming", 0) or 0
                ),
                "openai_calls_blocked_reserved_for_labeling": int(
                    run_meta.get("openai_calls_blocked_reserved_for_labeling", 0) or 0
                ),
                "openai_calls_blocked_task_quota": int(
                    run_meta.get("openai_calls_blocked_task_quota", 0) or 0
                ),
                "embedding_provider": str(run_meta.get("embedding_provider", "") or ""),
                "embedding_model": str(run_meta.get("embedding_model", "") or ""),
                "embedding_task": str(run_meta.get("embedding_task", "") or ""),
                "embedding_vector_dim": int(run_meta.get("embedding_vector_dim", 0) or 0),
                "cluster_reduction_summary": dict(
                    run_meta.get("cluster_reduction_summary", {}) or {}
                ),
                "cluster_parameter_summary": dict(
                    run_meta.get("cluster_parameter_summary", {}) or {}
                ),
                "cluster_assignment_confidence_avg": float(
                    run_meta.get("cluster_assignment_confidence_avg", 0.0) or 0.0
                ),
                "cluster_ambiguous_comment_share": float(
                    run_meta.get("cluster_ambiguous_comment_share", 0.0) or 0.0
                ),
                "degraded": bool(run_meta.get("degraded", False)),
                "degraded_reasons": list(run_meta.get("degraded_reasons", []) or []),
            },
            "labeling_diagnostics": {
                "fallback_topic_title_count": int(
                    labeling_diagnostics.get("fallback_topic_title_count", 0) or 0
                ),
                "fallback_position_title_count": int(
                    labeling_diagnostics.get("fallback_position_title_count", 0) or 0
                ),
                "llm_cluster_fail_count": int(
                    labeling_diagnostics.get("llm_cluster_fail_count", 0) or 0
                ),
                "llm_title_fail_count": int(
                    labeling_diagnostics.get("llm_title_fail_count", 0) or 0
                ),
                "llm_position_fail_count": int(
                    labeling_diagnostics.get("llm_position_fail_count", 0) or 0
                ),
                "llm_disabled_reason": str(
                    labeling_diagnostics.get("llm_disabled_reason", "") or ""
                ),
                "llm_disable_events": list(
                    labeling_diagnostics.get("llm_disable_events", []) or []
                ),
            },
            "clusters": cluster_rows,
        }

    @staticmethod
    def estimate_cluster_coherence(
        cluster: ClusterDraft,
        comments: list[ProcessedComment],
        vectors: list[list[float]],
    ) -> float:
        """Estimate the internal coherence of a single cluster.

        Computes a weighted average of cosine similarities between each
        member vector and the cluster centroid.  Higher values indicate
        that the cluster members are tightly grouped in embedding space.

        Args:
            cluster: The draft cluster whose coherence is being measured.
            comments: Full list of processed comments (indexed by position).
            vectors: Embedding vectors aligned with *comments* by index.

        Returns:
            A coherence score clamped to the ``[0.0, 1.0]`` range.  Returns
            ``0.0`` when the cluster has no members or no centroid.
        """
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

    def skip_low_coherence_cluster(self, cluster: ClusterDraft, coherence: float) -> bool:
        """Decide whether a cluster should be dropped due to low coherence.

        A cluster is skipped only when its coherence falls below the
        configured minimum **and** it is not large or prominent enough to
        be kept regardless.  Specifically, a cluster is retained if its
        ``weighted_share`` meets the minimum threshold for low-coherence
        clusters, or if its ``size_count`` is at least
        ``max(20, cluster_min_size * 3)``.

        Args:
            cluster: The candidate cluster to evaluate.
            coherence: Pre-computed coherence score for the cluster (see
                :meth:`estimate_cluster_coherence`).

        Returns:
            ``True`` if the cluster should be excluded from results,
            ``False`` otherwise.
        """
        if cluster.weighted_share >= self.settings.topic_min_weighted_share_for_low_coherence:
            return False
        if cluster.size_count >= max(20, self.settings.cluster_min_size * 3):
            return False
        return coherence < self.settings.topic_coherence_min
