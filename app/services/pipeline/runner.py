"""Main pipeline orchestrator for the YouTubeAnalyzer analysis pipeline.

This module contains the :class:`DailyRunService` class -- the slim,
refactored version of the original monolithic ``pipeline.py``.  Business
logic that was previously embedded in private methods has been extracted
into focused helper modules:

* :mod:`.position_extractor` -- audience-position identification
* :mod:`.cluster_enricher` -- cluster labeling, merging, deduplication
* :mod:`.report_builder` -- author-stance analysis and briefing polish
* :mod:`.quality_metrics` -- quality scoring and diagnostics
* :mod:`.text_utils` -- standalone text normalisation helpers
* :mod:`.constants` -- compiled regex patterns and numeric thresholds

``DailyRunService`` retains the 11-stage orchestration flow
(:meth:`_run_for_video_meta`), all persistence methods, LLM provider
construction, moderation logic, and JSON extraction -- delegating the
rest to the helper instances created in :meth:`__init__`.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import BudgetExceededError, ExternalServiceError, InvalidConfigurationError
from app.core.utils import utcnow
from app.db.models import Cluster, ClusterItem, Comment, Report, Run, Video
from app.schemas.domain import (
    ClusterDraft,
    EpisodeContext,
    ProcessedComment,
    TopicPosition,
    TopicSummary,
    VideoMeta,
)
from app.services.briefing import BriefingService
from app.services.budget import BudgetGovernor
from app.services.clustering import ClusteringService
from app.services.embeddings import (
    EmbeddingCacheStore,
    EmbeddingService,
    LocalSentenceTransformerProvider,
    OpenAIEmbeddingProvider,
)
from app.services.exporter import ReportExporter
from app.services.labeling import (
    LLMProvider,
    OpenAIChatProvider,
)
from app.services.moderation_llm import decision_from_payload
from app.services.openai_endpoint import (
    ensure_openai_endpoint_allowed,
    openai_base_url_host,
    openai_endpoint_mode,
)
from app.services.preprocessing import CommentPreprocessor, PreprocessResult
from app.services.youtube_client import YouTubeClient

from .cluster_enricher import ClusterEnricher
from .position_extractor import PositionExtractor
from .quality_metrics import QualityMetrics
from .report_builder import ReportBuilder
from .text_utils import _UNCERTAIN_TOPIC_LABEL


class DailyRunService:
    """Main orchestrator for the daily YouTube comment analysis pipeline.

    Coordinates an 11-stage flow that fetches comments, preprocesses and
    moderates them, builds embeddings, clusters the comments, labels the
    clusters, identifies audience positions, constructs a briefing
    report, and persists all artefacts to the database.

    Heavy-lifting sub-tasks are delegated to focused helper classes
    instantiated in :meth:`__init__`:

    * :class:`PositionExtractor` -- audience-position extraction
    * :class:`ClusterEnricher` -- cluster labeling and postprocessing
    * :class:`ReportBuilder` -- author-stance analysis and briefing polish
    * :class:`QualityMetrics` -- quality diagnostics

    Parameters
    ----------
    settings:
        Application-wide configuration.
    db:
        SQLAlchemy database session.
    """

    def __init__(self, settings: Settings, db: Session) -> None:
        self.settings = settings
        self.db = db
        self.logger = logging.getLogger(self.__class__.__name__)
        self.preprocessor = CommentPreprocessor(settings)
        self.clustering = ClusteringService(settings)
        self.briefing = BriefingService(settings)
        self.exporter = ReportExporter(settings)

        # Extracted helper instances
        self.position_extractor = PositionExtractor(settings, self.logger)
        self.cluster_enricher = ClusterEnricher(settings, self.logger)
        self.report_builder = ReportBuilder(settings, self.logger)
        self.quality_metrics = QualityMetrics(settings, self.logger)

    # ================================================================
    # Run metadata helpers
    # ================================================================

    def _merge_run_meta(self, run: Run, updates: dict[str, Any]) -> None:
        meta = run.meta_json if isinstance(run.meta_json, dict) else {}
        merged = dict(meta)
        merged.update(updates)
        run.meta_json = merged

    def _set_run_stage(self, run: Run, *, current: int, total: int, key: str, label: str) -> None:
        progress_pct = int(round((current / max(1, total)) * 100))
        self._merge_run_meta(
            run,
            {
                "stage_current": current,
                "stage_total": total,
                "stage_key": key,
                "stage_label": label,
                "progress_pct": progress_pct,
                "updated_at": utcnow().isoformat(),
            },
        )
        self.db.add(run)
        self.db.commit()
        self.logger.info("[%02d/%02d] %s | %s", current, total, key, label)

    @staticmethod
    def _pct(numerator: int | float, denominator: int | float) -> float:
        denom = float(denominator)
        if denom <= 0:
            return 0.0
        return float(numerator) / denom * 100.0

    def _mark_llm_disabled(self, diagnostics: dict[str, Any], *, reason: str) -> None:
        normalized = str(reason or "").strip().lower()
        if not normalized:
            return
        events = diagnostics.setdefault("llm_disable_events", [])
        if isinstance(events, list):
            events.append(normalized)
        if not diagnostics.get("llm_disabled_reason"):
            diagnostics["llm_disabled_reason"] = normalized

    # ================================================================
    # Top-level entry points
    # ================================================================

    def run_latest(
        self,
        skip_filtering: bool | None = None,
    ) -> dict[str, Any]:
        youtube = YouTubeClient(self.settings)
        try:
            video_meta = youtube.get_latest_video_from_playlist()
            return self._run_for_video_meta(
                video_meta,
                skip_filtering=skip_filtering,
            )
        finally:
            youtube.close()

    def run_video(
        self,
        video_url: str,
        skip_filtering: bool | None = None,
    ) -> dict[str, Any]:
        youtube = YouTubeClient(self.settings)
        try:
            video_meta = youtube.get_video_meta_by_url(video_url)
            return self._run_for_video_meta(
                video_meta,
                skip_filtering=skip_filtering,
            )
        finally:
            youtube.close()

    # ================================================================
    # Core 11-stage pipeline orchestrator
    # ================================================================

    def _run_for_video_meta(
        self,
        video_meta: VideoMeta,
        *,
        skip_filtering: bool | None = None,
    ) -> dict[str, Any]:
        run_id: int | None = None
        youtube = YouTubeClient(self.settings)
        budget = BudgetGovernor(self.settings, self.db)

        # Apply skip_filtering override: when True, disable preprocessing filters for this run.
        if skip_filtering is True:
            self.settings = self.settings.model_copy(update={"preprocessing_filter_enabled": False})
            self.preprocessor = CommentPreprocessor(self.settings)
        stage_total = 11
        llm_provider: LLMProvider | None = None
        try:
            video = self._upsert_video(video_meta)
            run = Run(
                video_id=video.id,
                mode="openai",
                status="running",
                started_at=utcnow(),
                ended_at=None,
                total_comments=0,
                processed_comments=0,
                meta_json={
                    "stage_current": 0,
                    "stage_total": stage_total,
                    "stage_key": "init",
                    "stage_label": "Initializing run",
                    "progress_pct": 0,
                    "cluster_noise_ratio": 0.0,
                    "emerging_cluster_count": 0,
                    "postprocess_merge_count": 0,
                    "postprocess_uncertain_collapsed": 0,
                    "fallback_topic_title_count": 0,
                    "fallback_position_title_count": 0,
                    "fallback_topic_title_rate": 0.0,
                    "fallback_position_title_rate": 0.0,
                    "fallback_title_rate": 0.0,
                    "llm_cluster_fail_count": 0,
                    "llm_title_fail_count": 0,
                    "llm_position_fail_count": 0,
                    "undetermined_comment_share": 0.0,
                    "llm_disabled_reason": "",
                    "llm_disable_events": [],
                    "degraded": False,
                    "degraded_reasons": [],
                    "cluster_diagnostics_path": "",
                    "context_reliability": "low",
                    "openai_base_url_host": "",
                    "openai_endpoint_mode": "",
                    "openai_calls_total": 0,
                    "openai_calls_moderation": 0,
                    "openai_calls_cluster_labeling": 0,
                    "openai_calls_cluster_title": 0,
                    "openai_calls_position_naming": 0,
                    "openai_calls_blocked_reserved_for_labeling": 0,
                    "openai_calls_blocked_task_quota": 0,
                    "moderation_dropped_count": 0,
                    "moderation_flagged_count": 0,
                    "moderation_kept_count": 0,
                    "moderation_drop_rate": 0.0,
                    "moderation_dropped_by_reason": {},
                    "moderation_llm_reviewed_count": 0,
                    "moderation_llm_drop_count": 0,
                    "moderation_llm_flag_count": 0,
                    "moderation_llm_fail_count": 0,
                    "moderation_llm_disabled_reason": "",
                },
            )
            self.db.add(run)
            self.db.commit()
            self.db.refresh(run)
            run_id = run.id

            openai_host = openai_base_url_host(self.settings.openai_base_url)
            openai_mode = openai_endpoint_mode(self.settings.openai_base_url)
            self._merge_run_meta(
                run,
                {
                    "openai_base_url_host": openai_host,
                    "openai_endpoint_mode": openai_mode,
                },
            )
            self.db.commit()
            # Fail fast if a non-official endpoint is forbidden by configuration.
            ensure_openai_endpoint_allowed(self.settings)

            # ----------------------------------------------------------
            # Stage 1: Context loading
            # ----------------------------------------------------------
            self._set_run_stage(
                run,
                current=1,
                total=stage_total,
                key="context",
                label="Loading previous report context",
            )
            previous_topics = self._load_previous_topics(video.youtube_video_id)
            episode_context = EpisodeContext(
                source="comments_only",
                topics=[],
                transcript_text="",
                transcript_language=None,
            )
            self._merge_run_meta(
                run,
                {
                    "previous_topic_count": len(previous_topics),
                },
            )
            self.db.commit()

            # ----------------------------------------------------------
            # Stage 2: Fetch YouTube comments
            # ----------------------------------------------------------
            self._set_run_stage(
                run,
                current=2,
                total=stage_total,
                key="comments_fetch",
                label=f"Fetching YouTube comments for {video_meta.youtube_video_id}",
            )
            raw_comments = youtube.fetch_comments(video_meta.youtube_video_id)

            # ----------------------------------------------------------
            # Stage 3: Preprocess & moderate
            # ----------------------------------------------------------
            self._set_run_stage(
                run,
                current=3,
                total=stage_total,
                key="preprocess",
                label="Preprocessing comments",
            )
            preprocessed = self.preprocessor.preprocess(raw_comments, video_meta)
            preprocessed, llm_provider = self._apply_llm_borderline_moderation(
                preprocessed=preprocessed,
                video_meta=video_meta,
                budget=budget,
                llm_provider=llm_provider,
            )
            self._log_moderation_summary(preprocessed)

            # ----------------------------------------------------------
            # Stage 4: Persist comments
            # ----------------------------------------------------------
            self._set_run_stage(
                run,
                current=4,
                total=stage_total,
                key="comments_persist",
                label="Persisting processed comments",
            )
            all_comment_models = self._upsert_comments(video.id, preprocessed.all_comments)
            comment_by_id = {comment.youtube_comment_id: comment for comment in all_comment_models}
            processed_comment_models = [
                comment_by_id[comment.youtube_comment_id]
                for comment in preprocessed.processed
                if comment.youtube_comment_id in comment_by_id
            ]
            if len(processed_comment_models) != len(preprocessed.processed):
                missing = len(preprocessed.processed) - len(processed_comment_models)
                msg = f"Missing {missing} processed comments during persistence mapping."
                raise RuntimeError(msg)

            run.total_comments = preprocessed.total_count
            run.processed_comments = len(preprocessed.processed)
            openai_call_stats = self._collect_openai_call_stats(llm_provider)
            self._merge_run_meta(
                run,
                {
                    "filtered_count": preprocessed.filtered_count,
                    "total_comments_raw": preprocessed.total_count,
                    "processed_comments": len(preprocessed.processed),
                    "moderation_dropped_count": preprocessed.dropped_count,
                    "moderation_flagged_count": preprocessed.flagged_count,
                    "moderation_kept_count": preprocessed.kept_count,
                    "moderation_drop_rate": round(
                        self._pct(preprocessed.dropped_count, preprocessed.total_count),
                        2,
                    ),
                    "moderation_dropped_by_reason": preprocessed.dropped_by_reason,
                    "moderation_llm_reviewed_count": int(
                        preprocessed.llm_moderation_stats.get("reviewed_count", 0) or 0
                    ),
                    "moderation_llm_drop_count": int(
                        preprocessed.llm_moderation_stats.get("drop_count", 0) or 0
                    ),
                    "moderation_llm_flag_count": int(
                        preprocessed.llm_moderation_stats.get("flag_count", 0) or 0
                    ),
                    "moderation_llm_fail_count": int(
                        preprocessed.llm_moderation_stats.get("fail_count", 0) or 0
                    ),
                    "moderation_llm_disabled_reason": str(
                        preprocessed.llm_moderation_stats.get("disabled_reason", "") or ""
                    ),
                    **openai_call_stats,
                },
            )
            self.db.commit()

            topics: list[TopicSummary] = []
            merged_clusters: list[ClusterDraft] = []
            episode_topic_matches: dict[str, str] = {}
            labeling_diagnostics: dict[str, Any] = {
                "fallback_topic_title_count": 0,
                "fallback_position_title_count": 0,
                "llm_cluster_fail_count": 0,
                "llm_title_fail_count": 0,
                "llm_position_fail_count": 0,
                "llm_disabled_reason": "",
                "llm_disable_events": [],
            }
            if preprocessed.processed:
                if budget is None:
                    budget = BudgetGovernor(self.settings, self.db)

                # ----------------------------------------------------------
                # Stage 5: Build embeddings
                # ----------------------------------------------------------
                self._set_run_stage(
                    run,
                    current=5,
                    total=stage_total,
                    key="embeddings",
                    label="Building embeddings for processed comments",
                )
                embedding_service = self._build_embedding_service(budget)
                vectors = embedding_service.get_embeddings(
                    [comment.text_normalized for comment in preprocessed.processed],
                    [comment.text_hash for comment in preprocessed.processed],
                    task="topic",
                )
                self._merge_run_meta(
                    run,
                    {
                        "embedding_provider": getattr(
                            getattr(embedding_service, "provider", None),
                            "provider_name",
                            "",
                        ),
                        "embedding_model": getattr(
                            getattr(embedding_service, "provider", None),
                            "model_name",
                            "",
                        ),
                        "embedding_task": "topic",
                        "embedding_vector_dim": len(vectors[0]) if vectors else 0,
                    },
                )
                self.db.commit()

                # ----------------------------------------------------------
                # Stage 6: Clustering
                # ----------------------------------------------------------
                self._set_run_stage(
                    run,
                    current=6,
                    total=stage_total,
                    key="clustering",
                    label="Clustering comment embeddings",
                )
                clustering_result = self.clustering.cluster(preprocessed.processed, vectors)
                merged_clusters = self.cluster_enricher._merge_similar_clusters(
                    clustering_result.clusters,
                    preprocessed.processed,
                    vectors,
                )
                if len(merged_clusters) != len(clustering_result.clusters):
                    self.logger.info(
                        "Cluster merge step reduced groups: %s -> %s.",
                        len(clustering_result.clusters),
                        len(merged_clusters),
                    )
                self.logger.info(
                    "Clustering produced %s groups from %s processed comments.",
                    len(merged_clusters),
                    len(preprocessed.processed),
                )
                emerging_cluster_count = sum(
                    1 for cluster in merged_clusters if cluster.is_emerging
                )
                emerging_comment_count = sum(
                    len(cluster.member_indices)
                    for cluster in merged_clusters
                    if cluster.is_emerging
                )
                cluster_noise_ratio = float(emerging_comment_count) / max(
                    1.0, float(len(preprocessed.processed))
                )
                self._merge_run_meta(
                    run,
                    {
                        "cluster_noise_ratio": round(cluster_noise_ratio, 4),
                        "emerging_cluster_count": int(emerging_cluster_count),
                        "cluster_reduction_summary": clustering_result.reduction_summary,
                        "cluster_parameter_summary": clustering_result.parameter_summary,
                        "cluster_assignment_confidence_avg": round(
                            float(
                                sum(
                                    assignment.primary_confidence
                                    for assignment in clustering_result.assignment_by_index.values()
                                )
                                / max(1, len(clustering_result.assignment_by_index))
                            ),
                            4,
                        ),
                        "cluster_ambiguous_comment_share": round(
                            float(
                                sum(
                                    1
                                    for assignment in clustering_result.assignment_by_index.values()
                                    if assignment.is_ambiguous
                                )
                                / max(1, len(clustering_result.assignment_by_index))
                                * 100.0
                            ),
                            2,
                        ),
                    },
                )
                self.db.commit()

                # ----------------------------------------------------------
                # Stage 7: Episode-topic matching (skipped, transcription removed)
                # ----------------------------------------------------------
                self._set_run_stage(
                    run,
                    current=7,
                    total=stage_total,
                    key="episode_match",
                    label="Skipped: transcription removed",
                )

                # ----------------------------------------------------------
                # Stage 8: Cluster labeling & position extraction
                # ----------------------------------------------------------
                self._set_run_stage(
                    run,
                    current=8,
                    total=stage_total,
                    key="labeling",
                    label="Labeling clusters with LLM/fallback provider",
                )
                if llm_provider is None:
                    llm_provider = self._build_llm_provider(budget)
                self._merge_run_meta(run, {"llm_provider_used": llm_provider.provider_name})
                self.db.commit()
                topics, labeling_diagnostics = self.cluster_enricher.label_topics(
                    merged_clusters,
                    preprocessed.processed,
                    vectors,
                    embedding_service,
                    llm_provider,
                    previous_topics,
                    episode_context,
                    episode_topic_matches,
                    request_llm_json=self._request_llm_json,
                    build_positions_fn=self._build_positions_for_cluster,
                )
                postprocessed_clusters, postprocessed_topics = (
                    self.cluster_enricher._postprocess_labeled_topics(
                        clusters=merged_clusters,
                        topics=topics,
                        comments=preprocessed.processed,
                        vectors=vectors,
                        episode_context=episode_context,
                        allow_harmonization=False,
                    )
                )
                raw_uncertain_count = sum(
                    1 for topic in topics if topic.label == _UNCERTAIN_TOPIC_LABEL
                )
                post_uncertain_count = sum(
                    1 for topic in postprocessed_topics if topic.label == _UNCERTAIN_TOPIC_LABEL
                )
                postprocess_uncertain_collapsed = max(0, raw_uncertain_count - post_uncertain_count)
                positions_quality = self.quality_metrics.position_quality_metrics(
                    postprocessed_topics
                )
                topic_count = len(postprocessed_topics)
                fallback_topic_count = int(
                    labeling_diagnostics.get("fallback_topic_title_count", 0) or 0
                )
                fallback_position_count = int(
                    labeling_diagnostics.get("fallback_position_title_count", 0) or 0
                )
                named_positions_count = int(positions_quality.get("named_positions_count", 0) or 0)
                fallback_topic_title_rate = self._pct(fallback_topic_count, topic_count)
                fallback_position_title_rate = self._pct(
                    fallback_position_count, named_positions_count
                )
                fallback_title_rate = self._pct(
                    fallback_topic_count + fallback_position_count,
                    topic_count + named_positions_count,
                )
                undetermined_comment_share = float(
                    positions_quality.get("undetermined_comment_share", 0.0) or 0.0
                )
                degraded, degraded_reasons = self.quality_metrics.evaluate_quality_watchdog(
                    undetermined_comment_share=undetermined_comment_share,
                    fallback_title_rate=fallback_title_rate,
                )
                openai_call_stats = self._collect_openai_call_stats(llm_provider)
                min_expected_label_calls = max(3, int(math.ceil(float(topic_count) / 2.0)))
                if (
                    fallback_title_rate > 65.0
                    or int(openai_call_stats.get("openai_calls_cluster_labeling", 0))
                    < min_expected_label_calls
                ):
                    degraded = True
                    degraded_reasons = list(dict.fromkeys([*degraded_reasons, "llm_starved"]))
                self._merge_run_meta(
                    run,
                    {
                        "postprocess_merge_count": max(0, len(topics) - len(postprocessed_topics)),
                        "postprocess_uncertain_collapsed": postprocess_uncertain_collapsed,
                        "fallback_topic_title_count": fallback_topic_count,
                        "fallback_position_title_count": fallback_position_count,
                        "fallback_topic_title_rate": round(fallback_topic_title_rate, 2),
                        "fallback_position_title_rate": round(fallback_position_title_rate, 2),
                        "fallback_title_rate": round(fallback_title_rate, 2),
                        "llm_cluster_fail_count": int(
                            labeling_diagnostics.get("llm_cluster_fail_count", 0) or 0
                        ),
                        "llm_title_fail_count": int(
                            labeling_diagnostics.get("llm_title_fail_count", 0) or 0
                        ),
                        "llm_position_fail_count": int(
                            labeling_diagnostics.get("llm_position_fail_count", 0) or 0
                        ),
                        "undetermined_comment_share": round(undetermined_comment_share, 2),
                        "llm_disabled_reason": str(
                            labeling_diagnostics.get("llm_disabled_reason", "") or ""
                        ),
                        "llm_disable_events": list(
                            labeling_diagnostics.get("llm_disable_events", []) or []
                        ),
                        "single_claim_pass_rate": round(
                            float(positions_quality.get("single_claim_pass_rate", 100.0) or 100.0),
                            2,
                        ),
                        "degraded": degraded,
                        "degraded_reasons": degraded_reasons,
                        **openai_call_stats,
                    },
                )
                self.logger.info(
                    "Run quality counters: fallback_topic_title_rate=%.2f%% "
                    "fallback_position_title_rate=%.2f%% fallback_title_rate=%.2f%% "
                    "undetermined_comment_share=%.2f%% llm_disabled_reason=%s degraded=%s",
                    fallback_topic_title_rate,
                    fallback_position_title_rate,
                    fallback_title_rate,
                    undetermined_comment_share,
                    str(labeling_diagnostics.get("llm_disabled_reason", "") or "-"),
                    degraded,
                )
                self.db.commit()
                merged_clusters = postprocessed_clusters
                topics = postprocessed_topics

                # Generate cluster names from position titles via LLM
                llm_was_disabled = bool(labeling_diagnostics.get("llm_disabled_reason"))
                topics = self.cluster_enricher.generate_cluster_names_from_positions(
                    topics,
                    llm_provider=llm_provider,
                    request_llm_json=self._request_llm_json,
                    llm_disabled=llm_was_disabled,
                    clusters=merged_clusters,
                    comments=preprocessed.processed,
                    vectors=vectors,
                )

                topic_keys = {topic.cluster_key for topic in topics}
                episode_topic_matches = {
                    cluster_key: title
                    for cluster_key, title in episode_topic_matches.items()
                    if cluster_key in topic_keys
                }

                # ----------------------------------------------------------
                # Stage 9: Persist clusters
                # ----------------------------------------------------------
                self._set_run_stage(
                    run,
                    current=9,
                    total=stage_total,
                    key="clusters_persist",
                    label="Persisting clusters and cluster items",
                )
                self._persist_clusters(
                    run.id,
                    video.id,
                    merged_clusters,
                    topics,
                    processed_comment_models,
                    vectors,
                )
                self.db.commit()
            else:
                self._set_run_stage(
                    run,
                    current=5,
                    total=stage_total,
                    key="embeddings",
                    label="Skipped: no processed comments",
                )
                self._set_run_stage(
                    run,
                    current=6,
                    total=stage_total,
                    key="clustering",
                    label="Skipped: no processed comments",
                )
                self._set_run_stage(
                    run,
                    current=7,
                    total=stage_total,
                    key="episode_match",
                    label="Skipped: no clusters available",
                )
                self._set_run_stage(
                    run,
                    current=8,
                    total=stage_total,
                    key="labeling",
                    label="Skipped: no clusters available",
                )
                self._set_run_stage(
                    run,
                    current=9,
                    total=stage_total,
                    key="clusters_persist",
                    label="Skipped: no clusters available",
                )

            # ----------------------------------------------------------
            # Stage 10: Build briefing
            # ----------------------------------------------------------
            self._set_run_stage(
                run,
                current=10,
                total=stage_total,
                key="briefing",
                label="Building daily briefing and trend deltas",
            )
            disagreement_comments = self._collect_disagreement_from_positions(topics)
            if not disagreement_comments:
                disagreement_comments = self.report_builder.extract_author_disagreement_comments(
                    preprocessed.processed
                )
            briefing = self.briefing.build(
                video=video_meta,
                mode="openai",
                topics=topics,
                previous_topics=previous_topics,
                disagreement_comments=disagreement_comments,
            )
            briefing.metadata["episode_topic_matches"] = episode_topic_matches
            briefing.metadata["llm_provider_used"] = (
                run.meta_json.get("llm_provider_used") if isinstance(run.meta_json, dict) else None
            )
            briefing.metadata["new_comment_topics"] = [
                topic.label for topic in topics if topic.source == "comment_topic"
            ][:10]
            briefing.metadata["previous_report_categories"] = [
                " ".join((topic.label or "").split()).strip()
                for topic in previous_topics
                if " ".join((topic.label or "").split()).strip()
            ][:16]
            briefing.metadata["low_coherence_topics"] = [
                topic.label
                for topic in topics
                if topic.coherence_score < self.settings.topic_coherence_min
            ][:10]
            run_meta = run.meta_json if isinstance(run.meta_json, dict) else {}
            briefing.metadata["fallback_topic_title_rate"] = float(
                run_meta.get("fallback_topic_title_rate", 0.0) or 0.0
            )
            briefing.metadata["fallback_position_title_rate"] = float(
                run_meta.get("fallback_position_title_rate", 0.0) or 0.0
            )
            briefing.metadata["fallback_title_rate"] = float(
                run_meta.get("fallback_title_rate", 0.0) or 0.0
            )
            briefing.metadata["undetermined_comment_share"] = float(
                run_meta.get("undetermined_comment_share", 0.0) or 0.0
            )
            briefing.metadata["llm_disabled_reason"] = str(
                run_meta.get("llm_disabled_reason", "") or ""
            )
            briefing.metadata["llm_disable_events"] = list(
                run_meta.get("llm_disable_events", []) or []
            )
            briefing.metadata["openai_base_url_host"] = str(
                run_meta.get("openai_base_url_host", "") or ""
            )
            briefing.metadata["openai_endpoint_mode"] = str(
                run_meta.get("openai_endpoint_mode", "") or ""
            )
            briefing.metadata["openai_calls_total"] = int(
                run_meta.get("openai_calls_total", 0) or 0
            )
            briefing.metadata["openai_calls_moderation"] = int(
                run_meta.get("openai_calls_moderation", 0) or 0
            )
            briefing.metadata["openai_calls_cluster_labeling"] = int(
                run_meta.get("openai_calls_cluster_labeling", 0) or 0
            )
            briefing.metadata["openai_calls_cluster_title"] = int(
                run_meta.get("openai_calls_cluster_title", 0) or 0
            )
            briefing.metadata["openai_calls_position_naming"] = int(
                run_meta.get("openai_calls_position_naming", 0) or 0
            )
            briefing.metadata["embedding_provider"] = str(
                run_meta.get("embedding_provider", "") or ""
            )
            briefing.metadata["embedding_model"] = str(
                run_meta.get("embedding_model", "") or ""
            )
            briefing.metadata["embedding_task"] = str(
                run_meta.get("embedding_task", "") or ""
            )
            briefing.metadata["embedding_vector_dim"] = int(
                run_meta.get("embedding_vector_dim", 0) or 0
            )
            briefing.metadata["cluster_reduction_summary"] = dict(
                run_meta.get("cluster_reduction_summary", {}) or {}
            )
            briefing.metadata["cluster_parameter_summary"] = dict(
                run_meta.get("cluster_parameter_summary", {}) or {}
            )
            briefing.metadata["cluster_assignment_confidence_avg"] = float(
                run_meta.get("cluster_assignment_confidence_avg", 0.0) or 0.0
            )
            briefing.metadata["cluster_ambiguous_comment_share"] = float(
                run_meta.get("cluster_ambiguous_comment_share", 0.0) or 0.0
            )
            briefing.metadata["degraded"] = bool(run_meta.get("degraded", False))
            briefing.metadata["degraded_reasons"] = list(run_meta.get("degraded_reasons", []) or [])
            if self.settings.openai_enable_polish_call:
                openai_provider = (
                    llm_provider if isinstance(llm_provider, OpenAIChatProvider) else None
                )
                briefing = self.report_builder.try_polish_briefing(
                    briefing,
                    budget,
                    openai_provider=openai_provider,
                )
            self._merge_run_meta(run, self._collect_openai_call_stats(llm_provider))
            self.db.commit()

            # ----------------------------------------------------------
            # Stage 11: Export report
            # ----------------------------------------------------------
            self._set_run_stage(
                run,
                current=11,
                total=stage_total,
                key="report_export",
                label="Rendering and saving report artifacts",
            )
            markdown_content = self.exporter.to_markdown(briefing)
            html_content = self.exporter.to_html(markdown_content)
            self.exporter.persist(video.youtube_video_id, markdown_content, html_content)
            diagnostics_path = ""
            if self.settings.cluster_diagnostics_enabled:
                try:
                    diagnostics_payload = self.quality_metrics.build_cluster_diagnostics_payload(
                        run=run,
                        video=video_meta,
                        topics=topics,
                        clusters=merged_clusters,
                        labeling_diagnostics=labeling_diagnostics,
                    )
                    diagnostics_artifact = self.exporter.persist_cluster_diagnostics(
                        video.youtube_video_id,
                        diagnostics_payload,
                    )
                    diagnostics_path = str(diagnostics_artifact)
                    briefing.metadata["cluster_diagnostics_path"] = diagnostics_path
                    self._merge_run_meta(run, {"cluster_diagnostics_path": diagnostics_path})
                except Exception as diagnostics_exc:
                    self.logger.warning(
                        "Failed to persist cluster diagnostics for run=%s video=%s: %s",
                        run.id,
                        video.youtube_video_id,
                        diagnostics_exc,
                    )

            report = Report(
                run_id=run.id,
                video_id=video.id,
                content_markdown=markdown_content,
                content_html=html_content,
                structured_json=briefing.model_dump(mode="json"),
            )
            self.db.add(report)
            run.status = "completed"
            run.ended_at = utcnow()
            self._merge_run_meta(
                run,
                {
                    "stage_current": stage_total,
                    "stage_total": stage_total,
                    "stage_key": "completed",
                    "stage_label": "Pipeline completed",
                    "progress_pct": 100,
                    "completed_at": utcnow().isoformat(),
                },
            )
            self.db.commit()
            return {
                "run_id": run.id,
                "video_id": video.youtube_video_id,
                "mode": "openai",
                "topics_count": len(topics),
                "report_created": True,
            }
        except Exception as exc:
            self.db.rollback()
            self.logger.exception("Pipeline failed: %s", exc)
            if run_id is not None:
                failed_run = self.db.get(Run, run_id)
                if failed_run:
                    failed_run.status = "failed"
                    failed_run.error = str(exc)
                    failed_run.ended_at = utcnow()
                    meta = failed_run.meta_json if isinstance(failed_run.meta_json, dict) else {}
                    meta.update(
                        {
                            "stage_key": "failed",
                            "stage_label": str(meta.get("stage_label") or "Pipeline failed"),
                            "failed_at": utcnow().isoformat(),
                            "failed_error": str(exc),
                        }
                    )
                    failed_run.meta_json = meta
                    self.db.commit()
            raise
        finally:
            youtube.close()

    # ================================================================
    # Database persistence
    # ================================================================

    def _upsert_video(self, video_meta: VideoMeta) -> Video:
        stmt = select(Video).where(Video.youtube_video_id == video_meta.youtube_video_id)
        existing = self.db.scalar(stmt)
        if existing:
            existing.title = video_meta.title
            existing.description = video_meta.description
            existing.published_at = video_meta.published_at
            existing.duration_seconds = video_meta.duration_seconds
            existing.url = video_meta.url
            existing.playlist_id = video_meta.playlist_id
            self.db.add(existing)
            self.db.commit()
            self.db.refresh(existing)
            return existing
        created = Video(
            youtube_video_id=video_meta.youtube_video_id,
            playlist_id=video_meta.playlist_id,
            title=video_meta.title,
            description=video_meta.description,
            published_at=video_meta.published_at,
            duration_seconds=video_meta.duration_seconds,
            url=video_meta.url,
        )
        self.db.add(created)
        self.db.commit()
        self.db.refresh(created)
        return created

    def _upsert_comments(self, video_id: int, comments: list[ProcessedComment]) -> list[Comment]:
        if not comments:
            return []
        ids = [comment.youtube_comment_id for comment in comments]
        unique_ids = list(dict.fromkeys(ids))
        duplicate_count = len(ids) - len(unique_ids)
        if duplicate_count > 0:
            self.logger.warning(
                "Detected %s duplicate youtube_comment_id entries in one batch; "
                "deduplicating inserts via in-memory upsert map.",
                duplicate_count,
            )

        stmt = select(Comment).where(Comment.youtube_comment_id.in_(unique_ids))
        existing_rows = list(self.db.scalars(stmt))
        existing = {row.youtube_comment_id: row for row in existing_rows}
        models: list[Comment] = []

        for comment in comments:
            row = existing.get(comment.youtube_comment_id)
            if row is None:
                row = Comment(
                    video_id=video_id,
                    youtube_comment_id=comment.youtube_comment_id,
                    parent_comment_id=comment.parent_comment_id,
                    author_name=comment.author_name,
                    author_channel_id=comment.author_channel_id,
                    text_raw=comment.text_raw,
                    text_normalized=comment.text_normalized,
                    text_hash=comment.text_hash,
                    language=comment.language,
                    like_count=comment.like_count,
                    reply_count=comment.reply_count,
                    published_at=comment.published_at,
                    weight=comment.weight,
                    is_top_level=comment.is_top_level,
                    is_filtered=comment.is_filtered,
                    filter_reason=comment.filter_reason,
                    moderation_action=comment.moderation_action,
                    moderation_reason=comment.moderation_reason,
                    moderation_source=comment.moderation_source,
                    moderation_score=comment.moderation_score,
                )
                self.db.add(row)
                # Prevent duplicate INSERT in same transaction when input list
                # contains repeated youtube_comment_id.
                existing[comment.youtube_comment_id] = row
            else:
                row.video_id = video_id
                row.author_name = comment.author_name
                row.author_channel_id = comment.author_channel_id
                row.text_raw = comment.text_raw
                row.text_normalized = comment.text_normalized
                row.text_hash = comment.text_hash
                row.language = comment.language
                row.like_count = comment.like_count
                row.reply_count = comment.reply_count
                row.published_at = comment.published_at
                row.weight = comment.weight
                row.is_top_level = comment.is_top_level
                row.is_filtered = comment.is_filtered
                row.filter_reason = comment.filter_reason
                row.moderation_action = comment.moderation_action
                row.moderation_reason = comment.moderation_reason
                row.moderation_source = comment.moderation_source
                row.moderation_score = comment.moderation_score
            models.append(row)
        self.db.commit()
        for model in models:
            self.db.refresh(model)
        return models

    def _persist_clusters(
        self,
        run_id: int,
        video_id: int,
        clusters: list[ClusterDraft],
        topics: list[TopicSummary],
        comments: list[Comment],
        vectors: list[list[float]],
    ) -> None:
        topic_by_key = {topic.cluster_key: topic for topic in topics}
        order = 0
        total_clusters = max(1, len(clusters))
        self.logger.info("Persisting clusters: %s candidates.", len(clusters))
        for idx, cluster in enumerate(clusters, start=1):
            topic = topic_by_key.get(cluster.cluster_key)
            if topic is None:
                self.logger.info(
                    "[persist %s/%s] skip cluster=%s (no topic)",
                    idx,
                    total_clusters,
                    cluster.cluster_key,
                )
                continue
            row = Cluster(
                run_id=run_id,
                video_id=video_id,
                cluster_key=cluster.cluster_key,
                sort_order=order,
                label=topic.label,
                description=topic.description,
                author_actions=topic.author_actions,
                sentiment=topic.sentiment,
                emotion_tags=topic.emotion_tags,
                intent_distribution=topic.intent_distribution,
                representative_quotes=topic.representative_quotes,
                size_count=topic.size_count,
                share_pct=topic.share_pct,
                weighted_share=topic.weighted_share,
                is_emerging=topic.is_emerging,
                centroid=topic.centroid,
            )
            self.db.add(row)
            self.db.flush()
            rep_set = set(cluster.representative_indices)
            for member_idx in cluster.member_indices:
                if member_idx < 0 or member_idx >= len(comments):
                    continue
                comment = comments[member_idx]
                vector = vectors[member_idx] if member_idx < len(vectors) else None
                relevance_score = self.cluster_enricher._cluster_member_relevance_score(
                    centroid=cluster.centroid,
                    vector=vector,
                    weight=float(comment.weight),
                    like_count=int(comment.like_count),
                    reply_count=int(comment.reply_count),
                )
                self.db.add(
                    ClusterItem(
                        cluster_id=row.id,
                        comment_id=comment.id,
                        score=relevance_score,
                        is_representative=member_idx in rep_set,
                    )
                )
            order += 1
            self.logger.info(
                "[persist %s/%s] saved cluster=%s members=%s",
                idx,
                total_clusters,
                cluster.cluster_key,
                len(cluster.member_indices),
            )

    def _load_previous_topics(self, current_video_id: str) -> list[TopicSummary]:
        stmt = (
            select(Report)
            .join(Video, Report.video_id == Video.id)
            .where(Video.youtube_video_id != current_video_id)
            .order_by(Report.created_at.desc())
            .limit(1)
        )
        prev = self.db.scalar(stmt)
        if not prev:
            return []
        raw_topics = prev.structured_json.get("top_topics", [])
        topics: list[TopicSummary] = []
        for item in raw_topics:
            try:
                topics.append(TopicSummary.model_validate(item))
            except Exception:
                continue
        return topics

    # ================================================================
    # Embedding & LLM provider factories
    # ================================================================

    def _build_embedding_service(self, budget: BudgetGovernor | None) -> EmbeddingService:
        if self.settings.resolved_embedding_mode == "openai":
            if budget is None:
                msg = "Budget governor is required when EMBEDDING_MODE=openai."
                raise InvalidConfigurationError(msg)
            provider = OpenAIEmbeddingProvider(self.settings, budget)
        else:
            provider = LocalSentenceTransformerProvider(
                self.settings.local_embedding_model,
                self.settings,
            )
        cache = EmbeddingCacheStore(
            self.settings, self.db, provider.provider_name, provider.model_name
        )
        return EmbeddingService(provider, cache)

    def _build_llm_provider(self, budget: BudgetGovernor) -> LLMProvider:
        if not self.settings.openai_api_key:
            raise ExternalServiceError("OpenAI chat provider is unavailable. Missing API key.")
        provider = OpenAIChatProvider(self.settings, budget)
        if provider.is_available():
            self.logger.info(
                "Using LLM provider=openai model=%s base_url_host=%s endpoint_mode=%s",
                self.settings.openai_chat_model,
                provider.base_url_host,
                provider.endpoint_mode,
            )
            return provider
        raise ExternalServiceError(
            "OpenAI chat provider is unavailable. Cannot proceed without LLM."
        )

    # ================================================================
    # LLM request & JSON extraction
    # ================================================================

    def _request_llm_json(
        self,
        llm_provider: LLMProvider,
        prompt: str,
        *,
        task: str = "json_generation",
        estimated_out_tokens: int = 900,
        max_output_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        try:
            return llm_provider.request_json(
                prompt=prompt,
                system_prompt=system_prompt or "Return strict JSON only.",
                task=task,
                temperature=0.1,
                estimated_out_tokens=estimated_out_tokens,
                max_output_tokens=min(
                    self.settings.openai_max_output_tokens,
                    max_output_tokens or 4096,
                ),
            )
        except (BudgetExceededError, ExternalServiceError):
            raise
        except Exception as exc:
            self.logger.warning(
                "JSON generation failed via %s: %s", llm_provider.provider_name, exc
            )
        return {}

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        raw_text = (text or "").strip()
        if not raw_text:
            return {}
        try:
            payload = json.loads(raw_text)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            pass
        if raw_text.startswith("```"):
            fenced = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE | re.DOTALL
            ).strip()
            if fenced:
                try:
                    payload = json.loads(fenced)
                    return payload if isinstance(payload, dict) else {}
                except Exception:
                    raw_text = fenced
        start = -1
        depth = 0
        in_string = False
        escape = False
        for idx, char in enumerate(raw_text):
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                if depth == 0:
                    start = idx
                depth += 1
                continue
            if char == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = raw_text[start : idx + 1]
                    try:
                        payload = json.loads(candidate)
                        return payload if isinstance(payload, dict) else {}
                    except Exception:
                        continue
        return {}

    # ================================================================
    # Moderation
    # ================================================================

    def _apply_llm_borderline_moderation(
        self,
        *,
        preprocessed: PreprocessResult,
        video_meta: VideoMeta,
        budget: BudgetGovernor,
        llm_provider: LLMProvider | None,
    ) -> tuple[PreprocessResult, LLMProvider | None]:
        stats: dict[str, int | bool | str] = {
            "reviewed_count": 0,
            "keep_count": 0,
            "flag_count": 0,
            "drop_count": 0,
            "fail_count": 0,
            "disabled": False,
            "disabled_reason": "",
        }
        preprocessed.llm_moderation_stats = stats

        if not self.settings.moderation_enabled:
            stats["disabled"] = True
            stats["disabled_reason"] = "moderation_disabled"
            return preprocessed, llm_provider

        candidate_ids = list(dict.fromkeys(preprocessed.borderline_comment_ids))
        if not candidate_ids:
            stats["disabled"] = True
            stats["disabled_reason"] = "no_borderline_candidates"
            return preprocessed, llm_provider

        processed_by_id = {
            comment.youtube_comment_id: comment for comment in preprocessed.processed
        }
        candidates = [
            processed_by_id[item_id] for item_id in candidate_ids if item_id in processed_by_id
        ]
        if not candidates:
            stats["disabled"] = True
            stats["disabled_reason"] = "no_borderline_candidates"
            return preprocessed, llm_provider

        if (
            not self.settings.moderation_enable_llm_borderline
            or self.settings.moderation_llm_scope == "disabled"
            or self.settings.moderation_llm_max_reviews_per_run <= 0
        ):
            for comment in candidates:
                self._set_moderation_flag(
                    comment,
                    reason="llm_disabled_fallback",
                    source="fallback",
                    score=comment.moderation_score,
                )
            stats["disabled"] = True
            stats["disabled_reason"] = "llm_disabled"
            stats["flag_count"] = len(candidates)
            self._recompute_preprocess_metrics(preprocessed)
            return preprocessed, llm_provider

        if llm_provider is None:
            try:
                llm_provider = self._build_llm_provider(budget)
            except ExternalServiceError:
                for comment in candidates:
                    self._set_moderation_flag(
                        comment,
                        reason="llm_unavailable_fallback",
                        source="fallback",
                        score=comment.moderation_score,
                    )
                stats["disabled"] = True
                stats["disabled_reason"] = "provider_unavailable"
                stats["flag_count"] = len(candidates)
                self._recompute_preprocess_metrics(preprocessed)
                return preprocessed, llm_provider
        max_reviews = max(0, int(self.settings.moderation_llm_max_reviews_per_run))
        review_candidates = candidates[:max_reviews]
        overflow_candidates = candidates[max_reviews:]
        for comment in overflow_candidates:
            self._set_moderation_flag(
                comment,
                reason="llm_review_cap_fallback",
                source="fallback",
                score=comment.moderation_score,
            )
            stats["flag_count"] = int(stats["flag_count"]) + 1

        for idx, comment in enumerate(review_candidates):
            prompt = self._build_moderation_prompt(comment=comment, video_meta=video_meta)
            try:
                payload = self._request_llm_json(
                    llm_provider,
                    prompt,
                    task="moderation_borderline",
                    estimated_out_tokens=140,
                    max_output_tokens=min(self.settings.openai_max_output_tokens, 220),
                )
            except BudgetExceededError as exc:
                stats["fail_count"] = int(stats["fail_count"]) + 1
                stats["disabled"] = True
                message = str(exc).lower()
                disable_reason = "budget_exceeded"
                fallback_reason = "llm_budget_fallback"
                if "reserved for topic labeling" in message:
                    disable_reason = "reserved_for_labeling"
                    fallback_reason = "llm_reserved_for_labeling"
                elif "task quota reached" in message:
                    disable_reason = "task_quota_reached"
                    fallback_reason = "llm_task_quota_fallback"
                elif "per-run call limit reached" in message:
                    disable_reason = "max_calls_reached"
                    fallback_reason = "llm_max_calls_fallback"
                stats["disabled_reason"] = disable_reason
                remaining = review_candidates[idx:]
                for fallback_comment in remaining:
                    self._set_moderation_flag(
                        fallback_comment,
                        reason=fallback_reason,
                        source="fallback",
                        score=fallback_comment.moderation_score,
                    )
                    stats["flag_count"] = int(stats["flag_count"]) + 1
                break
            except ExternalServiceError:
                stats["fail_count"] = int(stats["fail_count"]) + 1
                self._set_moderation_flag(
                    comment,
                    reason="llm_error_fallback",
                    source="fallback",
                    score=comment.moderation_score,
                )
                stats["flag_count"] = int(stats["flag_count"]) + 1
                continue

            decision = decision_from_payload(
                payload,
                fallback_action="flag",
                fallback_reason="llm_invalid_fallback",
            )
            stats["reviewed_count"] = int(stats["reviewed_count"]) + 1
            if decision.action == "drop":
                self._set_moderation_drop(
                    comment,
                    reason=decision.reason_code,
                    source="llm",
                    score=decision.score,
                )
                stats["drop_count"] = int(stats["drop_count"]) + 1
            elif decision.action == "flag":
                self._set_moderation_flag(
                    comment,
                    reason=decision.reason_code,
                    source="llm",
                    score=decision.score,
                )
                stats["flag_count"] = int(stats["flag_count"]) + 1
            else:
                self._set_moderation_keep(
                    comment,
                    reason=decision.reason_code,
                    source="llm",
                    score=decision.score,
                )
                stats["keep_count"] = int(stats["keep_count"]) + 1

        self._recompute_preprocess_metrics(preprocessed)
        return preprocessed, llm_provider

    def _build_moderation_prompt(self, *, comment: ProcessedComment, video_meta: VideoMeta) -> str:
        payload = {
            "video_title": video_meta.title,
            "video_description": (video_meta.description or "")[:1200],
            "comment_text": comment.text_raw,
            "comment_normalized": comment.text_normalized,
            "rule_reason": comment.moderation_reason or "",
            "rule_score": comment.moderation_score,
        }
        return (
            "Classify one borderline YouTube comment for moderation.\n"
            "Return strict JSON only.\n"
            "JSON schema:\n"
            "{\n"
            '  "action": "keep|flag|drop",\n'
            '  "reason_code": "short_snake_case",\n'
            '  "score": 0.0\n'
            "}\n"
            "Rules:\n"
            "- keep: meaningful position/commentary, acceptable for clustering;\n"
            "- flag: contains toxicity/low quality but still may contain useful stance;\n"
            "- drop: spam/meaningless/offtopic/profanity-only.\n"
            "Use conservative behavior: if uncertain choose flag.\n"
            f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    def _set_moderation_keep(
        self, comment: ProcessedComment, *, reason: str, source: str, score: float | None
    ) -> None:
        comment.is_filtered = False
        comment.filter_reason = None
        comment.moderation_action = "keep"
        comment.moderation_reason = reason or "llm_keep"
        comment.moderation_source = "llm" if source == "llm" else "fallback"
        comment.moderation_score = score

    def _set_moderation_flag(
        self, comment: ProcessedComment, *, reason: str, source: str, score: float | None
    ) -> None:
        if comment.moderation_action != "flag":
            comment.weight = min(
                self.settings.comment_weight_max,
                float(comment.weight) * float(self.settings.moderation_flagged_weight_multiplier),
            )
        comment.is_filtered = False
        comment.filter_reason = None
        comment.moderation_action = "flag"
        comment.moderation_reason = reason or "flagged"
        comment.moderation_source = "llm" if source == "llm" else "fallback"
        comment.moderation_score = score

    def _set_moderation_drop(
        self, comment: ProcessedComment, *, reason: str, source: str, score: float | None
    ) -> None:
        comment.is_filtered = True
        comment.filter_reason = reason or "moderation_drop"
        comment.moderation_action = "drop"
        comment.moderation_reason = reason or "moderation_drop"
        comment.moderation_source = "llm" if source == "llm" else "fallback"
        comment.moderation_score = score

    def _recompute_preprocess_metrics(self, preprocessed: PreprocessResult) -> None:
        preprocessed.processed = [
            comment for comment in preprocessed.processed if not comment.is_filtered
        ]
        dropped_counter: Counter[str] = Counter()
        flagged_counter: Counter[str] = Counter()
        for comment in preprocessed.all_comments:
            if comment.is_filtered:
                dropped_counter[str(comment.filter_reason or "unspecified")] += 1
            elif comment.moderation_action == "flag":
                flagged_counter[str(comment.moderation_reason or "unspecified")] += 1

        preprocessed.dropped_by_reason = dict(
            sorted(dropped_counter.items(), key=lambda item: (-item[1], item[0]))
        )
        preprocessed.flagged_by_reason = dict(
            sorted(flagged_counter.items(), key=lambda item: (-item[1], item[0]))
        )
        preprocessed.dropped_count = int(sum(preprocessed.dropped_by_reason.values()))
        preprocessed.flagged_count = int(sum(preprocessed.flagged_by_reason.values()))
        preprocessed.kept_count = max(0, len(preprocessed.processed) - preprocessed.flagged_count)
        preprocessed.filtered_count = preprocessed.dropped_count

    def _log_moderation_summary(self, preprocessed: PreprocessResult) -> None:
        drop_pct = self._pct(preprocessed.dropped_count, preprocessed.total_count)
        self.logger.info(
            "[moderation] raw=%s kept=%s flagged=%s dropped=%s drop_pct=%.2f%%",
            preprocessed.total_count,
            preprocessed.kept_count,
            preprocessed.flagged_count,
            preprocessed.dropped_count,
            drop_pct,
        )
        if self.settings.moderation_log_include_reason_breakdown:
            reason_text = (
                ", ".join(
                    f"{reason}={count}" for reason, count in preprocessed.dropped_by_reason.items()
                )
                or "-"
            )
            self.logger.info("[moderation] dropped_by_reason: %s", reason_text)
        stats = preprocessed.llm_moderation_stats
        self.logger.info(
            "[moderation] llm_borderline: reviewed=%s kept=%s flagged=%s dropped=%s failed=%s disabled_reason=%s",
            int(stats.get("reviewed_count", 0) or 0),
            int(stats.get("keep_count", 0) or 0),
            int(stats.get("flag_count", 0) or 0),
            int(stats.get("drop_count", 0) or 0),
            int(stats.get("fail_count", 0) or 0),
            str(stats.get("disabled_reason", "") or "-"),
        )

    def _collect_openai_call_stats(self, llm_provider: LLMProvider | None) -> dict[str, int]:
        if isinstance(llm_provider, OpenAIChatProvider):
            return llm_provider.get_call_stats()
        return {
            "openai_calls_total": 0,
            "openai_calls_moderation": 0,
            "openai_calls_cluster_labeling": 0,
            "openai_calls_cluster_title": 0,
            "openai_calls_position_naming": 0,
            "openai_calls_blocked_reserved_for_labeling": 0,
            "openai_calls_blocked_task_quota": 0,
        }

    # ================================================================
    # Disagreement collection from positions
    # ================================================================

    def _collect_disagreement_from_positions(self, topics: list[TopicSummary]) -> list[str]:
        """Collect comments from positions flagged as author disagreement by the LLM.

        Iterates over all topics and their positions, collecting comments
        from positions where ``is_author_disagreement`` is ``True``.
        Filters out offensive language and respects the disagreement limit.

        Args:
            topics: List of labeled topic summaries with position data.

        Returns:
            List of non-offensive disagreement comment texts.
        """
        items: list[str] = []
        seen: set[str] = set()
        for topic in topics:
            for position in topic.positions:
                if not position.is_author_disagreement:
                    continue
                for comment in position.comments:
                    text = " ".join(comment.split()).strip()
                    if not text:
                        continue
                    if not self.report_builder.has_author_reference(text):
                        continue
                    lowered = text.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    if self.report_builder.is_offensive_disagreement_text(text):
                        continue
                    items.append(text)
        return items

    # ================================================================
    # Bridge to extracted helpers
    # ================================================================

    def _build_positions_for_cluster(
        self,
        *,
        llm_provider: LLMProvider,
        cluster: ClusterDraft,
        comments: list[ProcessedComment],
        vectors: list[list[float]],
        llm_disabled: bool,
        cluster_title: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> tuple[list[TopicPosition], bool]:
        """Bridge method that delegates position extraction to :class:`PositionExtractor`.

        Passes ``self._request_llm_json`` and ``self.cluster_enricher._rank_cluster_member_indices``
        as callbacks so the position extractor can request LLM completions and rank cluster members
        without depending on the full pipeline service.
        """
        return self.position_extractor.build_positions_for_cluster(
            llm_provider=llm_provider,
            cluster=cluster,
            comments=comments,
            vectors=vectors,
            llm_disabled=llm_disabled,
            cluster_title=cluster_title,
            request_llm_json=self._request_llm_json,
            rank_cluster_member_indices=self.cluster_enricher._rank_cluster_member_indices,
            diagnostics=diagnostics,
        )
