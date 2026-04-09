from __future__ import annotations

from datetime import UTC, datetime

from app.db.models import Run
from app.schemas.domain import ClusterDraft, TopicPosition, TopicSummary, VideoMeta
from app.services.pipeline import DailyRunService


def _topic_with_positions(cluster_key: str) -> TopicSummary:
    return TopicSummary(
        cluster_key=cluster_key,
        label="Main topic label",
        description="Topic description",
        author_actions=["Action item"],
        sentiment="neutral",
        emotion_tags=[],
        intent_distribution={"other": 1},
        representative_quotes=["quote one"],
        question_comments=[],
        positions=[
            TopicPosition(
                key="pos_1",
                title="Single claim title",
                summary="Summary",
                prototype="Prototype",
                count=7,
                pct=70.0,
                weighted_count=7.0,
                weighted_pct=70.0,
                comments=["a"],
                single_claim_passed=True,
            ),
            TopicPosition(
                key="undetermined",
                title="Undetermined",
                summary="Undetermined comments",
                prototype="Prototype",
                count=3,
                pct=30.0,
                weighted_count=3.0,
                weighted_pct=30.0,
                comments=["b"],
                is_undetermined=True,
            ),
        ],
        size_count=10,
        share_pct=100.0,
        weighted_share=100.0,
        is_emerging=False,
        source="comment_topic",
        coherence_score=0.65,
        centroid=[1.0, 0.0],
    )


def test_quality_watchdog_marks_degraded_by_thresholds(db_session, test_settings) -> None:
    service = DailyRunService(test_settings, db_session)

    degraded, reasons = service.quality_metrics.evaluate_quality_watchdog(
        undetermined_comment_share=36.0,
        fallback_title_rate=10.0,
    )
    assert degraded is True
    assert any("undetermined_comment_share" in reason for reason in reasons)

    degraded, reasons = service.quality_metrics.evaluate_quality_watchdog(
        undetermined_comment_share=10.0,
        fallback_title_rate=41.0,
    )
    assert degraded is True
    assert any("fallback_title_rate" in reason for reason in reasons)

    degraded, reasons = service.quality_metrics.evaluate_quality_watchdog(
        undetermined_comment_share=10.0,
        fallback_title_rate=5.0,
    )
    assert degraded is False
    assert reasons == []


def test_build_cluster_diagnostics_payload_contains_metrics(db_session, test_settings) -> None:
    service = DailyRunService(test_settings, db_session)
    run = Run(
        id=7,
        video_id=1,
        mode="free",
        status="running",
        started_at=datetime(2026, 2, 23, 8, 0, tzinfo=UTC),
        ended_at=None,
        total_comments=10,
        processed_comments=10,
        meta_json={
            "cluster_noise_ratio": 0.21,
            "emerging_cluster_count": 1,
            "postprocess_merge_count": 2,
            "postprocess_uncertain_collapsed": 1,
            "fallback_topic_title_count": 1,
            "fallback_position_title_count": 2,
            "fallback_topic_title_rate": 20.0,
            "fallback_position_title_rate": 30.0,
            "fallback_title_rate": 25.0,
            "llm_cluster_fail_count": 1,
            "llm_disabled_reason": "rate_limited",
            "undetermined_comment_share": 30.0,
            "degraded": False,
            "degraded_reasons": [],
            "context_reliability": "high",
        },
    )
    video = VideoMeta(
        youtube_video_id="abc123xyz89",
        playlist_id="playlist",
        title="Video",
        description="desc",
        published_at=datetime(2026, 2, 23, 7, 0, tzinfo=UTC),
        duration_seconds=300,
        url="https://youtube.com/watch?v=abc123xyz89",
    )
    topic = _topic_with_positions("cluster_1")
    cluster = ClusterDraft(
        cluster_key="cluster_1",
        member_indices=list(range(10)),
        representative_indices=[0, 1],
        centroid=[1.0, 0.0],
        size_count=10,
        share_pct=100.0,
        weighted_share=100.0,
        is_emerging=False,
    )

    payload = service.quality_metrics.build_cluster_diagnostics_payload(
        run=run,
        video=video,
        topics=[topic],
        clusters=[cluster],
        labeling_diagnostics={
            "fallback_topic_title_count": 1,
            "fallback_position_title_count": 2,
            "llm_cluster_fail_count": 1,
            "llm_disabled_reason": "rate_limited",
        },
    )

    assert payload["run_id"] == 7
    assert payload["metrics"]["llm_disabled_reason"] == "rate_limited"
    assert payload["clusters"][0]["cluster_key"] == "cluster_1"
    assert payload["clusters"][0]["undetermined_comments"] == 3
