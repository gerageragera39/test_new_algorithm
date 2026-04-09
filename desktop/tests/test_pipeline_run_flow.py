from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.schemas.domain import (
    ClusterDraft,
    DailyBriefing,
    ProcessedComment,
    TopicSummary,
    VideoMeta,
)
from app.services.clustering import ClusteringResult
from app.services.labeling import NoLLMFallbackProvider
from app.services.pipeline import DailyRunService
from app.services.preprocessing import PreprocessResult


def _processed_comment() -> ProcessedComment:
    return ProcessedComment(
        youtube_comment_id="c1",
        text_raw="comment text",
        text_normalized="comment text",
        text_hash="hash-c1",
        published_at=datetime(2026, 2, 25, 10, 0, tzinfo=UTC),
        weight=1.0,
        like_count=0,
        reply_count=0,
    )


def _video_meta() -> VideoMeta:
    return VideoMeta(
        youtube_video_id="video_postprocess_test",
        playlist_id="playlist",
        title="Test video",
        description="desc",
        published_at=datetime(2026, 2, 25, 10, 0, tzinfo=UTC),
        duration_seconds=600,
        url="https://www.youtube.com/watch?v=video_postprocess_test",
    )


def _topic(cluster_key: str, label: str) -> TopicSummary:
    return TopicSummary(
        cluster_key=cluster_key,
        label=label,
        description="desc",
        author_actions=["action"],
        sentiment="neutral",
        emotion_tags=[],
        intent_distribution={"other": 1},
        representative_quotes=["quote"],
        question_comments=[],
        positions=[],
        size_count=1,
        share_pct=100.0,
        weighted_share=100.0,
        is_emerging=False,
        source="comment_topic",
        coherence_score=0.5,
        centroid=[1.0, 0.0],
    )


def test_run_flow_uses_postprocessed_topics_for_persist_and_briefing(
    db_session,
    test_settings,
    monkeypatch,
) -> None:
    service = DailyRunService(test_settings, db_session)
    video_meta = _video_meta()
    processed = _processed_comment()
    preprocess_result = PreprocessResult(
        processed=[processed],
        all_comments=[processed],
        filtered_count=0,
        total_count=1,
        dropped_count=0,
        flagged_count=0,
        kept_count=1,
        borderline_comment_ids=[],
    )

    class FakeYouTubeClient:
        def __init__(self, settings) -> None:  # noqa: ANN001
            _ = settings

        def fetch_comments(self, _video_id: str):  # noqa: ANN001
            return []

        def close(self) -> None:
            return None

    monkeypatch.setattr("app.services.pipeline.runner.YouTubeClient", FakeYouTubeClient)
    monkeypatch.setattr(service, "_load_previous_topics", lambda _video_id: [])
    monkeypatch.setattr(service.preprocessor, "preprocess", lambda _raw, _meta: preprocess_result)

    class FakeEmbeddingService:
        def get_embeddings(self, _texts, _hashes, **_kwargs):  # noqa: ANN001
            return [[1.0, 0.0]]

    monkeypatch.setattr(service, "_build_embedding_service", lambda _budget: FakeEmbeddingService())

    raw_cluster = ClusterDraft(
        cluster_key="cluster_raw",
        member_indices=[0],
        representative_indices=[0],
        centroid=[1.0, 0.0],
        size_count=1,
        share_pct=100.0,
        weighted_share=100.0,
        is_emerging=False,
    )
    monkeypatch.setattr(
        service.clustering,
        "cluster",
        lambda _comments, _vectors: ClusteringResult(
            clusters=[raw_cluster], label_by_index={0: "cluster_raw"}
        ),
    )
    monkeypatch.setattr(
        service.cluster_enricher,
        "_merge_similar_clusters",
        lambda clusters, *_: clusters,
    )
    monkeypatch.setattr(
        service,
        "_build_llm_provider",
        lambda *_: NoLLMFallbackProvider(test_settings),
    )

    raw_topic = _topic("cluster_raw", "Raw topic")
    monkeypatch.setattr(
        service.cluster_enricher,
        "label_topics",
        lambda *_args, **_kwargs: (
            [raw_topic],
            {
                "fallback_topic_title_count": 0,
                "fallback_position_title_count": 0,
                "llm_cluster_fail_count": 0,
            },
        ),
    )

    post_cluster = raw_cluster.model_copy(update={"cluster_key": "cluster_post"})
    post_topic = _topic("cluster_post", "Postprocessed topic")
    called: dict[str, object] = {"postprocess_called": False}

    def fake_postprocess(**kwargs):  # noqa: ANN003
        called["postprocess_called"] = True
        return [post_cluster], [post_topic]

    monkeypatch.setattr(service.cluster_enricher, "_postprocess_labeled_topics", fake_postprocess)

    def fake_persist(
        _run_id, _video_id, clusters, topics, _comment_models, _vectors
    ):  # noqa: ANN001, ANN002
        called["persist_clusters"] = clusters
        called["persist_topics"] = topics

    monkeypatch.setattr(service, "_persist_clusters", fake_persist)

    def fake_build_briefing(**kwargs):  # noqa: ANN003
        topics = kwargs["topics"]
        called["briefing_topics"] = topics
        return DailyBriefing(
            video_id=video_meta.youtube_video_id,
            video_title=video_meta.title,
            published_at=video_meta.published_at,
            mode="openai",
            executive_summary="summary",
            top_topics=topics,
            actions_for_tomorrow=[],
            misunderstandings_and_controversies=[],
            audience_requests_and_questions=[],
            risks_and_toxicity=[],
            representative_quotes=[],
        )

    monkeypatch.setattr(service.briefing, "build", fake_build_briefing)
    monkeypatch.setattr(service.exporter, "to_markdown", lambda _briefing: "md")
    monkeypatch.setattr(service.exporter, "to_html", lambda _markdown: "<p>md</p>")
    monkeypatch.setattr(service.exporter, "persist", lambda *_: None)
    monkeypatch.setattr(
        service.exporter,
        "persist_cluster_diagnostics",
        lambda *_: Path("data/reports/2026-03-02/video_postprocess_test.cluster_diagnostics.json"),
    )

    result = service._run_for_video_meta(video_meta, skip_filtering=False)

    assert result["topics_count"] == 1
    assert called["postprocess_called"] is True
    assert called["persist_clusters"][0].cluster_key == "cluster_post"
    assert called["persist_topics"][0].label == "Postprocessed topic"
    assert called["briefing_topics"][0].label == "Postprocessed topic"
