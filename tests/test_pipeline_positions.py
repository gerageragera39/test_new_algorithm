from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.schemas.domain import ClusterDraft, ProcessedComment
from app.services.labeling import NoLLMFallbackProvider
from app.services.pipeline import DailyRunService
from app.services.pipeline.text_utils import (
    position_title_single_claim_passed,
    sanitize_cluster_title,
)


def _comment(comment_id: str, text: str, weight: float = 1.0) -> ProcessedComment:
    return ProcessedComment(
        youtube_comment_id=comment_id,
        text_raw=text,
        text_normalized=text,
        text_hash=f"hash-{comment_id}",
        published_at=datetime(2026, 2, 23, 9, 0, tzinfo=UTC),
        weight=weight,
        like_count=0,
        reply_count=0,
    )


def test_subcluster_positions_cover_cluster_and_pct_is_100(db_session, test_settings) -> None:
    service = DailyRunService(test_settings, db_session)
    comments = [
        _comment("1", "I support the author point", 1.0),
        _comment("2", "Author argument is correct", 1.2),
        _comment("3", "I disagree and need facts", 1.0),
        _comment("4", "Counter-argument requires sources", 0.8),
        _comment("5", "Support and agreement", 1.1),
        _comment("6", "Critical feedback, show sources", 0.9),
    ]
    vectors = [
        [0.98, 0.02],
        [0.96, 0.04],
        [0.03, 0.97],
        [0.02, 0.98],
        [0.95, 0.05],
        [0.05, 0.95],
    ]
    cluster = ClusterDraft(
        cluster_key="cluster_positions",
        member_indices=[0, 1, 2, 3, 4, 5],
        representative_indices=[0, 2],
        centroid=[0.5, 0.5],
        size_count=6,
        share_pct=100.0,
        weighted_share=100.0,
    )

    positions, _ = service._build_positions_for_cluster(
        llm_provider=NoLLMFallbackProvider(test_settings),
        cluster=cluster,
        comments=comments,
        vectors=vectors,
        llm_disabled=False,
        cluster_title="Audience positions",
    )

    assert sum(position.count for position in positions) == cluster.size_count
    assert pytest.approx(sum(position.pct for position in positions), abs=0.001) == 100.0
    assert pytest.approx(sum(position.weighted_pct for position in positions), abs=0.001) == 100.0


def test_subcluster_positions_put_small_outlier_into_undetermined(
    db_session, test_settings
) -> None:
    service = DailyRunService(test_settings, db_session)
    comments = [
        _comment("1", "Support author"),
        _comment("2", "Agree with author"),
        _comment("3", "Fully support position"),
        _comment("4", "Mostly agree"),
        _comment("5", "Random off-topic sentence"),
    ]
    vectors = [
        [1.0, 0.0],
        [0.95, 0.05],
        [0.97, 0.03],
        [0.93, 0.07],
        [-1.0, 0.0],
    ]
    cluster = ClusterDraft(
        cluster_key="cluster_undetermined",
        member_indices=[0, 1, 2, 3, 4],
        representative_indices=[0],
        centroid=[0.5, 0.5],
        size_count=5,
        share_pct=100.0,
        weighted_share=100.0,
    )

    positions, _ = service._build_positions_for_cluster(
        llm_provider=NoLLMFallbackProvider(test_settings),
        cluster=cluster,
        comments=comments,
        vectors=vectors,
        llm_disabled=False,
        cluster_title="Audience positions",
    )

    undetermined = next(
        (position for position in positions if position.key == "undetermined"), None
    )
    assert undetermined is not None
    assert undetermined.count == 1


def test_subcluster_positions_omit_undetermined_when_not_needed(db_session, test_settings) -> None:
    service = DailyRunService(test_settings, db_session)
    comments = [
        _comment("1", "Support author"),
        _comment("2", "Agree with arguments"),
        _comment("3", "Fully agree with conclusion"),
        _comment("4", "I disagree with arguments"),
        _comment("5", "This is a weak argument"),
        _comment("6", "Need evidence and sources"),
    ]
    vectors = [
        [1.0, 0.0],
        [0.97, 0.03],
        [0.96, 0.04],
        [0.0, 1.0],
        [0.02, 0.98],
        [0.03, 0.97],
    ]
    cluster = ClusterDraft(
        cluster_key="cluster_without_undetermined",
        member_indices=[0, 1, 2, 3, 4, 5],
        representative_indices=[0],
        centroid=[0.5, 0.5],
        size_count=6,
        share_pct=100.0,
        weighted_share=100.0,
    )

    positions, _ = service._build_positions_for_cluster(
        llm_provider=NoLLMFallbackProvider(test_settings),
        cluster=cluster,
        comments=comments,
        vectors=vectors,
        llm_disabled=False,
        cluster_title="Audience positions",
    )

    assert all(position.key != "undetermined" for position in positions)


def test_subcluster_assignment_uses_similarity_and_margin_settings(
    db_session,
    test_settings,
    monkeypatch,
) -> None:
    settings = test_settings.model_copy(
        update={
            "position_assignment_min_similarity": 0.6,
            "position_assignment_min_margin": 0.12,
        }
    )
    service = DailyRunService(settings, db_session)
    vectors = [
        [1.0, 0.0],
        [0.98, 0.02],
        [0.97, 0.03],
        [0.0, 1.0],
        [0.02, 0.98],
        [0.03, 0.97],
        [0.72, 0.69],
    ]

    monkeypatch.setattr(
        service.position_extractor,
        "fit_subcluster_labels",
        lambda _matrix: np.array([0, 0, 0, 1, 1, 1, -1], dtype=np.int32),
    )

    groups, undetermined = service.position_extractor.subcluster_member_groups(
        member_indices=list(range(7)), vectors=vectors
    )

    assert len(groups) == 2
    assert 6 in undetermined


def test_title_sanitizer_rejects_bad_title_and_fallback_generates_valid(
    db_session, test_settings
) -> None:
    service = DailyRunService(test_settings, db_session)
    comments = [
        _comment("1", "sanctions and oil prices debate"),
        _comment("2", "oil market and sanctions topic"),
        _comment("3", "discussion about sanctions impact"),
    ]

    assert sanitize_cluster_title("misc comments topic", comments) == ""
    assert sanitize_cluster_title("politics/news", comments) == ""

    fallback = service.cluster_enricher._fallback_cluster_title(comments)
    assert fallback
    assert "/" not in fallback
    assert 2 <= len(fallback.split()) <= 10


def test_position_title_single_claim_validator() -> None:
    assert position_title_single_claim_passed("Arestovich lies about Zaluzhny")
    assert not position_title_single_claim_passed("Citizenship and origin: grandfather questions")
