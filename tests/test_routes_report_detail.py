from __future__ import annotations

from datetime import UTC, datetime

from app.api import routes
from app.schemas.domain import DailyBriefing, TopicPosition, TopicSummary


def _briefing() -> DailyBriefing:
    topic = TopicSummary(
        cluster_key="cluster_1",
        label="Main topic",
        description="Topic description",
        author_actions=["Do one practical step"],
        sentiment="neutral",
        emotion_tags=[],
        intent_distribution={"other": 1},
        representative_quotes=["quote"],
        question_comments=[],
        positions=[
            TopicPosition(
                key="pos_1",
                title="Single claim title",
                summary="One-claim summary",
                markers=["marker"],
                prototype="prototype",
                count=3,
                pct=100.0,
                weighted_count=3.0,
                weighted_pct=100.0,
                comments=["c1", "c2", "c3"],
                is_undetermined=False,
                coherence_score=0.72,
                single_claim_passed=True,
            )
        ],
        size_count=3,
        share_pct=100.0,
        weighted_share=100.0,
        is_emerging=False,
        source="comment_topic",
        coherence_score=0.72,
        centroid=[1.0, 0.0],
    )
    return DailyBriefing(
        video_id="vid123",
        video_title="Video",
        published_at=datetime(2026, 2, 25, 10, 0, tzinfo=UTC),
        mode="free",
        executive_summary="Summary",
        top_topics=[topic],
        actions_for_tomorrow=[],
        misunderstandings_and_controversies=[],
        audience_requests_and_questions=[],
        risks_and_toxicity=[],
        representative_quotes=[],
    )


def test_load_topic_positions_for_report_returns_positions() -> None:
    briefing = _briefing()

    topic_positions = routes._load_topic_positions_for_report(briefing)

    assert "cluster_1" in topic_positions
    assert len(topic_positions["cluster_1"]) == 1
    assert topic_positions["cluster_1"][0].title == "Single claim title"
    assert topic_positions["cluster_1"][0].single_claim_passed is True
