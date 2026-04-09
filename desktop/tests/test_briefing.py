from __future__ import annotations

from datetime import UTC, datetime

from app.schemas.domain import TopicSummary, VideoMeta
from app.services.briefing import BriefingService


def _topic(
    key: str,
    label: str,
    share: float,
    weighted: float,
    sentiment: str,
    centroid: list[float],
    *,
    representative_quotes: list[str] | None = None,
    question_comments: list[str] | None = None,
) -> TopicSummary:
    return TopicSummary(
        cluster_key=key,
        label=label,
        description="Topic description",
        author_actions=["Add one fact", "Clarify one point"],
        sentiment=sentiment,
        emotion_tags=[],
        intent_distribution={
            "question": 2,
            "request": 1,
            "complaint": 0,
            "praise": 0,
            "suggestion": 1,
            "joke": 0,
            "other": 0,
        },
        representative_quotes=representative_quotes or ["Quote 1", "Quote 2"],
        question_comments=question_comments or [],
        size_count=10,
        share_pct=share,
        weighted_share=weighted,
        is_emerging=False,
        centroid=centroid,
    )


def test_briefing_contains_actions_and_trends(test_settings) -> None:
    service = BriefingService(test_settings)
    video = VideoMeta(
        youtube_video_id="v123",
        playlist_id="pl",
        title="Test episode",
        published_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v123",
    )
    current = [
        _topic("c1", "Sanctions and market", 40.0, 45.0, "negative", [1.0, 0.0]),
        _topic("c2", "Requests for facts", 20.0, 18.0, "neutral", [0.0, 1.0]),
    ]
    previous = [_topic("p1", "Sanctions and market", 30.0, 33.0, "neutral", [0.99, 0.01])]
    briefing = service.build(
        video=video,
        mode="free",
        topics=current,
        previous_topics=previous,
        disagreement_comments=[
            "You are wrong on this point.",
            "You are wrong on this point.",
            "I disagree with your interpretation.",
        ],
    )
    assert briefing.actions_for_tomorrow
    assert briefing.action_items
    assert briefing.top_topics[0].label == "Sanctions and market"
    assert briefing.trend_vs_previous
    assert len(briefing.author_disagreement_comments) == 2


def test_briefing_trend_handles_embedding_dimension_change(test_settings) -> None:
    service = BriefingService(test_settings)
    video = VideoMeta(
        youtube_video_id="v456",
        playlist_id="pl",
        title="Embedding switch episode",
        published_at=datetime(2026, 2, 22, 9, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v456",
    )
    current = [_topic("c1", "New topic", 35.0, 37.0, "neutral", [0.1, 0.2, 0.3, 0.4])]
    previous = [_topic("p1", "Old topic", 25.0, 30.0, "neutral", [0.1, 0.2, 0.3])]

    briefing = service.build(
        video=video,
        mode="free",
        topics=current,
        previous_topics=previous,
        disagreement_comments=[],
    )

    assert briefing.trend_vs_previous
    assert "New topic" in briefing.trend_vs_previous[0]


def test_briefing_populates_actionable_sections(test_settings) -> None:
    """The briefing should expose practical sections for the report and UI."""
    service = BriefingService(test_settings)
    video = VideoMeta(
        youtube_video_id="v789",
        playlist_id="pl",
        title="Audience expectation check",
        published_at=datetime(2026, 2, 22, 10, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v789",
    )
    current = [
        _topic(
            "c1",
            "Corruption topic",
            42.0,
            46.0,
            "negative",
            [1.0, 0.0],
            representative_quotes=[
                "Нужны факты и ссылки, а не общие слова.",
                "Покажите статистику и дайте понятную логику по выводам.",
            ],
            question_comments=[
                "Здравствуйте Сергей. Смертную казнь в Украине надо ввести?",
            ],
        )
    ]

    briefing = service.build(
        video=video,
        mode="free",
        topics=current,
        previous_topics=[],
        disagreement_comments=[],
    )

    assert briefing.actions_for_tomorrow
    assert briefing.action_items
    assert briefing.audience_requests_and_questions == []
    assert briefing.representative_quotes == []
    assert briefing.action_items[0].topic_label == "Corruption topic"
    assert briefing.action_items[0].key_criticism == ""
    assert "Смертную казнь" in briefing.action_items[0].key_question


def test_build_topic_trend_series_tracks_matching_topics(test_settings) -> None:
    service = BriefingService(test_settings)
    current_briefing = service.build(
        video=VideoMeta(
            youtube_video_id="current",
            playlist_id="pl",
            title="Current video",
            published_at=datetime(2026, 2, 25, 10, 0, tzinfo=UTC),
            url="https://youtube.com/watch?v=current",
        ),
        mode="free",
        topics=[_topic("c1", "Sanctions and market", 40.0, 42.0, "negative", [1.0, 0.0])],
        previous_topics=[],
        disagreement_comments=[],
    )
    previous_briefing = service.build(
        video=VideoMeta(
            youtube_video_id="prev",
            playlist_id="pl",
            title="Previous video",
            published_at=datetime(2026, 2, 24, 10, 0, tzinfo=UTC),
            url="https://youtube.com/watch?v=prev",
        ),
        mode="free",
        topics=[_topic("p1", "Market and sanctions", 25.0, 27.0, "neutral", [0.99, 0.01])],
        previous_topics=[],
        disagreement_comments=[],
    )

    series = service.build_topic_trend_series(current_briefing, [previous_briefing])

    assert len(series) == 1
    assert len(series[0].points) == 2
    assert series[0].points[0].share_pct == 25.0
    assert series[0].points[1].is_current is True


def test_briefing_does_not_limit_topics_by_report_top_k(test_settings) -> None:
    low_topk_settings = test_settings.model_copy(
        update={"report_top_k_topics": 1, "report_quotes_per_topic": 1}
    )
    high_topk_settings = test_settings.model_copy(
        update={"report_top_k_topics": 10, "report_quotes_per_topic": 20}
    )
    service_low = BriefingService(low_topk_settings)
    service_high = BriefingService(high_topk_settings)
    video = VideoMeta(
        youtube_video_id="v101",
        playlist_id="pl",
        title="Top-k ignore check",
        published_at=datetime(2026, 2, 22, 11, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v101",
    )
    topics = [
        _topic("c1", "Тема 1", 30.0, 30.0, "neutral", [1.0, 0.0]),
        _topic("c2", "Тема 2", 25.0, 25.0, "neutral", [0.9, 0.1]),
        _topic("c3", "Тема 3", 20.0, 20.0, "neutral", [0.8, 0.2]),
    ]

    briefing_low = service_low.build(
        video=video, mode="free", topics=topics, previous_topics=[], disagreement_comments=[]
    )
    briefing_high = service_high.build(
        video=video, mode="free", topics=topics, previous_topics=[], disagreement_comments=[]
    )

    assert [topic.label for topic in briefing_low.top_topics] == [
        topic.label for topic in briefing_high.top_topics
    ]
    assert len(briefing_low.top_topics) == len(topics)
