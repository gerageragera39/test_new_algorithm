from __future__ import annotations

from datetime import UTC, datetime

from app.schemas.domain import RawComment, VideoMeta
from app.services.preprocessing import CommentPreprocessor


def test_preprocess_filters_duplicates_and_short_comments(test_settings) -> None:
    video = VideoMeta(
        youtube_video_id="v1",
        playlist_id="pl",
        title="t",
        published_at=datetime(2026, 2, 21, 8, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v1",
    )
    comments = [
        RawComment(
            youtube_comment_id="1",
            text_raw="Коротко",
            published_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        ),
        RawComment(
            youtube_comment_id="2",
            text_raw="Спасибо за выпуск, очень полезный разбор.",
            published_at=datetime(2026, 2, 21, 9, 1, tzinfo=UTC),
            like_count=5,
        ),
        RawComment(
            youtube_comment_id="3",
            text_raw="Спасибо за выпуск, очень полезный разбор.",
            published_at=datetime(2026, 2, 21, 9, 2, tzinfo=UTC),
            like_count=3,
        ),
    ]
    preprocessor = CommentPreprocessor(test_settings)
    result = preprocessor.preprocess(comments, video)
    assert result.total_count == 3
    assert result.filtered_count == 2
    assert len(result.processed) == 1
    assert result.processed[0].youtube_comment_id == "2"
    assert result.processed[0].weight > 1.0


def test_preprocess_filters_low_signal_comments(test_settings) -> None:
    video = VideoMeta(
        youtube_video_id="v2",
        playlist_id="pl",
        title="t2",
        published_at=datetime(2026, 2, 21, 8, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v2",
    )
    comments = [
        RawComment(
            youtube_comment_id="1",
            text_raw="тест тест тест тест тест тест",
            published_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        ),
        RawComment(
            youtube_comment_id="2",
            text_raw="Спасибо за выпуск, полезный разбор с фактами и ссылками.",
            published_at=datetime(2026, 2, 21, 9, 1, tzinfo=UTC),
            like_count=2,
        ),
    ]
    preprocessor = CommentPreprocessor(test_settings)
    result = preprocessor.preprocess(comments, video)

    assert result.total_count == 2
    assert result.filtered_count == 1
    assert len(result.processed) == 1
    assert result.processed[0].youtube_comment_id == "2"
    low_signal = next(
        comment for comment in result.all_comments if comment.youtube_comment_id == "1"
    )
    assert low_signal.is_filtered is True
    assert low_signal.filter_reason == "low_signal"


def test_preprocess_duplicate_volume_boosts_canonical_weight(test_settings) -> None:
    video = VideoMeta(
        youtube_video_id="v3",
        playlist_id="pl",
        title="t3",
        published_at=datetime(2026, 2, 21, 8, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v3",
    )
    canonical_text = "Спасибо за выпуск, это очень полезный и детальный разбор темы."
    comment_one = RawComment(
        youtube_comment_id="1",
        text_raw=canonical_text,
        published_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
    )
    comment_dup = RawComment(
        youtube_comment_id="2",
        text_raw=canonical_text,
        published_at=datetime(2026, 2, 21, 9, 1, tzinfo=UTC),
    )

    preprocessor = CommentPreprocessor(test_settings)
    single_result = preprocessor.preprocess([comment_one], video)
    duplicate_result = preprocessor.preprocess([comment_one, comment_dup], video)

    assert len(single_result.processed) == 1
    assert len(duplicate_result.processed) == 1
    assert duplicate_result.filtered_count == 1
    assert duplicate_result.processed[0].weight > single_result.processed[0].weight


def test_preprocess_keeps_full_comment_text_without_char_truncation(test_settings) -> None:
    settings = test_settings.model_copy(update={"comment_max_chars": 20})
    video = VideoMeta(
        youtube_video_id="v4",
        playlist_id="pl",
        title="t4",
        published_at=datetime(2026, 2, 21, 8, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v4",
    )
    long_text = (
        "Это очень длинный комментарий который должен остаться целиком "
        "после preprocessing и не должен обрезаться по лимиту символов."
    )
    comments = [
        RawComment(
            youtube_comment_id="1",
            text_raw=long_text,
            published_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        )
    ]

    result = CommentPreprocessor(settings).preprocess(comments, video)

    assert len(result.processed) == 1
    assert "не должен обрезаться по лимиту символов" in result.processed[0].text_normalized
    assert len(result.processed[0].text_normalized) > settings.comment_max_chars


def test_preprocess_moderation_drops_spam_link(test_settings) -> None:
    video = VideoMeta(
        youtube_video_id="v5",
        playlist_id="pl",
        title="Daily political analysis",
        published_at=datetime(2026, 2, 21, 8, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v5",
    )
    comments = [
        RawComment(
            youtube_comment_id="1",
            text_raw="Join t.me/somechannel for promo code now",
            published_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        )
    ]

    result = CommentPreprocessor(test_settings).preprocess(comments, video)
    assert result.filtered_count == 1
    assert len(result.processed) == 0
    assert result.dropped_by_reason.get("spam_link") == 1


def test_preprocess_moderation_drops_profanity_only(test_settings) -> None:
    settings = test_settings.model_copy(update={"comment_min_words": 1})
    video = VideoMeta(
        youtube_video_id="v6",
        playlist_id="pl",
        title="Defense strategy discussion",
        published_at=datetime(2026, 2, 21, 8, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v6",
    )
    comments = [
        RawComment(
            youtube_comment_id="1",
            text_raw="fuck idiot moron",
            published_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        )
    ]

    result = CommentPreprocessor(settings).preprocess(comments, video)
    assert result.filtered_count == 1
    assert result.dropped_by_reason.get("profanity_only") == 1


def test_preprocess_moderation_flags_toxic_with_position(test_settings) -> None:
    video = VideoMeta(
        youtube_video_id="v7",
        playlist_id="pl",
        title="Strategy and policy outcomes",
        published_at=datetime(2026, 2, 21, 8, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v7",
    )
    comments = [
        RawComment(
            youtube_comment_id="1",
            text_raw="fuck this strategy is wrong and harmful",
            published_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        )
    ]

    result = CommentPreprocessor(test_settings).preprocess(comments, video)
    assert result.filtered_count == 0
    assert len(result.processed) == 1
    assert result.processed[0].moderation_action == "flag"
    assert result.processed[0].moderation_reason == "toxic_with_position"
    assert result.flagged_count == 1


def test_preprocess_marks_borderline_for_llm(test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "moderation_enable_llm_borderline": True,
            "moderation_borderline_min_score": 0.35,
            "moderation_borderline_max_score": 0.55,
        }
    )
    video = VideoMeta(
        youtube_video_id="v8",
        playlist_id="pl",
        title="News review",
        published_at=datetime(2026, 2, 21, 8, 0, tzinfo=UTC),
        url="https://youtube.com/watch?v=v8",
    )
    comments = [
        RawComment(
            youtube_comment_id="1",
            text_raw="very unclear short statement",
            published_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        )
    ]

    result = CommentPreprocessor(settings).preprocess(comments, video)
    assert len(result.processed) == 1
    assert result.borderline_comment_ids == ["1"]
    assert result.processed[0].moderation_reason == "borderline_for_llm"
