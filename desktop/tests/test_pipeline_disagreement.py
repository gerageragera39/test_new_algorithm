from __future__ import annotations

from datetime import UTC, datetime

from app.schemas.domain import ProcessedComment
from app.services.pipeline import DailyRunService


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


def test_extract_author_disagreement_comments_filters_offensive_and_keeps_duplicates(
    db_session, test_settings
) -> None:
    service = DailyRunService(test_settings, db_session)
    comments = [
        _comment(
            "1",
            "\u0421\u0435\u0440\u0433\u0435\u0439, \u0432\u044b \u043d\u0435 \u043f\u0440\u0430\u0432\u044b, \u044d\u0442\u043e \u043b\u043e\u0436\u044c.",
            2.0,
        ),
        _comment(
            "2",
            "\u041d\u0435 \u0441\u043e\u0433\u043b\u0430\u0441\u0435\u043d \u0441 \u044d\u0442\u043e\u0439 \u043f\u043e\u0437\u0438\u0446\u0438\u0435\u0439, \u0421\u0435\u0440\u0433\u0435\u0439.",
            1.8,
        ),
        _comment(
            "3",
            "\u041d\u0435 \u0441\u043e\u0433\u043b\u0430\u0441\u0435\u043d \u0441 \u044d\u0442\u043e\u0439 \u043f\u043e\u0437\u0438\u0446\u0438\u0435\u0439, \u0421\u0435\u0440\u0433\u0435\u0439.",
            1.0,
        ),
        _comment(
            "4",
            "\u0421\u0435\u0440\u0433\u0435\u0439, \u0432\u044b \u0438\u0434\u0438\u043e\u0442, \u043d\u0435 \u0441\u043e\u0433\u043b\u0430\u0441\u0435\u043d.",
            0.9,
        ),
        _comment(
            "5",
            "\u0421\u043f\u0430\u0441\u0438\u0431\u043e \u0437\u0430 \u0432\u044b\u043f\u0443\u0441\u043a, \u0432\u0441\u0451 \u043f\u043e \u0434\u0435\u043b\u0443.",
            1.5,
        ),
        _comment(
            "6",
            "\u042d\u0442\u043e \u043b\u043e\u0436\u044c \u0438 \u043c\u0430\u043d\u0438\u043f\u0443\u043b\u044f\u0446\u0438\u044f.",
            1.0,
        ),
    ]

    result = service.report_builder.extract_author_disagreement_comments(comments)

    assert len(result) == 3
    assert any("\u043d\u0435 \u043f\u0440\u0430\u0432\u044b" in text.lower() for text in result)
    assert all("\u0441\u043f\u0430\u0441\u0438\u0431\u043e" not in text.lower() for text in result)
    assert all("\u0438\u0434\u0438\u043e\u0442" not in text.lower() for text in result)
    assert (
        sum(
            1
            for text in result
            if "\u043d\u0435 \u0441\u043e\u0433\u043b\u0430\u0441\u0435\u043d \u0441 \u044d\u0442\u043e\u0439 \u043f\u043e\u0437\u0438\u0446\u0438\u0435\u0439"
            in text.lower()
        )
        == 2
    )
