from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from app.db.models import Comment, Video
from app.schemas.domain import ProcessedComment
from app.services.pipeline import DailyRunService


def _processed_comment(comment_id: str, text: str, *, like_count: int = 0) -> ProcessedComment:
    return ProcessedComment(
        youtube_comment_id=comment_id,
        parent_comment_id=None,
        author_name="tester",
        text_raw=text,
        text_normalized=text,
        text_hash=f"hash-{comment_id}-{like_count}",
        language="ru",
        like_count=like_count,
        reply_count=0,
        published_at=datetime(2026, 2, 22, 12, 0, tzinfo=UTC),
        weight=1.0,
        is_top_level=True,
        is_filtered=False,
        filter_reason=None,
    )


def test_upsert_comments_handles_duplicate_ids_in_single_batch(db_session, test_settings) -> None:
    video = Video(
        youtube_video_id="video1234567",
        playlist_id="playlist",
        title="title",
        description="desc",
        published_at=datetime(2026, 2, 20, 9, 0, tzinfo=UTC),
        duration_seconds=3600,
        url="https://www.youtube.com/watch?v=video1234567",
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)

    service = DailyRunService(test_settings, db_session)
    comments = [
        _processed_comment("dup-id", "first version", like_count=1),
        _processed_comment("dup-id", "second version", like_count=2),
        _processed_comment("uniq-id", "unique comment", like_count=3),
    ]

    models = service._upsert_comments(video.id, comments)

    assert len(models) == 3
    assert models[0].youtube_comment_id == "dup-id"
    assert models[1].youtube_comment_id == "dup-id"
    assert models[0].id == models[1].id

    rows = list(
        db_session.scalars(
            select(Comment).where(Comment.youtube_comment_id.in_(["dup-id", "uniq-id"]))
        )
    )
    assert len(rows) == 2
