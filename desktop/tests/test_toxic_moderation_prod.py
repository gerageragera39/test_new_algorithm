from __future__ import annotations

from datetime import UTC, datetime


def _make_video(*, youtube_video_id: str, playlist_id: str = "PL1"):
    from app.db.models import Video

    return Video(
        youtube_video_id=youtube_video_id,
        playlist_id=playlist_id,
        title=f"Video {youtube_video_id}",
        url=f"https://youtube.com/watch?v={youtube_video_id}",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _make_comment(
    *, video_id: int, youtube_comment_id: str, author_name: str, author_channel_id: str | None
):
    from app.core.utils import hash_text
    from app.db.models import Comment

    text = f"Comment from {author_name}"
    normalized = " ".join(text.split())
    return Comment(
        video_id=video_id,
        youtube_comment_id=youtube_comment_id,
        parent_comment_id=None,
        author_name=author_name,
        author_channel_id=author_channel_id,
        text_raw=text,
        text_normalized=normalized,
        text_hash=hash_text(normalized),
        like_count=0,
        reply_count=0,
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        is_top_level=True,
        weight=1.0,
    )


def test_persist_raw_comments_keeps_author_channel_id(db_session, test_settings) -> None:
    from sqlalchemy import select

    from app.schemas.domain import RawComment
    from app.services.appeal_analytics.runner import AppealAnalyticsService

    video = _make_video(youtube_video_id="raw-channel-id")
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)

    service = AppealAnalyticsService(test_settings, db_session)
    service._persist_raw_comments(
        video.id,
        [
            RawComment(
                youtube_comment_id="yt-comment-1",
                author_name="Tester",
                author_channel_id="UC_TEST_123",
                text_raw="hello world",
                published_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ],
    )

    from app.db.models import Comment

    saved = db_session.scalar(select(Comment).where(Comment.youtube_comment_id == "yt-comment-1"))
    assert saved is not None
    assert saved.author_channel_id == "UC_TEST_123"


def test_get_toxic_review_excludes_globally_banned_channel_id(db_session) -> None:
    from app.api.routes import get_toxic_review
    from app.db.models import AppealBlock, AppealBlockItem, AppealRun, BannedUser

    banned_video = _make_video(youtube_video_id="ban-source")
    review_video = _make_video(youtube_video_id="review-target")
    db_session.add_all([banned_video, review_video])
    db_session.flush()

    review_comment = _make_comment(
        video_id=review_video.id,
        youtube_comment_id="yt-comment-review",
        author_name="Same Person",
        author_channel_id="UC_BANNED_GLOBAL",
    )
    db_session.add(review_comment)
    db_session.flush()

    appeal_run = AppealRun(
        video_id=review_video.id,
        status="completed",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        total_comments=1,
        processed_comments=1,
        meta_json={},
    )
    db_session.add(appeal_run)
    db_session.flush()

    block = AppealBlock(
        appeal_run_id=appeal_run.id,
        video_id=review_video.id,
        block_type="toxic_manual_review",
        display_label="Manual review",
        sort_order=5,
        item_count=1,
    )
    db_session.add(block)
    db_session.flush()

    db_session.add(
        AppealBlockItem(
            block_id=block.id,
            comment_id=review_comment.id,
            author_name=review_comment.author_name,
            detail_json={"confidence_score": 0.42, "insult_target": "author"},
            confidence_score=0.42,
            insult_target="author",
        )
    )
    db_session.add(
        BannedUser(
            video_id=banned_video.id,
            comment_id=None,
            username="Older Name",
            author_channel_id="UC_BANNED_GLOBAL",
            ban_reason="global hide",
            confidence_score=0.99,
            insult_target="author",
            banned_at=datetime(2026, 1, 1, tzinfo=UTC),
            youtube_banned=True,
            banned_by_admin=False,
        )
    )
    db_session.commit()

    response = get_toxic_review(review_video.youtube_video_id, db_session)
    assert response.total_review_items == 0
    assert response.items == []


def test_ban_service_deduplicates_by_author_channel_id_across_videos(
    db_session,
    test_settings,
    monkeypatch,
) -> None:
    from app.services.youtube_ban_service import YouTubeBanService

    video_a = _make_video(youtube_video_id="video-a")
    video_b = _make_video(youtube_video_id="video-b")
    db_session.add_all([video_a, video_b])
    db_session.flush()

    comment_a = _make_comment(
        video_id=video_a.id,
        youtube_comment_id="comment-a",
        author_name="Display A",
        author_channel_id="UC_DUPLICATE",
    )
    comment_b = _make_comment(
        video_id=video_b.id,
        youtube_comment_id="comment-b",
        author_name="Display B",
        author_channel_id="UC_DUPLICATE",
    )
    db_session.add_all([comment_a, comment_b])
    db_session.commit()

    service = YouTubeBanService(test_settings, db_session)
    monkeypatch.setattr(service, "_save_to_csv", lambda *args, **kwargs: None)

    first = service.ban_user(
        video_id=video_a.id,
        comment_id=comment_a.id,
        username=comment_a.author_name or "unknown",
        author_channel_id=comment_a.author_channel_id,
        ban_reason="first",
        confidence_score=0.95,
    )
    second = service.ban_user(
        video_id=video_b.id,
        comment_id=comment_b.id,
        username=comment_b.author_name or "unknown",
        author_channel_id=comment_b.author_channel_id,
        ban_reason="second",
        confidence_score=0.95,
    )

    assert first["status"] == "banned"
    assert second["status"] == "already_banned"


def test_ban_via_youtube_api_retries_once_after_401(
    db_session,
    test_settings,
    monkeypatch,
) -> None:
    from app.services.youtube_ban_service import YouTubeBanService

    test_settings.youtube_oauth_client_id = "cid"
    test_settings.youtube_oauth_client_secret = "secret"
    test_settings.youtube_oauth_refresh_token = "refresh"

    service = YouTubeBanService(test_settings, db_session)

    tokens = iter(["token-1", "token-2"])
    monkeypatch.setattr(service, "_get_access_token", lambda: next(tokens))

    class DummyResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.content = b""
            self.text = ""

    calls: list[str] = []

    def fake_post(url: str, headers: dict[str, str], timeout: int):
        calls.append(headers["Authorization"])
        return DummyResponse(401 if len(calls) == 1 else 204)

    monkeypatch.setattr("app.services.youtube_ban_service.requests.post", fake_post)

    success, error = service._ban_via_youtube_api("youtube-comment-id", "reason")
    assert success is True
    assert error is None
    assert calls == ["Bearer token-1", "Bearer token-2"]
