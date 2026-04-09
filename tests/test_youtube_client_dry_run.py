from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.schemas.domain import RawComment
from app.services.youtube_client import YouTubeClient, extract_youtube_video_id


def test_youtube_client_uses_fixture_in_dry_run(test_settings) -> None:
    client = YouTubeClient(test_settings)
    try:
        latest = client.get_latest_video_from_playlist()
        comments = client.fetch_comments(latest.youtube_video_id)
    finally:
        client.close()
    assert latest.youtube_video_id == "video_mock_001"
    assert latest.title
    assert len(comments) >= 5


def test_extract_youtube_video_id_from_supported_links() -> None:
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ?t=43") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_youtube_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_get_video_meta_by_url_in_dry_run(test_settings) -> None:
    client = YouTubeClient(test_settings)
    try:
        meta = client.get_video_meta_by_url("https://youtu.be/dQw4w9WgXcQ")
    finally:
        client.close()
    assert meta.youtube_video_id == "dQw4w9WgXcQ"
    assert meta.url.endswith("dQw4w9WgXcQ")


def test_merge_comment_samples_dedupes_and_respects_cap_without_relevance_share(
    test_settings,
) -> None:
    client = YouTubeClient(test_settings)
    try:
        ts = datetime(2026, 2, 23, 10, 0, tzinfo=UTC)
        time_comments = [
            RawComment(youtube_comment_id="a", text_raw="time a", published_at=ts),
            RawComment(youtube_comment_id="b", text_raw="time b", published_at=ts),
            RawComment(youtube_comment_id="c", text_raw="time c", published_at=ts),
            RawComment(youtube_comment_id="d", text_raw="time d", published_at=ts),
        ]
        relevance_comments = [
            RawComment(youtube_comment_id="x", text_raw="rel x", published_at=ts),
            RawComment(youtube_comment_id="b", text_raw="rel b dup", published_at=ts),
            RawComment(youtube_comment_id="y", text_raw="rel y", published_at=ts),
        ]
        merged = client._merge_comment_samples(
            time_comments=time_comments,
            relevance_comments=relevance_comments,
            max_comments=6,
        )
    finally:
        client.close()

    ids = [item.youtube_comment_id for item in merged]
    assert len(ids) == 6
    assert len(set(ids)) == 6
    assert "x" in ids and "y" in ids


def test_fetch_comments_ignores_relevance_share_for_final_pool(
    test_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    ts = datetime(2026, 2, 23, 10, 0, tzinfo=UTC)
    time_comments = [
        RawComment(youtube_comment_id="a", text_raw="time a", published_at=ts),
        RawComment(youtube_comment_id="b", text_raw="time b", published_at=ts),
        RawComment(youtube_comment_id="c", text_raw="time c", published_at=ts),
    ]
    relevance_comments = [
        RawComment(youtube_comment_id="x", text_raw="rel x", published_at=ts),
        RawComment(youtube_comment_id="b", text_raw="rel b dup", published_at=ts),
        RawComment(youtube_comment_id="y", text_raw="rel y", published_at=ts),
    ]

    def fake_fetch(
        self, *, video_id: str, order: str, include_replies: bool, max_comments: int, max_pages: int
    ):  # noqa: ANN001
        _ = video_id, include_replies, max_comments, max_pages
        return list(relevance_comments if order == "relevance" else time_comments)

    settings_a = test_settings.model_copy(
        update={
            "dry_run": False,
            "max_comments_per_video": 5,
            "youtube_mix_relevance_comments": True,
            "youtube_relevance_comments_share": 0.1,
        }
    )
    settings_b = test_settings.model_copy(
        update={
            "dry_run": False,
            "max_comments_per_video": 5,
            "youtube_mix_relevance_comments": True,
            "youtube_relevance_comments_share": 0.8,
        }
    )

    monkeypatch.setattr(YouTubeClient, "_fetch_comments_with_order", fake_fetch)
    client_a = YouTubeClient(settings_a)
    client_b = YouTubeClient(settings_b)
    try:
        ids_a = [item.youtube_comment_id for item in client_a.fetch_comments("video_x")]
        ids_b = [item.youtube_comment_id for item in client_b.fetch_comments("video_x")]
    finally:
        client_a.close()
        client_b.close()

    assert ids_a == ids_b
