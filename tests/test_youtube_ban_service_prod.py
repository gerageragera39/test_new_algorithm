from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.core.config import Settings


def test_ban_author_param_sent_as_lowercase_true(
    db_session: Session,
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime

    from app.db.models import Video
    from app.services.youtube_ban_service import YouTubeBanService

    test_settings.youtube_oauth_client_id = "cid"
    test_settings.youtube_oauth_client_secret = "secret"
    test_settings.youtube_oauth_refresh_token = "refresh"

    service = YouTubeBanService(test_settings, db_session)
    monkeypatch.setattr(service, "_get_access_token", lambda: "token")

    video = Video(
        youtube_video_id="BULdG_PCST4",
        playlist_id="PL1",
        title="Video",
        url="https://youtube.com/watch?v=BULdG_PCST4",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    db_session.add(video)
    db_session.commit()

    captured: dict[str, str] = {}

    class DummyResponse:
        status_code = 204
        content = b""
        text = ""

    class DummyGetResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.status_code = 200
            self._payload = payload
            self.text = ""

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_post(url: str, headers: dict[str, str], timeout: int) -> DummyResponse:
        captured["url"] = url
        return DummyResponse()

    def fake_get(
        url: str,
        params: dict[str, str],
        headers: dict[str, str],
        timeout: int,
    ) -> DummyGetResponse:
        if url.endswith("/channels"):
            return DummyGetResponse({"items": [{"id": "UC_OWNER"}]})
        if url.endswith("/videos"):
            assert params["id"] == "BULdG_PCST4"
            return DummyGetResponse({"items": [{"snippet": {"channelId": "UC_OWNER"}}]})
        raise AssertionError(f"Unexpected GET url: {url}")

    monkeypatch.setattr("app.services.youtube_ban_service.requests.get", fake_get)
    monkeypatch.setattr("app.services.youtube_ban_service.requests.post", fake_post)

    success, error = service._ban_via_youtube_api(
        "comment123", "reason", youtube_video_id="BULdG_PCST4"
    )

    assert success is True
    assert error is None
    assert "banAuthor=true" in captured["url"]


def test_ban_skips_api_when_oauth_channel_does_not_own_video(
    db_session: Session,
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.youtube_ban_service import YouTubeBanService

    test_settings.youtube_oauth_client_id = "cid"
    test_settings.youtube_oauth_client_secret = "secret"
    test_settings.youtube_oauth_refresh_token = "refresh"

    service = YouTubeBanService(test_settings, db_session)
    monkeypatch.setattr(service, "_get_access_token", lambda: "token")

    class DummyGetResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.status_code = 200
            self._payload = payload
            self.text = ""

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(
        url: str,
        params: dict[str, str],
        headers: dict[str, str],
        timeout: int,
    ) -> DummyGetResponse:
        if url.endswith("/channels"):
            return DummyGetResponse({"items": [{"id": "UC_AUTH"}]})
        if url.endswith("/videos"):
            return DummyGetResponse({"items": [{"snippet": {"channelId": "UC_VIDEO_OWNER"}}]})
        raise AssertionError(f"Unexpected GET url: {url}")

    def fake_post(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "comments.setModerationStatus must not be called when channel ownership mismatches"
        )

    monkeypatch.setattr("app.services.youtube_ban_service.requests.get", fake_get)
    monkeypatch.setattr("app.services.youtube_ban_service.requests.post", fake_post)

    success, error = service._ban_via_youtube_api(
        "comment123",
        "reason",
        youtube_video_id="BULdG_PCST4",
    )

    assert success is False
    assert error is not None
    assert "does not own the target YouTube channel" in error
