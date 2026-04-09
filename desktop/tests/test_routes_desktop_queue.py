from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.api import routes


class _FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], object]] = []

    def enqueue(self, kind: str, payload: dict[str, object], func: object) -> SimpleNamespace:
        self.calls.append((kind, payload, func))
        return SimpleNamespace(id=f"job-{kind}")

    def snapshot(self) -> dict[str, object]:
        return {"current": None, "queued": [], "recent": []}


def test_run_latest_async_queues_skip_filtering_from_query(
    db_session, test_settings, monkeypatch
) -> None:
    queue = _FakeQueue()
    monkeypatch.setattr(routes, "get_task_queue", lambda: queue)

    response = routes.run_latest(
        sync=False,
        skip_filtering="true",
        payload={"skip_filtering": False},
        db=db_session,
        settings=test_settings,
    )

    assert response.task_id == "job-run_latest"
    assert queue.calls[0][0] == "run_latest"
    assert queue.calls[0][1]["skip_filtering"] is True


def test_run_video_async_queues_video_url_and_skip_filtering(
    db_session, test_settings, monkeypatch
) -> None:
    queue = _FakeQueue()
    monkeypatch.setattr(routes, "get_task_queue", lambda: queue)

    response = routes.run_video(
        video_url="https://www.youtube.com/watch?v=testvideo1",
        sync=False,
        skip_filtering="1",
        payload={
            "video_url": "https://www.youtube.com/watch?v=testvideo1",
            "skip_filtering": False,
        },
        db=db_session,
        settings=test_settings,
    )

    assert response.task_id == "job-run_video"
    assert queue.calls[0][0] == "run_video"
    assert queue.calls[0][1]["video_url"] == "https://www.youtube.com/watch?v=testvideo1"
    assert queue.calls[0][1]["skip_filtering"] is True


def test_run_appeal_async_queues_guest_names(db_session, test_settings, monkeypatch) -> None:
    queue = _FakeQueue()
    monkeypatch.setattr(routes, "get_task_queue", lambda: queue)

    response = asyncio.run(
        routes.run_appeal_analytics(
            request=SimpleNamespace(),
            video_url=None,
            sync=False,
            payload={
                "video_url": "https://www.youtube.com/watch?v=demo",
                "guest_names": ["Guest One"],
            },
            db=db_session,
            settings=test_settings,
        )
    )

    assert response.task_id == "job-run_appeal_analytics"
    assert queue.calls[0][0] == "run_appeal_analytics"
    assert queue.calls[0][1]["video_url"] == "https://www.youtube.com/watch?v=demo"
    assert queue.calls[0][1]["guest_names"] == ["Guest One"]


def test_queue_snapshot_returns_serialized_queue(monkeypatch) -> None:
    queue = _FakeQueue()
    monkeypatch.setattr(routes, "get_task_queue", lambda: queue)

    snapshot = routes.queue_snapshot()

    assert snapshot.current is None
    assert snapshot.queued == []
    assert snapshot.recent == []
