from __future__ import annotations

from app.api import routes
from app.services.runtime_settings import RuntimeSettingsStore


class _TaskResult:
    def __init__(self, task_id: str) -> None:
        self.id = task_id


def test_run_latest_async_prefers_query_skip_filtering(
    db_session,
    test_settings,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_delay(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return _TaskResult("task-latest-1")

    monkeypatch.setattr(routes.run_latest_task, "delay", fake_delay)

    response = routes.run_latest(
        sync=False,
        skip_filtering="true",
        payload={"skip_filtering": False},
        db=db_session,
        settings=test_settings,
    )

    assert response.task_id == "task-latest-1"
    assert captured["skip_filtering"] is True


def test_run_latest_sync_passes_skip_filtering_override_to_service(
    db_session,
    test_settings,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeDailyRunService:
        def __init__(self, settings, db) -> None:  # noqa: ANN001
            _ = settings
            _ = db

        def run_latest(self, *, skip_filtering: bool | None = None) -> dict[str, object]:
            captured["skip_filtering"] = skip_filtering
            return {"run_id": 42, "video_id": "vid42"}

    monkeypatch.setattr(routes, "DailyRunService", FakeDailyRunService)

    response = routes.run_latest(
        sync=True,
        skip_filtering=False,
        payload=None,
        db=db_session,
        settings=test_settings,
    )

    assert response.task_id == "sync-42"
    assert captured["skip_filtering"] is False


def test_run_video_async_prefers_query_skip_filtering(
    db_session,
    test_settings,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_delay(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return _TaskResult("task-video-1")

    monkeypatch.setattr(routes.run_video_task, "delay", fake_delay)

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

    assert response.task_id == "task-video-1"
    assert captured["video_url"] == "https://www.youtube.com/watch?v=testvideo1"
    assert captured["skip_filtering"] is True


def test_run_latest_async_passes_runtime_overrides_from_dashboard_state(
    db_session,
    test_settings,
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    settings = test_settings.model_copy(update={"data_dir": data_dir})
    settings.ensure_directories()
    store = RuntimeSettingsStore(settings)
    store.save_patch({"beat_enabled": True, "cluster_max_count": 7})
    captured: dict[str, object] = {}

    def fake_delay(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return _TaskResult("task-latest-overrides")

    monkeypatch.setattr(routes.run_latest_task, "delay", fake_delay)

    response = routes.run_latest(
        sync=False,
        skip_filtering=None,
        payload=None,
        db=db_session,
        settings=settings,
    )

    assert response.task_id == "task-latest-overrides"
    runtime_overrides = captured.get("runtime_overrides")
    assert isinstance(runtime_overrides, dict)
    assert runtime_overrides["cluster_max_count"] == 7
