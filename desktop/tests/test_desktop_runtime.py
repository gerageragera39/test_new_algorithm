from __future__ import annotations

import time
from pathlib import Path

from app.core.config import Settings, clear_settings_cache, get_settings
from app.main import app as fastapi_app
from app.services.runtime_settings import RuntimeSettingsStore
from desktop import launcher
from desktop.bootstrap import (
    get_setup_status,
    load_effective_payload,
    save_first_run_setup,
    update_setup,
)
from desktop.paths import user_data_root
from desktop.queue import SequentialTaskQueue


def build_settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    return Settings(
        data_dir=data_dir,
        cache_dir=data_dir / "cache",
        raw_dir=data_dir / "raw",
        reports_dir=data_dir / "reports",
        database_url=f"sqlite:///{(data_dir / 'test.db').as_posix()}",
    )


def test_runtime_settings_persists_model_overrides(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    store = RuntimeSettingsStore(settings)

    state = store.save_patch(
        {
            "openai_chat_model": "gpt-5.2",
            "embedding_mode": "openai",
        }
    )

    assert state.openai_chat_model == "gpt-5.2"
    assert state.embedding_mode == "openai"
    assert store.load().openai_chat_model == "gpt-5.2"


def test_bootstrap_saves_secrets_into_local_appdata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    user_data_root.cache_clear()

    status_before = get_setup_status()
    assert status_before.is_configured is False

    status_after = save_first_run_setup(
        openai_api_key="sk-test",
        youtube_api_key="yt-test",
        youtube_playlist_id="PL_demo",
        youtube_oauth_client_id="oauth-client",
        youtube_oauth_client_secret="oauth-secret",
        youtube_oauth_refresh_token="oauth-refresh",
    )
    payload = load_effective_payload()

    assert status_after.is_configured is True
    assert status_after.has_playlist_id is True
    assert status_after.has_youtube_oauth_client_id is True
    assert status_after.has_youtube_oauth_client_secret is True
    assert status_after.has_youtube_oauth_refresh_token is True
    assert payload["OPENAI_API_KEY"] == "sk-test"
    assert payload["YOUTUBE_API_KEY"] == "yt-test"
    assert payload["YOUTUBE_PLAYLIST_ID"] == "PL_demo"


def test_bootstrap_can_store_optional_youtube_oauth_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    user_data_root.cache_clear()

    status = save_first_run_setup(
        openai_api_key="sk-test",
        youtube_api_key="yt-test",
        youtube_oauth_client_id="cid",
        youtube_oauth_client_secret="secret",
        youtube_oauth_refresh_token="refresh",
    )
    payload = load_effective_payload()

    assert status.has_youtube_oauth_client_id is True
    assert status.has_youtube_oauth_client_secret is True
    assert status.has_youtube_oauth_refresh_token is True
    assert payload["YOUTUBE_OAUTH_CLIENT_ID"] == "cid"
    assert payload["YOUTUBE_OAUTH_CLIENT_SECRET"] == "secret"
    assert payload["YOUTUBE_OAUTH_REFRESH_TOKEN"] == "refresh"


def test_empty_env_vars_do_not_override_saved_secrets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    user_data_root.cache_clear()

    save_first_run_setup(openai_api_key="sk-test", youtube_api_key="yt-test")
    clear_settings_cache()
    settings = get_settings()

    assert settings.openai_api_key == "sk-test"
    assert settings.youtube_api_key == "yt-test"


def test_sequential_task_queue_runs_jobs_in_order() -> None:
    task_queue = SequentialTaskQueue()
    events: list[str] = []

    def first_job() -> dict[str, str]:
        events.append("first-start")
        time.sleep(0.05)
        events.append("first-end")
        return {"ok": "first"}

    def second_job() -> dict[str, str]:
        events.append("second-start")
        events.append("second-end")
        return {"ok": "second"}

    first = task_queue.enqueue("first", {}, first_job)
    second = task_queue.enqueue("second", {}, second_job)

    deadline = time.time() + 2
    while time.time() < deadline:
        snapshot = task_queue.snapshot()
        recent_ids = {job["id"] for job in snapshot["recent"]}
        if first.id in recent_ids and second.id in recent_ids:
            break
        time.sleep(0.02)

    assert events == ["first-start", "first-end", "second-start", "second-end"]


def test_update_setup_can_rotate_keys_and_playlist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    user_data_root.cache_clear()

    save_first_run_setup(
        openai_api_key="sk-old", youtube_api_key="yt-old", youtube_playlist_id="PL_old"
    )
    status = update_setup(
        openai_api_key="sk-new",
        youtube_api_key="yt-new",
        youtube_playlist_id="PL_new",
        youtube_oauth_client_id="oauth-client-new",
        youtube_oauth_client_secret="oauth-secret-new",
        youtube_oauth_refresh_token="oauth-refresh-new",
    )
    payload = load_effective_payload()

    assert status.is_configured is True
    assert status.has_playlist_id is True
    assert status.has_youtube_oauth_client_id is True
    assert status.has_youtube_oauth_client_secret is True
    assert status.has_youtube_oauth_refresh_token is True
    assert payload["OPENAI_API_KEY"] == "sk-new"
    assert payload["YOUTUBE_API_KEY"] == "yt-new"
    assert payload["YOUTUBE_PLAYLIST_ID"] == "PL_new"
    assert payload["YOUTUBE_OAUTH_CLIENT_ID"] == "oauth-client-new"
    assert payload["YOUTUBE_OAUTH_CLIENT_SECRET"] == "oauth-secret-new"
    assert payload["YOUTUBE_OAUTH_REFRESH_TOKEN"] == "oauth-refresh-new"


def test_launcher_passes_imported_fastapi_app_to_uvicorn(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, config: object) -> None:
            captured["server_config"] = config

        def run(self) -> None:
            captured["server_ran"] = True

    class FakeThread:
        def __init__(self, target: object, daemon: bool) -> None:
            captured["thread_target"] = target
            captured["thread_daemon"] = daemon

        def start(self) -> None:
            captured["thread_started"] = True

        def join(self) -> None:
            captured["thread_joined"] = True

    def fake_config(app: object, **kwargs: object) -> object:
        captured["config_app"] = app
        captured["config_kwargs"] = kwargs
        return {"app": app, "kwargs": kwargs}

    monkeypatch.setattr(launcher.uvicorn, "Config", fake_config)
    monkeypatch.setattr(launcher.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(launcher.threading, "Thread", FakeThread)
    monkeypatch.setattr(launcher, "_wait_for_port", lambda host, port: False)
    monkeypatch.setattr(
        launcher.webbrowser, "open", lambda url: captured.setdefault("opened_url", url)
    )

    launcher.main()

    assert captured["config_app"] is fastapi_app
    assert captured["config_kwargs"] == {"host": "127.0.0.1", "port": 8765, "log_level": "info"}
    assert captured["thread_started"] is True
    assert captured["thread_joined"] is True
