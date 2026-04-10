from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from desktop.envfiles import parse_env_file, write_env_file
from desktop.paths import resource_root, runtime_env_path, secrets_path
from desktop.security import load_secret_payload, save_secret_payload

DEFAULTS_FILE = resource_root() / "desktop" / "defaults.env"


@dataclass(frozen=True)
class SetupStatus:
    is_configured: bool
    has_openai_api_key: bool
    has_youtube_api_key: bool
    has_playlist_id: bool
    has_youtube_oauth_client_id: bool
    has_youtube_oauth_client_secret: bool
    has_youtube_oauth_refresh_token: bool
    runtime_env_path: str


def ensure_runtime_env_exists() -> None:
    runtime_path = runtime_env_path()
    if runtime_path.exists():
        return
    defaults = parse_env_file(DEFAULTS_FILE)
    defaults.setdefault("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    defaults.pop("CELERY_BROKER_URL", None)
    defaults.pop("CELERY_RESULT_BACKEND", None)
    defaults.pop("ENABLE_SCHEDULED_RUNS", None)
    defaults.pop("SCHEDULE_DAILY_AT", None)
    write_env_file(runtime_path, defaults)


def get_setup_status() -> SetupStatus:
    ensure_runtime_env_exists()
    secrets = load_secret_payload(secrets_path())
    runtime = parse_env_file(runtime_env_path())
    has_openai = bool(str(secrets.get("OPENAI_API_KEY", "")).strip())
    has_youtube = bool(str(secrets.get("YOUTUBE_API_KEY", "")).strip())
    has_playlist = bool(str(runtime.get("YOUTUBE_PLAYLIST_ID", "")).strip())
    has_oauth_client_id = bool(str(secrets.get("YOUTUBE_OAUTH_CLIENT_ID", "")).strip())
    has_oauth_client_secret = bool(str(secrets.get("YOUTUBE_OAUTH_CLIENT_SECRET", "")).strip())
    has_oauth_refresh_token = bool(str(secrets.get("YOUTUBE_OAUTH_REFRESH_TOKEN", "")).strip())
    return SetupStatus(
        is_configured=has_openai and has_youtube,
        has_openai_api_key=has_openai,
        has_youtube_api_key=has_youtube,
        has_playlist_id=has_playlist,
        has_youtube_oauth_client_id=has_oauth_client_id,
        has_youtube_oauth_client_secret=has_oauth_client_secret,
        has_youtube_oauth_refresh_token=has_oauth_refresh_token,
        runtime_env_path=str(runtime_env_path()),
    )


def _save_runtime_patch(patch: dict[str, str]) -> None:
    ensure_runtime_env_exists()
    runtime_path = runtime_env_path()
    current = parse_env_file(runtime_path)
    current.update({key: value for key, value in patch.items() if value is not None})
    write_env_file(runtime_path, current)


def save_first_run_setup(
    *,
    openai_api_key: str,
    youtube_api_key: str,
    youtube_playlist_id: str | None = None,
    youtube_oauth_client_id: str | None = None,
    youtube_oauth_client_secret: str | None = None,
    youtube_oauth_refresh_token: str | None = None,
) -> SetupStatus:
    ensure_runtime_env_exists()
    from app.core.config import clear_settings_cache

    current = load_secret_payload(secrets_path())
    current.update({
        "OPENAI_API_KEY": openai_api_key.strip(),
        "YOUTUBE_API_KEY": youtube_api_key.strip(),
    })
    if youtube_oauth_client_id is not None and youtube_oauth_client_id.strip():
        current["YOUTUBE_OAUTH_CLIENT_ID"] = youtube_oauth_client_id.strip()
    if youtube_oauth_client_secret is not None and youtube_oauth_client_secret.strip():
        current["YOUTUBE_OAUTH_CLIENT_SECRET"] = youtube_oauth_client_secret.strip()
    if youtube_oauth_refresh_token is not None and youtube_oauth_refresh_token.strip():
        current["YOUTUBE_OAUTH_REFRESH_TOKEN"] = youtube_oauth_refresh_token.strip()
    save_secret_payload(secrets_path(), current)
    if youtube_playlist_id is not None:
        _save_runtime_patch({"YOUTUBE_PLAYLIST_ID": youtube_playlist_id.strip()})
    clear_settings_cache()
    return get_setup_status()


def update_setup(
    *,
    openai_api_key: str | None = None,
    youtube_api_key: str | None = None,
    youtube_playlist_id: str | None = None,
    youtube_oauth_client_id: str | None = None,
    youtube_oauth_client_secret: str | None = None,
    youtube_oauth_refresh_token: str | None = None,
) -> SetupStatus:
    ensure_runtime_env_exists()
    from app.core.config import clear_settings_cache

    current = load_secret_payload(secrets_path())
    if openai_api_key is not None and openai_api_key.strip():
        current["OPENAI_API_KEY"] = openai_api_key.strip()
    if youtube_api_key is not None and youtube_api_key.strip():
        current["YOUTUBE_API_KEY"] = youtube_api_key.strip()
    if youtube_oauth_client_id is not None and youtube_oauth_client_id.strip():
        current["YOUTUBE_OAUTH_CLIENT_ID"] = youtube_oauth_client_id.strip()
    if youtube_oauth_client_secret is not None and youtube_oauth_client_secret.strip():
        current["YOUTUBE_OAUTH_CLIENT_SECRET"] = youtube_oauth_client_secret.strip()
    if youtube_oauth_refresh_token is not None and youtube_oauth_refresh_token.strip():
        current["YOUTUBE_OAUTH_REFRESH_TOKEN"] = youtube_oauth_refresh_token.strip()
    save_secret_payload(secrets_path(), current)
    if youtube_playlist_id is not None:
        _save_runtime_patch({"YOUTUBE_PLAYLIST_ID": youtube_playlist_id.strip()})
    clear_settings_cache()
    return get_setup_status()


def load_effective_payload() -> dict[str, Any]:
    ensure_runtime_env_exists()
    payload = parse_env_file(DEFAULTS_FILE)
    payload.update(parse_env_file(runtime_env_path()))
    payload.update(load_secret_payload(secrets_path()))
    payload["OPENAI_CHAT_MODEL"] = "gpt-4o-mini"
    return payload
