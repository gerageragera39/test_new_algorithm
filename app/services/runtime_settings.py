"""Persistent runtime settings store with JSON file backing.

Manages mutable application settings (schedule, report limits) that can be
changed at runtime via the API without restarting the application.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.utils import utcnow, write_json

_TIME_HH_MM_RE = re.compile(r"^\d{2}:\d{2}$")
_DATE_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


_VALID_CHAT_MODELS = {
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-mini",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4-pro",
}
_VALID_EMBEDDING_MODES = {"local", "openai"}


@dataclass(frozen=True)
class RuntimeSettingsState:
    beat_enabled: bool
    beat_time_kyiv: str
    last_triggered_kyiv_date: str | None
    updated_at: datetime | None
    # Pipeline settings (overridable at runtime)
    author_name: str
    openai_chat_model: str
    embedding_mode: str
    local_embedding_model: str
    cluster_max_count: int
    max_comments_per_video: int
    youtube_include_replies: bool
    openai_enable_polish_call: bool


class RuntimeSettingsStore:
    """Reads and writes runtime-configurable settings from a JSON file.

    Provides safe loading with type coercion and defaults, and atomic
    patch-based updates that merge with existing values.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.data_dir / "runtime_settings.json"

    def defaults(self) -> dict[str, Any]:
        return {
            "beat_enabled": bool(self.settings.enable_scheduled_runs),
            "beat_time_kyiv": self.settings.schedule_daily_at,
            "last_triggered_kyiv_date": None,
            "updated_at": None,
            "author_name": self.settings.author_name,
            "openai_chat_model": self.settings.openai_chat_model,
            "embedding_mode": self.settings.embedding_mode,
            "local_embedding_model": self.settings.local_embedding_model,
            "cluster_max_count": self.settings.cluster_max_count,
            "max_comments_per_video": self.settings.max_comments_per_video,
            "youtube_include_replies": self.settings.youtube_include_replies,
            "openai_enable_polish_call": self.settings.openai_enable_polish_call,
        }

    def load(self) -> RuntimeSettingsState:
        """Load current runtime settings from disk, falling back to defaults.

        Returns:
            A RuntimeSettingsState with values merged from the JSON file
            and application defaults.
        """
        defaults = self.defaults()
        raw = self._read_json_dict(self.path)
        merged = {**defaults, **raw}
        return self._to_state(merged, defaults=defaults)

    def save_patch(self, patch: dict[str, Any]) -> RuntimeSettingsState:
        """Apply a partial update to runtime settings and persist to disk.

        Args:
            patch: Dictionary of setting keys and new values to merge.

        Returns:
            The updated RuntimeSettingsState after applying the patch.
        """
        current = self.load()
        merged: dict[str, Any] = {
            "beat_enabled": current.beat_enabled,
            "beat_time_kyiv": current.beat_time_kyiv,
            "last_triggered_kyiv_date": current.last_triggered_kyiv_date,
            "updated_at": current.updated_at.isoformat() if current.updated_at else None,
            "author_name": current.author_name,
            "openai_chat_model": current.openai_chat_model,
            "embedding_mode": current.embedding_mode,
            "local_embedding_model": current.local_embedding_model,
            "cluster_max_count": current.cluster_max_count,
            "max_comments_per_video": current.max_comments_per_video,
            "youtube_include_replies": current.youtube_include_replies,
            "openai_enable_polish_call": current.openai_enable_polish_call,
        }
        for key, value in patch.items():
            merged[key] = value
        merged["updated_at"] = utcnow().isoformat()
        state = self._to_state(merged, defaults=self.defaults())
        write_json(self.path, self._state_to_dict(state))
        return state

    def pipeline_overrides(self, runtime: RuntimeSettingsState | None = None) -> dict[str, Any]:
        if runtime is None:
            return {}
        return {
            "author_name": runtime.author_name,
            "openai_chat_model": runtime.openai_chat_model,
            "embedding_mode": runtime.embedding_mode,
            "local_embedding_model": runtime.local_embedding_model,
            "cluster_max_count": runtime.cluster_max_count,
            "max_comments_per_video": runtime.max_comments_per_video,
            "youtube_include_replies": runtime.youtube_include_replies,
            "openai_enable_polish_call": runtime.openai_enable_polish_call,
        }

    def build_pipeline_settings(
        self,
        runtime: RuntimeSettingsState | None = None,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> Settings:
        update = self.pipeline_overrides(runtime)
        if isinstance(overrides, dict):
            for key in update:
                if key in overrides:
                    update[key] = overrides[key]
        return self.settings.model_copy(update=update)

    def _to_state(
        self, payload: dict[str, Any], *, defaults: dict[str, Any]
    ) -> RuntimeSettingsState:
        updated_at = self._parse_datetime(payload.get("updated_at"))
        chat_model = self._coerce_choice(
            payload.get("openai_chat_model"),
            defaults["openai_chat_model"],
            _VALID_CHAT_MODELS,
        )
        embedding_mode = self._coerce_choice(
            payload.get("embedding_mode"),
            defaults["embedding_mode"],
            _VALID_EMBEDDING_MODES,
        )
        return RuntimeSettingsState(
            beat_enabled=self._coerce_bool(payload.get("beat_enabled"), defaults["beat_enabled"]),
            beat_time_kyiv=self._coerce_time(
                payload.get("beat_time_kyiv"), defaults["beat_time_kyiv"]
            ),
            last_triggered_kyiv_date=self._coerce_date(payload.get("last_triggered_kyiv_date")),
            updated_at=updated_at,
            author_name=str(payload.get("author_name") or defaults["author_name"] or ""),
            openai_chat_model=chat_model,
            embedding_mode=embedding_mode,
            local_embedding_model=str(
                payload.get("local_embedding_model") or defaults["local_embedding_model"] or ""
            ),
            cluster_max_count=self._coerce_int(
                payload.get("cluster_max_count"),
                defaults["cluster_max_count"],
                min_value=0,
                max_value=50,
            ),
            max_comments_per_video=self._coerce_int(
                payload.get("max_comments_per_video"),
                defaults["max_comments_per_video"],
                min_value=50,
                max_value=10000,
            ),
            youtube_include_replies=self._coerce_bool(
                payload.get("youtube_include_replies"),
                defaults["youtube_include_replies"],
            ),
            openai_enable_polish_call=self._coerce_bool(
                payload.get("openai_enable_polish_call"),
                defaults["openai_enable_polish_call"],
            ),
        )

    @staticmethod
    def _state_to_dict(state: RuntimeSettingsState) -> dict[str, Any]:
        return {
            "beat_enabled": state.beat_enabled,
            "beat_time_kyiv": state.beat_time_kyiv,
            "last_triggered_kyiv_date": state.last_triggered_kyiv_date,
            "updated_at": state.updated_at.isoformat() if state.updated_at else None,
            "author_name": state.author_name,
            "openai_chat_model": state.openai_chat_model,
            "embedding_mode": state.embedding_mode,
            "local_embedding_model": state.local_embedding_model,
            "cluster_max_count": state.cluster_max_count,
            "max_comments_per_video": state.max_comments_per_video,
            "youtube_include_replies": state.youtube_include_replies,
            "openai_enable_polish_call": state.openai_enable_polish_call,
        }

    def _read_json_dict(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _coerce_bool(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value == 0:
                return False
            if value == 1:
                return True
            return default
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "1", "yes", "y", "on"}:
                return True
            if text in {"false", "0", "no", "n", "off"}:
                return False
        return default

    def _coerce_int(self, value: Any, default: int, *, min_value: int, max_value: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            return default
        return min(max(parsed, min_value), max_value)

    def _coerce_float(
        self, value: Any, default: float, *, min_value: float, max_value: float
    ) -> float:
        try:
            parsed = float(value)
        except Exception:
            return default
        return min(max(parsed, min_value), max_value)

    @staticmethod
    def _coerce_choice(value: Any, default: str, valid: set[str]) -> str:
        if not isinstance(value, str):
            return default
        text = value.strip().lower()
        return text if text in valid else default

    def _coerce_time(self, value: Any, default: str) -> str:
        if not isinstance(value, str):
            return default
        text = value.strip()
        if not _TIME_HH_MM_RE.fullmatch(text):
            return default
        hour = int(text[:2])
        minute = int(text[3:5])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
        return default

    def _coerce_date(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not _DATE_ISO_RE.fullmatch(text):
            return None
        return text

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None
