from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

APP_DIR_NAME = "YouTubeIntelDesktop"


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def user_data_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    root = Path(base) / APP_DIR_NAME if base else Path.home() / f".{APP_DIR_NAME}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def config_dir() -> Path:
    path = user_data_root() / "config"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    path = user_data_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_env_path() -> Path:
    return config_dir() / "desktop.env"


def secrets_path() -> Path:
    return config_dir() / "secrets.json"


def sqlite_path() -> Path:
    return data_dir() / "youtube_intel.db"


def reports_dir() -> Path:
    path = data_dir() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    path = data_dir() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = data_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path
