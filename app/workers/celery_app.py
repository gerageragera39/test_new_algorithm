"""Celery application instance and broker configuration.

Initializes the Celery app with JSON serialization, beat schedule for the
daily run checker, and platform-specific worker pool settings for Windows.
"""

from __future__ import annotations

import os

from celery import Celery
from celery.schedules import crontab

from app.core.config import get_settings

settings = get_settings()
is_windows = os.name == "nt"

celery_app = Celery(
    "youtube_daily_comments_intelligence",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

conf: dict[str, object] = {
    "timezone": settings.app_timezone,
    "enable_utc": True,
    "task_serializer": "json",
    "result_serializer": "json",
    "accept_content": ["json"],
    "task_acks_late": False,
    "worker_prefetch_multiplier": 1,
    "broker_connection_retry_on_startup": True,
    "beat_schedule": {
        "runtime-daily-latest-video-checker": {
            "task": "app.workers.tasks.run_scheduled_latest_task",
            "schedule": crontab(minute="*"),
            "args": (),
        }
    },
}
if is_windows:
    conf["worker_pool"] = "solo"
    conf["worker_concurrency"] = 1

celery_app.conf.update(
    **conf,
)
