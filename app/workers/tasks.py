"""Celery task definitions for asynchronous and scheduled analysis runs.

Provides tasks to analyze the latest video, a specific video by URL, and a
scheduled beat task that checks whether a daily run should be triggered.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from celery import Task

from app.core.config import Settings, get_settings
from app.db.session import SessionLocal
from app.services.appeal_analytics import AppealAnalyticsService
from app.services.pipeline import DailyRunService
from app.services.runtime_settings import RuntimeSettingsStore
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


class BaseTaskWithRetry(Task):
    """Base Celery task configured with no automatic retries."""

    # Manual-only mode: no automatic retries.
    autoretry_for: tuple[type[BaseException], ...] = ()
    max_retries = 0


def _build_effective_settings(
    runtime_overrides: dict[str, Any] | None = None,
) -> tuple[Settings, RuntimeSettingsStore]:
    settings = get_settings()
    runtime_store = RuntimeSettingsStore(settings)
    runtime_state = runtime_store.load()
    effective_settings = runtime_store.build_pipeline_settings(
        runtime_state, overrides=runtime_overrides
    )
    return effective_settings, runtime_store


@celery_app.task(
    bind=True,
    base=BaseTaskWithRetry,
    name="app.workers.tasks.run_latest_task",
)
def run_latest_task(
    self: Task,
    skip_filtering: bool | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the analysis pipeline for the most recent video in the playlist."""
    settings, _ = _build_effective_settings(runtime_overrides=runtime_overrides)
    db = SessionLocal()
    try:
        service = DailyRunService(settings, db)
        return service.run_latest(
            skip_filtering=skip_filtering,
        )
    finally:
        db.close()


@celery_app.task(
    bind=True,
    base=BaseTaskWithRetry,
    name="app.workers.tasks.run_video_task",
)
def run_video_task(
    self: Task,
    video_url: str,
    skip_filtering: bool | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the analysis pipeline for a specific video URL."""
    settings, _ = _build_effective_settings(runtime_overrides=runtime_overrides)
    db = SessionLocal()
    try:
        service = DailyRunService(settings, db)
        return service.run_video(
            video_url=video_url,
            skip_filtering=skip_filtering,
        )
    finally:
        db.close()


@celery_app.task(
    bind=True,
    base=BaseTaskWithRetry,
    name="app.workers.tasks.run_appeal_analytics_task",
)
def run_appeal_analytics_task(
    self: Task,
    video_url: str | None = None,
    guest_names: list[str] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the appeal analytics pipeline for a video URL or the latest video."""
    settings, _ = _build_effective_settings(runtime_overrides=runtime_overrides)
    db = SessionLocal()
    try:
        service = AppealAnalyticsService(settings, db)
        if video_url:
            return service.run_for_video_url(video_url=video_url, guest_names=guest_names)
        return service.run_for_latest(guest_names=guest_names)
    finally:
        db.close()


@celery_app.task(
    bind=True,
    base=BaseTaskWithRetry,
    name="app.workers.tasks.run_scheduled_latest_task",
)
def run_scheduled_latest_task(self: Task) -> dict[str, Any]:
    """Check whether the daily scheduled run should fire and dispatch it if so."""
    _ = self
    settings = get_settings()
    runtime_store = RuntimeSettingsStore(settings)
    runtime = runtime_store.load()
    now_kyiv = datetime.now(ZoneInfo("Europe/Kyiv"))
    now_hhmm = now_kyiv.strftime("%H:%M")
    if not runtime.beat_enabled:
        return {"triggered": False, "reason": "beat_disabled"}
    if now_hhmm != runtime.beat_time_kyiv:
        return {
            "triggered": False,
            "reason": "time_mismatch",
            "target_time_kyiv": runtime.beat_time_kyiv,
        }
    today_kyiv = now_kyiv.date().isoformat()
    if runtime.last_triggered_kyiv_date == today_kyiv:
        return {"triggered": False, "reason": "already_triggered_today"}

    runtime_store.save_patch({"last_triggered_kyiv_date": today_kyiv})
    overrides = runtime_store.pipeline_overrides(runtime)
    task = run_latest_task.delay(
        runtime_overrides=overrides,
    )
    appeal_task = run_appeal_analytics_task.delay(
        runtime_overrides=overrides,
    )
    logger.info(
        "Beat triggered daily run at %s Kyiv (task_id=%s, appeal_task_id=%s).",
        runtime.beat_time_kyiv,
        task.id,
        appeal_task.id,
    )
    return {"triggered": True, "task_id": task.id, "appeal_task_id": appeal_task.id}
