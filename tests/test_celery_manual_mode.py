from __future__ import annotations

from app.workers.celery_app import celery_app
from app.workers.tasks import BaseTaskWithRetry


def test_celery_manual_mode_has_no_acks_late() -> None:
    assert bool(celery_app.conf.task_acks_late) is False


def test_celery_has_runtime_schedule_checker() -> None:
    schedule = dict(celery_app.conf.beat_schedule or {})
    assert "runtime-daily-latest-video-checker" in schedule


def test_celery_manual_mode_has_no_autoretry() -> None:
    assert BaseTaskWithRetry.autoretry_for == ()
    assert BaseTaskWithRetry.max_retries == 0
