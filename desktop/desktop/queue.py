from __future__ import annotations

import logging
import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class QueueJob:
    id: str
    kind: str
    payload: dict[str, Any]
    status: str = "queued"
    created_at: str = field(default_factory=_utcnow_iso)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class SequentialTaskQueue:
    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[QueueJob, Callable[[], dict[str, Any]]]] = queue.Queue()
        self._jobs: dict[str, QueueJob] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._thread = threading.Thread(
                target=self._worker_loop, name="desktop-task-queue", daemon=True
            )
            self._thread.start()
            self._started = True

    def enqueue(
        self, kind: str, payload: dict[str, Any], func: Callable[[], dict[str, Any]]
    ) -> QueueJob:
        self.start()
        job = QueueJob(id=str(uuid.uuid4()), kind=kind, payload=dict(payload))
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put((job, func))
        return job

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return {
            "current": next((asdict(job) for job in jobs if job.status == "running"), None),
            "queued": [asdict(job) for job in jobs if job.status == "queued"],
            "recent": [asdict(job) for job in jobs if job.status in {"completed", "failed"}][:20],
        }

    def _worker_loop(self) -> None:
        while True:
            job, func = self._queue.get()
            try:
                job.status = "running"
                job.started_at = _utcnow_iso()
                result = func()
                job.result = result
                job.status = "completed"
            except Exception as exc:
                logger.exception("Desktop queue job failed: %s", job.kind)
                job.status = "failed"
                job.error = str(exc)
            finally:
                job.finished_at = _utcnow_iso()
                self._queue.task_done()


_task_queue: SequentialTaskQueue | None = None


def get_task_queue() -> SequentialTaskQueue:
    global _task_queue
    if _task_queue is None:
        _task_queue = SequentialTaskQueue()
    return _task_queue
