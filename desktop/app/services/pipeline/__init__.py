"""
Pipeline package — decomposed analysis pipeline for YouTubeAnalyzer.

The pipeline is split into focused modules following the Single Responsibility
Principle.  ``DailyRunService`` in ``runner`` orchestrates the full flow and
delegates to specialised helpers in sibling modules.

Re-exports
----------
DailyRunService : class
    Main entry point imported by ``app.api.routes`` and ``app.workers.tasks``.
"""

from app.services.pipeline.runner import DailyRunService
from app.services.pipeline.text_utils import _UNCERTAIN_TOPIC_LABEL

__all__ = ["DailyRunService", "_UNCERTAIN_TOPIC_LABEL"]
