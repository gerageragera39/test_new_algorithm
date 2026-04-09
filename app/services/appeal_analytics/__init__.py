"""Appeal Analytics pipeline package.

Provides the :class:`AppealAnalyticsService` that classifies YouTube comments
into four exclusive blocks: toxic, constructive criticism, author appeals,
and constructive questions (with Question Refiner second-pass enrichment).
"""

from app.services.appeal_analytics.runner import AppealAnalyticsService

__all__ = ["AppealAnalyticsService"]
