"""Custom exception hierarchy for the application.

Defines domain-specific exceptions that map to distinct error conditions such as
external service failures, budget limits, and configuration problems.
"""


class AppError(Exception):
    """Base application exception from which all domain errors inherit."""


class ExternalServiceError(AppError):
    """Failure while calling an external dependency."""


class BudgetExceededError(AppError):
    """OpenAI budget exhausted for the current day."""


class InvalidConfigurationError(AppError):
    """Configuration is missing required values."""
