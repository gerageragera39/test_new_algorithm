"""OpenAI endpoint validation and routing utilities.

Determines whether the configured OpenAI base URL points to the official
API or a custom proxy, and enforces endpoint restrictions when configured.
"""

from __future__ import annotations

from urllib.parse import urlparse

from app.core.config import Settings
from app.core.exceptions import InvalidConfigurationError

_OFFICIAL_OPENAI_HOST = "api.openai.com"


def openai_base_url_host(base_url: str) -> str:
    """Extract and normalize the hostname from an OpenAI base URL.

    Args:
        base_url: Full base URL string (e.g. "https://api.openai.com/v1").

    Returns:
        Lowercased hostname, or empty string if parsing fails.
    """
    parsed = urlparse(str(base_url or "").strip())
    return (parsed.hostname or "").strip().lower()


def openai_endpoint_mode(base_url: str) -> str:
    """Determine whether the base URL targets the official OpenAI API or a custom endpoint.

    Args:
        base_url: Full base URL string.

    Returns:
        "official" if the host is api.openai.com, "custom" otherwise.
    """
    host = openai_base_url_host(base_url)
    if host == _OFFICIAL_OPENAI_HOST:
        return "official"
    return "custom"


def ensure_openai_endpoint_allowed(settings: Settings) -> tuple[str, str]:
    """Validate the OpenAI endpoint against configuration restrictions.

    Args:
        settings: Application settings containing endpoint configuration.

    Returns:
        A tuple of (host, mode) where mode is "official" or "custom".

    Raises:
        InvalidConfigurationError: If the official endpoint is required but a
            custom host is configured.
    """
    base_url = str(settings.openai_base_url).strip()
    host = openai_base_url_host(base_url)
    mode = openai_endpoint_mode(base_url)
    if settings.openai_require_official_endpoint and host != _OFFICIAL_OPENAI_HOST:
        msg = (
            "OPENAI_REQUIRE_OFFICIAL_ENDPOINT=true but OPENAI_BASE_URL host is "
            f"'{host or '-'}', expected '{_OFFICIAL_OPENAI_HOST}'."
        )
        raise InvalidConfigurationError(msg)
    return host, mode
