"""OpenAI API compatibility helpers for different model generations.

Provides utility functions to handle parameter differences between GPT-4
and GPT-5 model families, including token limit kwargs and temperature
support, as well as cached token extraction from usage responses.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def is_gpt5_chat_model(model: str) -> bool:
    """Check whether the given model identifier belongs to the GPT-5 family.

    Args:
        model: Model name string (e.g. "gpt-5-mini", "gpt-4o").

    Returns:
        True if the model is a GPT-5 variant, False otherwise.
    """
    return (model or "").strip().lower().startswith("gpt-5")


def build_completion_token_kwargs(model: str, token_limit: int) -> dict[str, Any]:
    """Build the appropriate max-token keyword argument for a chat completion call.

    GPT-5 models use ``max_completion_tokens`` while older models use ``max_tokens``.

    Args:
        model: Model name string.
        token_limit: Maximum number of completion tokens to allow.

    Returns:
        A single-entry dict suitable for unpacking into an API call.
    """
    # GPT-5 chat models require max_completion_tokens instead of max_tokens.
    if is_gpt5_chat_model(model):
        return {"max_completion_tokens": int(token_limit)}
    return {"max_tokens": int(token_limit)}


def build_temperature_kwargs(model: str, temperature: float | None) -> dict[str, Any]:
    """Build temperature kwargs, omitting the parameter for unsupported models.

    GPT-5 chat models should use the default temperature. To avoid passing
    unsupported or ineffective sampling overrides, this helper omits the
    parameter for the GPT-5 family.

    Args:
        model: Model name string.
        temperature: Desired sampling temperature, or None to use the default.

    Returns:
        A dict with the temperature key, or an empty dict if not applicable.
    """
    if temperature is None:
        return {}
    if is_gpt5_chat_model(model):
        return {}
    return {"temperature": float(temperature)}


def build_response_format_kwargs(model: str) -> dict[str, Any]:
    """Build the response_format kwarg for JSON mode.

    GPT-5 models work with json_object but may wrap responses in nested
    structures.  This function returns a consistent ``response_format``
    dict suitable for unpacking into an API call.

    Args:
        model: Model name string.

    Returns:
        A dict with the response_format key.
    """
    return {"response_format": {"type": "json_object"}}


def extract_cached_input_tokens(usage: Any) -> int:
    """Extract the number of cached input tokens from an API usage response.

    Handles both object-attribute and dict-style usage payloads returned by
    different OpenAI SDK versions.

    Args:
        usage: Usage object or dict from an OpenAI API response.

    Returns:
        Number of cached prompt tokens, or 0 if unavailable.
    """
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        return int(getattr(details, "cached_tokens", 0) or 0)
    if isinstance(usage, Mapping):
        raw_details = usage.get("prompt_tokens_details")
        if isinstance(raw_details, Mapping):
            return int(raw_details.get("cached_tokens", 0) or 0)
    return 0
