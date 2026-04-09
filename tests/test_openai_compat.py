from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.openai_compat import (
    build_completion_token_kwargs,
    build_temperature_kwargs,
    extract_cached_input_tokens,
)


def test_build_completion_token_kwargs_uses_gpt5_field() -> None:
    assert build_completion_token_kwargs("gpt-5-mini", 512) == {"max_completion_tokens": 512}
    assert build_completion_token_kwargs("gpt-4o-mini", 512) == {"max_tokens": 512}


def test_build_temperature_kwargs_omits_temperature_for_gpt5() -> None:
    assert build_temperature_kwargs("gpt-5-mini", 0.2) == {}
    assert build_temperature_kwargs("gpt-5.2", 0.7) == {}
    assert build_temperature_kwargs("gpt-4o-mini", 0.2) == {"temperature": 0.2}


def test_extract_cached_input_tokens_supports_mapping_payload() -> None:
    usage = {"prompt_tokens_details": {"cached_tokens": 123}}
    assert extract_cached_input_tokens(usage) == 123


def test_settings_reject_unknown_openai_embedding_model() -> None:
    with pytest.raises(ValueError, match="OPENAI_EMBEDDING_MODEL"):
        Settings(openai_embedding_model="text-embedding-ada-002")


def test_settings_accept_supported_openai_chat_model_alias() -> None:
    settings = Settings(openai_chat_model="gpt-5.2-2026-02-24")
    assert settings.openai_chat_model == "gpt-5.2-2026-02-24"


def test_settings_accept_supported_openai_chat_model_gpt_5_4_alias() -> None:
    settings = Settings(openai_chat_model="gpt-5.4-mini-2026-04-01")
    assert settings.openai_chat_model == "gpt-5.4-mini-2026-04-01"
