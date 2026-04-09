from __future__ import annotations

import pytest

from app.core.exceptions import BudgetExceededError
from app.services.budget import BudgetGovernor
from app.services.labeling import OpenAIChatProvider


def _settings_with_openai_limits(test_settings):
    return test_settings.model_copy(
        update={
            "mode": "openai",
            "openai_api_key": "test-openai-key",
            "openai_base_url": "https://api.openai.com/v1",
            "openai_max_calls_per_run": 10,
            "openai_calls_reserved_for_labeling": 4,
            "openai_max_moderation_calls_per_run": 4,
            "openai_max_position_naming_calls_per_run": 2,
        }
    )


def test_openai_provider_blocks_moderation_when_calls_reserved_for_labeling(
    db_session, test_settings
) -> None:
    settings = _settings_with_openai_limits(test_settings)
    provider = OpenAIChatProvider(settings, BudgetGovernor(settings, db_session))
    provider.calls_in_run = 6  # remaining=4, reserve=4 -> moderation must be blocked

    with pytest.raises(BudgetExceededError):
        provider._assert_call_allowed("moderation_borderline")

    stats = provider.get_call_stats()
    assert stats["openai_calls_blocked_reserved_for_labeling"] == 1


def test_openai_provider_enforces_position_naming_task_cap(db_session, test_settings) -> None:
    settings = _settings_with_openai_limits(test_settings)
    provider = OpenAIChatProvider(settings, BudgetGovernor(settings, db_session))
    provider.calls_by_task["position_naming"] = 2

    with pytest.raises(BudgetExceededError):
        provider._assert_call_allowed("position_naming")

    stats = provider.get_call_stats()
    assert stats["openai_calls_blocked_task_quota"] == 1


def test_openai_provider_counts_global_call_limit_as_task_quota_block(
    db_session, test_settings
) -> None:
    settings = _settings_with_openai_limits(test_settings)
    provider = OpenAIChatProvider(settings, BudgetGovernor(settings, db_session))
    provider.calls_in_run = settings.openai_max_calls_per_run

    with pytest.raises(BudgetExceededError):
        provider._assert_call_allowed("cluster_labeling")

    stats = provider.get_call_stats()
    assert stats["openai_calls_blocked_task_quota"] == 1
