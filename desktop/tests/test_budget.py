from __future__ import annotations

import pytest

from app.core.exceptions import BudgetExceededError
from app.services.budget import BudgetGovernor


def test_budget_snapshot_tracks_recorded_usage(test_settings, db_session) -> None:
    settings = test_settings.model_copy(update={"openai_max_usd_per_run": 0.01})
    governor = BudgetGovernor(settings, db_session)
    governor.record_usage(
        model="gpt-4o-mini",
        provider="openai_chat",
        tokens_input=100,
        tokens_output=50,
        estimated_cost_usd=0.003,
        meta={"test": True},
    )
    db_session.commit()

    snapshot = governor.get_snapshot()

    assert snapshot.spent_usd == pytest.approx(0.003, abs=1e-6)
    assert snapshot.tokens_used == 150
    assert governor.get_spent_usd() == pytest.approx(0.003, abs=1e-6)


def test_estimate_tokens_upper_bound_is_conservative(test_settings, db_session) -> None:
    governor = BudgetGovernor(test_settings, db_session)
    texts = ["Привет мир", "Это тестовая строка с русскими символами"]

    estimate = governor.estimate_tokens(texts)
    upper_bound = governor.estimate_tokens_upper_bound(texts)

    assert upper_bound >= estimate
    assert upper_bound == sum(len(text.encode("utf-8")) for text in texts)


def test_per_run_budget_cap_is_enforced(test_settings, db_session) -> None:
    settings = test_settings.model_copy(update={"openai_max_usd_per_run": 0.005})
    governor = BudgetGovernor(settings, db_session)
    governor.record_usage(
        model="gpt-4o-mini",
        provider="openai_chat",
        tokens_input=100,
        tokens_output=50,
        estimated_cost_usd=0.004,
        meta={"test": True},
    )

    governor.assert_can_spend(estimated_cost=0.001, estimated_tokens=1)
    with pytest.raises(BudgetExceededError, match="per-run budget exceeded"):
        governor.assert_can_spend(estimated_cost=0.0011, estimated_tokens=1)


def test_zero_run_budget_disables_cap(test_settings, db_session) -> None:
    governor = BudgetGovernor(
        test_settings.model_copy(update={"openai_max_usd_per_run": 0.0}), db_session
    )

    governor.assert_can_spend(estimated_cost=999.0, estimated_tokens=999999)


def test_estimate_chat_cost_supports_gpt_5_2_variants(test_settings, db_session) -> None:
    governor = BudgetGovernor(test_settings, db_session)

    expected = (800 * 1.75 / 1_000_000) + (200 * 0.175 / 1_000_000) + (500 * 14.00 / 1_000_000)
    base_cost = governor.estimate_chat_cost("gpt-5.2", 1000, 500, cached_input_tokens=200)
    dated_cost = governor.estimate_chat_cost(
        "GPT-5.2-2026-02-24", 1000, 500, cached_input_tokens=200
    )

    assert base_cost == pytest.approx(expected, abs=1e-12)
    assert dated_cost == pytest.approx(expected, abs=1e-12)


def test_estimate_chat_cost_applies_cached_pricing_for_gpt_4o_mini(
    test_settings, db_session
) -> None:
    governor = BudgetGovernor(test_settings, db_session)

    expected = (800 * 0.15 / 1_000_000) + (200 * 0.075 / 1_000_000) + (500 * 0.60 / 1_000_000)
    cost = governor.estimate_chat_cost("gpt-4o-mini", 1000, 500, cached_input_tokens=200)

    assert cost == pytest.approx(expected, abs=1e-12)


def test_estimate_chat_cost_supports_gpt_5_4_mini_variants(test_settings, db_session) -> None:
    governor = BudgetGovernor(test_settings, db_session)

    expected = (800 * 0.75 / 1_000_000) + (200 * 0.075 / 1_000_000) + (500 * 4.50 / 1_000_000)
    base_cost = governor.estimate_chat_cost("gpt-5.4-mini", 1000, 500, cached_input_tokens=200)
    dated_cost = governor.estimate_chat_cost(
        "gpt-5.4-mini-2026-04-01", 1000, 500, cached_input_tokens=200
    )

    assert base_cost == pytest.approx(expected, abs=1e-12)
    assert dated_cost == pytest.approx(expected, abs=1e-12)
