from __future__ import annotations

from app.services.runtime_settings import RuntimeSettingsStore


def test_runtime_settings_defaults_follow_env_values(test_settings, tmp_path) -> None:
    data_dir = tmp_path / "data"
    settings = test_settings.model_copy(update={"data_dir": data_dir})
    settings.ensure_directories()
    store = RuntimeSettingsStore(settings)

    state = store.load()

    assert state.beat_enabled is bool(settings.enable_scheduled_runs)
    assert state.beat_time_kyiv == settings.schedule_daily_at
    assert state.openai_chat_model == settings.openai_chat_model
    assert state.embedding_mode == settings.embedding_mode
    assert state.local_embedding_model == settings.local_embedding_model


def test_runtime_settings_save_patch_and_build_pipeline_settings(test_settings, tmp_path) -> None:
    data_dir = tmp_path / "data"
    settings = test_settings.model_copy(update={"data_dir": data_dir})
    settings.ensure_directories()
    store = RuntimeSettingsStore(settings)

    state = store.save_patch(
        {
            "beat_enabled": True,
            "beat_time_kyiv": "09:45",
            "openai_enable_polish_call": True,
            "local_embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "cluster_max_count": 8,
        }
    )

    assert state.beat_enabled is True
    assert state.beat_time_kyiv == "09:45"
    assert state.openai_enable_polish_call is True
    assert state.local_embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert state.cluster_max_count == 8

    effective_settings = store.build_pipeline_settings(state)
    assert effective_settings.openai_enable_polish_call is True
    assert effective_settings.local_embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert effective_settings.cluster_max_count == 8


def test_runtime_settings_build_pipeline_settings_applies_runtime_overrides(
    test_settings, tmp_path
) -> None:
    data_dir = tmp_path / "data"
    settings = test_settings.model_copy(update={"data_dir": data_dir})
    settings.ensure_directories()
    store = RuntimeSettingsStore(settings)
    state = store.save_patch({"beat_enabled": True, "embedding_mode": "local"})

    effective_settings = store.build_pipeline_settings(
        state,
        overrides={
            "embedding_mode": "openai",
            "unknown_field": 123,
        },
    )

    assert effective_settings.embedding_mode == "openai"
