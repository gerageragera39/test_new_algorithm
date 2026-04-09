from __future__ import annotations

from app.api import routes
from app.schemas.api import RuntimeSettingsUpdateRequest


def test_get_runtime_settings_uses_defaults(test_settings, tmp_path) -> None:
    settings = test_settings.model_copy(update={"data_dir": tmp_path / "data"})
    settings.ensure_directories()

    response = routes.get_runtime_settings(settings=settings)

    assert response.beat_time_kyiv == settings.schedule_daily_at
    assert response.beat_enabled is bool(settings.enable_scheduled_runs)
    assert response.author_name == settings.author_name
    assert response.embedding_mode == settings.embedding_mode
    assert response.local_embedding_model == settings.local_embedding_model
    assert response.openai_enable_polish_call is settings.openai_enable_polish_call


def test_update_runtime_settings_persists_values(test_settings, tmp_path) -> None:
    settings = test_settings.model_copy(update={"data_dir": tmp_path / "data"})
    settings.ensure_directories()

    payload = RuntimeSettingsUpdateRequest(
        beat_enabled=True,
        beat_time_kyiv="11:20",
        embedding_mode="openai",
        local_embedding_model="Qwen/Qwen3-Embedding-0.6B",
        openai_enable_polish_call=True,
        cluster_max_count=12,
    )
    updated = routes.update_runtime_settings(payload=payload, settings=settings)

    assert updated.beat_enabled is True
    assert updated.beat_time_kyiv == "11:20"
    assert updated.embedding_mode == "openai"
    assert updated.local_embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert updated.openai_enable_polish_call is True
    assert updated.cluster_max_count == 12

    loaded = routes.get_runtime_settings(settings=settings)
    assert loaded.beat_enabled is True
    assert loaded.beat_time_kyiv == "11:20"
    assert loaded.embedding_mode == "openai"
    assert loaded.local_embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert loaded.openai_enable_polish_call is True
    assert loaded.cluster_max_count == 12
