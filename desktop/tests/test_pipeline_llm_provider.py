from __future__ import annotations

import pytest

from app.core.exceptions import ExternalServiceError
from app.services.budget import BudgetGovernor
from app.services.embeddings import (
    EmbeddingCacheStore,
    LocalSentenceTransformerProvider,
    OpenAIEmbeddingProvider,
)
from app.services.labeling import OpenAIChatProvider
from app.services.pipeline import DailyRunService


def test_build_llm_provider_uses_openai_when_available(
    db_session, test_settings, monkeypatch
) -> None:
    settings = test_settings.model_copy(
        update={
            "openai_api_key": "test-openai-key",
            "openai_chat_model": "gpt-4o-mini",
        }
    )
    service = DailyRunService(settings, db_session)
    budget = BudgetGovernor(settings, db_session)
    monkeypatch.setattr(OpenAIChatProvider, "is_available", lambda self: True)

    provider = service._build_llm_provider(budget)

    assert isinstance(provider, OpenAIChatProvider)
    assert provider.provider_name == "openai"
    assert provider.settings.openai_chat_model == "gpt-4o-mini"


def test_build_llm_provider_raises_when_openai_unavailable(
    db_session, test_settings, monkeypatch
) -> None:
    settings = test_settings.model_copy(
        update={
            "openai_api_key": None,
        }
    )
    service = DailyRunService(settings, db_session)
    budget = BudgetGovernor(settings, db_session)
    monkeypatch.setattr(OpenAIChatProvider, "is_available", lambda self: False)

    with pytest.raises(ExternalServiceError, match="OpenAI chat provider is unavailable"):
        service._build_llm_provider(budget)


def test_build_embedding_service_uses_local_provider_when_embedding_mode_local(
    db_session, test_settings
) -> None:
    settings = test_settings.model_copy(
        update={
            "mode": "openai",
            "embedding_mode": "local",
            "local_embedding_model": "intfloat/multilingual-e5-large",
        }
    )
    service = DailyRunService(settings, db_session)
    budget = BudgetGovernor(settings, db_session)

    embedding_service = service._build_embedding_service(budget)

    assert isinstance(embedding_service.provider, LocalSentenceTransformerProvider)
    assert embedding_service.provider.model_name == "intfloat/multilingual-e5-large"


def test_build_embedding_service_uses_openai_provider_in_free_mode_when_configured(
    db_session, test_settings
) -> None:
    settings = test_settings.model_copy(
        update={
            "mode": "free",
            "embedding_mode": "openai",
            "openai_api_key": "test-openai-key",
            "openai_embedding_model": "text-embedding-3-small",
        }
    )
    service = DailyRunService(settings, db_session)
    budget = BudgetGovernor(settings, db_session)

    embedding_service = service._build_embedding_service(budget)

    assert isinstance(embedding_service.provider, OpenAIEmbeddingProvider)
    assert embedding_service.provider.model_name == "text-embedding-3-small"


def test_resolved_embedding_mode_reflects_explicit_setting(test_settings) -> None:
    local_settings = test_settings.model_copy(update={"embedding_mode": "local"})
    openai_settings = test_settings.model_copy(update={"embedding_mode": "openai"})

    assert local_settings.resolved_embedding_mode == "local"
    assert openai_settings.resolved_embedding_mode == "openai"


def test_local_embedding_provider_uses_instruction_prefix_for_instruction_models(test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "embedding_instruction_mode": "auto",
            "embedding_topic_task_prompt": "Represent the main topic of the comment.",
        }
    )
    provider = LocalSentenceTransformerProvider("Qwen/Qwen3-Embedding-0.6B", settings)

    prepared = provider._prepare_texts(["A comment about sanctions"], task="topic")

    assert prepared[0].startswith("Instruct:")
    assert provider.cache_namespace(task="topic").startswith("topic-instr-")


def test_bge_provider_uses_plain_cache_namespace_when_instruction_mode_off(test_settings) -> None:
    provider = LocalSentenceTransformerProvider("BAAI/bge-m3", test_settings)

    assert provider.cache_namespace(task="topic") == "topic-plain"


def test_embedding_cache_store_sanitizes_windows_unsafe_task_key(
    db_session, test_settings, tmp_path
) -> None:
    settings = test_settings.model_copy(update={"cache_dir": tmp_path / "cache"})
    store = EmbeddingCacheStore(
        settings,
        db_session,
        provider_name="local_st",
        model_name="Qwen/Qwen3-Embedding-0.6B",
    )

    path = store._file_path("topic:abc123")
    db_key = store._db_cache_key(
        "topic:6bb2eeaaa639430d4f6573c20329ec306d48eca4d48049a570683d5e5cdb4524"
    )

    assert ":" not in path.name
    assert path.name.endswith(".json")
    assert len(db_key) == 64


def test_embedding_cache_store_roundtrips_task_scoped_key(
    db_session, test_settings, tmp_path
) -> None:
    settings = test_settings.model_copy(update={"cache_dir": tmp_path / "cache"})
    store = EmbeddingCacheStore(
        settings,
        db_session,
        provider_name="local_st",
        model_name="Qwen/Qwen3-Embedding-0.6B",
    )
    key = "topic:6bb2eeaaa639430d4f6573c20329ec306d48eca4d48049a570683d5e5cdb4524"
    vector = [0.1, 0.2, 0.3]

    store.set(key, vector)
    db_session.commit()

    assert store.get(key) == vector


def test_local_embedding_provider_retries_with_smaller_batch_on_oom(test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "local_embedding_batch_size": 8,
            "local_embedding_oom_fallback_to_cpu": False,
        }
    )
    provider = LocalSentenceTransformerProvider("intfloat/multilingual-e5-large", settings)

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[int, str | None]] = []

        def encode(self, texts, **kwargs):  # noqa: ANN001
            batch_size = kwargs.get("batch_size")
            device = kwargs.get("device")
            self.calls.append((batch_size, device))
            if batch_size > 4:
                raise RuntimeError("CUDA out of memory")
            return [[0.1, 0.2] for _ in texts]

    provider._model = FakeModel()
    provider._active_device = "cuda"

    vectors = provider._encode_with_retries(["one", "two"])

    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert provider._model.calls[0][0] == 8
    assert provider._model.calls[-1][0] == 4


def test_local_embedding_provider_falls_back_to_cpu_on_oom(test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "local_embedding_batch_size": 4,
            "local_embedding_oom_fallback_to_cpu": True,
        }
    )
    provider = LocalSentenceTransformerProvider("Qwen/Qwen3-Embedding-0.6B", settings)

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[int, str | None]] = []

        def encode(self, texts, **kwargs):  # noqa: ANN001
            device = kwargs.get("device")
            batch_size = kwargs.get("batch_size")
            self.calls.append((batch_size, device))
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return [[0.3, 0.4] for _ in texts]

        def to(self, device: str) -> FakeModel:
            self.calls.append((-1, device))
            return self

    provider._model = FakeModel()
    provider._active_device = "cuda"

    vectors = provider._encode_with_retries(["one"])

    assert vectors == [[0.3, 0.4]]
    assert any(call == (-1, "cpu") for call in provider._model.calls)
