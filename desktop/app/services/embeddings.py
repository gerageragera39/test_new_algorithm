"""Embedding providers and caching layer.

Supports local sentence-transformers and OpenAI embedding API providers.
Embeddings are cached in both the database and the local filesystem for
reuse across runs. The service now supports task-aware text preparation so
that topic clustering can use a more stable semantic representation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import InvalidConfigurationError
from app.db.models import EmbeddingCache
from app.services.budget import BudgetGovernor
from app.services.openai_endpoint import ensure_openai_endpoint_allowed

_E5_QUERY_PREFIX = "query: "
_E5_MODEL_MARKERS = ("e5-large", "e5-base", "e5-small", "multilingual-e5")
_INSTRUCTION_AWARE_MODEL_MARKERS = (
    "qwen",
    "gte",
    "instruct",
)


class EmbeddingProvider(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def embed_texts(self, texts: Sequence[str], *, task: str = "topic") -> list[list[float]]:
        """Return embeddings for all texts preserving order."""

    def cache_namespace(self, *, task: str = "topic") -> str:
        return task


class LocalSentenceTransformerProvider(EmbeddingProvider):
    provider_name = "local_st"

    def __init__(self, model_name: str, settings: Settings) -> None:
        self.model_name = model_name
        self.settings = settings
        self._model = None
        self._active_device = "cpu"
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        self._active_device = self._resolve_device(torch)
        if self._active_device == "cuda" and self.settings.local_embedding_low_vram_mode:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        model_kwargs = self._build_model_kwargs(torch)
        self._model = SentenceTransformer(
            self.model_name,
            device=self._active_device,
            model_kwargs=model_kwargs or None,
        )
        max_seq_length = int(self.settings.local_embedding_max_seq_length)
        if max_seq_length > 0 and hasattr(self._model, "max_seq_length"):
            self._model.max_seq_length = min(int(self._model.max_seq_length), max_seq_length)

    def _resolve_device(self, torch_module: object) -> str:
        preference = self.settings.local_embedding_device
        cuda_available = bool(torch_module.cuda.is_available())
        if preference == "cpu":
            return "cpu"
        if preference == "cuda":
            return "cuda" if cuda_available else "cpu"
        return "cuda" if cuda_available else "cpu"

    def _build_model_kwargs(self, torch_module: object) -> dict[str, object]:
        if self._active_device != "cuda" or not self.settings.local_embedding_low_vram_mode:
            return {}
        dtype = getattr(torch_module, "float16", None)
        if dtype is None:
            return {}
        return {"torch_dtype": dtype}

    def _suggest_batch_size(self) -> int:
        configured = max(1, int(self.settings.local_embedding_batch_size))
        if self._active_device != "cuda":
            return configured
        if not self.settings.local_embedding_low_vram_mode:
            return configured
        name = self.model_name.lower()
        if "qwen" in name:
            return min(configured, 4)
        if "bge-m3" in name:
            return min(configured, 6)
        return configured

    def _clear_cuda_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _is_cuda_oom(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "cuda out of memory" in message or "cublas" in message and "alloc" in message

    def _encode_with_retries(self, prepared: Sequence[str]) -> list[list[float]]:
        assert self._model is not None
        batch_size = self._suggest_batch_size()
        current_device = self._active_device

        while True:
            try:
                embeddings = self._model.encode(
                    list(prepared),
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    device=current_device,
                )
                return [
                    vector.tolist() if hasattr(vector, "tolist") else list(vector)
                    for vector in embeddings
                ]
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc):
                    raise
                self.logger.warning(
                    "Embedding OOM on device=%s model=%s batch_size=%s; retrying with safer settings.",
                    current_device,
                    self.model_name,
                    batch_size,
                )
                self._clear_cuda_cache()
                if current_device == "cuda" and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    continue
                if (
                    current_device == "cuda"
                    and self.settings.local_embedding_oom_fallback_to_cpu
                ):
                    current_device = "cpu"
                    self._model.to("cpu")
                    self._active_device = "cpu"
                    batch_size = max(4, int(self.settings.local_embedding_batch_size))
                    continue
                raise

    def _is_e5_model(self) -> bool:
        name_lower = self.model_name.lower()
        return any(marker in name_lower for marker in _E5_MODEL_MARKERS)

    def _uses_instruction_prefix(self) -> bool:
        mode = self.settings.embedding_instruction_mode
        if mode == "off":
            return False
        if mode == "force":
            return True
        name_lower = self.model_name.lower()
        return any(marker in name_lower for marker in _INSTRUCTION_AWARE_MODEL_MARKERS)

    def _prepare_texts(self, texts: Sequence[str], *, task: str) -> list[str]:
        prepared = [" ".join(str(text).split()).strip() for text in texts]
        if self._is_e5_model():
            return [f"{_E5_QUERY_PREFIX}{text}" for text in prepared]
        if task == "topic" and self._uses_instruction_prefix():
            prompt = self.settings.embedding_topic_task_prompt.strip()
            return [f"Instruct: {prompt}\nText: {text}" for text in prepared]
        return prepared

    def cache_namespace(self, *, task: str = "topic") -> str:
        parts = [task]
        if self._is_e5_model():
            parts.append("e5query")
        elif task == "topic" and self._uses_instruction_prefix():
            prompt_hash = hashlib.sha256(
                self.settings.embedding_topic_task_prompt.encode("utf-8")
            ).hexdigest()[:8]
            parts.append(f"instr-{prompt_hash}")
        else:
            parts.append("plain")
        return "-".join(parts)

    def embed_texts(self, texts: Sequence[str], *, task: str = "topic") -> list[list[float]]:
        self._load_model()
        prepared = self._prepare_texts(texts, task=task)
        return self._encode_with_retries(prepared)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    provider_name = "openai"

    def __init__(self, settings: Settings, budget: BudgetGovernor) -> None:
        self.settings = settings
        self.model_name = settings.openai_embedding_model
        self.budget = budget
        self.logger = logging.getLogger(self.__class__.__name__)
        if not settings.openai_api_key:
            msg = "OPENAI_API_KEY is required when EMBEDDING_MODE=openai."
            raise InvalidConfigurationError(msg)
        self.base_url_host, self.endpoint_mode = ensure_openai_endpoint_allowed(settings)
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    def embed_texts(self, texts: Sequence[str], *, task: str = "topic") -> list[list[float]]:
        if not texts:
            return []
        _ = task
        if self.settings.openai_hard_budget_enforced:
            estimated_tokens = self.budget.estimate_tokens_upper_bound(list(texts))
        else:
            estimated_tokens = self.budget.estimate_tokens(list(texts))
        estimated_cost = self.budget.estimate_embedding_cost(self.model_name, estimated_tokens)
        self.budget.assert_can_spend(
            estimated_cost=estimated_cost,
            estimated_tokens=estimated_tokens,
        )

        response = self.client.embeddings.create(model=self.model_name, input=list(texts))
        vectors = [item.embedding for item in response.data]
        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        vectors = matrix.tolist()

        usage_tokens = (
            getattr(response.usage, "total_tokens", estimated_tokens)
            if response.usage
            else estimated_tokens
        )
        final_cost = self.budget.estimate_embedding_cost(self.model_name, usage_tokens)
        self.budget.record_usage(
            model=self.model_name,
            provider="openai_embedding",
            tokens_input=usage_tokens,
            tokens_output=0,
            estimated_cost_usd=final_cost,
            meta={"items": len(texts), "task": f"comment_embeddings:{task}"},
        )
        return vectors

    def cache_namespace(self, *, task: str = "topic") -> str:
        return f"{task}-openai"


class EmbeddingCacheStore:
    def __init__(
        self,
        settings: Settings,
        db: Session,
        provider_name: str,
        model_name: str,
    ) -> None:
        self.settings = settings
        self.db = db
        self.provider_name = provider_name
        self.model_name = model_name
        self.base_path = (
            settings.cache_dir / "embeddings" / provider_name / self._sanitize_model(model_name)
        )
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _sanitize_model(self, model_name: str) -> str:
        return model_name.replace("/", "__").replace(":", "__")

    def _db_cache_key(self, text_hash: str) -> str:
        raw = str(text_hash).strip()
        if raw and len(raw) <= 64 and all(char.isalnum() or char in {"-", "_", "."} for char in raw):
            return raw
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _sanitize_cache_key(self, text_hash: str) -> str:
        raw = str(text_hash).strip()
        if raw and all(char.isalnum() or char in {"-", "_", "."} for char in raw):
            return raw
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"cache_{digest}"

    def _file_path(self, text_hash: str) -> Path:
        return self.base_path / f"{self._sanitize_cache_key(text_hash)}.json"

    def _legacy_file_path(self, text_hash: str) -> Path:
        return self.base_path / f"{text_hash}.json"

    def get(self, text_hash: str) -> list[float] | None:
        lookup_keys = [self._db_cache_key(text_hash)]
        if lookup_keys[0] != text_hash and len(text_hash) <= 64:
            lookup_keys.append(text_hash)
        stmt = select(EmbeddingCache).where(
            EmbeddingCache.provider == self.provider_name,
            EmbeddingCache.model == self.model_name,
            EmbeddingCache.text_hash.in_(lookup_keys),
        )
        row = self.db.scalars(stmt).first()
        if row:
            return row.embedding

        for path in (self._file_path(text_hash), self._legacy_file_path(text_hash)):
            try:
                exists = path.exists()
            except OSError:
                continue
            if not exists:
                continue
            try:
                vector = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(vector, list):
                    return [float(item) for item in vector]
            except (json.JSONDecodeError, OSError):
                continue
        return None

    def set(self, text_hash: str, embedding: list[float]) -> None:
        db_key = self._db_cache_key(text_hash)
        stmt = select(EmbeddingCache).where(
            EmbeddingCache.provider == self.provider_name,
            EmbeddingCache.model == self.model_name,
            EmbeddingCache.text_hash == db_key,
        )
        row = self.db.scalar(stmt)
        if row:
            row.embedding = embedding
        else:
            self.db.add(
                EmbeddingCache(
                    provider=self.provider_name,
                    model=self.model_name,
                    text_hash=db_key,
                    embedding=embedding,
                )
            )
        self._file_path(text_hash).write_text(json.dumps(embedding), encoding="utf-8")


class EmbeddingService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_store: EmbeddingCacheStore,
    ) -> None:
        self.provider = provider
        self.cache_store = cache_store
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_embeddings(
        self,
        texts: Sequence[str],
        text_hashes: Sequence[str],
        *,
        task: str = "topic",
    ) -> list[list[float]]:
        if len(texts) != len(text_hashes):
            msg = "texts and text_hashes must have same length."
            raise ValueError(msg)
        vectors: list[list[float] | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_hashes: list[str] = []
        missing_indices: list[int] = []

        for idx, text_hash in enumerate(text_hashes):
            scoped_hash = f"{self.provider.cache_namespace(task=task)}:{text_hash}"
            cached = self.cache_store.get(scoped_hash)
            if cached is not None:
                vectors[idx] = cached
            else:
                missing_indices.append(idx)
                missing_hashes.append(scoped_hash)
                missing_texts.append(texts[idx])

        if missing_texts:
            embedded = self.provider.embed_texts(missing_texts, task=task)
            for idx, text_hash, vector in zip(
                missing_indices,
                missing_hashes,
                embedded,
                strict=True,
            ):
                vectors[idx] = vector
                self.cache_store.set(text_hash, vector)

        if any(vector is None for vector in vectors):
            msg = "Failed to resolve all embeddings."
            raise RuntimeError(msg)
        return [vector for vector in vectors if vector is not None]
