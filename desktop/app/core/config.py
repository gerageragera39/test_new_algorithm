"""
Application configuration via Pydantic Settings.

All settings can be overridden through environment variables or a ``.env``
file.  Default values are tuned for a typical production deployment with
a YouTube news channel producing daily episodes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SUPPORTED_OPENAI_CHAT_MODEL_PREFIXES = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-mini",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4",
    "gpt-5.2",
)
_SUPPORTED_OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
}


class Settings(BaseSettings):
    """Central application configuration.

    All fields can be overridden via environment variables (case-insensitive).
    A ``.env`` file in the project root is loaded automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── General ──────────────────────────────────────
    app_name: str = "YouTube Daily Comments Intelligence"
    app_timezone: str = "Europe/Berlin"
    log_level: str = "INFO"
    embedding_mode: Literal["local", "openai"] = "local"
    dry_run: bool = False
    dry_run_fixture_path: Path = Field(default=Path("tests/fixtures/mock_youtube.json"))

    # ── YouTube API ──────────────────────────────────
    youtube_api_key: str | None = None
    youtube_playlist_id: str = ""
    youtube_title_regex: str | None = None
    youtube_exclude_shorts: bool = True
    youtube_short_threshold_seconds: int = 120
    youtube_include_replies: bool = False
    youtube_request_timeout_sec: float = 20.0
    youtube_rate_limit_rps: float = 5.0
    youtube_max_pages: int = 20
    max_comments_per_video: int = 1500
    youtube_mix_relevance_comments: bool = True
    # Deprecated: ignored in fetch logic; kept for backward-compatible env parsing.
    youtube_relevance_comments_share: float = 0.35

    # ── OpenAI / LLM ────────────────────────────────
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_require_official_endpoint: bool = True
    openai_max_usd_per_run: float = 0.0
    openai_hard_budget_enforced: bool = False
    openai_max_output_tokens: int = 16_384
    openai_enable_polish_call: bool = True
    openai_max_calls_per_run: int = 999_999
    openai_calls_reserved_for_labeling: int = 0
    openai_max_moderation_calls_per_run: int = 999_999
    openai_max_position_naming_calls_per_run: int = 999_999

    # ── Embedding ────────────────────────────────────
    local_embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_instruction_mode: Literal["auto", "off", "force"] = "off"
    embedding_topic_task_prompt: str = (
        "Represent the main topical subject of this YouTube comment for semantic clustering. "
        "Ignore tone, insults, sarcasm, and wording differences."
    )
    local_embedding_device: Literal["auto", "cpu", "cuda"] = "auto"
    local_embedding_batch_size: int = 32
    local_embedding_max_seq_length: int = 0
    local_embedding_low_vram_mode: bool = False
    local_embedding_oom_fallback_to_cpu: bool = True
    author_name: str = ""

    # ── YouTube Ban Integration ──────────────────────
    # OAuth2 credentials for YouTube API to ban/moderate users
    # Required scopes: https://www.googleapis.com/auth/youtube.force-ssl
    youtube_oauth_client_id: str | None = None
    youtube_oauth_client_secret: str | None = None
    youtube_oauth_refresh_token: str | None = None
    # Legacy: kept for backward compatibility, but OAuth is preferred
    youtube_ban_token: str | None = None
    # Auto-ban threshold raised from 0.85 to 0.92 to reduce false positives
    # At 0.92 confidence, expected error rate is ~8% vs 15% at 0.85
    # For author/guest targets, any confidence >= 0.80 goes to manual review
    auto_ban_threshold: float = 0.92
    manual_review_threshold: float = 0.5

    # ── Clustering ───────────────────────────────────
    cluster_min_size: int = 6
    cluster_min_samples: int = 2
    cluster_max_count: int = 10
    cluster_reduction_enabled: bool = False
    cluster_reduction_target_dim: int = 32
    cluster_reduction_min_comments: int = 80
    cluster_accept_noise_ratio: float = 0.97
    cluster_smallset_accept_noise_ratio: float = 0.99
    cluster_soft_assignment_enabled: bool = False
    cluster_soft_assignment_min_similarity: float = 0.42
    cluster_ambiguity_confidence_threshold: float = 0.58
    cluster_ambiguity_margin_threshold: float = 0.08
    cluster_assignment_note_limit: int = 3
    cluster_kmeans_fallback_enabled: bool = True
    cluster_kmeans_fallback_min_size: int = 24
    cluster_noise_split_enabled: bool = True
    cluster_noise_split_min_size: int = 48
    cluster_noise_split_min_share_pct: float = 28.0
    cluster_noise_split_target_group_size: int = 20
    cluster_noise_split_max_groups: int = 4
    cluster_noise_split_min_silhouette_small: float = 0.0
    cluster_noise_split_min_silhouette_large: float = 0.02
    cluster_noise_split_min_avg_coherence: float = 0.12
    cluster_noise_split_min_group_coherence: float = 0.06
    cluster_large_split_enabled: bool = True
    cluster_large_split_min_share_pct: float = 25.0
    cluster_large_split_max_subgroups: int = 4
    cluster_merge_enabled: bool = True
    cluster_merge_similarity_threshold: float = 0.88
    cluster_merge_keyword_jaccard_min: float = 0.2
    cluster_merge_max_rounds: int = 3
    topic_merge_enabled: bool = False
    topic_merge_similarity_threshold: float = 0.88
    topic_merge_label_jaccard_min: float = 0.24
    topic_merge_label_subset_min_tokens: int = 2
    topic_cross_dedupe_enabled: bool = True
    topic_cross_dedupe_similarity_threshold: float = 0.9
    topic_uncertain_min_comments: int = 1
    topic_uncertain_support_min: float = 0.14

    # ── Preprocessing & Moderation ───────────────────
    preprocessing_filter_enabled: bool = True
    comment_min_words: int = 3
    comment_max_chars: int = 2000
    comment_weight_max: float = 5.0
    preprocessing_low_signal_filter_enabled: bool = True
    preprocessing_low_signal_min_alpha_chars: int = 5
    preprocessing_low_signal_max_symbol_ratio: float = 0.45
    preprocessing_low_signal_min_lexical_diversity: float = 0.34
    moderation_enabled: bool = True
    moderation_toxicity_policy: Literal["keep_flag", "hard_delete", "severe_delete"] = "keep_flag"
    moderation_enable_llm_borderline: bool = True
    moderation_llm_scope: Literal["borderline", "all", "disabled"] = "borderline"
    moderation_llm_max_reviews_per_run: int = 300
    moderation_flagged_weight_multiplier: float = 0.85
    moderation_profanity_only_max_words: int = 6
    moderation_topic_token_min_overlap: int = 1
    moderation_borderline_min_score: float = 0.35
    moderation_borderline_max_score: float = 0.55
    moderation_log_include_reason_breakdown: bool = True
    moderation_log_include_samples: bool = False
    moderation_log_sample_size: int = 3

    # ── Transcript / Episode Context ─────────────────
    enable_transcript_context: bool = True
    transcript_model: str = "medium"
    transcript_device: Literal["auto", "cpu", "cuda"] = "auto"
    transcript_language: str | None = "ru"
    transcript_beam_size: int = 5
    transcript_vad_filter: bool = True
    transcript_max_chars: int = 22_000
    transcript_topic_count: int = 8
    transcript_topic_match_min_similarity: float = 0.5
    transcript_compute_type_cuda: str = "float16"
    transcript_compute_type_cpu: str = "int8"
    transcript_keep_audio: bool = False

    # ── Position Extraction ──────────────────────────
    position_llm_sample_min: int = 80
    position_llm_sample_max: int = 200
    position_llm_sample_min_openai: int = 12
    position_llm_sample_max_openai: int = 24
    position_llm_comment_char_cap: int = 220
    position_llm_payload_char_budget: int = 5000
    position_title_retry_count: int = 1
    position_llm_naming_min_group_size: int = 3
    position_subcluster_max_k: int = 0
    position_subcluster_min_group_size: int = 2
    position_subcluster_min_group_share_pct: float = 3.0
    position_subcluster_hdbscan_scale: float = 0.75
    position_subcluster_hdbscan_max_cluster_size: int = 8
    position_assignment_min_similarity: float = 0.34
    position_assignment_borderline_similarity: float = 0.48
    position_assignment_min_margin: float = 0.035

    # ── Report & Briefing ───────────────────────────
    # Deprecated: report includes all clusters; kept for backward-compatible env parsing.
    report_top_k_topics: int = 6
    # Deprecated: no longer enforced; kept for backward-compatible env parsing.
    report_actions_limit: int = 50
    # Deprecated: report uses full per-position comment lists; kept for backward-compatible env parsing.
    report_quotes_per_topic: int = 8
    report_total_quotes_limit: int = 24
    topic_coherence_min: float = 0.38
    topic_min_weighted_share_for_low_coherence: float = 7.0
    max_representatives_per_cluster: int = 4
    quality_watchdog_undetermined_share_pct: float = 35.0
    quality_watchdog_fallback_title_rate_pct: float = 40.0
    cluster_diagnostics_enabled: bool = True
    trend_similarity_threshold: float = 0.72

    # ── Paths & Storage ─────────────────────────────
    cache_dir: Path = Field(default=Path("data/cache"))
    data_dir: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"))
    reports_dir: Path = Field(default=Path("data/reports"))

    # ── Runtime ──────────────────────────────────────
    database_url: str = "postgresql+psycopg://postgres:postgres@db:5432/youtube_intel"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    schedule_daily_at: str = "06:30"
    enable_scheduled_runs: bool = False
    # Legacy / compatibility
    include_replies: bool = False

    @field_validator("embedding_mode", mode="before")
    @classmethod
    def normalize_embedding_mode(cls, value: str) -> str:
        return str(value).strip().lower() or "local"

    @field_validator("moderation_toxicity_policy", mode="before")
    @classmethod
    def normalize_moderation_toxicity_policy(cls, value: str) -> str:
        return str(value).lower().strip()

    @field_validator("moderation_llm_scope", mode="before")
    @classmethod
    def normalize_moderation_llm_scope(cls, value: str) -> str:
        return str(value).lower().strip()

    @field_validator("openai_chat_model", mode="before")
    @classmethod
    def normalize_openai_chat_model(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if any(normalized.startswith(prefix) for prefix in _SUPPORTED_OPENAI_CHAT_MODEL_PREFIXES):
            return normalized
        msg = (
            "OPENAI_CHAT_MODEL must be one of the supported OpenAI chat families: "
            "gpt-4o-mini, gpt-4o, gpt-5-mini, gpt-5.2, gpt-5.4, gpt-5.4-mini, "
            "gpt-5.4-nano, gpt-5.4-pro."
        )
        raise ValueError(msg)

    @field_validator("openai_embedding_model", mode="before")
    @classmethod
    def validate_openai_embedding_model(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized in _SUPPORTED_OPENAI_EMBEDDING_MODELS:
            return normalized
        msg = "OPENAI_EMBEDDING_MODEL must be 'text-embedding-3-small' or 'text-embedding-3-large'."
        raise ValueError(msg)

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def normalize_openai_base_url(cls, value: str) -> str:
        base = str(value or "").strip().rstrip("/")
        if not base:
            return "https://api.openai.com/v1"
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    @field_validator("schedule_daily_at")
    @classmethod
    def validate_schedule_daily_at(cls, value: str) -> str:
        parts = value.split(":")
        if len(parts) != 2:
            msg = "SCHEDULE_DAILY_AT must be HH:MM format."
            raise ValueError(msg)
        hour, minute = parts
        if not hour.isdigit() or not minute.isdigit():
            msg = "SCHEDULE_DAILY_AT must be numeric HH:MM format."
            raise ValueError(msg)
        h, m = int(hour), int(minute)
        if not (0 <= h <= 23 and 0 <= m <= 59):
            msg = "SCHEDULE_DAILY_AT must be valid time."
            raise ValueError(msg)
        return f"{h:02d}:{m:02d}"

    @field_validator("youtube_relevance_comments_share")
    @classmethod
    def validate_youtube_relevance_comments_share(cls, value: float) -> float:
        share = float(value)
        if share < 0.0:
            return 0.0
        if share > 0.9:
            return 0.9
        return share

    @field_validator("position_llm_sample_min")
    @classmethod
    def validate_position_llm_sample_min(cls, value: int) -> int:
        sample_min = int(value)
        return min(200, max(20, sample_min))

    @field_validator("position_llm_sample_max")
    @classmethod
    def validate_position_llm_sample_max(cls, value: int) -> int:
        sample_max = int(value)
        return min(320, max(40, sample_max))

    @field_validator("position_llm_sample_min_openai")
    @classmethod
    def validate_position_llm_sample_min_openai(cls, value: int) -> int:
        sample_min = int(value)
        return min(128, max(4, sample_min))

    @field_validator("position_llm_sample_max_openai")
    @classmethod
    def validate_position_llm_sample_max_openai(cls, value: int) -> int:
        sample_max = int(value)
        return min(256, max(8, sample_max))

    @field_validator("position_llm_comment_char_cap")
    @classmethod
    def validate_position_llm_comment_char_cap(cls, value: int) -> int:
        char_cap = int(value)
        return min(1000, max(40, char_cap))

    @field_validator("position_llm_payload_char_budget")
    @classmethod
    def validate_position_llm_payload_char_budget(cls, value: int) -> int:
        budget = int(value)
        return min(30_000, max(800, budget))

    @field_validator("position_title_retry_count")
    @classmethod
    def validate_position_title_retry_count(cls, value: int) -> int:
        retries = int(value)
        return min(2, max(0, retries))

    @field_validator("position_llm_naming_min_group_size")
    @classmethod
    def validate_position_llm_naming_min_group_size(cls, value: int) -> int:
        min_size = int(value)
        return min(128, max(1, min_size))

    @field_validator("position_subcluster_max_k")
    @classmethod
    def validate_position_subcluster_max_k(cls, value: int) -> int:
        max_k = int(value)
        return max(0, max_k)

    @field_validator("position_subcluster_min_group_size")
    @classmethod
    def validate_position_subcluster_min_group_size(cls, value: int) -> int:
        min_size = int(value)
        return max(1, min_size)

    @field_validator("position_subcluster_min_group_share_pct")
    @classmethod
    def validate_position_subcluster_min_group_share_pct(cls, value: float) -> float:
        pct = float(value)
        return min(50.0, max(0.0, pct))

    @field_validator("position_subcluster_hdbscan_scale")
    @classmethod
    def validate_position_subcluster_hdbscan_scale(cls, value: float) -> float:
        scale = float(value)
        return min(2.0, max(0.2, scale))

    @field_validator("position_subcluster_hdbscan_max_cluster_size")
    @classmethod
    def validate_position_subcluster_hdbscan_max_cluster_size(cls, value: int) -> int:
        max_size = int(value)
        return min(40, max(2, max_size))

    @field_validator("position_assignment_min_similarity")
    @classmethod
    def validate_position_assignment_min_similarity(cls, value: float) -> float:
        threshold = float(value)
        return min(0.9, max(0.0, threshold))

    @field_validator("position_assignment_borderline_similarity")
    @classmethod
    def validate_position_assignment_borderline_similarity(cls, value: float) -> float:
        threshold = float(value)
        return min(0.95, max(0.0, threshold))

    @field_validator("position_assignment_min_margin")
    @classmethod
    def validate_position_assignment_min_margin(cls, value: float) -> float:
        margin = float(value)
        return min(0.4, max(0.0, margin))

    @field_validator("quality_watchdog_undetermined_share_pct")
    @classmethod
    def validate_quality_watchdog_undetermined_share_pct(cls, value: float) -> float:
        threshold = float(value)
        return min(100.0, max(0.0, threshold))

    @field_validator("quality_watchdog_fallback_title_rate_pct")
    @classmethod
    def validate_quality_watchdog_fallback_title_rate_pct(cls, value: float) -> float:
        threshold = float(value)
        return min(100.0, max(0.0, threshold))

    @field_validator("openai_max_usd_per_run")
    @classmethod
    def validate_openai_max_usd_per_run(cls, value: float) -> float:
        run_cap = float(value)
        return max(0.0, run_cap)

    @field_validator("cluster_max_count")
    @classmethod
    def validate_cluster_max_count(cls, value: int) -> int:
        count = int(value)
        return max(0, count)

    @field_validator("cluster_accept_noise_ratio")
    @classmethod
    def validate_cluster_accept_noise_ratio(cls, value: float) -> float:
        ratio = float(value)
        return min(1.0, max(0.0, ratio))

    @field_validator("cluster_smallset_accept_noise_ratio")
    @classmethod
    def validate_cluster_smallset_accept_noise_ratio(cls, value: float) -> float:
        ratio = float(value)
        return min(1.0, max(0.0, ratio))

    @field_validator("cluster_kmeans_fallback_min_size")
    @classmethod
    def validate_cluster_kmeans_fallback_min_size(cls, value: int) -> int:
        min_size = int(value)
        return min(50_000, max(6, min_size))

    @field_validator("cluster_noise_split_target_group_size")
    @classmethod
    def validate_cluster_noise_split_target_group_size(cls, value: int) -> int:
        target = int(value)
        return min(512, max(4, target))

    @field_validator("cluster_noise_split_min_silhouette_small")
    @classmethod
    def validate_cluster_noise_split_min_silhouette_small(cls, value: float) -> float:
        threshold = float(value)
        return min(1.0, max(-1.0, threshold))

    @field_validator("cluster_noise_split_min_silhouette_large")
    @classmethod
    def validate_cluster_noise_split_min_silhouette_large(cls, value: float) -> float:
        threshold = float(value)
        return min(1.0, max(-1.0, threshold))

    @field_validator("cluster_noise_split_min_avg_coherence")
    @classmethod
    def validate_cluster_noise_split_min_avg_coherence(cls, value: float) -> float:
        threshold = float(value)
        return min(1.0, max(0.0, threshold))

    @field_validator("cluster_noise_split_min_group_coherence")
    @classmethod
    def validate_cluster_noise_split_min_group_coherence(cls, value: float) -> float:
        threshold = float(value)
        return min(1.0, max(0.0, threshold))

    @field_validator("moderation_llm_max_reviews_per_run")
    @classmethod
    def validate_moderation_llm_max_reviews_per_run(cls, value: int) -> int:
        count = int(value)
        return min(2_000, max(0, count))

    @field_validator("moderation_flagged_weight_multiplier")
    @classmethod
    def validate_moderation_flagged_weight_multiplier(cls, value: float) -> float:
        multiplier = float(value)
        return min(1.0, max(0.1, multiplier))

    @field_validator("moderation_profanity_only_max_words")
    @classmethod
    def validate_moderation_profanity_only_max_words(cls, value: int) -> int:
        words = int(value)
        return min(40, max(1, words))

    @field_validator("moderation_topic_token_min_overlap")
    @classmethod
    def validate_moderation_topic_token_min_overlap(cls, value: int) -> int:
        overlap = int(value)
        return min(8, max(0, overlap))

    @field_validator("moderation_borderline_min_score")
    @classmethod
    def validate_moderation_borderline_min_score(cls, value: float) -> float:
        score = float(value)
        return min(1.0, max(0.0, score))

    @field_validator("moderation_borderline_max_score")
    @classmethod
    def validate_moderation_borderline_max_score(cls, value: float) -> float:
        score = float(value)
        return min(1.0, max(0.0, score))

    @field_validator("moderation_log_sample_size")
    @classmethod
    def validate_moderation_log_sample_size(cls, value: int) -> int:
        size = int(value)
        return min(20, max(1, size))

    @field_validator("openai_max_output_tokens")
    @classmethod
    def validate_openai_max_output_tokens(cls, value: int) -> int:
        tokens = int(value)
        return max(128, tokens)

    @field_validator("openai_max_calls_per_run")
    @classmethod
    def validate_openai_max_calls_per_run(cls, value: int) -> int:
        calls = int(value)
        return max(1, calls)

    @field_validator("openai_calls_reserved_for_labeling")
    @classmethod
    def validate_openai_calls_reserved_for_labeling(cls, value: int) -> int:
        reserve = int(value)
        return max(0, reserve)

    @field_validator("openai_max_moderation_calls_per_run")
    @classmethod
    def validate_openai_max_moderation_calls_per_run(cls, value: int) -> int:
        calls = int(value)
        return max(0, calls)

    @field_validator("openai_max_position_naming_calls_per_run")
    @classmethod
    def validate_openai_max_position_naming_calls_per_run(cls, value: int) -> int:
        calls = int(value)
        return max(0, calls)

    @property
    def schedule_hour(self) -> int:
        return int(self.schedule_daily_at.split(":")[0])

    @property
    def schedule_minute(self) -> int:
        return int(self.schedule_daily_at.split(":")[1])

    @property
    def resolved_embedding_mode(self) -> Literal["local", "openai"]:
        return self.embedding_mode

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.raw_dir, self.reports_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
