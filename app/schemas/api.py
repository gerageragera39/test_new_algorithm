"""Pydantic models for API request and response serialization.

Defines the schemas used by FastAPI route handlers to validate incoming data
and structure outgoing JSON responses for health, videos, reports, budget, and settings.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.domain import DailyBriefing, TopicPosition, TopicTrendSeries


class HealthResponse(BaseModel):
    """Response payload for the health-check endpoint."""

    status: str
    openai_endpoint_host: str | None = None
    openai_endpoint_mode: str | None = None
    timestamp: datetime


class RunResponse(BaseModel):
    """Response returned after triggering an analysis run."""

    task_id: str
    message: str


class VideoItemResponse(BaseModel):
    """Summary of a video for list endpoints."""

    youtube_video_id: str
    title: str
    published_at: datetime
    has_report: bool
    latest_run_status: str | None = None


class VideoDetailResponse(BaseModel):
    """Detailed information about a single video."""

    youtube_video_id: str
    title: str
    description: str | None
    published_at: datetime
    duration_seconds: int | None
    report_available: bool
    latest_run_status: str | None


class VideoStatusItemResponse(BaseModel):
    """Video summary with progress details for the status dashboard."""

    video_id: int
    youtube_video_id: str
    title: str
    published_at: datetime
    has_report: bool
    has_appeal_report: bool = False
    run_status: str | None
    run_status_text: str
    stage_current: int
    stage_total: int
    progress_pct: int
    stage_label: str | None
    has_error: bool
    appeal_run_status: str | None = None


class ReportResponse(BaseModel):
    """Serialized analysis report with markdown, HTML, and structured briefing."""

    video_id: str
    generated_at: datetime
    markdown: str
    html: str
    briefing: DailyBriefing


class ReportDetailResponse(BaseModel):
    """Extended report response including per-topic comments and positions."""

    report: ReportResponse
    author_name: str
    topic_comments: dict[str, list[str]] = Field(default_factory=dict)
    topic_question_comments: dict[str, list[str]] = Field(default_factory=dict)
    topic_positions: dict[str, list[TopicPosition]] = Field(default_factory=dict)
    topic_trends: list[TopicTrendSeries] = Field(default_factory=list)


class BudgetUsageResponse(BaseModel):
    """Current day budget snapshot with cost breakdown entries."""

    usage_date: str
    spent_usd: float
    tokens_used: int
    entries: list[dict[str, Any]] = Field(default_factory=list)


class AppealBlockItemResponse(BaseModel):
    """A single comment within an appeal analytics block."""

    comment_id: int
    author_name: str | None = None
    text: str = ""
    score: int | None = None
    author_channel_id: str | None = None


class AppealAuthorGroup(BaseModel):
    """Group of comments from the same author within a block."""

    author_name: str
    author_channel_id: str | None = None
    banned_user_id: int | None = None
    is_banned_active: bool = False
    youtube_banned: bool = False
    comment_count: int = 0
    comments: list[AppealBlockItemResponse] = Field(default_factory=list)


class AppealBlockResponse(BaseModel):
    """One of the 5 blocks in the appeal analytics result."""

    block_type: str
    display_label: str
    sort_order: int
    item_count: int
    authors: list[AppealAuthorGroup] = Field(default_factory=list)
    items: list[AppealBlockItemResponse] = Field(default_factory=list)


class AppealAnalyticsResponse(BaseModel):
    """Full appeal analytics result for a video."""

    video_id: str
    video_title: str
    generated_at: datetime
    total_comments: int
    classified_comments: int
    blocks: list[AppealBlockResponse] = Field(default_factory=list)


class AuthorCommentsResponse(BaseModel):
    """All comments by a specific author under a video."""

    author_name: str
    video_id: str
    comments: list[AppealBlockItemResponse] = Field(default_factory=list)


class RuntimeSettingsResponse(BaseModel):
    """Current runtime settings for the scheduled beat and pipeline."""

    beat_enabled: bool
    beat_time_kyiv: str
    updated_at: datetime | None = None
    author_name: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    embedding_mode: str = "local"
    local_embedding_model: str = "intfloat/multilingual-e5-large"
    cluster_max_count: int = 10
    max_comments_per_video: int = 1500
    youtube_include_replies: bool = False
    openai_enable_polish_call: bool = True


class SetupStatusResponse(BaseModel):
    """Desktop bootstrap status for the first-run setup flow."""

    is_configured: bool
    has_openai_api_key: bool
    has_youtube_api_key: bool
    has_playlist_id: bool
    has_youtube_oauth_client_id: bool = False
    has_youtube_oauth_client_secret: bool = False
    has_youtube_oauth_refresh_token: bool = False
    runtime_env_path: str


class SetupRequest(BaseModel):
    """Payload for first-run desktop setup."""

    openai_api_key: str
    youtube_api_key: str
    youtube_playlist_id: str | None = None
    youtube_oauth_client_id: str | None = None
    youtube_oauth_client_secret: str | None = None
    youtube_oauth_refresh_token: str | None = None


class SetupUpdateRequest(BaseModel):
    """Partial update for stored desktop secrets/runtime bootstrap values."""

    openai_api_key: str | None = None
    youtube_api_key: str | None = None
    youtube_playlist_id: str | None = None
    youtube_oauth_client_id: str | None = None
    youtube_oauth_client_secret: str | None = None
    youtube_oauth_refresh_token: str | None = None


class RuntimeSettingsUpdateRequest(BaseModel):
    """Partial update payload for modifying runtime settings."""

    beat_enabled: bool | None = None
    beat_time_kyiv: str | None = None
    author_name: str | None = None
    openai_chat_model: str | None = None
    embedding_mode: str | None = None
    local_embedding_model: str | None = None
    cluster_max_count: int | None = None
    max_comments_per_video: int | None = None
    youtube_include_replies: bool | None = None
    openai_enable_polish_call: bool | None = None

    @field_validator("beat_time_kyiv")
    @classmethod
    def validate_beat_time_kyiv(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        parts = text.split(":")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            msg = "beat_time_kyiv must be HH:MM."
            raise ValueError(msg)
        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            msg = "beat_time_kyiv must be a valid time."
            raise ValueError(msg)
        return f"{hour:02d}:{minute:02d}"

    @field_validator("openai_chat_model")
    @classmethod
    def validate_openai_chat_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        valid = {
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-5-mini",
            "gpt-5.2",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.4-pro",
        }
        normalized = value.strip().lower()
        if normalized not in valid:
            msg = f"openai_chat_model must be one of: {', '.join(sorted(valid))}"
            raise ValueError(msg)
        return normalized

    @field_validator("embedding_mode")
    @classmethod
    def validate_embedding_mode(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in {"local", "openai"}:
            msg = "embedding_mode must be 'local' or 'openai'."
            raise ValueError(msg)
        return normalized


# ── Toxic Moderation Schemas ──────────────────────────────────────────────────


class ToxicReviewItemResponse(BaseModel):
    """Single toxic comment for manual review."""

    comment_id: int
    author_name: str | None
    text: str
    confidence_score: float
    insult_target: str | None


class ToxicReviewResponse(BaseModel):
    """Response for toxic comments requiring manual review."""

    video_id: str
    video_title: str
    total_review_items: int
    items: list[ToxicReviewItemResponse]


class BanUserRequest(BaseModel):
    """Request to ban a user."""

    video_id: str
    comment_id: int
    author_name: str
    ban_reason: str | None = None


class BanUserResponse(BaseModel):
    """Response after banning a user."""

    status: str
    banned_user_id: int | None = None
    youtube_banned: bool
    youtube_error: str | None = None
    csv_saved: bool


class UnbanUserRequest(BaseModel):
    """Request to restore a previously banned commenter."""

    banned_user_id: int
    unban_reason: str | None = None


class UnbanUserResponse(BaseModel):
    """Response after attempting to restore a banned commenter."""

    status: str
    banned_user_id: int | None = None
    youtube_unbanned: bool
    youtube_error: str | None = None


class VideoGuestsResponse(BaseModel):
    """Response with video guests list."""

    video_id: str
    guest_names: list[str]


class UpdateVideoGuestsRequest(BaseModel):
    """Request to update video guests."""

    guest_names: list[str]
