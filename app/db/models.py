"""SQLAlchemy ORM model definitions for the YouTube analyzer database.

Contains all table models including videos, comments, analysis runs, clusters,
reports, budget tracking, and embedding cache.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Video(TimestampMixin, Base):
    """Represents a YouTube video and its metadata."""

    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    youtube_video_id: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False
    )
    playlist_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    url: Mapped[str] = mapped_column(String(256), nullable=False)

    comments: Mapped[list[Comment]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    runs: Mapped[list[Run]] = relationship(back_populates="video", cascade="all, delete-orphan")
    reports: Mapped[list[Report]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    appeal_runs: Mapped[list[AppealRun]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )


class Comment(TimestampMixin, Base):
    """Stores a single YouTube comment with its normalized text, moderation status, and weight."""

    __tablename__ = "comments"
    __table_args__ = (
        UniqueConstraint("youtube_comment_id", name="uq_comments_youtube_comment_id"),
        Index("ix_comments_video_text_hash", "video_id", "text_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    youtube_comment_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    parent_comment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    author_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    author_channel_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    text_raw: Mapped[str] = mapped_column(Text, nullable=False)
    text_normalized: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    like_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reply_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    is_top_level: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_filtered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    filter_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    moderation_action: Mapped[str | None] = mapped_column(String(16), nullable=True)
    moderation_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    moderation_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    moderation_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    video: Mapped[Video] = relationship(back_populates="comments")
    cluster_items: Mapped[list[ClusterItem]] = relationship(
        back_populates="comment", cascade="all, delete-orphan"
    )


class Run(TimestampMixin, Base):
    """Tracks a single analysis pipeline execution for a video."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_comments: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_comments: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    video: Mapped[Video] = relationship(back_populates="runs")
    clusters: Mapped[list[Cluster]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    report: Mapped[Report | None] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        uselist=False,
    )


class Cluster(TimestampMixin, Base):
    """Represents a topic cluster of related comments within an analysis run."""

    __tablename__ = "clusters"
    __table_args__ = (Index("ix_clusters_run_sort", "run_id", "sort_order"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    cluster_key: Mapped[str] = mapped_column(String(64), nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    author_actions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    sentiment: Mapped[str] = mapped_column(String(16), nullable=False, default="neutral")
    emotion_tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    intent_distribution: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False, default=dict)
    representative_quotes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    size_count: Mapped[int] = mapped_column(Integer, nullable=False)
    share_pct: Mapped[float] = mapped_column(Float, nullable=False)
    weighted_share: Mapped[float] = mapped_column(Float, nullable=False)
    is_emerging: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    centroid: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)

    run: Mapped[Run] = relationship(back_populates="clusters")
    items: Mapped[list[ClusterItem]] = relationship(
        back_populates="cluster", cascade="all, delete-orphan"
    )


class ClusterItem(TimestampMixin, Base):
    """Maps a comment to a cluster with its similarity score."""

    __tablename__ = "cluster_items"
    __table_args__ = (
        UniqueConstraint("cluster_id", "comment_id", name="uq_cluster_items_cluster_comment"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cluster_id: Mapped[int] = mapped_column(
        ForeignKey("clusters.id", ondelete="CASCADE"), nullable=False, index=True
    )
    comment_id: Mapped[int] = mapped_column(
        ForeignKey("comments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    is_representative: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    cluster: Mapped[Cluster] = relationship(back_populates="items")
    comment: Mapped[Comment] = relationship(back_populates="cluster_items")


class Report(TimestampMixin, Base):
    """Stores a generated analysis report in markdown, HTML, and structured JSON formats."""

    __tablename__ = "reports"
    __table_args__ = (UniqueConstraint("run_id", name="uq_reports_run_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    content_html: Mapped[str] = mapped_column(Text, nullable=False)
    structured_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    run: Mapped[Run] = relationship(back_populates="report")
    video: Mapped[Video] = relationship(back_populates="reports")


class AppealRun(TimestampMixin, Base):
    """Tracks a single Appeal Analytics pipeline execution for a video."""

    __tablename__ = "appeal_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_comments: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_comments: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    video: Mapped[Video] = relationship(back_populates="appeal_runs")
    blocks: Mapped[list[AppealBlock]] = relationship(
        back_populates="appeal_run", cascade="all, delete-orphan"
    )


class AppealBlock(TimestampMixin, Base):
    """Stores one of the 5 classification blocks within an appeal analytics run."""

    __tablename__ = "appeal_blocks"
    __table_args__ = (Index("ix_appeal_blocks_run_sort", "appeal_run_id", "sort_order"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    appeal_run_id: Mapped[int] = mapped_column(
        ForeignKey("appeal_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    block_type: Mapped[str] = mapped_column(String(32), nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    display_label: Mapped[str] = mapped_column(String(128), nullable=False)
    item_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    appeal_run: Mapped[AppealRun] = relationship(back_populates="blocks")
    items: Mapped[list[AppealBlockItem]] = relationship(
        back_populates="block", cascade="all, delete-orphan"
    )


class AppealBlockItem(TimestampMixin, Base):
    """Links a comment to an appeal analytics block."""

    __tablename__ = "appeal_block_items"
    __table_args__ = (
        UniqueConstraint("block_id", "comment_id", name="uq_appeal_block_items_block_comment"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    block_id: Mapped[int] = mapped_column(
        ForeignKey("appeal_blocks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    comment_id: Mapped[int] = mapped_column(
        ForeignKey("comments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    author_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    detail_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    insult_target: Mapped[str | None] = mapped_column(String(32), nullable=True)

    block: Mapped[AppealBlock] = relationship(back_populates="items")
    comment: Mapped[Comment] = relationship()


class BannedUser(TimestampMixin, Base):
    """Tracks users banned for toxic comments (auto or manual).

    Uses author_channel_id as primary identifier (stable), username as fallback.
    The row stores the source video/comment for audit, but the underlying
    YouTube hide-user action is channel-wide.
    """

    __tablename__ = "banned_users"
    __table_args__ = (
        Index("ix_banned_users_video_username", "video_id", "username"),
        Index("ix_banned_users_video_channel", "video_id", "author_channel_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    comment_id: Mapped[int | None] = mapped_column(
        ForeignKey("comments.id", ondelete="SET NULL"), nullable=True, index=True
    )
    username: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    author_channel_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    ban_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    insult_target: Mapped[str | None] = mapped_column(String(32), nullable=True)
    banned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    youtube_banned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    youtube_ban_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    banned_by_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    unbanned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    youtube_unbanned: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    youtube_unban_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    unbanned_by_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    unban_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    video: Mapped[Video] = relationship()
    comment: Mapped[Comment | None] = relationship()


class ToxicTrainingData(TimestampMixin, Base):
    """Stores labeled toxic/non-toxic comments for future model training."""

    __tablename__ = "toxic_training_data"
    __table_args__ = (
        Index("ix_toxic_training_is_toxic", "is_toxic"),
        Index("ix_toxic_training_labeled_by", "labeled_by"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    comment_id: Mapped[int] = mapped_column(
        ForeignKey("comments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    author_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    is_toxic: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    insult_target: Mapped[str | None] = mapped_column(String(32), nullable=True)
    labeled_by: Mapped[str] = mapped_column(String(16), nullable=False)
    labeled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    video: Mapped[Video] = relationship()
    comment: Mapped[Comment] = relationship()


class VideoSettings(TimestampMixin, Base):
    """Stores video-specific settings like guest names."""

    __tablename__ = "video_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    guest_names: Mapped[str | None] = mapped_column(Text, nullable=True)

    video: Mapped[Video] = relationship()


class BudgetUsage(TimestampMixin, Base):
    """Records daily API token usage and estimated cost per provider and model."""

    __tablename__ = "budget_usage"
    __table_args__ = (Index("ix_budget_usage_day_provider", "usage_date", "provider"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    usage_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="openai")
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    tokens_input: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_output: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, default=0.0)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    meta_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class EmbeddingCache(TimestampMixin, Base):
    """Caches computed embedding vectors keyed by provider, model, and text hash."""

    __tablename__ = "embedding_cache"
    __table_args__ = (
        UniqueConstraint(
            "provider", "model", "text_hash", name="uq_embedding_cache_provider_model_hash"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)
