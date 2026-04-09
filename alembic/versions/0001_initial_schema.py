"""Initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-02-22 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "videos",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("youtube_video_id", sa.String(length=32), nullable=False),
        sa.Column("playlist_id", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("url", sa.String(length=256), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("youtube_video_id"),
    )
    op.create_index("ix_videos_youtube_video_id", "videos", ["youtube_video_id"], unique=True)
    op.create_index("ix_videos_playlist_id", "videos", ["playlist_id"], unique=False)
    op.create_index("ix_videos_published_at", "videos", ["published_at"], unique=False)

    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_comments", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_comments", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("meta_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_runs_video_id", "runs", ["video_id"], unique=False)
    op.create_index("ix_runs_status", "runs", ["status"], unique=False)

    op.create_table(
        "comments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("youtube_comment_id", sa.String(length=64), nullable=False),
        sa.Column("parent_comment_id", sa.String(length=64), nullable=True),
        sa.Column("author_name", sa.String(length=256), nullable=True),
        sa.Column("text_raw", sa.Text(), nullable=False),
        sa.Column("text_normalized", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=True),
        sa.Column("like_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reply_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("is_top_level", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("is_filtered", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("filter_reason", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("youtube_comment_id", name="uq_comments_youtube_comment_id"),
    )
    op.create_index("ix_comments_video_id", "comments", ["video_id"], unique=False)
    op.create_index(
        "ix_comments_youtube_comment_id", "comments", ["youtube_comment_id"], unique=False
    )
    op.create_index("ix_comments_text_hash", "comments", ["text_hash"], unique=False)
    op.create_index("ix_comments_published_at", "comments", ["published_at"], unique=False)
    op.create_index(
        "ix_comments_video_text_hash", "comments", ["video_id", "text_hash"], unique=False
    )

    op.create_table(
        "clusters",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("cluster_key", sa.String(length=64), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("label", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("author_actions", sa.JSON(), nullable=False),
        sa.Column("sentiment", sa.String(length=16), nullable=False),
        sa.Column("emotion_tags", sa.JSON(), nullable=False),
        sa.Column("intent_distribution", sa.JSON(), nullable=False),
        sa.Column("representative_quotes", sa.JSON(), nullable=False),
        sa.Column("size_count", sa.Integer(), nullable=False),
        sa.Column("share_pct", sa.Float(), nullable=False),
        sa.Column("weighted_share", sa.Float(), nullable=False),
        sa.Column("is_emerging", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("centroid", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_clusters_run_id", "clusters", ["run_id"], unique=False)
    op.create_index("ix_clusters_video_id", "clusters", ["video_id"], unique=False)
    op.create_index("ix_clusters_run_sort", "clusters", ["run_id", "sort_order"], unique=False)

    op.create_table(
        "cluster_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        sa.Column("comment_id", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column(
            "is_representative", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["cluster_id"], ["clusters.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["comment_id"], ["comments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("cluster_id", "comment_id", name="uq_cluster_items_cluster_comment"),
    )
    op.create_index("ix_cluster_items_cluster_id", "cluster_items", ["cluster_id"], unique=False)
    op.create_index("ix_cluster_items_comment_id", "cluster_items", ["comment_id"], unique=False)

    op.create_table(
        "reports",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("content_markdown", sa.Text(), nullable=False),
        sa.Column("content_html", sa.Text(), nullable=False),
        sa.Column("structured_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", name="uq_reports_run_id"),
    )
    op.create_index("ix_reports_run_id", "reports", ["run_id"], unique=False)
    op.create_index("ix_reports_video_id", "reports", ["video_id"], unique=False)

    op.create_table(
        "budget_usage",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("usage_date", sa.Date(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("tokens_input", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens_output", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("estimated_cost_usd", sa.Numeric(10, 6), nullable=False, server_default="0"),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("meta_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_budget_usage_usage_date", "budget_usage", ["usage_date"], unique=False)
    op.create_index(
        "ix_budget_usage_day_provider", "budget_usage", ["usage_date", "provider"], unique=False
    )

    op.create_table(
        "embedding_cache",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider", "model", "text_hash", name="uq_embedding_cache_provider_model_hash"
        ),
    )
    op.create_index("ix_embedding_cache_text_hash", "embedding_cache", ["text_hash"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_embedding_cache_text_hash", table_name="embedding_cache")
    op.drop_table("embedding_cache")

    op.drop_index("ix_budget_usage_day_provider", table_name="budget_usage")
    op.drop_index("ix_budget_usage_usage_date", table_name="budget_usage")
    op.drop_table("budget_usage")

    op.drop_index("ix_reports_video_id", table_name="reports")
    op.drop_index("ix_reports_run_id", table_name="reports")
    op.drop_table("reports")

    op.drop_index("ix_cluster_items_comment_id", table_name="cluster_items")
    op.drop_index("ix_cluster_items_cluster_id", table_name="cluster_items")
    op.drop_table("cluster_items")

    op.drop_index("ix_clusters_run_sort", table_name="clusters")
    op.drop_index("ix_clusters_video_id", table_name="clusters")
    op.drop_index("ix_clusters_run_id", table_name="clusters")
    op.drop_table("clusters")

    op.drop_index("ix_comments_video_text_hash", table_name="comments")
    op.drop_index("ix_comments_published_at", table_name="comments")
    op.drop_index("ix_comments_text_hash", table_name="comments")
    op.drop_index("ix_comments_youtube_comment_id", table_name="comments")
    op.drop_index("ix_comments_video_id", table_name="comments")
    op.drop_table("comments")

    op.drop_index("ix_runs_status", table_name="runs")
    op.drop_index("ix_runs_video_id", table_name="runs")
    op.drop_table("runs")

    op.drop_index("ix_videos_published_at", table_name="videos")
    op.drop_index("ix_videos_playlist_id", table_name="videos")
    op.drop_index("ix_videos_youtube_video_id", table_name="videos")
    op.drop_table("videos")
