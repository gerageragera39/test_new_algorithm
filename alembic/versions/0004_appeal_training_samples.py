"""Add appeal training samples table.

Revision ID: 0004_appeal_training_samples
Revises: 0003_appeal_analytics_tables
Create Date: 2026-03-31
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004_appeal_training_samples"
down_revision = "0003_appeal_analytics_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "appeal_training_samples",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "appeal_run_id",
            sa.Integer(),
            sa.ForeignKey("appeal_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "video_id", sa.Integer(), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "comment_id",
            sa.Integer(),
            sa.ForeignKey("comments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("youtube_comment_id", sa.String(length=64), nullable=False),
        sa.Column("text_raw_snapshot", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("author_name", sa.String(length=256), nullable=True),
        sa.Column("guest_names_snapshot", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("llm_category", sa.String(length=32), nullable=False, server_default="skip"),
        sa.Column("target", sa.String(length=16), nullable=False, server_default="none"),
        sa.Column("guest_name", sa.String(length=256), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("prefilter_score", sa.Float(), nullable=True),
        sa.Column("moderation_decision", sa.String(length=32), nullable=True),
        sa.Column("model_label", sa.String(length=16), nullable=False, server_default="other"),
        sa.Column("review_status", sa.String(length=32), nullable=False, server_default="ignored"),
        sa.Column("final_label", sa.String(length=16), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "appeal_run_id", "comment_id", name="uq_appeal_training_samples_run_comment"
        ),
    )
    op.create_index(
        "ix_appeal_training_samples_review_status",
        "appeal_training_samples",
        ["review_status"],
    )
    op.create_index(
        "ix_appeal_training_samples_model_label",
        "appeal_training_samples",
        ["model_label"],
    )


def downgrade() -> None:
    op.drop_index("ix_appeal_training_samples_model_label", table_name="appeal_training_samples")
    op.drop_index("ix_appeal_training_samples_review_status", table_name="appeal_training_samples")
    op.drop_table("appeal_training_samples")
