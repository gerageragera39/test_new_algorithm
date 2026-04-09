"""Create appeal analytics tables.

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-06
"""

import sqlalchemy as sa

from alembic import op

revision = "0003_appeal_analytics_tables"
down_revision = "0002_comment_moderation_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "appeal_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "video_id",
            sa.Integer(),
            sa.ForeignKey("videos.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("status", sa.String(16), nullable=False, default="pending", index=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_comments", sa.Integer(), nullable=False, default=0),
        sa.Column("processed_comments", sa.Integer(), nullable=False, default=0),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("meta_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )

    op.create_table(
        "appeal_blocks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "appeal_run_id",
            sa.Integer(),
            sa.ForeignKey("appeal_runs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "video_id",
            sa.Integer(),
            sa.ForeignKey("videos.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("block_type", sa.String(32), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False, default=0),
        sa.Column("display_label", sa.String(128), nullable=False),
        sa.Column("item_count", sa.Integer(), nullable=False, default=0),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_appeal_blocks_run_sort", "appeal_blocks", ["appeal_run_id", "sort_order"])

    op.create_table(
        "appeal_block_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "block_id",
            sa.Integer(),
            sa.ForeignKey("appeal_blocks.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "comment_id",
            sa.Integer(),
            sa.ForeignKey("comments.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("author_name", sa.String(256), nullable=True),
        sa.Column("detail_json", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_unique_constraint(
        "uq_appeal_block_items_block_comment", "appeal_block_items", ["block_id", "comment_id"]
    )


def downgrade() -> None:
    op.drop_table("appeal_block_items")
    op.drop_table("appeal_blocks")
    op.drop_table("appeal_runs")
