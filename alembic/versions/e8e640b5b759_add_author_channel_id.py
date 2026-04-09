"""add_author_channel_id

Revision ID: e8e640b5b759
Revises: 0005_toxic_schema_fix
Create Date: 2026-04-06 20:08:57.470628
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "e8e640b5b759"
down_revision = "0005_toxic_schema_fix"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add author_channel_id to comments table
    op.add_column("comments", sa.Column("author_channel_id", sa.String(length=64), nullable=True))
    op.create_index(
        "ix_comments_author_channel_id", "comments", ["author_channel_id"], unique=False
    )

    # Add author_channel_id to banned_users table
    op.add_column(
        "banned_users", sa.Column("author_channel_id", sa.String(length=64), nullable=True)
    )
    op.create_index(
        "ix_banned_users_author_channel_id", "banned_users", ["author_channel_id"], unique=False
    )
    op.create_index(
        "ix_banned_users_video_channel",
        "banned_users",
        ["video_id", "author_channel_id"],
        unique=False,
    )


def downgrade() -> None:
    # Remove from banned_users
    op.drop_index("ix_banned_users_video_channel", table_name="banned_users")
    op.drop_index("ix_banned_users_author_channel_id", table_name="banned_users")
    op.drop_column("banned_users", "author_channel_id")

    # Remove from comments
    op.drop_index("ix_comments_author_channel_id", table_name="comments")
    op.drop_column("comments", "author_channel_id")
