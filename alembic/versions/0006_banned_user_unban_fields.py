"""Add unban audit fields to banned_users.

Revision ID: 0006_banned_user_unban_fields
Revises: e8e640b5b759
Create Date: 2026-04-09 16:35:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0006_banned_user_unban_fields"
down_revision = "e8e640b5b759"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("banned_users", sa.Column("unbanned_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("banned_users", sa.Column("youtube_unbanned", sa.Boolean(), nullable=True))
    op.add_column("banned_users", sa.Column("youtube_unban_error", sa.Text(), nullable=True))
    op.add_column(
        "banned_users",
        sa.Column("unbanned_by_admin", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("banned_users", sa.Column("unban_reason", sa.Text(), nullable=True))

    op.alter_column("banned_users", "unbanned_by_admin", server_default=None)


def downgrade() -> None:
    op.drop_column("banned_users", "unban_reason")
    op.drop_column("banned_users", "unbanned_by_admin")
    op.drop_column("banned_users", "youtube_unban_error")
    op.drop_column("banned_users", "youtube_unbanned")
    op.drop_column("banned_users", "unbanned_at")
