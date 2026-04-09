"""Add moderation fields to comments

Revision ID: 0002_comment_moderation_fields
Revises: 0001_initial_schema
Create Date: 2026-03-02 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_comment_moderation_fields"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("comments", sa.Column("moderation_action", sa.String(length=16), nullable=True))
    op.add_column("comments", sa.Column("moderation_reason", sa.String(length=128), nullable=True))
    op.add_column("comments", sa.Column("moderation_source", sa.String(length=16), nullable=True))
    op.add_column("comments", sa.Column("moderation_score", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("comments", "moderation_score")
    op.drop_column("comments", "moderation_source")
    op.drop_column("comments", "moderation_reason")
    op.drop_column("comments", "moderation_action")
