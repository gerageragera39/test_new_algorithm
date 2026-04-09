"""Repair toxic moderation schema for already-stamped databases.

Revision ID: 0005_toxic_schema_fix
Revises: ca9a9ad66c80
Create Date: 2026-04-04 19:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0005_toxic_schema_fix"
down_revision = "ca9a9ad66c80"
branch_labels = None
depends_on = None


def _table_names(bind: sa.engine.Connection) -> set[str]:
    return set(sa.inspect(bind).get_table_names())


def _column_names(bind: sa.engine.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _index_names(bind: sa.engine.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(table_name)}


def _ensure_index(
    bind: sa.engine.Connection,
    index_name: str,
    table_name: str,
    columns: list[str],
    *,
    unique: bool = False,
) -> None:
    if index_name not in _index_names(bind, table_name):
        op.create_index(index_name, table_name, columns, unique=unique)


def upgrade() -> None:
    bind = op.get_bind()

    if "video_settings" not in _table_names(bind):
        op.create_table(
            "video_settings",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("video_id", sa.Integer(), nullable=False),
            sa.Column("guest_names", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    _ensure_index(bind, "ix_video_settings_video_id", "video_settings", ["video_id"], unique=True)

    if "banned_users" not in _table_names(bind):
        op.create_table(
            "banned_users",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("video_id", sa.Integer(), nullable=False),
            sa.Column("comment_id", sa.Integer(), nullable=True),
            sa.Column("username", sa.String(length=256), nullable=False),
            sa.Column("ban_reason", sa.Text(), nullable=True),
            sa.Column("confidence_score", sa.Float(), nullable=False),
            sa.Column("insult_target", sa.String(length=32), nullable=True),
            sa.Column("banned_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column(
                "youtube_banned", sa.Boolean(), nullable=False, server_default=sa.text("false")
            ),
            sa.Column("youtube_ban_error", sa.Text(), nullable=True),
            sa.Column(
                "banned_by_admin", sa.Boolean(), nullable=False, server_default=sa.text("false")
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.ForeignKeyConstraint(["comment_id"], ["comments.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    _ensure_index(bind, "ix_banned_users_video_id", "banned_users", ["video_id"])
    _ensure_index(bind, "ix_banned_users_comment_id", "banned_users", ["comment_id"])
    _ensure_index(bind, "ix_banned_users_username", "banned_users", ["username"])
    _ensure_index(bind, "ix_banned_users_video_username", "banned_users", ["video_id", "username"])

    if "toxic_training_data" not in _table_names(bind):
        op.create_table(
            "toxic_training_data",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("comment_id", sa.Integer(), nullable=False),
            sa.Column("video_id", sa.Integer(), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("author_name", sa.String(length=256), nullable=True),
            sa.Column("is_toxic", sa.Boolean(), nullable=False),
            sa.Column("confidence_score", sa.Float(), nullable=False),
            sa.Column("insult_target", sa.String(length=32), nullable=True),
            sa.Column("labeled_by", sa.String(length=16), nullable=False),
            sa.Column("labeled_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.ForeignKeyConstraint(["comment_id"], ["comments.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    _ensure_index(bind, "ix_toxic_training_data_comment_id", "toxic_training_data", ["comment_id"])
    _ensure_index(bind, "ix_toxic_training_data_video_id", "toxic_training_data", ["video_id"])
    _ensure_index(bind, "ix_toxic_training_is_toxic", "toxic_training_data", ["is_toxic"])
    _ensure_index(bind, "ix_toxic_training_labeled_by", "toxic_training_data", ["labeled_by"])

    existing_columns = _column_names(bind, "appeal_block_items")
    if "confidence_score" not in existing_columns:
        op.add_column(
            "appeal_block_items", sa.Column("confidence_score", sa.Float(), nullable=True)
        )
    if "insult_target" not in existing_columns:
        op.add_column(
            "appeal_block_items", sa.Column("insult_target", sa.String(length=32), nullable=True)
        )


def downgrade() -> None:
    # ca9a9ad66c80 is expected to own the final schema for fresh databases.
    # This repair migration only backfills already-stamped environments.
    pass
