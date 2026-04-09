"""Add toxic moderation tables, fields, and video settings.

Revision ID: ca9a9ad66c80
Revises: 0004_appeal_training_samples
Create Date: 2026-04-04 16:52:29.083031
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "ca9a9ad66c80"
down_revision = "0004_appeal_training_samples"
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


def _ensure_appeal_block_item_columns(bind: sa.engine.Connection) -> None:
    existing_columns = _column_names(bind, "appeal_block_items")
    if "confidence_score" not in existing_columns:
        op.add_column(
            "appeal_block_items", sa.Column("confidence_score", sa.Float(), nullable=True)
        )
    if "insult_target" not in existing_columns:
        op.add_column(
            "appeal_block_items", sa.Column("insult_target", sa.String(length=32), nullable=True)
        )


def _ensure_video_settings_table(bind: sa.engine.Connection) -> None:
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


def _ensure_banned_users_table(bind: sa.engine.Connection) -> None:
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


def _ensure_toxic_training_data_table(bind: sa.engine.Connection) -> None:
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


def upgrade() -> None:
    bind = op.get_bind()
    _ensure_banned_users_table(bind)
    _ensure_toxic_training_data_table(bind)
    _ensure_video_settings_table(bind)
    _ensure_appeal_block_item_columns(bind)


def downgrade() -> None:
    bind = op.get_bind()

    appeal_block_item_columns = _column_names(bind, "appeal_block_items")
    if "insult_target" in appeal_block_item_columns:
        op.drop_column("appeal_block_items", "insult_target")
    if "confidence_score" in appeal_block_item_columns:
        op.drop_column("appeal_block_items", "confidence_score")

    if "video_settings" in _table_names(bind):
        if "ix_video_settings_video_id" in _index_names(bind, "video_settings"):
            op.drop_index("ix_video_settings_video_id", table_name="video_settings")
        op.drop_table("video_settings")

    if "toxic_training_data" in _table_names(bind):
        for index_name in (
            "ix_toxic_training_labeled_by",
            "ix_toxic_training_is_toxic",
            "ix_toxic_training_data_video_id",
            "ix_toxic_training_data_comment_id",
        ):
            if index_name in _index_names(bind, "toxic_training_data"):
                op.drop_index(index_name, table_name="toxic_training_data")
        op.drop_table("toxic_training_data")

    if "banned_users" in _table_names(bind):
        for index_name in (
            "ix_banned_users_video_username",
            "ix_banned_users_username",
            "ix_banned_users_comment_id",
            "ix_banned_users_video_id",
        ):
            if index_name in _index_names(bind, "banned_users"):
                op.drop_index(index_name, table_name="banned_users")
        op.drop_table("banned_users")
