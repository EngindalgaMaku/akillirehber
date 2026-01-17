"""Add refresh tokens table

Revision ID: 014_add_refresh_tokens
Revises: 013_add_custom_llm_models
Create Date: 2026-01-12 20:47:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "014_add_refresh_tokens"
down_revision: Union[str, None] = "013_add_custom_llm_models"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create refresh_tokens table."""
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("token", sa.String(500), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("revoked", sa.Boolean(), nullable=True, default=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_refresh_tokens_id"),
        "refresh_tokens",
        ["id"],
        unique=False
    )
    op.create_index(
        op.f("ix_refresh_tokens_token"),
        "refresh_tokens",
        ["token"],
        unique=True
    )


def downgrade() -> None:
    """Drop refresh_tokens table."""
    op.drop_index(
        op.f("ix_refresh_tokens_token"),
        table_name="refresh_tokens"
    )
    op.drop_index(
        op.f("ix_refresh_tokens_id"),
        table_name="refresh_tokens"
    )
    op.drop_table("refresh_tokens")