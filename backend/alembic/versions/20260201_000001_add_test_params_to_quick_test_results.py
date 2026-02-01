"""add test-time params to quick_test_results

Revision ID: 20260201_000001
Revises: 20260130_000003
Create Date: 2026-02-01 00:00:01.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260201_000001"
down_revision = "20260130_000003"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "quick_test_results",
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "quick_test_results",
        sa.Column("search_top_k", sa.Integer(), nullable=True),
    )
    op.add_column(
        "quick_test_results",
        sa.Column("search_alpha", sa.Float(), nullable=True),
    )
    op.add_column(
        "quick_test_results",
        sa.Column("reranker_used", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "quick_test_results",
        sa.Column("reranker_provider", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "quick_test_results",
        sa.Column("reranker_model", sa.String(length=255), nullable=True),
    )


def downgrade():
    op.drop_column("quick_test_results", "reranker_model")
    op.drop_column("quick_test_results", "reranker_provider")
    op.drop_column("quick_test_results", "reranker_used")
    op.drop_column("quick_test_results", "search_alpha")
    op.drop_column("quick_test_results", "search_top_k")
    op.drop_column("quick_test_results", "embedding_model")
