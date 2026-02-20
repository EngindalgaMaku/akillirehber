"""add_original_bertscore_to_semantic_similarity_results

Revision ID: 0d8a5e6f7a1b
Revises: bfc4c1885c2e
Create Date: 2026-01-26 11:24:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0d8a5e6f7a1b"
down_revision: Union[str, None] = "bfc4c1885c2e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "semantic_similarity_results",
        sa.Column("original_bertscore_precision", sa.Float(), nullable=True),
    )
    op.add_column(
        "semantic_similarity_results",
        sa.Column("original_bertscore_recall", sa.Float(), nullable=True),
    )
    op.add_column(
        "semantic_similarity_results",
        sa.Column("original_bertscore_f1", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [
        col["name"]
        for col in inspector.get_columns("semantic_similarity_results")
    ]

    if "original_bertscore_f1" in columns:
        op.drop_column("semantic_similarity_results", "original_bertscore_f1")
    if "original_bertscore_recall" in columns:
        op.drop_column(
            "semantic_similarity_results",
            "original_bertscore_recall",
        )
    if "original_bertscore_precision" in columns:
        op.drop_column(
            "semantic_similarity_results",
            "original_bertscore_precision",
        )
