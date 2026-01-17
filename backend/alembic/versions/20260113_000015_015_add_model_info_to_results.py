"""Add model info to evaluation results

Revision ID: 015_add_model_info
Revises: 014_add_refresh_tokens
Create Date: 2026-01-13 01:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '015_add_model_info'
down_revision: Union[str, None] = '014_add_refresh_tokens'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add model information columns to evaluation_results
    op.add_column('evaluation_results', sa.Column('llm_provider', sa.String(), nullable=True))
    op.add_column('evaluation_results', sa.Column('llm_model', sa.String(), nullable=True))
    op.add_column('evaluation_results', sa.Column('embedding_model', sa.String(), nullable=True))
    op.add_column('evaluation_results', sa.Column('evaluation_model', sa.String(), nullable=True))
    op.add_column('evaluation_results', sa.Column('search_alpha', sa.Float(), nullable=True))
    op.add_column('evaluation_results', sa.Column('search_top_k', sa.Integer(), nullable=True))


def downgrade() -> None:
    # Remove model information columns from evaluation_results
    op.drop_column('evaluation_results', 'search_top_k')
    op.drop_column('evaluation_results', 'search_alpha')
    op.drop_column('evaluation_results', 'evaluation_model')
    op.drop_column('evaluation_results', 'embedding_model')
    op.drop_column('evaluation_results', 'llm_model')
    op.drop_column('evaluation_results', 'llm_provider')
