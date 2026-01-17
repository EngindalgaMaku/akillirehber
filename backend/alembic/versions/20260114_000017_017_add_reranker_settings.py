"""Add reranker settings to course_settings

Revision ID: 017_add_reranker_settings
Revises: 016_semantic_sim
Create Date: 2026-01-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '017_add_reranker'
down_revision: Union[str, None] = '016_semantic_sim'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add reranker fields to course_settings table
    op.add_column('course_settings', sa.Column('enable_reranker', sa.Boolean(), nullable=False, server_default='0'))
    op.add_column('course_settings', sa.Column('reranker_provider', sa.String(50), nullable=True))
    op.add_column('course_settings', sa.Column('reranker_model', sa.String(100), nullable=True))
    op.add_column('course_settings', sa.Column('reranker_top_k', sa.Integer(), nullable=False, server_default='100'))


def downgrade() -> None:
    # Remove reranker fields from course_settings table
    op.drop_column('course_settings', 'reranker_top_k')
    op.drop_column('course_settings', 'reranker_model')
    op.drop_column('course_settings', 'reranker_provider')
    op.drop_column('course_settings', 'enable_reranker')
