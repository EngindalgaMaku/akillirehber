"""Add min_relevance_score to course_settings.

Revision ID: 009_add_min_relevance_score
Revises: 008_add_ragas_models
Create Date: 2026-01-11
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '009_add_min_relevance_score'
down_revision = '008_add_ragas_models'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add min_relevance_score column
    op.add_column(
        'course_settings',
        sa.Column('min_relevance_score', sa.Float(), nullable=True, server_default='0.0')
    )


def downgrade() -> None:
    op.drop_column('course_settings', 'min_relevance_score')
