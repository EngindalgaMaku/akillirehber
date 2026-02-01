"""add bloom_level to semantic_similarity_results

Revision ID: 20260130_000003
Revises: 20260130_000002_bloom_backfill
Create Date: 2026-01-30 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260130_000003'
down_revision = '20260130_000002_bloom_backfill'
branch_labels = None
depends_on = None


def upgrade():
    # Add bloom_level column to semantic_similarity_results
    op.add_column('semantic_similarity_results', 
                  sa.Column('bloom_level', sa.String(length=50), nullable=True))


def downgrade():
    # Remove bloom_level column
    op.drop_column('semantic_similarity_results', 'bloom_level')
