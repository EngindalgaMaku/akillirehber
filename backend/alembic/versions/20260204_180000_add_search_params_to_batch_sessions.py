"""add search params to batch sessions

Revision ID: 20260204_180000
Revises: 305b5391cb41
Create Date: 2026-02-04 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260204_180000'
down_revision = '305b5391cb41'
branch_labels = None
depends_on = None


def upgrade():
    # Add search_top_k and search_alpha to batch_test_sessions
    op.add_column('batch_test_sessions', sa.Column('search_top_k', sa.Integer(), nullable=True))
    op.add_column('batch_test_sessions', sa.Column('search_alpha', sa.Float(), nullable=True))


def downgrade():
    op.drop_column('batch_test_sessions', 'search_alpha')
    op.drop_column('batch_test_sessions', 'search_top_k')
