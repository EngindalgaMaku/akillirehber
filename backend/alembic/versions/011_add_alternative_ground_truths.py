"""add alternative ground truths

Revision ID: 011_alt_ground_truths
Revises: 010_add_system_settings
Create Date: 2026-01-12 11:47:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '011_alt_ground_truths'
down_revision = '010_add_system_settings'
branch_labels = None
depends_on = None


def upgrade():
    # Add alternative_ground_truths column to test_questions table
    op.add_column(
        'test_questions',
        sa.Column('alternative_ground_truths', sa.JSON(), nullable=True)
    )


def downgrade():
    # Remove alternative_ground_truths column
    op.drop_column('test_questions', 'alternative_ground_truths')