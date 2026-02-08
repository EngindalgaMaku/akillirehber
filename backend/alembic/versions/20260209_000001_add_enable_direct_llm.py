"""add enable_direct_llm to course_settings

Revision ID: 20260209_000001
Revises: 20260204_180000
Create Date: 2026-02-09 00:00:01.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260209_000001'
down_revision = '20260204_180000'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'course_settings',
        sa.Column('enable_direct_llm', sa.Boolean(), nullable=False, server_default=sa.text('false'))
    )


def downgrade():
    op.drop_column('course_settings', 'enable_direct_llm')
