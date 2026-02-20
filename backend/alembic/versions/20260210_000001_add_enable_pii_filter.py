"""add enable_pii_filter to course_settings

Revision ID: 20260210_000001
Revises: 20260209_000001
Create Date: 2026-02-10 00:00:01.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260210_000001'
down_revision = '20260209_000001'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'course_settings',
        sa.Column('enable_pii_filter', sa.Boolean(), nullable=False, server_default=sa.text('false'))
    )


def downgrade():
    op.drop_column('course_settings', 'enable_pii_filter')
