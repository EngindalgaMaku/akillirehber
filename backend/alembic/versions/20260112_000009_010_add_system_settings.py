"""Add system settings table for registration keys and captcha

Revision ID: 010_add_system_settings
Revises: 009_add_min_relevance_score
Create Date: 2026-01-12 01:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '010_add_system_settings'
down_revision: Union[str, None] = '009_add_min_relevance_score'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create system_settings table
    op.create_table(
        'system_settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('teacher_registration_key', sa.String(255), nullable=True),
        sa.Column('student_registration_key', sa.String(255), nullable=True),
        sa.Column('hcaptcha_site_key', sa.String(255), nullable=True),
        sa.Column('hcaptcha_secret_key', sa.String(255), nullable=True),
        sa.Column('captcha_enabled', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_system_settings_id'), 'system_settings', ['id'])
    
    # Insert default settings row
    op.execute("""
        INSERT INTO system_settings (
            teacher_registration_key, 
            student_registration_key, 
            captcha_enabled,
            created_at, 
            updated_at
        ) VALUES (
            'TEACHER2026',
            'STUDENT2026', 
            false,
            NOW(), 
            NOW()
        )
    """)


def downgrade() -> None:
    op.drop_index(op.f('ix_system_settings_id'), table_name='system_settings')
    op.drop_table('system_settings')
