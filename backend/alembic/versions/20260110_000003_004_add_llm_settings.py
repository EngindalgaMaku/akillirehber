"""Add LLM configuration columns to course_settings

Revision ID: 004_add_llm_settings
Revises: 003_add_course_settings
Create Date: 2026-01-10 00:00:03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '004_add_llm_settings'
down_revision: Union[str, None] = '003_add_course_settings'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add LLM configuration columns to course_settings table
    # Check if columns exist before adding
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('course_settings')]
    
    if 'llm_provider' not in columns:
        op.add_column('course_settings', sa.Column(
            'llm_provider',
            sa.String(50),
            nullable=True,
            server_default='openrouter'
        ))
    if 'llm_model' not in columns:
        op.add_column('course_settings', sa.Column(
            'llm_model',
            sa.String(255),
            nullable=True,
            server_default='openai/gpt-4o-mini'
        ))
    if 'llm_temperature' not in columns:
        op.add_column('course_settings', sa.Column(
            'llm_temperature',
            sa.Float(),
            nullable=True,
            server_default='0.7'
        ))
    if 'llm_max_tokens' not in columns:
        op.add_column('course_settings', sa.Column(
            'llm_max_tokens',
            sa.Integer(),
            nullable=True,
            server_default='1000'
        ))


def downgrade() -> None:
    op.drop_column('course_settings', 'llm_max_tokens')
    op.drop_column('course_settings', 'llm_temperature')
    op.drop_column('course_settings', 'llm_model')
    op.drop_column('course_settings', 'llm_provider')