"""Add course_settings table

Revision ID: 003_add_course_settings
Revises: 002_add_embedding_columns
Create Date: 2026-01-10 00:00:02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_add_course_settings'
down_revision: Union[str, None] = '002_add_embedding_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'course_settings',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('course_id', sa.Integer(), sa.ForeignKey('courses.id'), unique=True, nullable=False),
        sa.Column('default_chunk_strategy', sa.String(50), default='recursive'),
        sa.Column('default_chunk_size', sa.Integer(), default=500),
        sa.Column('default_overlap', sa.Integer(), default=50),
        sa.Column('default_embedding_model', sa.String(255), default='openai/text-embedding-3-small'),
        sa.Column('search_alpha', sa.Float(), default=0.5),
        sa.Column('search_top_k', sa.Integer(), default=5),
        sa.Column('llm_provider', sa.String(50), default='openrouter'),
        sa.Column('llm_model', sa.String(255), default='openai/gpt-4o-mini'),
        sa.Column('llm_temperature', sa.Float(), default=0.7),
        sa.Column('llm_max_tokens', sa.Integer(), default=1000),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('course_settings')
