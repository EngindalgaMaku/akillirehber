"""Add custom LLM models table

Revision ID: 013_add_custom_llm_models
Revises: 012_add_quick_test_results
Create Date: 2026-01-12 16:41:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '013_add_custom_llm_models'
down_revision: Union[str, None] = '012_add_quick_test_results'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'custom_llm_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(50), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('display_name', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_by', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_custom_llm_models_id'),
        'custom_llm_models',
        ['id'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_custom_llm_models_id'),
        table_name='custom_llm_models'
    )
    op.drop_table('custom_llm_models')