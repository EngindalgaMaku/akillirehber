"""Add quick_test_results table

Revision ID: 012_add_quick_test_results
Revises: 011_add_alternative_ground_truths
Create Date: 2026-01-12 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '012_add_quick_test_results'
down_revision: Union[str, None] = '011_alt_ground_truths'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'quick_test_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('group_name', sa.String(255), nullable=True),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('ground_truth', sa.Text(), nullable=False),
        sa.Column('alternative_ground_truths', sa.JSON(), nullable=True),
        sa.Column('system_prompt', sa.Text(), nullable=True),
        sa.Column('llm_provider', sa.String(100), nullable=False),
        sa.Column('llm_model', sa.String(255), nullable=False),
        sa.Column('generated_answer', sa.Text(), nullable=False),
        sa.Column('retrieved_contexts', sa.JSON(), nullable=True),
        sa.Column('faithfulness', sa.Float(), nullable=True),
        sa.Column('answer_relevancy', sa.Float(), nullable=True),
        sa.Column('context_precision', sa.Float(), nullable=True),
        sa.Column('context_recall', sa.Float(), nullable=True),
        sa.Column('answer_correctness', sa.Float(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_quick_test_results_id'),
        'quick_test_results',
        ['id'],
        unique=False
    )
    op.create_index(
        'ix_quick_test_results_course_id',
        'quick_test_results',
        ['course_id'],
        unique=False
    )
    op.create_index(
        'ix_quick_test_results_group_name',
        'quick_test_results',
        ['group_name'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index('ix_quick_test_results_group_name', table_name='quick_test_results')
    op.drop_index('ix_quick_test_results_course_id', table_name='quick_test_results')
    op.drop_index(op.f('ix_quick_test_results_id'), table_name='quick_test_results')
    op.drop_table('quick_test_results')