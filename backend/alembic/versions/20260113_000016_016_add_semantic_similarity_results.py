"""Add semantic_similarity_results table

Revision ID: 016_add_semantic_similarity_results
Revises: 015_add_model_info
Create Date: 2026-01-13 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '016_semantic_sim'
down_revision: Union[str, None] = '015_add_model_info'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'semantic_similarity_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('group_name', sa.String(255), nullable=True),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('ground_truth', sa.Text(), nullable=False),
        sa.Column('alternative_ground_truths', sa.JSON(), nullable=True),
        sa.Column('generated_answer', sa.Text(), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=False),
        sa.Column('best_match_ground_truth', sa.Text(), nullable=False),
        sa.Column('all_scores', sa.JSON(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=False),
        sa.Column('embedding_model_used', sa.String(255), nullable=False),
        sa.Column('llm_model_used', sa.String(255), nullable=True),
        sa.Column('created_by', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_semantic_similarity_results_id'),
        'semantic_similarity_results',
        ['id'],
        unique=False
    )
    op.create_index(
        'ix_semantic_similarity_results_course_id',
        'semantic_similarity_results',
        ['course_id'],
        unique=False
    )
    op.create_index(
        'ix_semantic_similarity_results_group_name',
        'semantic_similarity_results',
        ['group_name'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index('ix_semantic_similarity_results_group_name', table_name='semantic_similarity_results')
    op.drop_index('ix_semantic_similarity_results_course_id', table_name='semantic_similarity_results')
    op.drop_index(op.f('ix_semantic_similarity_results_id'), table_name='semantic_similarity_results')
    op.drop_table('semantic_similarity_results')
