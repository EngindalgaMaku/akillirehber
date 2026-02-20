"""Add batch test sessions table

Revision ID: 023_add_batch_test_sessions
Revises: 022_course_id_nullable
Create Date: 2026-01-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '023_add_batch_test_sessions'
down_revision: Union[str, None] = "022_course_id_nullable"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create batch_test_sessions table
    op.create_table(
        'batch_test_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('course_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('group_name', sa.String(length=255), nullable=False),
        sa.Column('test_cases', sa.Text(), nullable=False),
        sa.Column('total_tests', sa.Integer(), nullable=False, default=0),
        sa.Column('completed_tests', sa.Integer(), nullable=False, default=0),
        sa.Column('failed_tests', sa.Integer(), nullable=False, default=0),
        sa.Column('current_index', sa.Integer(), nullable=False, default=0),
        sa.Column('status', sa.String(length=50), nullable=False, default='in_progress'),
        sa.Column('llm_provider', sa.String(length=50), nullable=True),
        sa.Column('llm_model', sa.String(length=255), nullable=True),
        sa.Column('embedding_model_used', sa.String(length=255), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_batch_test_sessions_id'), 'batch_test_sessions', ['id'], unique=False)
    op.create_index(op.f('ix_batch_test_sessions_course_id'), 'batch_test_sessions', ['course_id'], unique=False)
    op.create_index(op.f('ix_batch_test_sessions_user_id'), 'batch_test_sessions', ['user_id'], unique=False)
    op.create_index(op.f('ix_batch_test_sessions_group_name'), 'batch_test_sessions', ['group_name'], unique=False)

    # Add batch_session_id column to semantic_similarity_results table
    op.add_column(
        'semantic_similarity_results',
        sa.Column('batch_session_id', sa.Integer(), nullable=True)
    )
    op.create_index(
        op.f('ix_semantic_similarity_results_batch_session_id'),
        'semantic_similarity_results',
        ['batch_session_id'],
        unique=False
    )
    op.create_foreign_key(
        'fk_semantic_similarity_results_batch_session_id',
        'semantic_similarity_results',
        'batch_test_sessions',
        ['batch_session_id'],
        ['id']
    )


def downgrade() -> None:
    # Remove batch_session_id column from semantic_similarity_results
    op.drop_index(
        op.f('ix_semantic_similarity_results_batch_session_id'),
        table_name='semantic_similarity_results'
    )
    op.drop_constraint(
        'fk_semantic_similarity_results_batch_session_id',
        'semantic_similarity_results',
        type_='foreignkey'
    )
    op.drop_column('semantic_similarity_results', 'batch_session_id')

    # Drop batch_test_sessions table
    op.drop_index(op.f('ix_batch_test_sessions_group_name'), table_name='batch_test_sessions')
    op.drop_index(op.f('ix_batch_test_sessions_user_id'), table_name='batch_test_sessions')
    op.drop_index(op.f('ix_batch_test_sessions_course_id'), table_name='batch_test_sessions')
    op.drop_index(op.f('ix_batch_test_sessions_id'), table_name='batch_test_sessions')
    op.drop_table('batch_test_sessions')
