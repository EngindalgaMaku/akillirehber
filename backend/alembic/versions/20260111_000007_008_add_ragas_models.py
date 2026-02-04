"""Add RAGAS evaluation models

Revision ID: 008_add_ragas_models
Revises: 007_fix_course_settings_defaults
Create Date: 2026-01-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '008_add_ragas_models'
down_revision: Union[str, None] = '007_fix_course_settings_defaults'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check existing tables
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()
    
    # Create test_sets table
    if 'test_sets' not in existing_tables:
        op.create_table(
            'test_sets',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('course_id', sa.Integer(), nullable=False),
            sa.Column('name', sa.String(255), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('created_by', sa.Integer(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ),
            sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_test_sets_id'), 'test_sets', ['id'], unique=False)

    # Create test_questions table
    if 'test_questions' not in existing_tables:
        op.create_table(
            'test_questions',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('test_set_id', sa.Integer(), nullable=False),
            sa.Column('question', sa.Text(), nullable=False),
            sa.Column('ground_truth', sa.Text(), nullable=False),
            sa.Column('expected_contexts', sa.JSON(), nullable=True),
            sa.Column('question_metadata', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['test_set_id'], ['test_sets.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_test_questions_id'), 'test_questions', ['id'], unique=False)

    # Create evaluation_runs table
    if 'evaluation_runs' not in existing_tables:
        op.create_table(
            'evaluation_runs',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('test_set_id', sa.Integer(), nullable=False),
            sa.Column('course_id', sa.Integer(), nullable=False),
            sa.Column('name', sa.String(255), nullable=True),
            sa.Column('status', sa.String(50), nullable=False, server_default='pending'),
            sa.Column('config', sa.JSON(), nullable=True),
            sa.Column('total_questions', sa.Integer(), nullable=True, server_default='0'),
            sa.Column('processed_questions', sa.Integer(), nullable=True, server_default='0'),
            sa.Column('started_at', sa.DateTime(), nullable=True),
            sa.Column('completed_at', sa.DateTime(), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['test_set_id'], ['test_sets.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_evaluation_runs_id'), 'evaluation_runs', ['id'], unique=False)

    # Create evaluation_results table
    if 'evaluation_results' not in existing_tables:
        op.create_table(
            'evaluation_results',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('run_id', sa.Integer(), nullable=False),
            sa.Column('question_id', sa.Integer(), nullable=False),
            sa.Column('question_text', sa.Text(), nullable=False),
            sa.Column('ground_truth_text', sa.Text(), nullable=False),
            sa.Column('generated_answer', sa.Text(), nullable=True),
            sa.Column('retrieved_contexts', sa.JSON(), nullable=True),
            sa.Column('faithfulness', sa.Float(), nullable=True),
            sa.Column('answer_relevancy', sa.Float(), nullable=True),
            sa.Column('context_precision', sa.Float(), nullable=True),
            sa.Column('context_recall', sa.Float(), nullable=True),
            sa.Column('answer_correctness', sa.Float(), nullable=True),
            sa.Column('latency_ms', sa.Integer(), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['run_id'], ['evaluation_runs.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['question_id'], ['test_questions.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_evaluation_results_id'), 'evaluation_results', ['id'], unique=False)

    # Create run_summaries table
    if 'run_summaries' not in existing_tables:
        op.create_table(
            'run_summaries',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('run_id', sa.Integer(), nullable=False),
            sa.Column('avg_faithfulness', sa.Float(), nullable=True),
            sa.Column('avg_answer_relevancy', sa.Float(), nullable=True),
            sa.Column('avg_context_precision', sa.Float(), nullable=True),
            sa.Column('avg_context_recall', sa.Float(), nullable=True),
            sa.Column('avg_answer_correctness', sa.Float(), nullable=True),
            sa.Column('avg_latency_ms', sa.Float(), nullable=True),
            sa.Column('total_questions', sa.Integer(), nullable=True, server_default='0'),
            sa.Column('successful_questions', sa.Integer(), nullable=True, server_default='0'),
            sa.Column('failed_questions', sa.Integer(), nullable=True, server_default='0'),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['run_id'], ['evaluation_runs.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('run_id')
        )
        op.create_index(op.f('ix_run_summaries_id'), 'run_summaries', ['id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_run_summaries_id'), table_name='run_summaries')
    op.drop_table('run_summaries')
    
    op.drop_index(op.f('ix_evaluation_results_id'), table_name='evaluation_results')
    op.drop_table('evaluation_results')
    
    op.drop_index(op.f('ix_evaluation_runs_id'), table_name='evaluation_runs')
    op.drop_table('evaluation_runs')
    
    op.drop_index(op.f('ix_test_questions_id'), table_name='test_questions')
    op.drop_table('test_questions')
    
    op.drop_index(op.f('ix_test_sets_id'), table_name='test_sets')
    op.drop_table('test_sets')
