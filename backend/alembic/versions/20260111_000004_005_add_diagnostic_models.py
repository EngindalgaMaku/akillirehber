"""Add diagnostic models for PDF debugging

Revision ID: 005_add_diagnostic_models
Revises: 004_add_llm_settings
Create Date: 2026-01-11 00:00:04.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005_add_diagnostic_models'
down_revision = '004_add_llm_settings'
branch_labels = None
depends_on = None


def upgrade():
    # Create processing_status table (using string for status to avoid enum issues)
    op.create_table('processing_status',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_details', sa.JSON(), nullable=True),
        sa.Column('processing_duration', sa.Float(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=True),
        sa.Column('last_retry_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id')
    )
    op.create_index(op.f('ix_processing_status_id'), 'processing_status', ['id'], unique=False)

    # Create diagnostic_reports table
    op.create_table('diagnostic_reports',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('report_type', sa.String(length=50), nullable=False),
        sa.Column('file_info', sa.JSON(), nullable=True),
        sa.Column('extraction_info', sa.JSON(), nullable=True),
        sa.Column('chunking_info', sa.JSON(), nullable=True),
        sa.Column('error_log', sa.JSON(), nullable=True),
        sa.Column('performance_metrics', sa.JSON(), nullable=True),
        sa.Column('recommendations', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_diagnostic_reports_id'), 'diagnostic_reports', ['id'], unique=False)

    # Create chunk_quality_metrics table
    op.create_table('chunk_quality_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('total_chunks', sa.Integer(), nullable=False),
        sa.Column('avg_chunk_size', sa.Integer(), nullable=False),
        sa.Column('min_chunk_size', sa.Integer(), nullable=False),
        sa.Column('max_chunk_size', sa.Integer(), nullable=False),
        sa.Column('size_distribution', sa.JSON(), nullable=True),
        sa.Column('overlap_analysis', sa.JSON(), nullable=True),
        sa.Column('content_quality_score', sa.Float(), nullable=False),
        sa.Column('recommendations', sa.JSON(), nullable=True),
        sa.Column('chunking_strategy', sa.String(length=50), nullable=True),
        sa.Column('chunk_size_config', sa.Integer(), nullable=True),
        sa.Column('overlap_config', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id')
    )
    op.create_index(op.f('ix_chunk_quality_metrics_id'), 'chunk_quality_metrics', ['id'], unique=False)


def downgrade():
    # Drop tables
    op.drop_index(op.f('ix_chunk_quality_metrics_id'), table_name='chunk_quality_metrics')
    op.drop_table('chunk_quality_metrics')
    op.drop_index(op.f('ix_diagnostic_reports_id'), table_name='diagnostic_reports')
    op.drop_table('diagnostic_reports')
    op.drop_index(op.f('ix_processing_status_id'), table_name='processing_status')
    op.drop_table('processing_status')