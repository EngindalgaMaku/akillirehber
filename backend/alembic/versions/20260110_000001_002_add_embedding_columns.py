"""Add embedding columns to documents table

Revision ID: 002_add_embedding_columns
Revises: 001
Create Date: 2026-01-10 00:00:01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002_add_embedding_columns'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add embedding-related columns to documents table
    op.add_column('documents', sa.Column(
        'embedding_status',
        sa.String(20),
        nullable=True,
        server_default='PENDING'
    ))
    op.add_column('documents', sa.Column(
        'embedding_model',
        sa.String(100),
        nullable=True
    ))
    op.add_column('documents', sa.Column(
        'embedded_at',
        sa.DateTime(timezone=True),
        nullable=True
    ))
    op.add_column('documents', sa.Column(
        'vector_count',
        sa.Integer(),
        nullable=True,
        server_default='0'
    ))


def downgrade() -> None:
    op.drop_column('documents', 'vector_count')
    op.drop_column('documents', 'embedded_at')
    op.drop_column('documents', 'embedding_model')
    op.drop_column('documents', 'embedding_status')
