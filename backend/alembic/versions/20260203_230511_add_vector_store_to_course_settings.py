"""add vector_store to course_settings

Revision ID: 20260203_230511
Revises: 20260202_000001
Create Date: 2026-02-03 23:05:11

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260203_230511'
down_revision = '20260202_000001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add vector_store column to course_settings table.
    
    This allows choosing between Weaviate (default) and ChromaDB per course.
    """
    # Add vector_store column with default 'weaviate'
    op.add_column(
        'course_settings',
        sa.Column(
            'vector_store',
            sa.String(50),
            nullable=False,
            server_default='weaviate'
        )
    )


def downgrade() -> None:
    """Remove vector_store column from course_settings table."""
    op.drop_column('course_settings', 'vector_store')
