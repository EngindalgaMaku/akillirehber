"""add contexts to semantic similarity

Revision ID: 018_add_contexts
Revises: 017_add_reranker
Create Date: 2026-01-15 14:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '018_add_contexts'
down_revision: Union[str, None] = '017_add_reranker'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add retrieved_contexts and system_prompt_used columns if they don't exist
    from sqlalchemy import inspect
    from sqlalchemy.engine import reflection
    
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('semantic_similarity_results')]
    
    if 'retrieved_contexts' not in columns:
        op.add_column('semantic_similarity_results', 
                      sa.Column('retrieved_contexts', sa.JSON(), nullable=True))
    
    if 'system_prompt_used' not in columns:
        op.add_column('semantic_similarity_results', 
                      sa.Column('system_prompt_used', sa.Text(), nullable=True))


def downgrade() -> None:
    # Remove the columns
    op.drop_column('semantic_similarity_results', 'system_prompt_used')
    op.drop_column('semantic_similarity_results', 'retrieved_contexts')
