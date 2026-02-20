"""Fix course_settings default values

Revision ID: 007_fix_course_settings_defaults
Revises: 006_add_system_prompt
Create Date: 2026-01-11 00:00:06

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '007_fix_course_settings_defaults'
down_revision: Union[str, None] = '006_add_system_prompt'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Update existing NULL values with defaults
    op.execute("""
        UPDATE course_settings 
        SET default_chunk_strategy = 'recursive' 
        WHERE default_chunk_strategy IS NULL
    """)
    op.execute("""
        UPDATE course_settings 
        SET default_chunk_size = 500 
        WHERE default_chunk_size IS NULL
    """)
    op.execute("""
        UPDATE course_settings 
        SET default_overlap = 50 
        WHERE default_overlap IS NULL
    """)
    op.execute("""
        UPDATE course_settings 
        SET default_embedding_model = 'openai/text-embedding-3-small' 
        WHERE default_embedding_model IS NULL
    """)
    op.execute("""
        UPDATE course_settings 
        SET search_alpha = 0.5 
        WHERE search_alpha IS NULL
    """)
    op.execute("""
        UPDATE course_settings 
        SET search_top_k = 5 
        WHERE search_top_k IS NULL
    """)
    op.execute("""
        UPDATE course_settings 
        SET llm_provider = 'openrouter' 
        WHERE llm_provider IS NULL
    """)
    op.execute("""
        UPDATE course_settings 
        SET llm_model = 'openai/gpt-4o-mini' 
        WHERE llm_model IS NULL
    """)
    op.execute("""
        UPDATE course_settings 
        SET llm_temperature = 0.7 
        WHERE llm_temperature IS NULL
    """)
    op.execute("""
        UPDATE course_settings 
        SET llm_max_tokens = 1000 
        WHERE llm_max_tokens IS NULL
    """)


def downgrade() -> None:
    # No downgrade needed - we're just fixing NULL values
    pass
