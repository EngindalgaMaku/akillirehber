"""add_test_generation_persona_to_course_settings

Revision ID: 938b9bf03e20
Revises: 8a5a340c8ee4
Create Date: 2026-01-29 00:12:01.656946

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '938b9bf03e20'
down_revision: Union[str, None] = '8a5a340c8ee4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add test_generation_persona column to course_settings
    op.add_column('course_settings', 
        sa.Column('test_generation_persona', sa.String(500), nullable=True)
    )
    
    # Set default persona for existing courses
    op.execute("""
        UPDATE course_settings 
        SET test_generation_persona = 'Bilişim Teknolojileri öğrencisi, teknik terimleri öğreniyor.'
        WHERE test_generation_persona IS NULL
    """)


def downgrade() -> None:
    # Drop test_generation_persona column
    op.drop_column('course_settings', 'test_generation_persona')
