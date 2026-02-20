"""Add system_prompt to course_settings

Revision ID: 006_add_system_prompt
Revises: 005_add_diagnostic_models
Create Date: 2026-01-11 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '006_add_system_prompt'
down_revision = '005_add_diagnostic_models'
branch_labels = None
depends_on = None

# Default system prompt for educational assistant
DEFAULT_SYSTEM_PROMPT = """Sen yardımcı bir eğitim asistanısın. Öğrencilere ve öğretmenlere ders materyalleri hakkında açık, doğru ve eğitici yanıtlar veriyorsun. 

Görevlerin:
- Ders içeriğini anlaşılır şekilde açıklamak
- Öğrenci sorularını sabırla yanıtlamak  
- Kaynak materyallere dayalı bilgi vermek
- Öğrenmeyi teşvik edici bir ton kullanmak

Her zaman:
- Nazik ve profesyonel ol
- Eğitici ve yapıcı yanıtlar ver
- Kaynaklarını belirt
- Anlaşılır dil kullan"""


def upgrade() -> None:
    """Add system_prompt column to course_settings table."""
    # Add the system_prompt column
    op.add_column('course_settings', sa.Column('system_prompt', sa.Text(), nullable=True))
    
    # Update existing records with default system prompt
    connection = op.get_bind()
    connection.execute(
        sa.text("UPDATE course_settings SET system_prompt = :prompt WHERE system_prompt IS NULL"),
        {"prompt": DEFAULT_SYSTEM_PROMPT}
    )


def downgrade() -> None:
    """Remove system_prompt column from course_settings table."""
    op.drop_column('course_settings', 'system_prompt')