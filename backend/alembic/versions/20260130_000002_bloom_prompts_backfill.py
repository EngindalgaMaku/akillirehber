"""Backfill Bloom-specific system prompts with defaults

Revision ID: 20260130_000002_bloom_backfill
Revises: 20260130_000001_bloom_prompts
Create Date: 2026-01-30 00:00:02.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260130_000002_bloom_backfill"
down_revision = "20260130_000001_bloom_prompts"
branch_labels = None
depends_on = None


DEFAULT_TEST_SYSTEM_PROMPT_REMEMBERING = (
    "Sen bir eğitim uzmanısın ve Bloom Taksonomisi 'Hatırlama' seviyesinde soru "
    "üretiyorsun. Görev: İçerikten DOĞRUDAN bilgi çekmeyi gerektiren tanım/liste/"
    "terim odaklı bir soru üret. KURALLAR: (1) Soru self-contained olmalı. "
    "(2) Cevap içerikte geçmeli veya parafraz olmalı. (3) Cevap 2-3 cümle. "
    "(4) Çıktı: 'SORU:' ve 'CEVAP:'"
)

DEFAULT_TEST_SYSTEM_PROMPT_UNDERSTANDING_APPLYING = (
    "Sen bir eğitim uzmanısın ve Bloom Taksonomisi 'Anlama/Uygulama' seviyesinde "
    "soru üretiyorsun. Görev: Yorumlama veya uygulama gerektiren senaryolu soru üret. "
    "KURALLAR: (1) Self-contained. (2) Cevap 3-4 cümle akıl yürütme içersin. "
    "(3) İçeriğe dayalı kal. (4) Çıktı: 'SORU:' ve 'CEVAP:'"
)

DEFAULT_TEST_SYSTEM_PROMPT_ANALYZING_EVALUATING = (
    "Sen bir eğitim uzmanısın ve Bloom Taksonomisi 'Analiz/Değerlendirme' seviyesinde "
    "soru üretiyorsun. Görev: karşılaştırma/analiz/değerlendirme sorusu üret. "
    "KURALLAR: (1) Self-contained. (2) Cevap 4-5 cümle, sentez içersin. "
    "(3) İçeriğe dayalı kal. (4) Çıktı: 'SORU:' ve 'CEVAP:'"
)


def upgrade() -> None:
    """Backfill bloom prompts.

    If bloom prompt columns are NULL or equal to system_prompt (generic copy),
    replace them with meaningful defaults.
    """

    connection = op.get_bind()
    connection.execute(
        sa.text(
            """
            UPDATE course_settings
            SET system_prompt_remembering = :remembering
            WHERE system_prompt_remembering IS NULL
               OR system_prompt_remembering = system_prompt
            """
        ),
        {"remembering": DEFAULT_TEST_SYSTEM_PROMPT_REMEMBERING},
    )

    connection.execute(
        sa.text(
            """
            UPDATE course_settings
            SET system_prompt_understanding_applying = :ua
            WHERE system_prompt_understanding_applying IS NULL
               OR system_prompt_understanding_applying = system_prompt
            """
        ),
        {"ua": DEFAULT_TEST_SYSTEM_PROMPT_UNDERSTANDING_APPLYING},
    )

    connection.execute(
        sa.text(
            """
            UPDATE course_settings
            SET system_prompt_analyzing_evaluating = :ae
            WHERE system_prompt_analyzing_evaluating IS NULL
               OR system_prompt_analyzing_evaluating = system_prompt
            """
        ),
        {"ae": DEFAULT_TEST_SYSTEM_PROMPT_ANALYZING_EVALUATING},
    )


def downgrade() -> None:
    """No-op downgrade."""

    # Intentionally left blank: do not destroy user-customized prompts.
    return
