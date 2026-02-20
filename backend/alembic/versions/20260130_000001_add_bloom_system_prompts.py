"""Add Bloom-specific system prompts to course_settings

Revision ID: 20260130_000001_bloom_prompts
Revises: 938b9bf03e20
Create Date: 2026-01-30 00:00:01.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260130_000001_bloom_prompts"
down_revision = "938b9bf03e20"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add Bloom-specific system prompt columns to course_settings table."""

    op.add_column(
        "course_settings",
        sa.Column(
            "system_prompt_remembering",
            sa.Text(),
            nullable=True,
        ),
    )
    op.add_column(
        "course_settings",
        sa.Column(
            "system_prompt_understanding_applying",
            sa.Text(),
            nullable=True,
        ),
    )
    op.add_column(
        "course_settings",
        sa.Column(
            "system_prompt_analyzing_evaluating",
            sa.Text(),
            nullable=True,
        ),
    )

    # Backfill old system_prompt into Bloom-specific prompts.
    connection = op.get_bind()
    connection.execute(
        sa.text(
            """
            UPDATE course_settings
            SET
              system_prompt_remembering = COALESCE(
                system_prompt_remembering,
                system_prompt
              ),
              system_prompt_understanding_applying = COALESCE(
                system_prompt_understanding_applying,
                system_prompt
              ),
              system_prompt_analyzing_evaluating = COALESCE(
                system_prompt_analyzing_evaluating,
                system_prompt
              )
            """
        )
    )


def downgrade() -> None:
    """Remove Bloom-specific system prompt columns.

    Removes the Bloom-specific system prompt columns from course_settings.
    """

    op.drop_column("course_settings", "system_prompt_analyzing_evaluating")
    op.drop_column("course_settings", "system_prompt_understanding_applying")
    op.drop_column("course_settings", "system_prompt_remembering")
