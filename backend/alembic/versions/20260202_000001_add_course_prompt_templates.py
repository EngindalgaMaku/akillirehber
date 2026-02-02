"""Add per-course prompt templates and active template selection

Revision ID: 20260202_000001
Revises: 20260201_000001
Create Date: 2026-02-02 00:01:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260202_000001"
down_revision = "20260201_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "course_prompt_templates",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column(
            "course_id",
            sa.Integer(),
            sa.ForeignKey("courses.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "course_id",
            "name",
            name="uq_course_prompt_templates_course_id_name",
        ),
    )

    op.add_column(
        "course_settings",
        sa.Column(
            "active_prompt_template_id",
            sa.Integer(),
            sa.ForeignKey("course_prompt_templates.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("course_settings", "active_prompt_template_id")
    op.drop_table("course_prompt_templates")
