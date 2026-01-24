"""Add Giskard tables

Revision ID: 024_add_giskard_tables
Revises: 023_add_batch_test_sessions
Create Date: 2026-01-22 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "024_add_giskard_tables"
down_revision: Union[str, None] = "023_add_batch_test_sessions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "giskard_test_sets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "question_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("created_by", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["course_id"], ["courses.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_giskard_test_sets_id"),
        "giskard_test_sets",
        ["id"],
        unique=False,
    )
    op.create_index(
        "ix_giskard_test_sets_course_id",
        "giskard_test_sets",
        ["course_id"],
        unique=False,
    )

    op.create_table(
        "giskard_questions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("test_set_id", sa.Integer(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("question_type", sa.String(length=50), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=False),
        sa.Column("question_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["test_set_id"], ["giskard_test_sets.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_giskard_questions_id"),
        "giskard_questions",
        ["id"],
        unique=False,
    )
    op.create_index(
        "ix_giskard_questions_test_set_id",
        "giskard_questions",
        ["test_set_id"],
        unique=False,
    )

    op.create_table(
        "giskard_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("test_set_id", sa.Integer(), nullable=False),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column(
            "total_questions",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "processed_questions",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["course_id"], ["courses.id"]),
        sa.ForeignKeyConstraint(["test_set_id"], ["giskard_test_sets.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_giskard_runs_id"),
        "giskard_runs",
        ["id"],
        unique=False,
    )
    op.create_index(
        "ix_giskard_runs_test_set_id",
        "giskard_runs",
        ["test_set_id"],
        unique=False,
    )
    op.create_index(
        "ix_giskard_runs_course_id",
        "giskard_runs",
        ["course_id"],
        unique=False,
    )

    op.create_table(
        "giskard_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=False),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=False),
        sa.Column("generated_answer", sa.Text(), nullable=False),
        sa.Column("question_type", sa.String(length=50), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("correct_refusal", sa.Boolean(), nullable=True),
        sa.Column("hallucinated", sa.Boolean(), nullable=True),
        sa.Column("provided_answer", sa.Boolean(), nullable=True),
        sa.Column("language", sa.String(length=50), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("llm_provider", sa.String(length=100), nullable=True),
        sa.Column("llm_model", sa.String(length=100), nullable=True),
        sa.Column("embedding_model", sa.String(length=100), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["question_id"], ["giskard_questions.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["giskard_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_giskard_results_id"),
        "giskard_results",
        ["id"],
        unique=False,
    )
    op.create_index(
        "ix_giskard_results_run_id",
        "giskard_results",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "ix_giskard_results_question_id",
        "giskard_results",
        ["question_id"],
        unique=False,
    )

    op.create_table(
        "giskard_summaries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column(
            "relevant_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("relevant_avg_score", sa.Float(), nullable=True),
        sa.Column("relevant_success_rate", sa.Float(), nullable=True),
        sa.Column(
            "irrelevant_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("irrelevant_avg_score", sa.Float(), nullable=True),
        sa.Column("irrelevant_success_rate", sa.Float(), nullable=True),
        sa.Column("hallucination_rate", sa.Float(), nullable=True),
        sa.Column("correct_refusal_rate", sa.Float(), nullable=True),
        sa.Column("language_consistency", sa.Float(), nullable=True),
        sa.Column("turkish_response_rate", sa.Float(), nullable=True),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column(
            "total_questions",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "successful_questions",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "failed_questions",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("avg_latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["giskard_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )
    op.create_index(
        op.f("ix_giskard_summaries_id"),
        "giskard_summaries",
        ["id"],
        unique=False,
    )
    op.create_index(
        "ix_giskard_summaries_run_id",
        "giskard_summaries",
        ["run_id"],
        unique=True,
    )

    op.create_table(
        "giskard_quick_test_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column("group_name", sa.String(length=255), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("question_type", sa.String(length=50), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=False),
        sa.Column("generated_answer", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("correct_refusal", sa.Boolean(), nullable=True),
        sa.Column("hallucinated", sa.Boolean(), nullable=True),
        sa.Column("provided_answer", sa.Boolean(), nullable=True),
        sa.Column("language", sa.String(length=50), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("llm_provider", sa.String(length=100), nullable=True),
        sa.Column("llm_model", sa.String(length=100), nullable=True),
        sa.Column("embedding_model", sa.String(length=100), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_by", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["course_id"], ["courses.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_giskard_quick_test_results_id"),
        "giskard_quick_test_results",
        ["id"],
        unique=False,
    )
    op.create_index(
        "ix_giskard_quick_test_results_course_id",
        "giskard_quick_test_results",
        ["course_id"],
        unique=False,
    )
    op.create_index(
        "ix_giskard_quick_test_results_group_name",
        "giskard_quick_test_results",
        ["group_name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_giskard_quick_test_results_group_name",
        table_name="giskard_quick_test_results",
    )
    op.drop_index(
        "ix_giskard_quick_test_results_course_id",
        table_name="giskard_quick_test_results",
    )
    op.drop_index(
        op.f("ix_giskard_quick_test_results_id"),
        table_name="giskard_quick_test_results",
    )
    op.drop_table("giskard_quick_test_results")

    op.drop_index(
        "ix_giskard_summaries_run_id",
        table_name="giskard_summaries",
    )
    op.drop_index(
        op.f("ix_giskard_summaries_id"),
        table_name="giskard_summaries",
    )
    op.drop_table("giskard_summaries")

    op.drop_index(
        "ix_giskard_results_question_id",
        table_name="giskard_results",
    )
    op.drop_index(
        "ix_giskard_results_run_id",
        table_name="giskard_results",
    )
    op.drop_index(
        op.f("ix_giskard_results_id"),
        table_name="giskard_results",
    )
    op.drop_table("giskard_results")

    op.drop_index(
        "ix_giskard_runs_course_id",
        table_name="giskard_runs",
    )
    op.drop_index(
        "ix_giskard_runs_test_set_id",
        table_name="giskard_runs",
    )
    op.drop_index(
        op.f("ix_giskard_runs_id"),
        table_name="giskard_runs",
    )
    op.drop_table("giskard_runs")

    op.drop_index(
        "ix_giskard_questions_test_set_id",
        table_name="giskard_questions",
    )
    op.drop_index(
        op.f("ix_giskard_questions_id"),
        table_name="giskard_questions",
    )
    op.drop_table("giskard_questions")

    op.drop_index(
        "ix_giskard_test_sets_course_id",
        table_name="giskard_test_sets",
    )
    op.drop_index(
        op.f("ix_giskard_test_sets_id"),
        table_name="giskard_test_sets",
    )
    op.drop_table("giskard_test_sets")
