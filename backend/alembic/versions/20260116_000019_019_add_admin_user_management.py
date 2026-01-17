"""Add admin user management models

Revision ID: 019_add_admin_user_management
Revises: 018_add_contexts_to_semantic_similarity
Create Date: 2026-01-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "019_add_admin_user_management"
down_revision: Union[str, None] = "018_add_contexts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add admin user management tables and columns."""
    
    # Add last_login column to users table if it doesn't exist
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(sa.Column("last_login", sa.DateTime(), nullable=True))
    
    # Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("admin_user_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("target_user_id", sa.Integer(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.ForeignKeyConstraint(
            ["admin_user_id"],
            ["users.id"],
            ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["target_user_id"],
            ["users.id"],
            ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_audit_logs_id"),
        "audit_logs",
        ["id"],
        unique=False
    )
    op.create_index(
        op.f("ix_audit_logs_timestamp"),
        "audit_logs",
        ["timestamp"],
        unique=False
    )
    
    # Create temporary_passwords table
    op.create_table(
        "temporary_passwords",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("used", sa.Boolean(), nullable=False, default=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_temporary_passwords_id"),
        "temporary_passwords",
        ["id"],
        unique=False
    )


def downgrade() -> None:
    """Remove admin user management tables and columns."""
    
    # Drop temporary_passwords table
    op.drop_index(
        op.f("ix_temporary_passwords_id"),
        table_name="temporary_passwords"
    )
    op.drop_table("temporary_passwords")
    
    # Drop audit_logs table
    op.drop_index(
        op.f("ix_audit_logs_timestamp"),
        table_name="audit_logs"
    )
    op.drop_index(
        op.f("ix_audit_logs_id"),
        table_name="audit_logs"
    )
    op.drop_table("audit_logs")
    
    # Remove last_login column from users table
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_column("last_login")
