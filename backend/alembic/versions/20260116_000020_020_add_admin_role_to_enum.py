"""Add admin role to userrole enum

Revision ID: 020_add_admin_role
Revises: 019_add_admin_user_management
Create Date: 2026-01-16 20:25:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '020_add_admin_role'
down_revision = '019_add_admin_user_management'
branch_labels = None
depends_on = None


def upgrade():
    """Add ADMIN value to userrole enum."""
    # PostgreSQL requires special handling for enum alterations
    # We need to use raw SQL to add a new enum value
    op.execute("ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'ADMIN'")


def downgrade():
    """Remove ADMIN value from userrole enum."""
    # Note: PostgreSQL doesn't support removing enum values directly
    # This would require recreating the enum type, which is complex
    # For now, we'll leave the enum value in place
    pass
