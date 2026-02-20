from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "022_course_id_nullable"
down_revision: Union[str, None] = "021_add_chat_messages"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "documents",
        "course_id",
        existing_type=sa.Integer(),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "documents",
        "course_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
