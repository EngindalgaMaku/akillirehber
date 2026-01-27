#!/usr/bin/env python3
"""Fix migration for embedding_status enum name typo."""

import sys
from alembic import op
from alembic.config import Config

print("Running migration...")
try:
    # Execute the ALTER TABLE statement directly
    op.execute('ALTER TABLE documents ALTER COLUMN embedding_status TYPE VARCHAR(20) USING embeddingstatus;')
    print("Migration SQL generated successfully")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
