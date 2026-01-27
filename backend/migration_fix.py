"from alembic import op; from alembic.config import Config; print('Creating migration...'); op.execute('ALTER TABLE documents ALTER COLUMN embedding_status TYPE VARCHAR(20) USING embeddingstatus;')" 
