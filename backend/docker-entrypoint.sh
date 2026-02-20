#!/bin/sh
set -e

# Backend Docker Entrypoint Script
# This script handles both development and production modes

echo "=========================================="
echo "Starting RAG Backend"
echo "=========================================="
echo "Environment: ${ENVIRONMENT:-development}"
echo "Workers: ${WORKERS:-auto}"
echo "=========================================="

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
if echo "${DATABASE_URL:-}" | grep -q "@postgres:"; then
  while ! nc -z postgres 5432; do
    sleep 1
  done
  echo "PostgreSQL is ready!"
else
  echo "Skipping PostgreSQL wait (remote DATABASE_URL)"
fi

# Wait for Weaviate to be ready
echo "Waiting for Weaviate..."
if echo "${WEAVIATE_URL:-}" | grep -q "://weaviate\(:\|/\)"; then
  while ! nc -z weaviate 8080; do
    sleep 1
  done
  echo "Weaviate is ready!"
else
  echo "Skipping Weaviate wait (remote WEAVIATE_URL)"
fi

# Run database migrations with error handling
echo "Running database migrations..."
if ! alembic upgrade head; then
  echo "Migration failed, trying to create tables from models..."
  python -c "
from app.database import engine, Base
from app.models.db_models import *
Base.metadata.create_all(bind=engine)
print('Tables created successfully from models')
"
fi

# Auto-restore from backups if available
BACKUP_DIR="/app/backups"
if [ -d "$BACKUP_DIR" ] && [ "$(ls -A $BACKUP_DIR 2>/dev/null)" ]; then
  echo "=========================================="
  echo "Backups found in $BACKUP_DIR"
  echo "Checking for auto-restore..."
  echo "=========================================="
  
  # Check if database is empty (first deployment)
  DB_EMPTY=$(python -c "
from app.database import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    result = db.execute(text('SELECT COUNT(*) FROM users')).scalar()
    print('empty' if result == 0 else 'not_empty')
except:
    print('empty')
finally:
    db.close()
" 2>/dev/null || echo "empty")
  
  if [ "$DB_EMPTY" = "empty" ]; then
    echo "Database is empty - attempting auto-restore..."
    
    # Find latest PostgreSQL backup (support both .sql and .sql.zip)
    LATEST_PG=$(ls -t $BACKUP_DIR/postgres-*.sql $BACKUP_DIR/postgres-*.sql.zip 2>/dev/null | head -1)
    if [ -n "$LATEST_PG" ]; then
      echo "Restoring PostgreSQL from: $(basename $LATEST_PG)"
      
      # Check if file is zipped
      case "$LATEST_PG" in
      *.zip)
        echo "Extracting zipped backup..."
        python -c "
import zipfile
import sys

try:
    with zipfile.ZipFile('$LATEST_PG', 'r') as zip_ref:
        # Extract to temp location
        zip_ref.extractall('/tmp/')
        # Get the extracted SQL file name
        sql_file = [f for f in zip_ref.namelist() if f.endswith('.sql')][0]
        print(f'Extracted: {sql_file}')
        
        # Now restore from extracted file
        from app.database import SessionLocal
        from sqlalchemy import text
        
        db = SessionLocal()
        try:
            with open(f'/tmp/{sql_file}', 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            for statement in sql_content.split(';'):
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    try:
                        db.execute(text(statement))
                    except Exception as e:
                        print(f'Warning: {e}', file=sys.stderr)
            
            db.commit()
            print('✅ PostgreSQL restored successfully')
        except Exception as e:
            db.rollback()
            print(f'❌ PostgreSQL restore failed: {e}', file=sys.stderr)
            sys.exit(1)
        finally:
            db.close()
except Exception as e:
    print(f'❌ Failed to extract backup: {e}', file=sys.stderr)
    sys.exit(1)
"
      ;;
      *)
        # Direct SQL file restore
        python -c "
from app.database import SessionLocal
from sqlalchemy import text
import sys

db = SessionLocal()
try:
    with open('$LATEST_PG', 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    for statement in sql_content.split(';'):
        statement = statement.strip()
        if statement and not statement.startswith('--'):
            try:
                db.execute(text(statement))
            except Exception as e:
                print(f'Warning: {e}', file=sys.stderr)
    
    db.commit()
    print('✅ PostgreSQL restored successfully')
except Exception as e:
    db.rollback()
    print(f'❌ PostgreSQL restore failed: {e}', file=sys.stderr)
    sys.exit(1)
finally:
    db.close()
"
      ;;
      esac
    fi
    
    # Find latest Weaviate backup (support both .json and .json.zip)
    LATEST_WV=$(ls -t $BACKUP_DIR/weaviate-*.json $BACKUP_DIR/weaviate-*.json.zip 2>/dev/null | head -1)
    if [ -n "$LATEST_WV" ]; then
      echo "Restoring Weaviate from: $(basename $LATEST_WV)"
      
      # Check if file is zipped
      case "$LATEST_WV" in
      *.zip)
        echo "Extracting zipped backup..."
        python -c "
import zipfile
import json
import sys
sys.path.insert(0, '/app')

from app.services.weaviate_service import WeaviateService

try:
    with zipfile.ZipFile('$LATEST_WV', 'r') as zip_ref:
        # Get the JSON file name
        json_file = [f for f in zip_ref.namelist() if f.endswith('.json')][0]
        
        # Read JSON directly from zip
        with zip_ref.open(json_file) as f:
            backup_data = json.load(f)
    
    weaviate_service = WeaviateService()
    total_imported = 0
    
    for collection_name, collection_data in backup_data.get('collections', {}).items():
        if 'error' in collection_data:
            continue
        
        course_id = collection_data.get('course_id')
        objects = collection_data.get('objects', [])
        
        if course_id and objects:
            imported = weaviate_service.import_collection(course_id, objects)
            total_imported += imported
            print(f'  - Imported {imported} objects for course {course_id}')
    
    print(f'✅ Weaviate restored successfully ({total_imported} objects)')
except Exception as e:
    print(f'❌ Weaviate restore failed: {e}', file=sys.stderr)
"
      ;;
      *)
        # Direct JSON file restore
        python -c "
import json
import sys
sys.path.insert(0, '/app')

from app.services.weaviate_service import WeaviateService

try:
    with open('$LATEST_WV', 'r', encoding='utf-8') as f:
        backup_data = json.load(f)
    
    weaviate_service = WeaviateService()
    total_imported = 0
    
    for collection_name, collection_data in backup_data.get('collections', {}).items():
        if 'error' in collection_data:
            continue
        
        course_id = collection_data.get('course_id')
        objects = collection_data.get('objects', [])
        
        if course_id and objects:
            imported = weaviate_service.import_collection(course_id, objects)
            total_imported += imported
            print(f'  - Imported {imported} objects for course {course_id}')
    
    print(f'✅ Weaviate restored successfully ({total_imported} objects)')
except Exception as e:
    print(f'❌ Weaviate restore failed: {e}', file=sys.stderr)
"
      ;;
      esac
    fi
    
    echo "=========================================="
    echo "Auto-restore completed!"
    echo "=========================================="
  else
    echo "Database is not empty - skipping auto-restore"
    echo "To restore manually, use the /dashboard/backup page"
  fi
else
  echo "No backups found in $BACKUP_DIR - skipping auto-restore"
fi

# Check if we should run in development or production mode
if [ "${ENVIRONMENT}" = "production" ]; then
  echo "Starting production server with Gunicorn..."
  
  # Use gunicorn with uvicorn workers for production
  exec gunicorn app.main:app \
    --config gunicorn_conf.py \
    --log-level ${LOG_LEVEL:-info}
else
  echo "Starting development server with uvicorn..."
  
  # Use uvicorn with auto-reload for development
  exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level ${LOG_LEVEL:-info}
fi
