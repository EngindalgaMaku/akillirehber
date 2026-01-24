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
while ! nc -z postgres 5432; do
  sleep 1
done
echo "PostgreSQL is ready!"

# Wait for Weaviate to be ready
echo "Waiting for Weaviate..."
while ! nc -z weaviate 8080; do
  sleep 1
done
echo "Weaviate is ready!"

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

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
