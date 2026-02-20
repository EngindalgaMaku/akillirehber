# Development Setup Guide

This guide explains how to run the application in development mode with hot reload support.

## Overview

The development setup uses `docker-compose.override.yml` which automatically extends the base `docker-compose.yml` configuration when present. This provides:

- **Faster builds** - No multi-stage builds, no wheel caching issues
- **Hot reload** - Code changes are automatically detected and applied
- **Debug logging** - Verbose output for troubleshooting
- **Writable volumes** - Full access to mounted source code

## Quick Start

### First Time Setup

1. **Ensure you have Docker and Docker Compose installed**
   ```bash
   docker --version
   docker-compose --version
   ```

2. **Create your environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start the development environment**
   ```bash
   docker-compose up --build
   ```

   This will:
   - Build all services using development Dockerfiles
   - Start backend with uvicorn (auto-reload enabled)
   - Start frontend with Next.js dev server (hot reload enabled)
   - Start supporting services (PostgreSQL, Weaviate, RAGAS)

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Stopping the Environment

```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (cleans database data)
docker-compose down -v
```

## Development Features

### Backend (FastAPI)

- **Hot Reload**: Changes to Python files in `backend/app/` are automatically detected
- **Debug Logging**: Set `LOG_LEVEL=debug` in `.env` for verbose output
- **Single Worker**: Development mode uses 1 worker (no Gunicorn)
- **Database Migrations**: Automatically run on startup

### Frontend (Next.js)

- **Hot Reload**: Changes to React/TypeScript files in `frontend/src/` are automatically detected
- **Fast Refresh**: Preserves component state during updates
- **Source Maps**: Full debugging support in browser DevTools

### Supporting Services

- **PostgreSQL**: Database with persisted data in volume
- **Weaviate**: Vector database for embeddings
- **RAGAS**: Evaluation service for RAG quality metrics

## Common Development Tasks

### View Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
```

### Rebuild a Specific Service

```bash
# Rebuild backend only
docker-compose up --build backend

# Rebuild frontend only
docker-compose up --build frontend
```

### Execute Commands in Containers

```bash
# Access backend container
docker-compose exec backend bash

# Access frontend container
docker-compose exec frontend sh

# Run Python tests
docker-compose exec backend pytest

# Run database migrations
docker-compose exec backend alembic upgrade head
```

### Clear Build Cache

If you experience build issues, clear the Docker cache:

```bash
# Clear all build cache
docker builder prune -a

# Rebuild from scratch
docker-compose build --no-cache
```

## Troubleshooting

### Build is Too Slow

The development Dockerfiles are optimized for speed. If builds are still slow:

1. **Check disk space**: Docker can slow down with low disk space
2. **Clear cache**: `docker builder prune -a`
3. **Check resources**: Ensure Docker has enough CPU/RAM allocated

### Hot Reload Not Working

1. **Check volume mounts**: Ensure volumes are mounted correctly
   ```bash
   docker-compose config
   ```
2. **Verify file permissions**: Ensure mounted files are writable
3. **Check logs**: Look for reload messages in service logs

### Port Already in Use

If ports 3000, 8000, 5432, or 8080 are already in use:

1. **Stop conflicting services**
2. **Or change ports in docker-compose.override.yml**:
   ```yaml
   services:
     backend:
       ports:
         - "8001:8000"  # Use port 8001 instead
   ```

### Database Connection Issues

1. **Check PostgreSQL is healthy**:
   ```bash
   docker-compose ps postgres
   ```
2. **View PostgreSQL logs**:
   ```bash
   docker-compose logs postgres
   ```
3. **Restart services**:
   ```bash
   docker-compose restart backend
   ```

## Production vs Development

### Development Mode (docker-compose.override.yml)
- Uses `Dockerfile.dev` for faster builds
- Enables hot reload with `--reload` flag
- Single worker, debug logging
- Writable volume mounts
- Runs on localhost

### Production Mode (docker-compose.yml only)
- Uses optimized multi-stage `Dockerfile`
- Gunicorn with multiple workers
- Production logging level
- Read-only volume mounts
- Optimized for performance and security

To run in production:
```bash
# Remove or rename docker-compose.override.yml
mv docker-compose.override.yml docker-compose.override.yml.bak

# Start production environment
docker-compose up --build
```

## Environment Variables

Key environment variables for development (in `.env`):

```bash
# Application
ENVIRONMENT=development
LOG_LEVEL=debug

# Backend
DATABASE_URL=postgresql://raguser:ragpassword@postgres:5432/ragchatbot
WEAVIATE_URL=http://weaviate:8080

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000

# API Keys (required for features)
OPENAI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
# ... other API keys
```

## Additional Resources

- [Backend API Documentation](http://localhost:8000/docs)
- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
