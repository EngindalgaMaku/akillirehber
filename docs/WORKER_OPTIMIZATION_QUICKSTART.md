# Worker Optimization - Quick Start Guide

## Problem Solved

Backend was blocking during long-running operations (semantic similarity tests, RAGAS evaluations, document processing). Multiple workers now allow concurrent request handling.

## Quick Start

### 1. Rebuild and Restart (Development Mode)

```bash
# Stop existing containers
docker-compose down

# Rebuild and start with new worker configuration
docker-compose up --build
```

### 2. Rebuild and Restart (Production Mode)

```bash
# Stop existing containers
docker-compose -f docker-compose.prod.yml down

# Rebuild and start with production worker configuration
docker-compose -f docker-compose.prod.yml up --build -d
```

## Verify Worker Configuration

Check if workers are running correctly:

```bash
# View backend logs to see worker configuration
docker logs rag-backend

# You should see output like:
# ========================================
# Gunicorn Configuration
# ========================================
# Workers: 9
# Threads per worker: 4
# Worker class: uvicorn.workers.UvicornWorker
# ========================================
```

## Custom Worker Settings

Edit your `.env` file to customize worker configuration:

```bash
# Number of worker processes (default: auto based on CPU)
WORKERS=8

# Threads per worker (default: 4)
THREADS=4

# Worker timeout in seconds (default: 300)
GUNICORN_TIMEOUT=600
```

Then restart:

```bash
docker-compose down
docker-compose up --build
```

## Testing

### Test 1: Run Semantic Similarity Test

1. Open the frontend at `http://localhost:3000`
2. Navigate to Semantic Similarity test
3. Run a test with multiple questions
4. While the test is running, try to navigate to other pages
5. **Result**: Other pages should load normally (no blocking)

### Test 2: Run Multiple Concurrent Requests

```bash
# Run multiple requests in parallel
curl http://localhost:8000/health &
curl http://localhost:8000/health &
curl http://localhost:8000/health &
wait
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Set to `production` for Gunicorn |
| `WORKERS` | `(2 * CPU) + 1` | Number of worker processes |
| `THREADS` | `4` | Threads per worker |
| `GUNICORN_TIMEOUT` | `300` | Worker timeout (seconds) |

## Troubleshooting

### Backend Still Blocking

1. Check if `ENVIRONMENT=production` is set
2. Verify worker count in logs
3. Increase `GUNICORN_TIMEOUT` for long tasks

### Workers Not Starting

```bash
# Check backend logs
docker logs rag-backend

# Verify environment variables
docker exec rag-backend env | grep WORKER
```

## Full Documentation

See [`WORKER_OPTIMIZATION.md`](./WORKER_OPTIMIZATION.md) for detailed documentation.
