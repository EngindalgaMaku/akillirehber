# Docker Worker Optimization Guide

## Overview

This document describes the worker optimization implemented to prevent backend blocking during long-running operations like semantic similarity tests, RAGAS evaluations, and document processing.

## Problem

Previously, the backend ran with a single worker process. When long-running operations were executed (e.g., semantic similarity tests with LLM calls, embedding API calls, database queries), the entire worker was blocked, preventing other requests from being processed. This caused the backend to appear "frozen" during these operations.

## Solution

The solution involves:

1. **Multiple Worker Processes**: Running multiple Gunicorn worker processes to handle concurrent requests
2. **Uvicorn Workers**: Using `uvicorn.workers.UvicornWorker` for async support
3. **Configurable Worker Settings**: Environment variables for fine-tuning worker configuration
4. **Automatic Worker Recycling**: Workers restart after processing a certain number of requests to prevent memory leaks

## Architecture

### Development Mode

In development mode (default), the application runs with a single Uvicorn worker with auto-reload:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

In production mode, Gunicorn manages multiple Uvicorn workers:

```bash
gunicorn app.main:app --config gunicorn_conf.py
```

## Configuration

### Environment Variables

The following environment variables can be set in your `.env` file or `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Set to `production` to enable Gunicorn |
| `WORKERS` | `(2 * CPU cores) + 1` | Number of worker processes |
| `THREADS` | `4` | Number of threads per worker |
| `WORKER_CONNECTIONS` | `1000` | Max concurrent connections per worker |
| `MAX_REQUESTS` | `1000` | Max requests before worker restart |
| `MAX_REQUESTS_JITTER` | `100` | Random variation in max requests |
| `GUNICORN_TIMEOUT` | `300` | Worker timeout in seconds (5 minutes) |
| `GUNICORN_KEEPALIVE` | `5` | Keep-alive timeout in seconds |
| `GUNICORN_GRACEFUL_TIMEOUT` | `30` | Graceful shutdown timeout in seconds |
| `GUNICORN_BIND` | `0.0.0.0:8000` | Bind address |
| `LOG_LEVEL` | `info` | Logging level |

### Worker Count Formula

The default worker count follows the formula: `(2 * CPU cores) + 1`

This is recommended for I/O-bound workloads like your application, which makes many external API calls (LLM, embeddings, database).

For example:
- 2 CPU cores → 5 workers
- 4 CPU cores → 9 workers
- 8 CPU cores → 17 workers

### Thread Count

Each worker runs with multiple threads (default: 4). This allows each worker to handle multiple concurrent requests, further improving throughput.

## Files Modified

1. **`backend/gunicorn_conf.py`** - Gunicorn configuration file
2. **`backend/docker-entrypoint.sh`** - Entrypoint script that chooses between dev/prod modes
3. **`backend/Dockerfile`** - Updated to install Gunicorn and use the entrypoint script
4. **`backend/requirements.txt`** - Added Gunicorn dependency
5. **`docker-compose.yml`** - Added worker configuration environment variables
6. **`docker-compose.prod.yml`** - Added worker configuration environment variables

## Usage

### Development Mode (Default)

Start the application in development mode:

```bash
docker-compose up --build
```

This will:
- Run with a single Uvicorn worker
- Enable auto-reload for code changes
- Use the `--reload` flag

### Production Mode

Start the application in production mode:

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

This will:
- Run with multiple Gunicorn workers (default: 4 or based on CPU)
- Use Uvicorn workers for async support
- Disable auto-reload
- Enable worker recycling

### Custom Worker Configuration

To customize worker settings, set the environment variables in your `.env` file:

```bash
# Use 8 workers
WORKERS=8

# Use 8 threads per worker
THREADS=8

# Increase timeout for long-running tasks
GUNICORN_TIMEOUT=600
```

Then restart the containers:

```bash
docker-compose down
docker-compose up --build
```

## Monitoring

### Check Worker Status

You can check the number of running workers by examining the container logs:

```bash
docker logs rag-backend
```

You should see output like:

```
========================================
Gunicorn Configuration
========================================
Workers: 9
Threads per worker: 4
Worker class: uvicorn.workers.UvicornWorker
Worker connections: 1000
Max requests: 1000
Timeout: 300s
Bind: 0.0.0.0:8000
========================================
```

### Monitor Worker Health

Check if all workers are healthy:

```bash
docker exec rag-backend ps aux | grep gunicorn
```

You should see multiple worker processes.

## Troubleshooting

### Backend Still Blocking

If the backend still appears to block during operations:

1. **Check worker count**: Ensure `WORKERS` is set correctly
2. **Check timeout**: Increase `GUNICORN_TIMEOUT` for long-running operations
3. **Check logs**: Look for worker crashes or restarts in logs
4. **Monitor resources**: Check CPU and memory usage

### Workers Crashing

If workers are crashing frequently:

1. **Increase `MAX_REQUESTS`**: Workers may be hitting memory limits
2. **Check for memory leaks**: Use tools like `memory_profiler`
3. **Review error logs**: Check for unhandled exceptions

### Performance Issues

If performance is poor:

1. **Adjust worker count**: Too many workers can cause context switching overhead
2. **Adjust thread count**: More threads per worker increases memory usage
3. **Monitor resources**: Ensure you have enough CPU and memory

## Best Practices

1. **Start with default settings**: The default configuration is optimized for most use cases
2. **Monitor performance**: Use monitoring tools to track worker utilization
3. **Adjust based on workload**: Tune settings based on your specific workload
4. **Test in staging**: Always test configuration changes in a staging environment
5. **Keep workers healthy**: Regular worker recycling prevents memory leaks

## Performance Impact

With the new worker configuration:

- **Concurrent requests**: Multiple workers can handle requests simultaneously
- **No blocking**: Long-running operations don't block other requests
- **Better resource utilization**: CPU cores are used more efficiently
- **Improved reliability**: Worker recycling prevents memory leaks

## Example Scenarios

### Scenario 1: Semantic Similarity Test

**Before**: Running a semantic similarity test with 10 questions would block the backend for 2-3 minutes.

**After**: The test runs in one worker, while other workers continue handling requests. Users can still navigate the application, upload documents, etc.

### Scenario 2: Document Processing

**Before**: Processing a large PDF would block the backend.

**After**: Document processing happens in one worker, while other workers handle user requests.

### Scenario 3: Multiple Users

**Before**: Multiple users making requests simultaneously would experience slow response times.

**After**: Multiple workers handle requests concurrently, providing better response times.

## Additional Resources

- [Gunicorn Documentation](https://docs.gunicorn.org/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Worker Configuration Guide](https://docs.gunicorn.org/en/stable/design.html#how-many-workers)

## Support

If you encounter issues with the worker configuration:

1. Check the logs: `docker logs rag-backend`
2. Verify environment variables: `docker exec rag-backend env | grep WORKER`
3. Review this documentation
4. Check the Gunicorn and Uvicorn documentation

## Version History

- **v1.0** - Initial implementation with Gunicorn + Uvicorn workers
  - Added multi-worker support
  - Added worker recycling
  - Added configurable worker settings
