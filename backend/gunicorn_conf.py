"""Gunicorn configuration for production deployment.

This configuration optimizes worker processes for handling concurrent
requests, especially for blocking operations like LLM calls and embeddings.
"""

import multiprocessing
import os

# Worker class - uvicorn workers for async support
worker_class = "uvicorn.workers.UvicornWorker"

# Number of worker processes
# Formula: (2 * CPU cores) + 1 for I/O bound workloads
workers = int(os.getenv("WORKERS", (2 * multiprocessing.cpu_count()) + 1))

# Number of worker threads per worker (for I/O bound operations)
threads = int(os.getenv("THREADS", 4))

# Worker connections (max concurrent connections per worker)
worker_connections = int(os.getenv("WORKER_CONNECTIONS", 1000))

# Maximum number of requests a worker will process before restarting
# Helps prevent memory leaks
max_requests = int(os.getenv("MAX_REQUESTS", 1000))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", 100))

# Timeout settings
timeout = int(os.getenv("GUNICORN_TIMEOUT", 1800))  # 30 minutes for long-running batch tests
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", 5))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", 120))

# Process naming
proc_name = "rag-backend"

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = os.getenv("LOG_LEVEL", "info")

# Bind address
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# Preload app for faster worker startup and memory sharing
preload_app = True

# Worker lifecycle
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Server mechanics
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# SSL (if needed)
keyfile = None
certfile = None

# Process naming
proc_name = "rag-backend"

# Enable stats
statsd_host = None
statsd_prefix = ""

# Print configuration on startup for debugging
print(f"""
========================================
Gunicorn Configuration
========================================
Workers: {workers}
Threads per worker: {threads}
Worker class: {worker_class}
Worker connections: {worker_connections}
Max requests: {max_requests}
Timeout: {timeout}s
Bind: {bind}
========================================
""")
