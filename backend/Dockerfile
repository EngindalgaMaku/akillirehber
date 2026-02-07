# Backend Dockerfile - RAG Educational Chatbot
# Multi-stage build for optimized production image

# syntax=docker/dockerfile:1.4

# Stage 1: Builder
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with cache mount
# First install large packages that rarely change (for better caching)
COPY requirements-base.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
    && pip wheel --retries 10 --timeout 120 --wheel-dir /app/wheels -r requirements-base.txt

# Then install other packages that may change more frequently
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --retries 10 --timeout 120 --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Production
FROM python:3.12-slim as production

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files for wheel-based installation
COPY requirements-base.txt requirements.txt ./

# Set up Numba cache directory for UMAP
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -p /tmp/numba_cache && \
    chmod 777 /tmp/numba_cache

# Copy wheels from builder and install with cache mount
COPY --from=builder /app/wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/wheels -r requirements-base.txt -r requirements.txt

# Download NLTK data to a shared location
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True, download_dir='/usr/share/nltk_data')"

# Copy application code
COPY --chown=appuser:appgroup . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Copy entrypoint script
COPY --chown=appuser:appgroup docker-entrypoint.sh /app/
RUN chown -R appuser:appgroup /app \
    && sed -i 's/\r$//' /app/docker-entrypoint.sh \
    && chmod +x /app/docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Run the application using entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]
