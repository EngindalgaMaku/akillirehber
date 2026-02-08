FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Copy application code
COPY . .

# Create non-root user with REAL home directory
RUN useradd -m -d /home/appuser -s /bin/bash appuser

# Ensure all cache directories are writable
RUN mkdir -p /home/appuser/.cache/huggingface \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /home/appuser

# Force HuggingFace to use writable cache
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/home/appuser/.cache/huggingface
ENV XDG_CACHE_HOME=/home/appuser/.cache
ENV HOME=/home/appuser

USER appuser

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
