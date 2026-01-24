# Giskard UMAP Dependency Fix

## Problem Description

The Giskard RAG Evaluation Toolkit (RAGET) was failing to import with the following error:

```
Failed to import Giskard RAGET dependencies in the backend container: cannot cache function 'rdist': no locator available for file '/usr/local/lib/python3.12/site-packages/umap/layouts.py'.
```

This error occurred because:

1. **UMAP (Uniform Manifold Approximation and Projection)** is a dependency of Giskard
2. UMAP uses **Numba** for JIT (Just-In-Time) compilation
3. Numba needs to cache compiled functions, but the cache directory was not properly configured
4. The error "cannot cache function 'rdist'" indicates Numba cannot write to its cache directory

## Solution Implemented

The fix involves three main changes:

### 1. Updated Dependencies (`backend/requirements.txt`)

Added explicit UMAP dependencies with specific versions:

```python
# UMAP dependencies (required by Giskard)
umap-learn>=0.5.0,<0.6.0
numba>=0.60.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

This ensures:
- UMAP is installed with a compatible version
- Numba is available for JIT compilation
- Required scientific computing libraries are present

### 2. Updated Dockerfiles

Both `backend/Dockerfile` and `backend/Dockerfile.dev` now include:

```dockerfile
# Set up Numba cache directory for UMAP
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -p /tmp/numba_cache && \
    chmod 777 /tmp/numba_cache
```

This creates:
- A dedicated cache directory at `/tmp/numba_cache`
- Proper permissions (777) for Numba to write cache files
- Environment variable pointing to the cache directory

### 3. Updated Docker Compose Files

Both `docker-compose.yml` and `docker-compose.override.yml` now include:

```yaml
environment:
  # Numba cache directory for UMAP (required by Giskard)
  - NUMBA_CACHE_DIR=/tmp/numba_cache
```

This ensures the environment variable is passed to the container at runtime.

## How to Apply the Fix

### Option 1: Using the Rebuild Script (Recommended)

**Windows:**
```cmd
rebuild-backend-giskard.bat
```

**Linux/Mac:**
```bash
chmod +x rebuild-backend-giskard.sh
./rebuild-backend-giskard.sh
```

### Option 2: Manual Rebuild

1. Stop and remove existing containers:
   ```bash
   docker-compose down
   ```

2. Rebuild the backend:
   ```bash
   docker-compose up --build -d backend
   ```

3. Verify the fix:
   ```bash
   docker-compose logs -f backend
   ```

## Verification

After rebuilding, verify the fix by:

### 1. Check Backend Logs

Look for successful import of Giskard dependencies:

```bash
docker-compose logs backend | grep -i giskard
```

You should NOT see the UMAP error anymore.

### 2. Test Giskard Functionality

Try generating test questions via the API:

```bash
curl -X POST http://localhost:8000/api/giskard/generate-test-questions \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": 1,
    "num_relevant": 5,
    "num_irrelevant": 10,
    "language": "tr"
  }'
```

### 3. Verify Numba Cache Directory

Check that the cache directory exists and is writable:

```bash
docker exec -it rag-backend ls -la /tmp/numba_cache
```

## Troubleshooting

### Issue: Still seeing UMAP errors

**Solution:**
1. Ensure you completely rebuilt the image (use `--no-cache` if needed):
   ```bash
   docker-compose build --no-cache backend
   docker-compose up -d backend
   ```

2. Check that all dependencies are installed:
   ```bash
   docker exec -it rag-backend pip list | grep -E "(umap|numba|scikit)"
   ```

### Issue: Permission denied on cache directory

**Solution:**
The Dockerfile sets permissions to 777, but if you still have issues:

```bash
docker exec -it rag-backend chmod 777 /tmp/numba_cache
```

### Issue: Giskard still not working

**Solution:**
1. Check if Giskard is properly installed:
   ```bash
   docker exec -it rag-backend python -c "import giskard; print(giskard.__version__)"
   ```

2. Test UMAP import:
   ```bash
   docker exec -it rag-backend python -c "import umap; print('UMAP imported successfully')"
   ```

## Technical Details

### Why This Fix Works

1. **Explicit Dependencies**: By explicitly declaring UMAP and its dependencies, we ensure compatible versions are installed.

2. **Numba Cache Directory**: Numba compiles Python functions to machine code for better performance. It needs a writable directory to cache these compiled functions. Without this directory, UMAP fails when trying to use Numba-compiled functions like `rdist`.

3. **Permissions**: Setting 777 permissions ensures Numba can write to the cache directory regardless of the user context.

### UMAP and Numba Relationship

- **UMAP** uses dimensionality reduction algorithms
- These algorithms are computationally intensive
- **Numba** JIT-compiles these algorithms for performance
- Numba caches compiled functions to avoid recompilation
- The cache directory is essential for this caching mechanism

## Related Files

- `backend/requirements.txt` - Updated with UMAP dependencies
- `backend/requirements-base.txt` - Large packages that rarely change (for better caching)
- `backend/Dockerfile` - Production build with Numba cache setup and optimized dependency caching
- `backend/Dockerfile.dev` - Development build with Numba cache setup and optimized dependency caching (also fixed useradd command)
- `docker-compose.yml` - Production environment variables
- `docker-compose.override.yml` - Development environment variables
- `rebuild-backend-giskard.sh` - Linux/Mac rebuild script
- `rebuild-backend-giskard.bat` - Windows rebuild script

## Additional Fixes Applied

### Fixed Dockerfile.dev User Creation

The original [`Dockerfile.dev`](backend/Dockerfile.dev:34) had an incorrect useradd command:

**Before (incorrect):**
```dockerfile
RUN groupadd -r appgroup && useradd -r -g appgroup -G appuser appuser
```

This failed because `-G appuser` tried to add the user to a non-existent group.

**After (correct):**
```dockerfile
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
```

This properly creates the group `appgroup` and user `appuser` with `appgroup` as their primary group.

### Optimized Docker Layer Caching

To solve the problem of re-downloading large packages (like torch, giskard, umap-learn) every time `requirements.txt` changes, we've split the dependencies into two files:

#### 1. `requirements-base.txt` - Large, rarely changing packages
Contains:
- numpy, pandas, scikit-learn, scipy (scientific computing)
- giskard[llm] (RAG evaluation toolkit)
- umap-learn, numba (UMAP dependencies)

These packages are installed first and cached separately.

#### 2. `requirements.txt` - Frequently changing packages
Contains:
- FastAPI, uvicorn, gunicorn (web framework)
- pydantic, sqlalchemy (data validation & database)
- openai, cohere (API clients)
- Other smaller packages

#### How This Works:

**Before (slow):**
```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt
```
- Any change to `requirements.txt` → All packages re-downloaded
- torch (915 MB) downloaded every time

**After (fast):**
```dockerfile
COPY requirements-base.txt .
RUN pip install -r requirements-base.txt  # Cached separately

COPY requirements.txt .
RUN pip install -r requirements.txt  # Only installs new/changed packages
```

#### Benefits:

1. **Faster builds**: Large packages only downloaded once
2. **Faster iterations**: Changing small packages doesn't trigger large package re-installation
3. **Better cache utilization**: Docker layers are optimized for common use cases

#### Example Scenarios:

**Scenario 1: Add a new small package to requirements.txt**
- requirements-base.txt: Not changed → Uses cache (instant)
- requirements.txt: Changed → Only installs the new package (fast)

**Scenario 2: Update a version in requirements.txt**
- requirements-base.txt: Not changed → Uses cache (instant)
- requirements.txt: Changed → Only re-installs updated packages (fast)

**Scenario 3: Update a large package version in requirements-base.txt**
- requirements-base.txt: Changed → Re-installs large packages (slow, but rare)
- requirements.txt: Not changed → Uses cache (instant)

This optimization dramatically reduces build times for most development scenarios!

## Additional Resources

- [Giskard Documentation](https://docs.giskard.ai/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [Numba Documentation](https://numba.pydata.org/)
