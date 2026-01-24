# Docker Cache Optimization Guide

## Problem

When `requirements.txt` changes, Docker re-downloads ALL packages including large ones like:
- torch (~915 MB)
- giskard[llm] (~200+ MB)
- umap-learn, numba, scikit-learn, scipy, pandas, numpy (~500+ MB total)

This makes builds extremely slow even for small changes.

## Solution: Split Requirements

We've split dependencies into two files:

### 1. `requirements-base.txt` - Large, rarely changing packages
```python
# Scientific computing and ML (large packages)
numpy>=1.26.0,<2
pandas>=2.2.0,<3
scikit-learn>=1.3.0
scipy>=1.11.0

# Giskard RAG Evaluation Toolkit (RAGET) with dependencies
giskard[llm]>=2.0.0

# UMAP dependencies (required by Giskard)
umap-learn>=0.5.0,<0.6.0
numba>=0.60.0
```

### 2. `requirements.txt` - Frequently changing packages
```python
# FastAPI and ASGI server
fastapi>=0.115.0,<0.116.0
uvicorn[standard]>=0.34.0,<0.35.0
gunicorn>=23.0.0,<24.0.0

# Data validation
pydantic>=2.12.0,<3.0.0
pydantic-settings>=2.0.0
email-validator>=2.0.0

# Database
sqlalchemy>=2.0.0,<3.0.0
psycopg2-binary>=2.9.0
alembic>=1.13.0

# ... other smaller packages
```

## Dockerfile Changes

### Before (Slow)
```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt
```
- Any change → All packages re-downloaded
- Build time: 5-10 minutes

### After (Fast)
```dockerfile
# Install large packages first (cached separately)
COPY requirements-base.txt .
RUN pip install -r requirements-base.txt

# Then install other packages
COPY requirements.txt .
RUN pip install -r requirements.txt
```
- Small changes → Only small packages re-installed
- Build time: 30 seconds - 2 minutes

## How It Works

### Docker Layer Caching

Docker caches each layer separately. If a layer's content hasn't changed, Docker uses the cached version.

```
Layer 1: FROM python:3.12-slim
Layer 2: COPY requirements-base.txt .
Layer 3: RUN pip install -r requirements-base.txt  ← Cached if requirements-base.txt unchanged
Layer 4: COPY requirements.txt .
Layer 5: RUN pip install -r requirements.txt  ← Only re-runs if requirements.txt changed
```

### Build Scenarios

#### Scenario 1: Add new small package to requirements.txt
```
requirements-base.txt: Unchanged → Uses cache (instant)
requirements.txt: Changed → Only installs new package (30 seconds)
Total time: ~30 seconds
```

#### Scenario 2: Update version in requirements.txt
```
requirements-base.txt: Unchanged → Uses cache (instant)
requirements.txt: Changed → Only re-installs updated packages (1-2 minutes)
Total time: ~1-2 minutes
```

#### Scenario 3: Update large package in requirements-base.txt
```
requirements-base.txt: Changed → Re-installs large packages (5-10 minutes)
requirements.txt: Unchanged → Uses cache (instant)
Total time: ~5-10 minutes (but this is rare)
```

## Best Practices

### When to Update requirements-base.txt
- Adding new large ML/scientific packages
- Updating numpy, pandas, scikit-learn versions
- Updating giskard or umap-learn versions
- Adding new heavy dependencies (>50 MB)

### When to Update requirements.txt
- Adding small utility packages
- Updating FastAPI, uvicorn, gunicorn
- Adding API clients (openai, cohere)
- Updating database packages (sqlalchemy, alembic)
- Adding testing packages (pytest, httpx)

### Tips for Maximum Cache Utilization

1. **Keep requirements-base.txt stable**: Only update when necessary
2. **Order packages by size**: Put largest packages first in requirements-base.txt
3. **Pin versions**: Use exact versions in requirements-base.txt for stability
4. **Monitor build times**: Check which packages trigger cache misses

## Performance Comparison

| Scenario | Before | After | Improvement |
|-----------|---------|--------|-------------|
| Add small package | 5-10 min | 30 sec | 10-20x faster |
| Update small package | 5-10 min | 1-2 min | 5-10x faster |
| Update large package | 5-10 min | 5-10 min | Same (rare) |
| First build | 5-10 min | 5-10 min | Same |

## Troubleshooting

### Cache Not Working?

If builds are still slow, check:

1. **Docker cache disabled?**
   ```bash
   # Check if --no-cache is being used
   docker-compose build backend  # Should NOT have --no-cache
   ```

2. **requirements-base.txt changed?**
   ```bash
   # Check if file was modified
   git diff backend/requirements-base.txt
   ```

3. **Force cache invalidation?**
   ```bash
   # Clear Docker cache (last resort)
   docker system prune -a
   ```

### Large Package Still Downloading?

If a large package is re-downloading when it shouldn't:

1. Check if it's in the correct file:
   ```bash
   # Large packages should be in requirements-base.txt
   cat backend/requirements-base.txt
   ```

2. Check if version changed:
   ```bash
   # Version changes invalidate cache
   git diff backend/requirements-base.txt
   ```

3. Check Dockerfile order:
   ```dockerfile
   # Make sure requirements-base.txt is installed FIRST
   COPY requirements-base.txt .
   RUN pip install -r requirements-base.txt  # This must come before requirements.txt
   ```

## Related Files

- `backend/requirements-base.txt` - Large, rarely changing packages
- `backend/requirements.txt` - Frequently changing packages
- `backend/Dockerfile` - Production build with optimized caching
- `backend/Dockerfile.dev` - Development build with optimized caching
- `docker-compose.override.yml` - Mounts both requirement files for development

## Additional Resources

- [Docker Best Practices for Images](https://docs.docker.com/develop/dev-best-practices/)
- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
- [Docker Build Cache](https://docs.docker.com/build/cache/)
