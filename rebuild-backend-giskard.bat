@echo off
REM Rebuild script for fixing Giskard UMAP dependency issue (Windows)
REM This script rebuilds the backend container with updated dependencies and Numba cache configuration
REM Uses Docker layer caching for faster builds (large packages will be cached separately)

echo ==========================================
echo Rebuilding Backend for Giskard Fix
echo ==========================================
echo.

REM Stop and remove existing containers
echo Stopping and removing existing containers...
docker-compose down

REM Rebuild with Docker layer caching (faster for subsequent builds)
echo Rebuilding backend with new configuration (using cache)...
echo.
echo Caching Strategy:
echo   - Large packages (numpy, pandas, giskard, umap-learn, numba) are cached separately
echo   - requirements-base.txt changes rarely, so these packages stay cached
echo   - requirements.txt changes frequently, but won't trigger large package re-download
echo.
docker-compose build backend

REM Start the containers
echo Starting containers...
docker-compose up -d backend

echo.
echo ==========================================
echo Rebuild Complete!
echo ==========================================
echo.
echo The backend container has been rebuilt with:
echo   - Updated UMAP dependencies (umap-learn, numba, scikit-learn, scipy)
echo   - Numba cache directory configured at /tmp/numba_cache
echo   - NUMBA_CACHE_DIR environment variable set
echo   - Fixed user creation command in Dockerfile.dev
echo   - Optimized dependency caching (requirements-base.txt + requirements.txt)
echo.
echo To verify the fix, check the backend logs:
echo   docker-compose logs -f backend
echo.
echo To test Giskard functionality, try generating test questions via the API
echo.
echo NOTE: If you need a completely clean rebuild (no cache), run:
echo   docker-compose build --no-cache backend
echo.

pause
