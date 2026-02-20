"""Public restore endpoints for initial setup (no authentication required)."""

import json
import zipfile
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

router = APIRouter(prefix="/api/restore", tags=["restore"])

# Upload directory inside container (created by entrypoint script)
UPLOAD_DIR = Path("/app/backups/uploads")


@router.post("/upload/postgres")
async def upload_postgres(file: UploadFile = File(...)):
    """Upload PostgreSQL backup file (no auth required)."""
    try:
        if not file.filename:
            raise HTTPException(400, "No filename provided")
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"postgres_{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save file
        content = await file.read()
        file_path.write_bytes(content)
        
        file_size_mb = len(content) / 1024 / 1024
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"PostgreSQL backup uploaded: {safe_filename}",
                "details": {
                    "filename": safe_filename,
                    "path": str(file_path),
                    "size_mb": round(file_size_mb, 2)
                }
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to upload file: {str(e)}")


@router.post("/upload/weaviate")
async def upload_weaviate(file: UploadFile = File(...)):
    """Upload Weaviate backup file (no auth required)."""
    try:
        if not file.filename:
            raise HTTPException(400, "No filename provided")
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"weaviate_{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save file
        content = await file.read()
        file_path.write_bytes(content)
        
        file_size_mb = len(content) / 1024 / 1024
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"Weaviate backup uploaded: {safe_filename}",
                "details": {
                    "filename": safe_filename,
                    "path": str(file_path),
                    "size_mb": round(file_size_mb, 2)
                }
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to upload file: {str(e)}")


@router.get("/list")
async def list_uploads():
    """List uploaded backup files (no auth required)."""
    try:
        files = []
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return JSONResponse(
            content={
                "success": True,
                "files": sorted(files, key=lambda x: x["modified"], reverse=True)
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to list files: {str(e)}")


@router.options("/upload/postgres")
async def options_upload_postgres():
    """Handle CORS preflight for postgres upload."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


@router.options("/upload/weaviate")
async def options_upload_weaviate():
    """Handle CORS preflight for weaviate upload."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


@router.options("/list")
async def options_list():
    """Handle CORS preflight for list."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )
