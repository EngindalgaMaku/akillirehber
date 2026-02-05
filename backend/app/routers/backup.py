"""Backup and restore API endpoints for admin."""

import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import get_db
from app.models.db_models import User
from app.services.auth_service import require_admin
from app.services.weaviate_service import WeaviateService
from app.models.schemas import (
    BackupListResponse,
    BackupInfo,
    BackupCreateResponse,
    BackupRestoreResponse,
)

router = APIRouter(prefix="/api/admin/backup", tags=["admin-backup"])

# Backup directory - use /app/backups which is mounted from host
BACKUP_DIR = Path("/app/backups")
# Don't create directory here - it's mounted from host and may not have write permissions
# BACKUP_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/list", response_model=BackupListResponse)
async def list_backups(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """List all available backups."""
    backups: List[BackupInfo] = []
    
    if not BACKUP_DIR.exists():
        return BackupListResponse(backups=backups, total=0)
    
    for file_path in BACKUP_DIR.glob("*"):
        if file_path.is_file():
            stat = file_path.stat()
            backup_type = "unknown"
            if "postgres" in file_path.name.lower():
                backup_type = "postgres"
            elif "weaviate" in file_path.name.lower():
                backup_type = "weaviate"
            elif "full" in file_path.name.lower():
                backup_type = "full"
            
            backups.append(
                BackupInfo(
                    filename=file_path.name,
                    size=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_mtime),
                    type=backup_type,
                )
            )
    
    backups.sort(key=lambda x: x.created_at, reverse=True)
    return BackupListResponse(backups=backups, total=len(backups))


@router.post("/create/postgres", response_model=BackupCreateResponse)
async def create_postgres_backup(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a PostgreSQL database backup using SQL dump."""
    try:
        # Ensure backup directory exists and is writable
        if not BACKUP_DIR.exists():
            raise HTTPException(
                status_code=500,
                detail="Backup directory not found. Ensure /app/backups is mounted."
            )
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"postgres-{timestamp}.sql"
        backup_path = BACKUP_DIR / filename
        
        # Get all table names
        tables_result = db.execute(text("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename
        """))
        tables = [row[0] for row in tables_result]
        
        # Create backup file
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write("-- PostgreSQL Database Backup\n")
            f.write(f"-- Created: {datetime.now().isoformat()}\n")
            f.write("-- Tables: " + ", ".join(tables) + "\n\n")
            
            # Dump each table
            for table in tables:
                try:
                    f.write(f"\n-- Table: {table}\n")
                    
                    # Get table data
                    data_result = db.execute(text(f"SELECT * FROM {table}"))
                    columns = list(data_result.keys())
                    rows = data_result.fetchall()
                    
                    if not rows:
                        f.write(f"-- No data in {table}\n")
                        continue
                    
                    f.write(f"TRUNCATE TABLE {table} CASCADE;\n")
                    
                    for row in rows:
                        values = []
                        for val in row:
                            if val is None:
                                values.append("NULL")
                            elif isinstance(val, str):
                                # Escape single quotes
                                escaped = val.replace("'", "''")
                                values.append(f"'{escaped}'")
                            elif isinstance(val, (int, float, bool)):
                                values.append(str(val))
                            else:
                                # For other types (datetime, etc)
                                values.append(f"'{str(val)}'")
                        
                        cols = ", ".join(columns)
                        vals = ", ".join(values)
                        f.write(f"INSERT INTO {table} ({cols}) VALUES ({vals});\n")
                    
                    f.write(f"-- {len(rows)} rows inserted into {table}\n")
                    
                except Exception as table_error:
                    f.write(f"-- Error backing up {table}: {str(table_error)}\n")
        
        size = backup_path.stat().st_size
        
        return BackupCreateResponse(
            success=True,
            message=f"PostgreSQL backup created successfully ({len(tables)} tables)",
            filename=filename,
            size=size,
            created_at=datetime.now(),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Backup failed: {str(e)}"
        )


@router.post("/create/weaviate", response_model=BackupCreateResponse)
async def create_weaviate_backup(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a Weaviate vector database backup using API export."""
    try:
        # Ensure backup directory exists and is writable
        if not BACKUP_DIR.exists():
            raise HTTPException(
                status_code=500,
                detail="Backup directory not found. Ensure /app/backups is mounted."
            )
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"weaviate-{timestamp}.json"
        backup_path = BACKUP_DIR / filename
        
        weaviate_service = WeaviateService()
        
        # Get all courses to backup their collections
        courses_result = db.execute(text("SELECT id, name FROM courses"))
        courses = courses_result.fetchall()
        
        backup_data = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "collections": {}
        }
        
        total_objects = 0
        
        for course_id, course_name in courses:
            try:
                collection_name = f"Course_{course_id}"
                
                # Get all objects from this collection
                objects = weaviate_service.export_collection(course_id)
                
                if objects:
                    backup_data["collections"][collection_name] = {
                        "course_id": course_id,
                        "course_name": course_name,
                        "objects": objects,
                        "count": len(objects)
                    }
                    total_objects += len(objects)
                    
            except Exception as e:
                backup_data["collections"][f"Course_{course_id}_error"] = {
                    "error": str(e)
                }
        
        # Save to file
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        size = backup_path.stat().st_size
        
        return BackupCreateResponse(
            success=True,
            message=f"Weaviate backup created successfully ({total_objects} objects from {len(courses)} courses)",
            filename=filename,
            size=size,
            created_at=datetime.now(),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Backup failed: {str(e)}"
        )


@router.post("/create/full", response_model=BackupCreateResponse)
async def create_full_backup(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a full backup (PostgreSQL + Weaviate)."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create PostgreSQL backup
        pg_response = await create_postgres_backup(admin_user, db)
        
        # Create Weaviate backup
        wv_response = await create_weaviate_backup(admin_user, db)
        
        # Create combined archive
        filename = f"full-backup-{timestamp}.tar.gz"
        backup_path = BACKUP_DIR / filename
        
        import tarfile
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(BACKUP_DIR / pg_response.filename, arcname=pg_response.filename)
            tar.add(BACKUP_DIR / wv_response.filename, arcname=wv_response.filename)
        
        # Clean up individual files
        (BACKUP_DIR / pg_response.filename).unlink()
        (BACKUP_DIR / wv_response.filename).unlink()
        
        size = backup_path.stat().st_size
        
        return BackupCreateResponse(
            success=True,
            message="Full backup created successfully (PostgreSQL + Weaviate)",
            filename=filename,
            size=size,
            created_at=datetime.now(),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Backup failed: {str(e)}"
        )


@router.get("/download/{filename}")
async def download_backup(
    filename: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Download a backup file.
    
    Returns the backup file for download.
    """
    backup_path = BACKUP_DIR / filename
    
    if not backup_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Backup file '{filename}' not found"
        )
    
    # Security check: ensure file is within backup directory
    if not backup_path.resolve().is_relative_to(BACKUP_DIR.resolve()):
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    return FileResponse(
        path=backup_path,
        filename=filename,
        media_type="application/octet-stream",
    )


@router.post("/restore/postgres", response_model=BackupRestoreResponse)
async def restore_postgres_backup(
    file: UploadFile = File(...),
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Restore PostgreSQL database from uploaded SQL file."""
    try:
        content = await file.read()
        sql_content = content.decode('utf-8')
        
        # Execute SQL statements
        for statement in sql_content.split(';'):
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                try:
                    db.execute(text(statement))
                except Exception as e:
                    # Continue on error (some statements might fail)
                    print(f"Warning: {e}")
        
        db.commit()
        
        return BackupRestoreResponse(
            success=True,
            message="PostgreSQL database restored successfully",
        )
            
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Restore failed: {str(e)}"
        )


@router.post("/restore/weaviate", response_model=BackupRestoreResponse)
async def restore_weaviate_backup(
    file: UploadFile = File(...),
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Restore Weaviate database from uploaded JSON backup."""
    try:
        content = await file.read()
        backup_data = json.loads(content.decode('utf-8'))
        
        weaviate_service = WeaviateService()
        
        total_imported = 0
        collections = backup_data.get("collections", {})
        
        for collection_name, collection_data in collections.items():
            if "error" in collection_data:
                continue
                
            course_id = collection_data.get("course_id")
            objects = collection_data.get("objects", [])
            
            if course_id and objects:
                imported = weaviate_service.import_collection(course_id, objects)
                total_imported += imported
        
        return BackupRestoreResponse(
            success=True,
            message=f"Weaviate database restored successfully ({total_imported} objects)",
        )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Restore failed: {str(e)}"
        )



@router.delete("/delete/{filename}")
async def delete_backup(
    filename: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Delete a backup file.
    """
    backup_path = BACKUP_DIR / filename
    
    if not backup_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Backup file '{filename}' not found"
        )
    
    # Security check: ensure file is within backup directory
    if not backup_path.resolve().is_relative_to(BACKUP_DIR.resolve()):
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    try:
        backup_path.unlink()
        return {
            "success": True,
            "message": f"Backup '{filename}' deleted successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete backup: {str(e)}"
        )
