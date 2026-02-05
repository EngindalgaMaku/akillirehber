"""Public restore endpoints for initial setup (no authentication required)."""

import json
import zipfile
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import get_db
from app.services.weaviate_service import WeaviateService

router = APIRouter(prefix="/api/restore", tags=["restore"])


@router.post("/postgres")
async def restore_postgres(file: UploadFile = File(...)):
    """Restore PostgreSQL database from uploaded file (no auth required)."""
    try:
        content = await file.read()
        
        # Check if file is zipped
        if file.filename and file.filename.endswith('.zip'):
            # Extract SQL from zip
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                tmp_zip.write(content)
                tmp_zip_path = tmp_zip.name
            
            with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                # Find SQL file in zip
                sql_files = [f for f in zip_ref.namelist() if f.endswith('.sql')]
                if not sql_files:
                    raise HTTPException(400, "No SQL file found in zip")
                
                with zip_ref.open(sql_files[0]) as sql_file:
                    sql_content = sql_file.read().decode('utf-8')
            
            Path(tmp_zip_path).unlink()
        else:
            sql_content = content.decode('utf-8')
        
        # Execute SQL
        from app.database import SessionLocal
        db = SessionLocal()
        errors = []
        success_count = 0
        
        try:
            statements = [s.strip() for s in sql_content.split(';') if s.strip() and not s.strip().startswith('--')]
            
            for i, statement in enumerate(statements):
                try:
                    db.execute(text(statement))
                    success_count += 1
                except Exception as e:
                    error_msg = f"Statement {i+1}: {str(e)[:100]}"
                    errors.append(error_msg)
                    print(f"Warning: {error_msg}")
            
            db.commit()
            
            # Check if any users were created
            user_count = db.execute(text("SELECT COUNT(*) FROM users")).scalar()
            
            message = f"PostgreSQL restored: {success_count} statements executed"
            if errors:
                message += f", {len(errors)} errors (check logs)"
            message += f", {user_count} users in database"
            
            return {
                "success": True,
                "message": message,
                "details": {
                    "statements_executed": success_count,
                    "errors": len(errors),
                    "users": user_count
                }
            }
        except Exception as e:
            db.rollback()
            raise HTTPException(500, f"Restore failed: {str(e)}")
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to process file: {str(e)}")


@router.post("/weaviate")
async def restore_weaviate(file: UploadFile = File(...)):
    """Restore Weaviate database from uploaded file (no auth required)."""
    try:
        content = await file.read()
        
        # Check if file is zipped
        if file.filename and file.filename.endswith('.zip'):
            # Extract JSON from zip
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                tmp_zip.write(content)
                tmp_zip_path = tmp_zip.name
            
            with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                # Find JSON file in zip
                json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]
                if not json_files:
                    raise HTTPException(400, "No JSON file found in zip")
                
                with zip_ref.open(json_files[0]) as json_file:
                    backup_data = json.load(json_file)
            
            Path(tmp_zip_path).unlink()
        else:
            backup_data = json.loads(content.decode('utf-8'))
        
        # Restore to Weaviate
        weaviate_service = WeaviateService()
        total_imported = 0
        errors = []
        
        for collection_name, collection_data in backup_data.get('collections', {}).items():
            if 'error' in collection_data:
                errors.append(f"{collection_name}: {collection_data['error']}")
                continue
            
            course_id = collection_data.get('course_id')
            objects = collection_data.get('objects', [])
            
            if course_id and objects:
                try:
                    imported = weaviate_service.import_collection(course_id, objects)
                    total_imported += imported
                except Exception as e:
                    error_msg = f"Course {course_id}: {str(e)[:100]}"
                    errors.append(error_msg)
                    print(f"Error importing course {course_id}: {e}")
        
        message = f"Weaviate restored: {total_imported} objects imported"
        if errors:
            message += f", {len(errors)} errors (check logs)"
        
        return {
            "success": True,
            "message": message,
            "details": {
                "objects_imported": total_imported,
                "errors": len(errors),
                "error_messages": errors[:5]  # First 5 errors
            }
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to process file: {str(e)}")
