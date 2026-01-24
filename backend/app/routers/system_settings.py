"""System settings API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.database import get_db
from app.models.db_models import User, SystemSettings
from app.services.auth_service import require_admin

router = APIRouter(prefix="/api/system", tags=["system"])


class SystemSettingsResponse(BaseModel):
    """Response model for system settings."""
    id: int
    teacher_registration_key: Optional[str] = None
    student_registration_key: Optional[str] = None
    hcaptcha_site_key: Optional[str] = None
    captcha_enabled: bool = True
    
    model_config = {"from_attributes": True}


class SystemSettingsUpdate(BaseModel):
    """Update model for system settings."""
    teacher_registration_key: Optional[str] = None
    student_registration_key: Optional[str] = None
    hcaptcha_site_key: Optional[str] = None
    hcaptcha_secret_key: Optional[str] = None
    captcha_enabled: Optional[bool] = None


class PublicSettingsResponse(BaseModel):
    """Public settings for registration page."""
    captcha_enabled: bool
    hcaptcha_site_key: Optional[str] = None
    registration_key_required: bool


class VerifyKeyRequest(BaseModel):
    """Request to verify registration key."""
    role: str
    key: str


class VerifyKeyResponse(BaseModel):
    """Response for key verification."""
    valid: bool


def get_or_create_settings(db: Session) -> SystemSettings:
    """Get or create system settings."""
    settings = db.query(SystemSettings).first()
    if not settings:
        settings = SystemSettings(
            teacher_registration_key="TEACHER2026",
            student_registration_key="STUDENT2026",
            captcha_enabled=False
        )
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


@router.get("/settings", response_model=SystemSettingsResponse)
async def get_system_settings(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Get system settings (admin only - user id=1)."""
    settings = get_or_create_settings(db)
    return settings


@router.put("/settings", response_model=SystemSettingsResponse)
async def update_system_settings(
    data: SystemSettingsUpdate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Update system settings (admin only - user id=1)."""
    settings = get_or_create_settings(db)
    
    if data.teacher_registration_key is not None:
        settings.teacher_registration_key = data.teacher_registration_key
    if data.student_registration_key is not None:
        settings.student_registration_key = data.student_registration_key
    if data.hcaptcha_site_key is not None:
        settings.hcaptcha_site_key = data.hcaptcha_site_key
    if data.hcaptcha_secret_key is not None:
        settings.hcaptcha_secret_key = data.hcaptcha_secret_key
    if data.captcha_enabled is not None:
        settings.captcha_enabled = data.captcha_enabled
    
    db.commit()
    db.refresh(settings)
    return settings


@router.get("/public-settings", response_model=PublicSettingsResponse)
async def get_public_settings(db: Session = Depends(get_db)):
    """Get public settings for registration page (no auth required)."""
    settings = get_or_create_settings(db)
    
    # Check if registration keys are set
    key_required = bool(
        settings.teacher_registration_key or 
        settings.student_registration_key
    )
    
    return PublicSettingsResponse(
        captcha_enabled=settings.captcha_enabled,
        hcaptcha_site_key=settings.hcaptcha_site_key if settings.captcha_enabled else None,
        registration_key_required=key_required
    )


@router.post("/verify-key", response_model=VerifyKeyResponse)
async def verify_registration_key(
    data: VerifyKeyRequest,
    db: Session = Depends(get_db)
):
    """Verify registration key for a role (no auth required)."""
    settings = get_or_create_settings(db)
    
    if data.role == "teacher":
        expected_key = settings.teacher_registration_key
    elif data.role == "student":
        expected_key = settings.student_registration_key
    else:
        return VerifyKeyResponse(valid=False)
    
    # If no key is set, allow registration
    if not expected_key:
        return VerifyKeyResponse(valid=True)
    
    return VerifyKeyResponse(valid=data.key == expected_key)
