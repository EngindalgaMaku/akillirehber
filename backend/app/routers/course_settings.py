"""Course settings API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import User, CourseSettings, CustomLLMModel
from app.models.schemas import CourseSettingsResponse, CourseSettingsUpdate
from app.services.auth_service import get_current_user, get_current_teacher
from app.services.course_service import (
    verify_course_access,
    verify_course_ownership,
    get_or_create_settings,
)

router = APIRouter(prefix="/api", tags=["course-settings"])


@router.get(
    "/courses/{course_id}/settings",
    response_model=CourseSettingsResponse
)
async def get_course_settings(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get course settings.

    All authenticated users can view settings.
    """
    verify_course_access(db, course_id, current_user)
    settings = get_or_create_settings(db, course_id)
    return CourseSettingsResponse.model_validate(settings)


@router.put(
    "/courses/{course_id}/settings",
    response_model=CourseSettingsResponse
)
async def update_course_settings(
    course_id: int,
    request: CourseSettingsUpdate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Update course settings.

    Only teachers can update settings for their own courses.
    """
    verify_course_ownership(db, course_id, current_user)
    settings = get_or_create_settings(db, course_id)
    
    # Update only provided fields
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(settings, field, value)
    
    db.commit()
    db.refresh(settings)
    
    return CourseSettingsResponse.model_validate(settings)


@router.get("/llm-providers")
async def get_llm_providers(
    db: Session = Depends(get_db),
):
    """Get available LLM providers and their models (default + custom)."""
    from app.services.llm_providers import LLM_PROVIDERS
    
    providers = {}
    for provider in LLM_PROVIDERS.keys():
        # Get default models
        default_models = LLM_PROVIDERS[provider]["models"]
        
        # Get custom models for this provider
        custom_models = db.query(CustomLLMModel).filter(
            CustomLLMModel.provider == provider,
            CustomLLMModel.is_active.is_(True)
        ).all()
        
        # Combine: default models + custom model IDs
        all_models = list(default_models)
        for cm in custom_models:
            if cm.model_id not in all_models:
                all_models.append(cm.model_id)
        
        providers[provider] = all_models
    
    return providers
