"""Course prompt templates API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import CoursePromptTemplate, CourseSettings, User
from app.models.schemas import (
    CoursePromptTemplateCreate,
    CoursePromptTemplateListResponse,
    CoursePromptTemplateResponse,
    CoursePromptTemplateUpdate,
)
from app.services.auth_service import get_current_user, get_current_teacher
from app.services.course_service import (
    get_or_create_settings,
    verify_course_access,
    verify_course_ownership,
)

router = APIRouter(prefix="/api", tags=["course-prompt-templates"])


@router.get(
    "/courses/{course_id}/prompt-templates",
    response_model=CoursePromptTemplateListResponse,
)
async def list_course_prompt_templates(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    verify_course_access(db, course_id, current_user)
    templates = (
        db.query(CoursePromptTemplate)
        .filter(CoursePromptTemplate.course_id == course_id)
        .order_by(CoursePromptTemplate.created_at.desc())
        .all()
    )
    return {
        "templates": [
            CoursePromptTemplateResponse.model_validate(t) for t in templates
        ]
    }


@router.post(
    "/courses/{course_id}/prompt-templates",
    response_model=CoursePromptTemplateResponse,
)
async def create_course_prompt_template(
    course_id: int,
    request: CoursePromptTemplateCreate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    verify_course_ownership(db, course_id, current_user)
    template = CoursePromptTemplate(
        course_id=course_id,
        name=request.name.strip(),
        content=request.content,
    )
    db.add(template)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Bu ders için aynı isimde bir prompt template zaten var.",
        ) from exc
    db.refresh(template)
    return CoursePromptTemplateResponse.model_validate(template)


@router.put(
    "/courses/{course_id}/prompt-templates/{template_id}",
    response_model=CoursePromptTemplateResponse,
)
async def update_course_prompt_template(
    course_id: int,
    template_id: int,
    request: CoursePromptTemplateUpdate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    verify_course_ownership(db, course_id, current_user)
    template = (
        db.query(CoursePromptTemplate)
        .filter(CoursePromptTemplate.id == template_id)
        .filter(CoursePromptTemplate.course_id == course_id)
        .first()
    )
    if not template:
        raise HTTPException(status_code=404, detail="Template bulunamadı")

    update_data = request.model_dump(exclude_unset=True)
    if "name" in update_data and update_data["name"] is not None:
        update_data["name"] = update_data["name"].strip()

    for field, value in update_data.items():
        setattr(template, field, value)

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Bu ders için aynı isimde bir prompt template zaten var.",
        ) from exc

    db.refresh(template)
    return CoursePromptTemplateResponse.model_validate(template)


@router.delete("/courses/{course_id}/prompt-templates/{template_id}")
async def delete_course_prompt_template(
    course_id: int,
    template_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    verify_course_ownership(db, course_id, current_user)

    template = (
        db.query(CoursePromptTemplate)
        .filter(CoursePromptTemplate.id == template_id)
        .filter(CoursePromptTemplate.course_id == course_id)
        .first()
    )
    if not template:
        raise HTTPException(status_code=404, detail="Template bulunamadı")

    settings = get_or_create_settings(db, course_id)
    if settings.active_prompt_template_id == template_id:
        settings.active_prompt_template_id = None

    db.delete(template)
    db.commit()

    return {"success": True}


@router.post("/courses/{course_id}/prompt-templates/activate")
async def activate_course_prompt_template(
    course_id: int,
    template_id: Optional[int] = None,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    verify_course_ownership(db, course_id, current_user)
    settings: CourseSettings = get_or_create_settings(db, course_id)

    if template_id is None:
        settings.active_prompt_template_id = None
        db.commit()
        db.refresh(settings)
        return {"success": True, "active_prompt_template_id": None}

    template = (
        db.query(CoursePromptTemplate)
        .filter(CoursePromptTemplate.id == template_id)
        .filter(CoursePromptTemplate.course_id == course_id)
        .first()
    )
    if not template:
        raise HTTPException(status_code=404, detail="Template bulunamadı")

    settings.active_prompt_template_id = template_id
    db.commit()
    db.refresh(settings)

    return {"success": True, "active_prompt_template_id": template_id}
