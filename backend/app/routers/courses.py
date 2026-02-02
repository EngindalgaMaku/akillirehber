"""Course management API endpoints."""

from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models.db_models import User, UserRole, Course, Document, Chunk
from app.models.schemas import (
    CourseCreate,
    CourseUpdate,
    CourseResponse,
    CourseListResponse,
)
from app.services.auth_service import get_current_user, get_current_teacher
from app.services.course_service import (
    create_course,
    delete_course,
    get_all_active_courses,
    get_course_by_id,
    get_course_with_document_count,
    get_courses_by_teacher,
    update_course,
    verify_course_access,
)

router = APIRouter(prefix="/api/courses", tags=["courses"])


@router.get("/dashboard/stats")
async def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get dashboard statistics for the current user.

    Returns course count, document count, and chunk count.
    """
    if current_user.role == UserRole.TEACHER:
        # Teacher sees their own courses
        courses = get_courses_by_teacher(db, current_user.id)
        course_ids = [c.id for c in courses]
    else:
        # Student sees all active courses
        courses = get_all_active_courses(db)
        course_ids = [c.id for c in courses]

    course_count = len(courses)

    # Count documents in user's courses
    document_count = (
        db.query(func.count(Document.id))
        .filter(Document.course_id.in_(course_ids))
        .scalar()
    ) if course_ids else 0

    # Count chunks in user's courses
    chunk_count = (
        db.query(func.count(Chunk.id))
        .join(Document, Chunk.document_id == Document.id)
        .filter(Document.course_id.in_(course_ids))
        .scalar()
    ) if course_ids else 0

    return {
        "course_count": course_count,
        "document_count": document_count,
        "chunk_count": chunk_count,
    }


@router.get("", response_model=CourseListResponse)
async def list_courses(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """
    List courses based on user role.

    - **Teachers**: See only their own courses (including inactive)
    - **Admins**: See all courses (including inactive)
    - **Students**: See all active courses
    """
    if current_user.role == UserRole.ADMIN:
        # Admin sees all courses (including inactive)
        courses = db.query(Course).order_by(Course.created_at.desc()).all()
    elif current_user.role == UserRole.TEACHER:
        courses = get_courses_by_teacher(db, current_user.id)
    else:
        courses = get_all_active_courses(db)

    course_responses = [
        CourseResponse(**get_course_with_document_count(db, course))
        for course in courses
    ]

    return CourseListResponse(courses=course_responses, total=len(course_responses))


@router.post("", response_model=CourseResponse, status_code=status.HTTP_201_CREATED)
async def create_new_course(
    course_data: CourseCreate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Create a new course.

    Only teachers and admins can create courses.

    - **name**: Course name (required)
    - **description**: Course description (optional)
    """
    course = create_course(db, course_data, current_user)
    return CourseResponse(**get_course_with_document_count(db, course))


@router.get("/{course_id}", response_model=CourseResponse)
async def get_course(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get course details by ID.

    All authenticated users can view course details.
    """
    course = verify_course_access(db, course_id, current_user)
    return CourseResponse(**get_course_with_document_count(db, course))


@router.put("/{course_id}", response_model=CourseResponse)
async def update_existing_course(
    course_id: int,
    course_data: CourseUpdate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Update an existing course.

    Only the course owner (teacher) can update it.

    - **name**: New course name (optional)
    - **description**: New course description (optional)
    """
    course = update_course(db, course_id, course_data, current_user)
    return CourseResponse(**get_course_with_document_count(db, course))


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_course(
    course_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Delete a course.

    Only the course owner (teacher) can delete it.
    This is a soft delete - the course is marked as inactive.
    """
    delete_course(db, course_id, current_user)
    return None


@router.get("/{course_id}/chunks")
async def get_course_chunks(
    course_id: int,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get chunks from a course's documents for test generation."""
    # Verify access
    verify_course_access(db, course_id, current_user)
    
    # Get chunks from all documents in this course
    chunks = (
        db.query(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .filter(Document.course_id == course_id)
        .filter(Chunk.content.isnot(None))
        .limit(limit)
        .all()
    )
    
    return [{"id": c.id, "content": c.content} for c in chunks]
