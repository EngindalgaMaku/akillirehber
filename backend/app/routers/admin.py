"""Admin user management API endpoints."""

from typing import Optional
from math import ceil
from datetime import datetime, timedelta, UTC
import secrets
import string

from fastapi import APIRouter, Depends, Query, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import or_, func

from app.database import get_db
from app.models.db_models import User, UserRole, TemporaryPassword
from app.models.schemas import (
    AdminUserListResponse,
    AdminUserCreate,
    AdminUserUpdate,
    AdminUserUpdateResponse,
    AdminUserDeleteResponse,
    AdminUserDeactivateResponse,
    AdminStatisticsResponse,
    AdminPasswordResetResponse,
)
from app.services.auth_service import require_admin, get_password_hash
from app.services.audit_service import log_admin_action
from app.services.email_service import send_password_reset_email

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/users", response_model=AdminUserUpdateResponse)
async def create_user(
    user_data: AdminUserCreate,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Create a new user as admin (no registration key required).
    """
    from sqlalchemy import exists

    # Check if email already exists
    email_exists = db.query(
        exists().where(User.email == user_data.email)
    ).scalar()
    if email_exists:
        raise HTTPException(
            status_code=400,
            detail=f"Bu e-posta zaten kayıtlı: {user_data.email}",
        )

    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Log admin action
    client_ip = request.client.host if request.client else None
    log_admin_action(
        db=db,
        admin_user_id=admin_user.id,
        action="create",
        target_user_id=new_user.id,
        details={
            "user_email": new_user.email,
            "user_role": new_user.role.value,
        },
        ip_address=client_ip,
    )

    return AdminUserUpdateResponse(
        success=True,
        user=new_user,
        message=f"Kullanıcı {new_user.email} başarıyla oluşturuldu",
    )


@router.get("/users", response_model=AdminUserListResponse)
async def get_users(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    role: Optional[str] = Query(
        None,
        description="Filter by role (teacher/student)",
    ),
    search: Optional[str] = Query(None, description="Search by name or email"),
    sort_by: str = Query("id", description="Column to sort by"),
    sort_order: str = Query("asc", description="Sort order (asc/desc)"),
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Get list of all users with pagination, filtering, and sorting.
    Requires admin privileges (user ID=1).
    - **page**: Page number (default: 1)
    - **limit**: Items per page (default: 50, max: 100)
    - **role**: Filter by role (teacher or student)
    - **search**: Search by name or email (case-insensitive)
    - **sort_by**: Column to sort by (id, full_name, email, role, created_at,
      last_login)
    - **sort_order**: Sort order (asc or desc)
    Returns paginated list of users with all required fields.
    """
    # Build base query
    query = db.query(User)

    # Apply role filter
    if role:
        try:
            role_enum = UserRole(role.lower())
            query = query.filter(User.role == role_enum)
        except ValueError:
            # Invalid role, ignore filter
            pass

    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                User.full_name.ilike(search_term),
                User.email.ilike(search_term)
            )
        )

    # Get total count before pagination
    total = query.count()

    # Apply sorting
    valid_sort_columns = {
        "id": User.id,
        "full_name": User.full_name,
        "email": User.email,
        "role": User.role,
        "created_at": User.created_at,
        "last_login": User.last_login,
        "is_active": User.is_active,
    }

    sort_column = valid_sort_columns.get(sort_by, User.id)
    if sort_order.lower() == "desc":
        sort_column = sort_column.desc()
    else:
        sort_column = sort_column.asc()

    query = query.order_by(sort_column)

    # Apply pagination
    offset = (page - 1) * limit
    users = query.offset(offset).limit(limit).all()

    # Calculate total pages
    total_pages = ceil(total / limit) if total > 0 else 1

    return AdminUserListResponse(
        users=users,
        total=total,
        page=page,
        total_pages=total_pages,
    )


@router.put("/users/{user_id}", response_model=AdminUserUpdateResponse)
async def update_user(
    user_id: int,
    user_update: AdminUserUpdate,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Update a user's information.
    Requires admin privileges (user ID=1).
    - **user_id**: ID of the user to update
    - **full_name**: New full name (optional)
    - **email**: New email address (optional)
    - **role**: New role (teacher or student, optional)
    - **is_active**: Active status (optional)
    Restrictions:
    - Cannot edit user with ID=1 (admin user)
    - Email must be unique if changed
    Returns updated user information."""
    # Get the user to update
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User with ID {user_id} not found"
        )

    # Prevent editing admin users
    if user.role == UserRole.ADMIN:
        raise HTTPException(
            status_code=400,
            detail="Cannot edit admin users"
        )

    # Track changed fields for audit log
    changed_fields = []

    # Validate email uniqueness if email is being changed
    if user_update.email is not None and user_update.email != user.email:
        existing_user = db.query(User).filter(
            User.email == user_update.email,
            User.id != user_id
        ).first()

        if existing_user:
            msg = (
                f"Email {user_update.email} is already in use by another user"
            )
            raise HTTPException(
                status_code=400,
                detail=msg,
            )

        user.email = user_update.email
        changed_fields.append("email")

    # Update full name if provided
    if user_update.full_name is not None:
        user.full_name = user_update.full_name
        changed_fields.append("full_name")

    # Update role if provided
    if user_update.role is not None:
        user.role = user_update.role
        changed_fields.append("role")

    # Update active status if provided
    if user_update.is_active is not None:
        user.is_active = user_update.is_active
        changed_fields.append("is_active")

    # Save changes to database
    db.commit()
    db.refresh(user)

    # Log admin action
    client_ip = request.client.host if request.client else None
    log_admin_action(
        db=db,
        admin_user_id=admin_user.id,
        action="edit",
        target_user_id=user_id,
        details={
            "changed_fields": changed_fields,
            "new_values": {
                field: getattr(user, field) for field in changed_fields
            },
        },
        ip_address=client_ip,
    )

    return AdminUserUpdateResponse(
        success=True,
        user=user,
        message="User updated successfully"
    )


@router.delete("/users/{user_id}", response_model=AdminUserDeleteResponse)
async def delete_user(
    user_id: int,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete a user and all related data.
    Requires admin privileges (user ID=1).
    - **user_id**: ID of the user to delete
    Restrictions:
    - Cannot delete user with ID=1 (admin user)
    Cascade deletion includes:
    - User's courses (if teacher) and all related documents
    - User's test sets and evaluation runs
    - User's quick test results
    - User's semantic similarity results
    - User's custom LLM models
    - User's refresh tokens
    - User's temporary passwords
    - Audit logs where user is target
    Returns success message and summary of deleted data."""
    # Get the user to delete
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User with ID {user_id} not found"
        )

    # Prevent deleting admin users
    if user.role == UserRole.ADMIN:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete admin users"
        )

    # Track deleted data for response
    deleted_data = {
        "user_id": user_id,
        "user_email": user.email,
        "user_role": user.role.value,
    }

    # Import models for cascade deletion
    from app.models.db_models import (
        Course,
        TestSet,
        QuickTestResult,
        SemanticSimilarityResult,
        CustomLLMModel,
        RefreshToken,
        TemporaryPassword,
        AuditLog,
    )

    # Delete courses (if teacher) - cascade will handle documents, chunks
    if user.role.value == "teacher":
        courses = db.query(Course).filter(
            Course.teacher_id == user_id
        ).all()
        deleted_data["courses_deleted"] = len(courses)
        for course in courses:
            db.delete(course)

    # Delete test sets created by user
    test_sets = db.query(TestSet).filter(
        TestSet.created_by == user_id
    ).all()
    deleted_data["test_sets_deleted"] = len(test_sets)
    for test_set in test_sets:
        db.delete(test_set)

    # Delete quick test results
    quick_tests = db.query(QuickTestResult).filter(
        QuickTestResult.created_by == user_id
    ).all()
    deleted_data["quick_tests_deleted"] = len(quick_tests)
    for quick_test in quick_tests:
        db.delete(quick_test)

    # Delete semantic similarity results
    similarity_results = db.query(SemanticSimilarityResult).filter(
        SemanticSimilarityResult.created_by == user_id
    ).all()
    deleted_data["similarity_results_deleted"] = len(similarity_results)
    for result in similarity_results:
        db.delete(result)

    # Delete custom LLM models
    custom_models = db.query(CustomLLMModel).filter(
        CustomLLMModel.created_by == user_id
    ).all()
    deleted_data["custom_models_deleted"] = len(custom_models)
    for model in custom_models:
        db.delete(model)

    # Delete refresh tokens
    refresh_tokens = db.query(RefreshToken).filter(
        RefreshToken.user_id == user_id
    ).all()
    deleted_data["refresh_tokens_deleted"] = len(refresh_tokens)
    for token in refresh_tokens:
        db.delete(token)

    # Delete temporary passwords
    temp_passwords = db.query(TemporaryPassword).filter(
        TemporaryPassword.user_id == user_id
    ).all()
    deleted_data["temp_passwords_deleted"] = len(temp_passwords)
    for temp_pass in temp_passwords:
        db.delete(temp_pass)

    # Delete audit logs where user is target
    # (Keep audit logs where user was admin to maintain audit trail)
    target_audit_logs = db.query(AuditLog).filter(
        AuditLog.target_user_id == user_id
    ).all()
    deleted_data["audit_logs_deleted"] = len(target_audit_logs)
    for log in target_audit_logs:
        db.delete(log)

    # Finally, delete the user
    db.delete(user)

    # Commit all deletions
    db.commit()

    # Log admin action
    client_ip = request.client.host if request.client else None
    log_admin_action(
        db=db,
        admin_user_id=admin_user.id,
        action="delete",
        target_user_id=user_id,
        details={
            "deleted_user_email": user.email,
            "deleted_user_role": user.role.value,
            "deleted_data_summary": deleted_data,
        },
        ip_address=client_ip,
    )

    return AdminUserDeleteResponse(
        success=True,
        message=f"User {user.email} (ID={user_id}) deleted successfully",
        deleted_data=deleted_data,
    )


@router.patch(
    "/users/{user_id}/deactivate",
    response_model=AdminUserDeactivateResponse
)
async def deactivate_user(
    user_id: int,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Deactivate a user account.

    Requires admin privileges (user ID=1).

    - **user_id**: ID of the user to deactivate

    Restrictions:
    - Cannot deactivate user with ID=1 (admin user)

    When a user is deactivated:
    - is_active is set to False
    - User cannot log in
    - All admin actions are logged

    Returns success message and updated user information.
    """
    # Get the user to deactivate
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User with ID {user_id} not found"
        )

    # Prevent deactivating admin users
    if user.role == UserRole.ADMIN:
        raise HTTPException(
            status_code=400,
            detail="Cannot deactivate admin users"
        )

    # Check if already inactive
    if not user.is_active:
        raise HTTPException(
            status_code=400,
            detail=f"User {user.email} is already inactive"
        )

    # Deactivate the user
    user.is_active = False
    db.commit()
    db.refresh(user)

    # Log admin action
    client_ip = request.client.host if request.client else None
    log_admin_action(
        db=db,
        admin_user_id=admin_user.id,
        action="deactivate",
        target_user_id=user_id,
        details={
            "user_email": user.email,
            "user_role": user.role.value,
        },
        ip_address=client_ip,
    )

    return AdminUserDeactivateResponse(
        success=True,
        message=f"User {user.email} (ID={user_id}) deactivated successfully",
        user=user,
    )


@router.patch(
    "/users/{user_id}/activate",
    response_model=AdminUserDeactivateResponse
)
async def activate_user(
    user_id: int,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Activate a user account.

    Requires admin privileges (user ID=1).

    - **user_id**: ID of the user to activate

    When a user is activated:
    - is_active is set to True
    - User can log in again
    - All admin actions are logged

    Returns success message and updated user information.
    """
    # Get the user to activate
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User with ID {user_id} not found"
        )

    # Check if already active
    if user.is_active:
        raise HTTPException(
            status_code=400,
            detail=f"User {user.email} is already active"
        )

    # Activate the user
    user.is_active = True
    db.commit()
    db.refresh(user)

    # Log admin action
    client_ip = request.client.host if request.client else None
    log_admin_action(
        db=db,
        admin_user_id=admin_user.id,
        action="activate",
        target_user_id=user_id,
        details={
            "user_email": user.email,
            "user_role": user.role.value,
        },
        ip_address=client_ip,
    )

    return AdminUserDeactivateResponse(
        success=True,
        message=f"User {user.email} (ID={user_id}) activated successfully",
        user=user,
    )


@router.get("/statistics", response_model=AdminStatisticsResponse)
async def get_statistics(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Get admin statistics.

    Requires admin privileges (user ID=1).

    Returns:
    - **total_users**: Total number of users in the system
    - **active_teachers**: Number of active teachers
    - **active_students**: Number of active students
    - **inactive_users**: Number of inactive users
    - **new_users_this_month**: Number of users created this month
    """
    # Calculate total users
    total_users = db.query(func.count(User.id)).scalar()

    # Calculate active teachers
    active_teachers = db.query(func.count(User.id)).filter(
        User.role == UserRole.TEACHER,
        User.is_active == True  # noqa: E712
    ).scalar()

    # Calculate active students
    active_students = db.query(func.count(User.id)).filter(
        User.role == UserRole.STUDENT,
        User.is_active == True  # noqa: E712
    ).scalar()

    # Calculate inactive users
    inactive_users = db.query(func.count(User.id)).filter(
        User.is_active == False  # noqa: E712
    ).scalar()

    # Calculate new users this month
    # Get first day of current month
    now = datetime.now(UTC)
    first_day_of_month = datetime(now.year, now.month, 1, tzinfo=UTC)

    new_users_this_month = db.query(func.count(User.id)).filter(
        User.created_at >= first_day_of_month
    ).scalar()

    return AdminStatisticsResponse(
        total_users=total_users,
        active_teachers=active_teachers,
        active_students=active_students,
        inactive_users=inactive_users,
        new_users_this_month=new_users_this_month,
    )


@router.post(
    "/users/{user_id}/reset-password",
    response_model=AdminPasswordResetResponse
)
async def reset_user_password(
    user_id: int,
    request: Request,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Reset a user's password and generate a temporary password.

    Requires admin privileges (user ID=1).

    - **user_id**: ID of the user whose password to reset

    Restrictions:
    - Cannot reset password for user with ID=1 (admin user)

    Process:
    1. Generate a random temporary password (12 characters)
    2. Hash and store in TemporaryPassword table
    3. Set expiration to 24 hours from now
    4. Send email to user with temporary password
    5. Log admin action
    6. Return temporary password to admin

    Note: If email is not configured (SMTP settings missing), the
    temporary password will still be generated and returned to admin
    for manual communication to the user.

    Returns temporary password and expiration time.
    """
    # Get the user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User with ID {user_id} not found"
        )

    # Prevent resetting password for admin users
    if user.role == UserRole.ADMIN:
        raise HTTPException(
            status_code=400,
            detail="Cannot reset password for admin users"
        )

    # Generate random temporary password (12 characters)
    # Use letters, digits, and some special characters
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    temporary_password = ''.join(
        secrets.choice(alphabet) for _ in range(12)
    )

    # Hash the temporary password
    hashed_temp_password = get_password_hash(temporary_password)

    # Set expiration to 24 hours from now
    expires_at = datetime.now(UTC) + timedelta(hours=24)

    # Invalidate any existing temporary passwords for this user
    db.query(TemporaryPassword).filter(
        TemporaryPassword.user_id == user_id,
        TemporaryPassword.used == False  # noqa: E712
    ).update({"used": True})

    # Create new temporary password record
    temp_password_record = TemporaryPassword(
        user_id=user_id,
        hashed_password=hashed_temp_password,
        expires_at=expires_at,
        used=False,
    )
    db.add(temp_password_record)
    db.commit()

    # Log admin action
    client_ip = request.client.host if request.client else None
    log_admin_action(
        db=db,
        admin_user_id=admin_user.id,
        action="reset_password",
        target_user_id=user_id,
        details={
            "user_email": user.email,
            "expires_at": expires_at.isoformat(),
        },
        ip_address=client_ip,
    )

    # Send email to user with temporary password
    email_sent = send_password_reset_email(
        to_email=user.email,
        user_name=user.full_name,
        temporary_password=temporary_password,
        expires_hours=24
    )

    # Prepare response message
    if email_sent:
        message = (
            f"Temporary password generated and sent to {user.email}. "
            "Valid for 24 hours."
        )
    else:
        message = (
            f"Temporary password generated for {user.email}. "
            "Valid for 24 hours. "
            "Email could not be sent - please communicate this to the user."
        )

    return AdminPasswordResetResponse(
        success=True,
        message=message,
        temporary_password=temporary_password,
        expires_at=expires_at,
    )
