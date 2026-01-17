"""
Tests for admin user management endpoints.

Feature: admin-user-management, Task 4.1
Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.7
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.models.db_models import User, UserRole
from app.services.auth_service import create_access_token, get_password_hash


@pytest.fixture
def regular_user(db_session: Session, hashed_password):
    """Create regular non-admin user."""
    user = User(
        email="user@test.com",
        hashed_password=hashed_password,
        full_name="Regular User",
        role=UserRole.TEACHER,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def regular_token(regular_user):
    """Create regular user JWT token (no admin flag)."""
    return create_access_token(
        data={
            "sub": str(regular_user.id),
            "email": regular_user.email,
            "role": regular_user.role.value,
            "is_admin": False,
        }
    )


def test_get_users_requires_admin(client, regular_token):
    """Test that non-admin users cannot access user list."""
    response = client.get(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {regular_token}"}
    )
    assert response.status_code == 403
    assert "Admin privileges required" in response.json()["detail"]


def test_get_users_requires_authentication(client):
    """Test that unauthenticated requests are rejected."""
    response = client.get("/api/admin/users")
    assert response.status_code == 401


def test_get_users_basic(client, admin_token, sample_users, admin_user):
    """Test basic user list retrieval."""
    response = client.get(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "users" in data
    assert "total" in data
    assert "page" in data
    assert "total_pages" in data
    
    # Should have admin + 4 sample users = 5 total
    assert data["total"] == 5
    assert len(data["users"]) == 5
    assert data["page"] == 1
    assert data["total_pages"] == 1
    
    # Check user fields (Requirement 4.2)
    user = data["users"][0]
    assert "id" in user
    assert "full_name" in user
    assert "email" in user
    assert "role" in user
    assert "is_active" in user
    assert "created_at" in user
    assert "last_login" in user


def test_get_users_pagination(client, admin_token, sample_users, admin_user):
    """Test pagination works correctly (Requirement 4.7)."""
    # Get first page with limit 2
    response = client.get(
        "/api/admin/users?page=1&limit=2",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 5
    assert len(data["users"]) == 2
    assert data["page"] == 1
    assert data["total_pages"] == 3
    
    # Get second page
    response = client.get(
        "/api/admin/users?page=2&limit=2",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 5
    assert len(data["users"]) == 2
    assert data["page"] == 2
    
    # Get last page
    response = client.get(
        "/api/admin/users?page=3&limit=2",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 5
    assert len(data["users"]) == 1  # Only 1 user on last page
    assert data["page"] == 3


def test_get_users_role_filter(client, admin_token, sample_users, admin_user):
    """Test filtering by role (Requirement 4.4)."""
    # Filter for teachers
    response = client.get(
        "/api/admin/users?role=teacher",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Admin + 2 teachers = 3
    assert data["total"] == 3
    for user in data["users"]:
        assert user["role"] == "teacher"
    
    # Filter for students
    response = client.get(
        "/api/admin/users?role=student",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 2
    for user in data["users"]:
        assert user["role"] == "student"


def test_get_users_search_by_name(client, admin_token, sample_users, admin_user):
    """Test search by name (Requirement 4.5)."""
    response = client.get(
        "/api/admin/users?search=Teacher One",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 1
    assert data["users"][0]["full_name"] == "Teacher One"


def test_get_users_search_by_email(client, admin_token, sample_users, admin_user):
    """Test search by email (Requirement 4.5)."""
    response = client.get(
        "/api/admin/users?search=student1@test.com",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 1
    assert data["users"][0]["email"] == "student1@test.com"


def test_get_users_search_partial(client, admin_token, sample_users, admin_user):
    """Test partial search (case-insensitive) (Requirement 4.5)."""
    response = client.get(
        "/api/admin/users?search=student",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should find both students
    assert data["total"] == 2
    for user in data["users"]:
        assert "student" in user["email"].lower() or "student" in user["full_name"].lower()


def test_get_users_sort_by_name(client, admin_token, sample_users, admin_user):
    """Test sorting by name (Requirement 4.3)."""
    # Sort ascending
    response = client.get(
        "/api/admin/users?sort_by=full_name&sort_order=asc",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    names = [user["full_name"] for user in data["users"]]
    assert names == sorted(names)
    
    # Sort descending
    response = client.get(
        "/api/admin/users?sort_by=full_name&sort_order=desc",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    names = [user["full_name"] for user in data["users"]]
    assert names == sorted(names, reverse=True)


def test_get_users_sort_by_created_at(client, admin_token, sample_users, admin_user):
    """Test sorting by creation date (Requirement 4.3)."""
    response = client.get(
        "/api/admin/users?sort_by=created_at&sort_order=desc",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Most recent should be first
    dates = [user["created_at"] for user in data["users"]]
    assert dates == sorted(dates, reverse=True)


def test_get_users_combined_filters(client, admin_token, sample_users, admin_user):
    """Test combining multiple filters."""
    response = client.get(
        "/api/admin/users?role=teacher&search=teacher&sort_by=full_name&sort_order=asc",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should find 2 teachers with "teacher" in name
    assert data["total"] == 2
    for user in data["users"]:
        assert user["role"] == "teacher"
        assert "teacher" in user["full_name"].lower()
    
    # Check sorting
    names = [user["full_name"] for user in data["users"]]
    assert names == sorted(names)


def test_get_users_empty_result(client, admin_token, sample_users, admin_user):
    """Test empty result set."""
    response = client.get(
        "/api/admin/users?search=nonexistent",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] == 0
    assert len(data["users"]) == 0
    assert data["page"] == 1
    assert data["total_pages"] == 1


def test_get_users_invalid_role_filter(client, admin_token, sample_users, admin_user):
    """Test that invalid role filter is ignored."""
    response = client.get(
        "/api/admin/users?role=invalid",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should return all users (filter ignored)
    assert data["total"] == 5



# ==================== PUT /api/admin/users/{userId} Tests ====================


def test_update_user_requires_admin(client, regular_token, sample_users):
    """Test that non-admin users cannot update users."""
    user_id = sample_users[0].id
    response = client.put(
        f"/api/admin/users/{user_id}",
        headers={"Authorization": f"Bearer {regular_token}"},
        json={"full_name": "Updated Name"}
    )
    assert response.status_code == 403
    assert "Admin privileges required" in response.json()["detail"]


def test_update_user_requires_authentication(client, sample_users):
    """Test that unauthenticated requests are rejected."""
    user_id = sample_users[0].id
    response = client.put(
        f"/api/admin/users/{user_id}",
        json={"full_name": "Updated Name"}
    )
    assert response.status_code == 401


def test_update_user_full_name(client, admin_token, sample_users, db_session):
    """Test updating user's full name (Requirement 5.5)."""
    user = sample_users[0]
    original_name = user.full_name
    
    response = client.put(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"full_name": "Updated Teacher Name"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["user"]["full_name"] == "Updated Teacher Name"
    assert data["user"]["id"] == user.id
    assert data["message"] == "User updated successfully"
    
    # Verify in database
    db_session.refresh(user)
    assert user.full_name == "Updated Teacher Name"


def test_update_user_email(client, admin_token, sample_users, db_session):
    """Test updating user's email (Requirement 5.5)."""
    user = sample_users[0]
    
    response = client.put(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"email": "newemail@test.com"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["user"]["email"] == "newemail@test.com"
    
    # Verify in database
    db_session.refresh(user)
    assert user.email == "newemail@test.com"


def test_update_user_role(client, admin_token, sample_users, db_session):
    """Test updating user's role (Requirement 5.5)."""
    user = sample_users[2]  # Student
    assert user.role == UserRole.STUDENT
    
    response = client.put(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"role": "teacher"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["user"]["role"] == "teacher"
    
    # Verify in database
    db_session.refresh(user)
    assert user.role == UserRole.TEACHER


def test_update_user_is_active(client, admin_token, sample_users, db_session):
    """Test updating user's active status (Requirement 5.5)."""
    user = sample_users[0]
    assert user.is_active is True
    
    response = client.put(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"is_active": False}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["user"]["is_active"] is False
    
    # Verify in database
    db_session.refresh(user)
    assert user.is_active is False


def test_update_user_multiple_fields(client, admin_token, sample_users, db_session):
    """Test updating multiple fields at once (Requirement 5.5)."""
    user = sample_users[0]
    
    response = client.put(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "full_name": "New Name",
            "email": "newemail@test.com",
            "role": "student",
            "is_active": False
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["user"]["full_name"] == "New Name"
    assert data["user"]["email"] == "newemail@test.com"
    assert data["user"]["role"] == "student"
    assert data["user"]["is_active"] is False
    
    # Verify in database
    db_session.refresh(user)
    assert user.full_name == "New Name"
    assert user.email == "newemail@test.com"
    assert user.role == UserRole.STUDENT
    assert user.is_active is False


def test_update_user_email_uniqueness(client, admin_token, sample_users, db_session):
    """Test that duplicate email is rejected (Requirement 5.4)."""
    user1 = sample_users[0]
    user2 = sample_users[1]
    
    # Try to update user1's email to user2's email
    response = client.put(
        f"/api/admin/users/{user1.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"email": user2.email}
    )
    
    assert response.status_code == 400
    assert "already in use" in response.json()["detail"]
    
    # Verify user1's email was not changed
    db_session.refresh(user1)
    assert user1.email != user2.email


def test_update_user_cannot_edit_admin(client, admin_token, admin_user):
    """Test that admin user (ID=1) cannot be edited (Requirement 5.7)."""
    response = client.put(
        f"/api/admin/users/{admin_user.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"full_name": "Hacked Admin"}
    )
    
    assert response.status_code == 400
    assert "Cannot edit the admin user" in response.json()["detail"]


def test_update_user_not_found(client, admin_token):
    """Test updating non-existent user returns 404."""
    response = client.put(
        "/api/admin/users/99999",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"full_name": "New Name"}
    )
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_update_user_audit_log(client, admin_token, sample_users, db_session, admin_user):
    """Test that admin action is logged (Requirement 8.5)."""
    from app.models.db_models import AuditLog
    
    user = sample_users[0]
    
    # Count audit logs before
    before_count = db_session.query(AuditLog).count()
    
    response = client.put(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"full_name": "Updated Name", "email": "updated@test.com"}
    )
    
    assert response.status_code == 200
    
    # Check audit log was created
    after_count = db_session.query(AuditLog).count()
    assert after_count == before_count + 1
    
    # Get the audit log entry
    audit_log = db_session.query(AuditLog).order_by(AuditLog.timestamp.desc()).first()
    
    assert audit_log.admin_user_id == admin_user.id
    assert audit_log.action == "edit"
    assert audit_log.target_user_id == user.id
    assert "changed_fields" in audit_log.details
    assert "full_name" in audit_log.details["changed_fields"]
    assert "email" in audit_log.details["changed_fields"]
    assert "new_values" in audit_log.details


def test_update_user_empty_update(client, admin_token, sample_users, db_session):
    """Test that empty update (no fields) still succeeds."""
    user = sample_users[0]
    original_name = user.full_name
    
    response = client.put(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["user"]["full_name"] == original_name
    
    # Verify nothing changed in database
    db_session.refresh(user)
    assert user.full_name == original_name


def test_update_user_invalid_email_format(client, admin_token, sample_users):
    """Test that invalid email format is rejected."""
    user = sample_users[0]
    
    response = client.put(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"email": "not-an-email"}
    )
    
    assert response.status_code == 422  # Validation error


def test_update_user_invalid_role(client, admin_token, sample_users):
    """Test that invalid role is rejected."""
    user = sample_users[0]
    
    response = client.put(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"role": "invalid_role"}
    )
    
    assert response.status_code == 422  # Validation error



# ==================== DELETE User Tests ====================


def test_delete_user_requires_admin(client, regular_token, sample_users):
    """Test that delete user endpoint requires admin privileges."""
    user_id = sample_users[0].id
    response = client.delete(
        f"/api/admin/users/{user_id}",
        headers={"Authorization": f"Bearer {regular_token}"}
    )
    assert response.status_code == 403
    assert "Admin privileges required" in response.json()["detail"]


def test_delete_user_requires_authentication(client, sample_users):
    """Test that delete user endpoint requires authentication."""
    user_id = sample_users[0].id
    response = client.delete(f"/api/admin/users/{user_id}")
    assert response.status_code == 401


def test_delete_user_basic(client, admin_token, sample_users, db_session):
    """Test basic user deletion."""
    # Get a user to delete
    user_to_delete = sample_users[0]
    user_id = user_to_delete.id
    
    response = client.delete(
        f"/api/admin/users/{user_id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "deleted successfully" in data["message"]
    assert data["deleted_data"]["user_id"] == user_id
    
    # Verify user is deleted
    deleted_user = db_session.query(User).filter(User.id == user_id).first()
    assert deleted_user is None


def test_delete_user_cannot_delete_admin(client, admin_token, admin_user):
    """Test that admin user (ID=1) cannot be deleted."""
    response = client.delete(
        "/api/admin/users/1",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 400
    assert "Cannot delete the admin user" in response.json()["detail"]


def test_delete_user_not_found(client, admin_token):
    """Test deleting non-existent user returns 404."""
    response = client.delete(
        "/api/admin/users/99999",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_delete_user_cascade_courses(client, admin_token, db_session, hashed_password):
    """Test that deleting a teacher cascades to their courses."""
    from app.models.db_models import Course, Document
    
    # Create a teacher with courses
    teacher = User(
        email="teacher_with_courses@test.com",
        hashed_password=hashed_password,
        full_name="Teacher With Courses",
        role=UserRole.TEACHER,
        is_active=True,
    )
    db_session.add(teacher)
    db_session.commit()
    db_session.refresh(teacher)
    
    # Create courses for the teacher
    course1 = Course(
        name="Course 1",
        description="Test course 1",
        teacher_id=teacher.id,
    )
    course2 = Course(
        name="Course 2",
        description="Test course 2",
        teacher_id=teacher.id,
    )
    db_session.add(course1)
    db_session.add(course2)
    db_session.commit()
    
    # Delete the teacher
    response = client.delete(
        f"/api/admin/users/{teacher.id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_data"]["courses_deleted"] == 2
    
    # Verify courses are deleted
    courses = db_session.query(Course).filter(
        Course.teacher_id == teacher.id
    ).all()
    assert len(courses) == 0


def test_delete_user_cascade_refresh_tokens(client, admin_token, db_session, hashed_password):
    """Test that deleting a user cascades to their refresh tokens."""
    from app.models.db_models import RefreshToken
    
    # Create a user with refresh tokens
    user = User(
        email="user_with_tokens@test.com",
        hashed_password=hashed_password,
        full_name="User With Tokens",
        role=UserRole.STUDENT,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    # Create refresh tokens
    token1 = RefreshToken(
        token="token1",
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    token2 = RefreshToken(
        token="token2",
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db_session.add(token1)
    db_session.add(token2)
    db_session.commit()
    
    # Delete the user
    response = client.delete(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_data"]["refresh_tokens_deleted"] == 2
    
    # Verify tokens are deleted
    tokens = db_session.query(RefreshToken).filter(
        RefreshToken.user_id == user.id
    ).all()
    assert len(tokens) == 0


def test_delete_user_audit_log(client, admin_token, db_session, hashed_password):
    """Test that user deletion is logged in audit log."""
    from app.models.db_models import AuditLog
    
    # Create a user to delete
    user = User(
        email="user_to_audit@test.com",
        hashed_password=hashed_password,
        full_name="User To Audit",
        role=UserRole.STUDENT,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    user_id = user.id
    
    # Delete the user
    response = client.delete(
        f"/api/admin/users/{user_id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    
    # Verify audit log entry
    audit_log = db_session.query(AuditLog).filter(
        AuditLog.action == "delete",
        AuditLog.target_user_id == user_id,
    ).first()
    
    assert audit_log is not None
    assert audit_log.admin_user_id == 1
    assert audit_log.details["deleted_user_email"] == "user_to_audit@test.com"
    assert audit_log.details["deleted_user_role"] == "student"


def test_delete_user_cascade_audit_logs_as_target(client, admin_token, db_session, hashed_password):
    """Test that audit logs where user is target are deleted."""
    from app.models.db_models import AuditLog
    
    # Create a user
    user = User(
        email="user_with_audit@test.com",
        hashed_password=hashed_password,
        full_name="User With Audit",
        role=UserRole.STUDENT,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    # Create audit logs where user is target
    log1 = AuditLog(
        admin_user_id=1,
        action="edit",
        target_user_id=user.id,
        details={"test": "data"},
    )
    log2 = AuditLog(
        admin_user_id=1,
        action="deactivate",
        target_user_id=user.id,
        details={"test": "data"},
    )
    db_session.add(log1)
    db_session.add(log2)
    db_session.commit()
    
    # Delete the user
    response = client.delete(
        f"/api/admin/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_data"]["audit_logs_deleted"] == 2
    
    # Verify old audit logs are deleted, but deletion log remains
    logs = db_session.query(AuditLog).filter(
        AuditLog.target_user_id == user.id
    ).all()
    # Should have 1 log remaining - the deletion log itself
    assert len(logs) == 1
    assert logs[0].action == "delete"



# ==================== PATCH /api/admin/users/{userId}/deactivate Tests ====================


def test_deactivate_user_requires_admin(client, regular_token, sample_users):
    """Test that deactivate user endpoint requires admin privileges."""
    user_id = sample_users[0].id
    response = client.patch(
        f"/api/admin/users/{user_id}/deactivate",
        headers={"Authorization": f"Bearer {regular_token}"}
    )
    assert response.status_code == 403
    assert "Admin privileges required" in response.json()["detail"]


def test_deactivate_user_requires_authentication(client, sample_users):
    """Test that deactivate user endpoint requires authentication."""
    user_id = sample_users[0].id
    response = client.patch(f"/api/admin/users/{user_id}/deactivate")
    assert response.status_code == 401


def test_deactivate_user_basic(client, admin_token, sample_users, db_session):
    """Test basic user deactivation (Requirement 6.1)."""
    user = sample_users[0]
    assert user.is_active is True
    
    response = client.patch(
        f"/api/admin/users/{user.id}/deactivate",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "deactivated successfully" in data["message"]
    assert data["user"]["id"] == user.id
    assert data["user"]["is_active"] is False
    
    # Verify in database
    db_session.refresh(user)
    assert user.is_active is False


def test_deactivate_user_cannot_deactivate_admin(client, admin_token, admin_user):
    """Test that admin user (ID=1) cannot be deactivated (Requirement 6.1)."""
    response = client.patch(
        "/api/admin/users/1/deactivate",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 400
    assert "Cannot deactivate the admin user" in response.json()["detail"]


def test_deactivate_user_not_found(client, admin_token):
    """Test deactivating non-existent user returns 404."""
    response = client.patch(
        "/api/admin/users/99999/deactivate",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_deactivate_user_already_inactive(client, admin_token, sample_users, db_session):
    """Test deactivating already inactive user returns error."""
    user = sample_users[1]  # This user is already inactive
    assert user.is_active is False
    
    response = client.patch(
        f"/api/admin/users/{user.id}/deactivate",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 400
    assert "already inactive" in response.json()["detail"]


def test_deactivate_user_audit_log(client, admin_token, sample_users, db_session, admin_user):
    """Test that deactivation is logged in audit log (Requirement 6.6)."""
    from app.models.db_models import AuditLog
    
    user = sample_users[0]
    
    # Count audit logs before
    before_count = db_session.query(AuditLog).count()
    
    response = client.patch(
        f"/api/admin/users/{user.id}/deactivate",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    
    # Check audit log was created
    after_count = db_session.query(AuditLog).count()
    assert after_count == before_count + 1
    
    # Get the audit log entry
    audit_log = db_session.query(AuditLog).order_by(AuditLog.timestamp.desc()).first()
    
    assert audit_log.admin_user_id == admin_user.id
    assert audit_log.action == "deactivate"
    assert audit_log.target_user_id == user.id
    assert audit_log.details["user_email"] == user.email
    assert audit_log.details["user_role"] == user.role.value


# ==================== PATCH /api/admin/users/{userId}/activate Tests ====================


def test_activate_user_requires_admin(client, regular_token, sample_users):
    """Test that activate user endpoint requires admin privileges."""
    user_id = sample_users[1].id  # Inactive user
    response = client.patch(
        f"/api/admin/users/{user_id}/activate",
        headers={"Authorization": f"Bearer {regular_token}"}
    )
    assert response.status_code == 403
    assert "Admin privileges required" in response.json()["detail"]


def test_activate_user_requires_authentication(client, sample_users):
    """Test that activate user endpoint requires authentication."""
    user_id = sample_users[1].id  # Inactive user
    response = client.patch(f"/api/admin/users/{user_id}/activate")
    assert response.status_code == 401


def test_activate_user_basic(client, admin_token, sample_users, db_session):
    """Test basic user activation (Requirement 6.4)."""
    user = sample_users[1]  # This user is inactive
    assert user.is_active is False
    
    response = client.patch(
        f"/api/admin/users/{user.id}/activate",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "activated successfully" in data["message"]
    assert data["user"]["id"] == user.id
    assert data["user"]["is_active"] is True
    
    # Verify in database
    db_session.refresh(user)
    assert user.is_active is True


def test_activate_user_not_found(client, admin_token):
    """Test activating non-existent user returns 404."""
    response = client.patch(
        "/api/admin/users/99999/activate",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_activate_user_already_active(client, admin_token, sample_users, db_session):
    """Test activating already active user returns error."""
    user = sample_users[0]  # This user is already active
    assert user.is_active is True
    
    response = client.patch(
        f"/api/admin/users/{user.id}/activate",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 400
    assert "already active" in response.json()["detail"]


def test_activate_user_audit_log(client, admin_token, sample_users, db_session, admin_user):
    """Test that activation is logged in audit log (Requirement 6.4)."""
    from app.models.db_models import AuditLog
    
    user = sample_users[1]  # Inactive user
    
    # Count audit logs before
    before_count = db_session.query(AuditLog).count()
    
    response = client.patch(
        f"/api/admin/users/{user.id}/activate",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    
    # Check audit log was created
    after_count = db_session.query(AuditLog).count()
    assert after_count == before_count + 1
    
    # Get the audit log entry
    audit_log = db_session.query(AuditLog).order_by(AuditLog.timestamp.desc()).first()
    
    assert audit_log.admin_user_id == admin_user.id
    assert audit_log.action == "activate"
    assert audit_log.target_user_id == user.id
    assert audit_log.details["user_email"] == user.email
    assert audit_log.details["user_role"] == user.role.value


# ==================== Login with Inactive User Tests ====================


def test_login_inactive_user_rejected(client, db_session, hashed_password):
    """Test that inactive users cannot log in (Requirement 6.2, 6.3)."""
    # Create an inactive user
    inactive_user = User(
        email="inactive@test.com",
        hashed_password=hashed_password,
        full_name="Inactive User",
        role=UserRole.TEACHER,
        is_active=False,
    )
    db_session.add(inactive_user)
    db_session.commit()
    
    # Try to login
    response = client.post(
        "/api/auth/login",
        data={
            "username": "inactive@test.com",
            "password": "testpassword123"
        }
    )
    
    assert response.status_code == 403
    assert response.json()["detail"] == "Hesabınız devre dışı bırakılmış"


def test_login_active_user_succeeds(client, db_session, hashed_password):
    """Test that active users can log in normally."""
    # Create an active user
    active_user = User(
        email="active@test.com",
        hashed_password=hashed_password,
        full_name="Active User",
        role=UserRole.TEACHER,
        is_active=True,
    )
    db_session.add(active_user)
    db_session.commit()
    
    # Try to login
    response = client.post(
        "/api/auth/login",
        data={
            "username": "active@test.com",
            "password": "testpassword123"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


def test_deactivate_then_login_rejected(client, admin_token, sample_users, db_session):
    """Test full workflow: deactivate user, then verify they cannot log in."""
    user = sample_users[0]
    assert user.is_active is True
    
    # Deactivate the user
    response = client.patch(
        f"/api/admin/users/{user.id}/deactivate",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    
    # Try to login
    response = client.post(
        "/api/auth/login",
        data={
            "username": user.email,
            "password": "testpassword123"
        }
    )
    
    assert response.status_code == 403
    assert response.json()["detail"] == "Hesabınız devre dışı bırakılmış"


def test_activate_then_login_succeeds(client, admin_token, sample_users, db_session):
    """Test full workflow: activate user, then verify they can log in."""
    user = sample_users[1]  # Inactive user
    assert user.is_active is False
    
    # Activate the user
    response = client.patch(
        f"/api/admin/users/{user.id}/activate",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    
    # Try to login
    response = client.post(
        "/api/auth/login",
        data={
            "username": user.email,
            "password": "testpassword123"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data



# ==================== Statistics Endpoint Tests ====================


def test_get_statistics_requires_admin(client, regular_token):
    """Test that statistics endpoint requires admin privileges."""
    response = client.get(
        "/api/admin/statistics",
        headers={"Authorization": f"Bearer {regular_token}"}
    )
    assert response.status_code == 403


def test_get_statistics_requires_authentication(client):
    """Test that statistics endpoint requires authentication."""
    response = client.get("/api/admin/statistics")
    assert response.status_code == 401


def test_get_statistics_basic(client, admin_token, db_session, hashed_password):
    """Test basic statistics retrieval."""
    # Create some test users
    users = [
        User(
            email=f"teacher{i}@test.com",
            hashed_password=hashed_password,
            full_name=f"Teacher {i}",
            role=UserRole.TEACHER,
            is_active=True,
            created_at=datetime.utcnow() - timedelta(days=i),
        )
        for i in range(3)
    ]
    users.extend([
        User(
            email=f"student{i}@test.com",
            hashed_password=hashed_password,
            full_name=f"Student {i}",
            role=UserRole.STUDENT,
            is_active=True,
            created_at=datetime.utcnow() - timedelta(days=i),
        )
        for i in range(2)
    ])
    # Add an inactive user
    users.append(
        User(
            email="inactive@test.com",
            hashed_password=hashed_password,
            full_name="Inactive User",
            role=UserRole.TEACHER,
            is_active=False,
            created_at=datetime.utcnow() - timedelta(days=60),
        )
    )
    
    for user in users:
        db_session.add(user)
    db_session.commit()
    
    # Get statistics
    response = client.get(
        "/api/admin/statistics",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Admin user (ID=1) + 6 test users = 7 total
    assert data["total_users"] >= 7
    assert data["active_teachers"] >= 3  # 3 active teachers
    assert data["active_students"] >= 2  # 2 active students
    assert data["inactive_users"] >= 1  # 1 inactive user
    # New users this month should include all users created in current month
    assert data["new_users_this_month"] >= 6


# ==================== Password Reset Endpoint Tests ====================


def test_reset_password_requires_admin(client, regular_token, sample_users):
    """Test that password reset requires admin privileges."""
    user = sample_users[0]
    response = client.post(
        f"/api/admin/users/{user.id}/reset-password",
        headers={"Authorization": f"Bearer {regular_token}"}
    )
    assert response.status_code == 403


def test_reset_password_requires_authentication(client, sample_users):
    """Test that password reset requires authentication."""
    user = sample_users[0]
    response = client.post(f"/api/admin/users/{user.id}/reset-password")
    assert response.status_code == 401


def test_reset_password_basic(client, admin_token, sample_users, db_session):
    """Test basic password reset functionality."""
    user = sample_users[0]
    
    response = client.post(
        f"/api/admin/users/{user.id}/reset-password",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "temporary_password" in data
    assert len(data["temporary_password"]) == 12
    assert "expires_at" in data
    
    # Verify temporary password was created in database
    from app.models.db_models import TemporaryPassword
    temp_password = db_session.query(TemporaryPassword).filter(
        TemporaryPassword.user_id == user.id,
        TemporaryPassword.used == False  # noqa: E712
    ).first()
    
    assert temp_password is not None
    assert temp_password.expires_at > datetime.utcnow()


def test_reset_password_cannot_reset_admin(client, admin_token):
    """Test that admin user (ID=1) password cannot be reset."""
    response = client.post(
        "/api/admin/users/1/reset-password",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 400
    assert "Cannot reset password for the admin user" in response.json()["detail"]


def test_reset_password_user_not_found(client, admin_token):
    """Test password reset for non-existent user."""
    response = client.post(
        "/api/admin/users/99999/reset-password",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    
    assert response.status_code == 404


def test_reset_password_invalidates_old_temp_passwords(
    client, admin_token, sample_users, db_session
):
    """Test that resetting password invalidates old temporary passwords."""
    user = sample_users[0]
    
    # Create first temporary password
    response1 = client.post(
        f"/api/admin/users/{user.id}/reset-password",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response1.status_code == 200
    temp_pass1 = response1.json()["temporary_password"]
    
    # Create second temporary password
    response2 = client.post(
        f"/api/admin/users/{user.id}/reset-password",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response2.status_code == 200
    temp_pass2 = response2.json()["temporary_password"]
    
    # Verify only one unused temporary password exists
    from app.models.db_models import TemporaryPassword
    unused_temp_passwords = db_session.query(TemporaryPassword).filter(
        TemporaryPassword.user_id == user.id,
        TemporaryPassword.used == False  # noqa: E712
    ).all()
    
    assert len(unused_temp_passwords) == 1
    assert temp_pass1 != temp_pass2


def test_login_with_temporary_password(
    client, admin_token, sample_users, db_session
):
    """Test that user can login with temporary password."""
    user = sample_users[0]
    
    # Reset password
    response = client.post(
        f"/api/admin/users/{user.id}/reset-password",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    temp_password = response.json()["temporary_password"]
    
    # Login with temporary password
    response = client.post(
        "/api/auth/login",
        data={
            "username": user.email,
            "password": temp_password
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    
    # Verify temporary password was marked as used
    from app.models.db_models import TemporaryPassword
    temp_pass_record = db_session.query(TemporaryPassword).filter(
        TemporaryPassword.user_id == user.id
    ).first()
    assert temp_pass_record.used is True


def test_login_with_expired_temporary_password(
    client, admin_token, sample_users, db_session, hashed_password
):
    """Test that expired temporary password cannot be used."""
    user = sample_users[0]
    
    # Create temporary password
    temp_password = "TempPass123!"
    hashed_temp = hashed_password  # Use fixture instead of hashing
    
    # Create expired temporary password
    from app.models.db_models import TemporaryPassword
    expired_temp = TemporaryPassword(
        user_id=user.id,
        hashed_password=hashed_temp,
        expires_at=datetime.utcnow() - timedelta(hours=1),  # Expired
        used=False,
    )
    db_session.add(expired_temp)
    db_session.commit()
    
    # Try to login with expired temporary password
    response = client.post(
        "/api/auth/login",
        data={
            "username": user.email,
            "password": temp_password
        }
    )
    
    assert response.status_code == 401


def test_login_with_used_temporary_password(
    client, admin_token, sample_users, db_session
):
    """Test that used temporary password cannot be reused."""
    user = sample_users[0]
    
    # Reset password
    response = client.post(
        f"/api/admin/users/{user.id}/reset-password",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    temp_password = response.json()["temporary_password"]
    
    # Login with temporary password (first time)
    response = client.post(
        "/api/auth/login",
        data={
            "username": user.email,
            "password": temp_password
        }
    )
    assert response.status_code == 200
    
    # Try to login again with same temporary password
    response = client.post(
        "/api/auth/login",
        data={
            "username": user.email,
            "password": temp_password
        }
    )
    
    # Should fail because temporary password was already used
    assert response.status_code == 401
