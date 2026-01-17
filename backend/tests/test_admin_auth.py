"""Tests for admin authentication middleware."""

import pytest
from fastapi import HTTPException
from app.services.auth_service import (
    is_admin,
    get_admin_flag_from_jwt,
    create_access_token,
    require_admin,
)
from app.models.db_models import User, UserRole


@pytest.fixture
def hashed_password():
    """Provide a pre-hashed password for testing."""
    # Pre-hashed version of "password123"
    # Using bcrypt hash to avoid calling get_password_hash during test
    return "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYqVr/qviu."


def test_is_admin_with_admin_user():
    """Test that user with ID=1 is recognized as admin."""
    assert is_admin(1) is True


def test_is_admin_with_non_admin_user():
    """Test that users with ID != 1 are not recognized as admin."""
    assert is_admin(2) is False
    assert is_admin(100) is False
    assert is_admin(999) is False


def test_get_admin_flag_from_jwt_with_admin_token():
    """Test extracting admin flag from JWT token for admin user."""
    # Create token with admin flag
    token = create_access_token(
        data={
            "sub": "1",
            "email": "admin@example.com",
            "role": "admin",
            "is_admin": True,
        }
    )

    assert get_admin_flag_from_jwt(token) is True


def test_get_admin_flag_from_jwt_with_non_admin_token():
    """Test extracting admin flag from JWT token for non-admin user."""
    # Create token without admin flag
    token = create_access_token(
        data={
            "sub": "2",
            "email": "teacher@example.com",
            "role": "teacher",
            "is_admin": False,
        }
    )

    assert get_admin_flag_from_jwt(token) is False


def test_get_admin_flag_from_jwt_with_missing_flag():
    """Test extracting admin flag when flag is missing from token."""
    # Create token without admin flag at all
    token = create_access_token(
        data={
            "sub": "3",
            "email": "student@example.com",
            "role": "student",
        }
    )

    # Should default to False
    assert get_admin_flag_from_jwt(token) is False


def test_get_admin_flag_from_jwt_with_invalid_token():
    """Test extracting admin flag from invalid JWT token."""
    invalid_token = "invalid.jwt.token"

    # Should return False for invalid tokens
    assert get_admin_flag_from_jwt(invalid_token) is False


@pytest.mark.asyncio
async def test_require_admin_with_admin_user(db_session, hashed_password):
    """Test require_admin allows admin user (ID=1)."""
    # Create admin user with ID=1
    admin_user = User(
        id=1,
        email="admin@example.com",
        hashed_password=hashed_password,
        full_name="Admin User",
        role=UserRole.TEACHER,
        is_active=True,
    )
    db_session.add(admin_user)
    db_session.commit()

    # Create admin token
    token = create_access_token(
        data={
            "sub": "1",
            "email": "admin@example.com",
            "role": "teacher",
            "is_admin": True,
        }
    )

    # Should not raise exception
    user = await require_admin(token=token, db=db_session)
    assert user.id == 1
    assert user.is_admin is True


@pytest.mark.asyncio
async def test_require_admin_with_non_admin_user(db_session, hashed_password):
    """Test require_admin rejects non-admin user."""
    # Create non-admin user
    teacher_user = User(
        id=2,
        email="teacher@example.com",
        hashed_password=hashed_password,
        full_name="Teacher User",
        role=UserRole.TEACHER,
        is_active=True,
    )
    db_session.add(teacher_user)
    db_session.commit()

    # Create non-admin token
    token = create_access_token(
        data={
            "sub": "2",
            "email": "teacher@example.com",
            "role": "teacher",
            "is_admin": False,
        }
    )

    # Should raise 403 Forbidden
    with pytest.raises(HTTPException) as exc_info:
        await require_admin(token=token, db=db_session)

    assert exc_info.value.status_code == 403
    assert "Admin privileges required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_require_admin_with_invalid_token(db_session):
    """Test require_admin rejects invalid token."""
    invalid_token = "invalid.jwt.token"

    # Should raise 401 Unauthorized
    with pytest.raises(HTTPException) as exc_info:
        await require_admin(token=invalid_token, db=db_session)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_admin_with_inactive_user(db_session, hashed_password):
    """Test require_admin rejects inactive admin user."""
    # Create inactive admin user
    admin_user = User(
        id=1,
        email="admin@example.com",
        hashed_password=hashed_password,
        full_name="Admin User",
        role=UserRole.TEACHER,
        is_active=False,  # Inactive
    )
    db_session.add(admin_user)
    db_session.commit()

    # Create admin token
    token = create_access_token(
        data={
            "sub": "1",
            "email": "admin@example.com",
            "role": "teacher",
            "is_admin": True,
        }
    )

    # Should raise 400 Bad Request
    with pytest.raises(HTTPException) as exc_info:
        await require_admin(token=token, db=db_session)

    assert exc_info.value.status_code == 400
    assert "Inactive user" in exc_info.value.detail


@pytest.mark.asyncio
async def test_require_admin_without_admin_flag_in_jwt(db_session, hashed_password):
    """Test require_admin rejects token without admin flag."""
    # Create admin user with ID=1
    admin_user = User(
        id=1,
        email="admin@example.com",
        hashed_password=hashed_password,
        full_name="Admin User",
        role=UserRole.TEACHER,
        is_active=True,
    )
    db_session.add(admin_user)
    db_session.commit()

    # Create token WITHOUT admin flag (simulating old token)
    token = create_access_token(
        data={
            "sub": "1",
            "email": "admin@example.com",
            "role": "teacher",
            # Missing is_admin flag
        }
    )

    # Should raise 403 Forbidden
    with pytest.raises(HTTPException) as exc_info:
        await require_admin(token=token, db=db_session)

    assert exc_info.value.status_code == 403
    assert "Admin privileges required" in exc_info.value.detail
