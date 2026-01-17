"""
Test temporary password login functionality.

Feature: admin-user-management, Task 7.5
Validates: Requirements 10.4
"""

from datetime import datetime, timedelta

from jose import jwt

from app.config import get_settings
from app.models.db_models import TemporaryPassword

settings = get_settings()


def test_login_with_temporary_password_includes_flag(
    client, sample_users, db_session, hashed_password
):
    """
    Test that logging in with a temporary password includes
    requires_password_change flag in JWT.

    Validates: Requirement 10.4 - Force password change on first login
    """
    user = sample_users[0]

    # Create a temporary password manually
    temp_password = "testpassword123"  # Use same password as fixture
    hashed_temp = hashed_password  # Use fixture instead of hashing

    temp_password_record = TemporaryPassword(
        user_id=user.id,
        hashed_password=hashed_temp,
        expires_at=datetime.utcnow() + timedelta(hours=24),
        used=False,
    )
    db_session.add(temp_password_record)
    db_session.commit()

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

    # Decode the JWT token and verify requires_password_change flag
    access_token = data["access_token"]
    decoded = jwt.decode(
        access_token,
        settings.secret_key,
        algorithms=[settings.algorithm]
    )

    # Verify the flag is present and set to True
    assert "requires_password_change" in decoded
    assert decoded["requires_password_change"] is True

    # Verify temporary password was marked as used
    db_session.refresh(temp_password_record)
    assert temp_password_record.used is True


def test_login_with_regular_password_no_flag(
    client, sample_users
):
    """
    Test that logging in with regular password does NOT include
    requires_password_change flag.

    Validates: Requirement 10.4 - Flag only present for temporary passwords
    """
    user = sample_users[0]

    # Login with regular password (testpassword123 from fixture)
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

    # Decode the JWT token
    access_token = data["access_token"]
    decoded = jwt.decode(
        access_token,
        settings.secret_key,
        algorithms=[settings.algorithm]
    )

    # Verify the flag is NOT present for regular login
    assert "requires_password_change" not in decoded


def test_temporary_password_marked_as_used(
    client, sample_users, db_session, hashed_password
):
    """
    Test that temporary password is marked as used after first login.

    Validates: Requirement 10.4 - Mark temporary password as used
    """
    user = sample_users[0]

    # Create a temporary password
    temp_password = "testpassword123"  # Use same password as fixture
    hashed_temp = hashed_password  # Use fixture instead of hashing

    temp_password_record = TemporaryPassword(
        user_id=user.id,
        hashed_password=hashed_temp,
        expires_at=datetime.utcnow() + timedelta(hours=24),
        used=False,
    )
    db_session.add(temp_password_record)
    db_session.commit()

    # Verify it's not used initially
    assert temp_password_record.used is False

    # Login with temporary password
    response = client.post(
        "/api/auth/login",
        data={
            "username": user.email,
            "password": temp_password
        }
    )

    assert response.status_code == 200

    # Verify temporary password was marked as used
    db_session.refresh(temp_password_record)
    assert temp_password_record.used is True

    # Try to login again with the same temporary password
    response = client.post(
        "/api/auth/login",
        data={
            "username": user.email,
            "password": temp_password
        }
    )

    # Should fail because temporary password is already used
    assert response.status_code == 401


def test_expired_temporary_password_rejected(
    client, sample_users, db_session, hashed_password
):
    """
    Test that expired temporary password cannot be used.

    Validates: Requirement 10.4 - Check if password is temporary and valid
    """
    user = sample_users[0]

    # Create an expired temporary password
    temp_password = "testpassword123"  # Use same password as fixture
    hashed_temp = hashed_password  # Use fixture instead of hashing

    temp_password_record = TemporaryPassword(
        user_id=user.id,
        hashed_password=hashed_temp,
        expires_at=datetime.utcnow() - timedelta(hours=1),  # Expired
        used=False,
    )
    db_session.add(temp_password_record)
    db_session.commit()

    # Try to login with expired temporary password
    response = client.post(
        "/api/auth/login",
        data={
            "username": user.email,
            "password": temp_password
        }
    )

    # Should fail because temporary password is expired
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


def test_temporary_password_with_inactive_user(
    client, sample_users, db_session, hashed_password
):
    """
    Test that temporary password login fails for inactive users.

    Validates: Requirement 10.4 - Check user is active before login
    """
    user = sample_users[0]

    # Deactivate the user
    user.is_active = False
    db_session.commit()

    # Create a temporary password
    temp_password = "testpassword123"  # Use same password as fixture
    hashed_temp = hashed_password  # Use fixture instead of hashing

    temp_password_record = TemporaryPassword(
        user_id=user.id,
        hashed_password=hashed_temp,
        expires_at=datetime.utcnow() + timedelta(hours=24),
        used=False,
    )
    db_session.add(temp_password_record)
    db_session.commit()

    # Try to login with temporary password
    response = client.post(
        "/api/auth/login",
        data={
            "username": user.email,
            "password": temp_password
        }
    )

    # Should fail because user is inactive
    assert response.status_code == 403
    detail = response.json()["detail"]
    # Check for Turkish message about deactivated account
    assert "devre" in detail.lower() or "inactive" in detail.lower()
