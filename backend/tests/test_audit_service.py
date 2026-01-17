"""Tests for audit logging service."""

import pytest
from datetime import datetime
from sqlalchemy.orm import Session

from app.services.audit_service import log_admin_action
from app.models.db_models import AuditLog, User, UserRole


@pytest.fixture
def hashed_password():
    """Provide a pre-hashed password for testing."""
    # Pre-hashed version of "password123"
    return "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYqVr/qviu."


@pytest.fixture
def test_admin_user(db_session: Session, hashed_password: str):
    """Create a test admin user with ID=1."""
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
    db_session.refresh(admin_user)
    return admin_user


def test_log_admin_action_basic(db_session: Session, test_admin_user: User):
    """Test basic audit log creation."""
    # Create a target user
    target_user = User(
        email="target@example.com",
        hashed_password="hashed",
        full_name="Target User",
        role=UserRole.STUDENT,
    )
    db_session.add(target_user)
    db_session.commit()
    db_session.refresh(target_user)

    # Log an admin action
    audit_entry = log_admin_action(
        db=db_session,
        admin_user_id=test_admin_user.id,
        action="edit",
        target_user_id=target_user.id,
        details={"changed_fields": ["email", "role"]},
        ip_address="192.168.1.1",
    )

    # Verify the audit entry was created
    assert audit_entry.id is not None
    assert audit_entry.admin_user_id == test_admin_user.id
    assert audit_entry.action == "edit"
    assert audit_entry.target_user_id == target_user.id
    assert audit_entry.details == {"changed_fields": ["email", "role"]}
    assert audit_entry.ip_address == "192.168.1.1"
    assert isinstance(audit_entry.timestamp, datetime)

    # Verify it's in the database
    db_entry = (
        db_session.query(AuditLog).filter(AuditLog.id == audit_entry.id).first()
    )
    assert db_entry is not None
    assert db_entry.admin_user_id == test_admin_user.id


def test_log_admin_action_without_optional_fields(
    db_session: Session, test_admin_user: User
):
    """Test audit log creation without optional fields."""
    # Log an admin action without target_user_id, details, or ip_address
    audit_entry = log_admin_action(
        db=db_session,
        admin_user_id=test_admin_user.id,
        action="delete",
    )

    # Verify the audit entry was created
    assert audit_entry.id is not None
    assert audit_entry.admin_user_id == test_admin_user.id
    assert audit_entry.action == "delete"
    assert audit_entry.target_user_id is None
    assert audit_entry.details is None
    assert audit_entry.ip_address is None
    assert isinstance(audit_entry.timestamp, datetime)


def test_log_admin_action_different_actions(
    db_session: Session, test_admin_user: User
):
    """Test logging different types of admin actions."""
    actions = ["edit", "delete", "deactivate", "activate", "reset_password"]

    for action in actions:
        audit_entry = log_admin_action(
            db=db_session,
            admin_user_id=test_admin_user.id,
            action=action,
        )

        assert audit_entry.action == action
        assert audit_entry.admin_user_id == test_admin_user.id


def test_log_admin_action_with_complex_details(
    db_session: Session, test_admin_user: User
):
    """Test audit log with complex details object."""
    complex_details = {
        "changed_fields": ["email", "role", "full_name"],
        "old_values": {
            "email": "old@example.com",
            "role": "student",
            "full_name": "Old Name",
        },
        "new_values": {
            "email": "new@example.com",
            "role": "teacher",
            "full_name": "New Name",
        },
        "reason": "User requested role change",
    }

    audit_entry = log_admin_action(
        db=db_session,
        admin_user_id=test_admin_user.id,
        action="edit",
        details=complex_details,
    )

    assert audit_entry.details == complex_details
    assert audit_entry.details["changed_fields"] == [
        "email",
        "role",
        "full_name",
    ]
    assert audit_entry.details["reason"] == "User requested role change"
