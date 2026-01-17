"""Audit logging service for tracking admin operations."""

from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session

from app.models.db_models import AuditLog


def log_admin_action(
    db: Session,
    admin_user_id: int,
    action: str,
    target_user_id: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> AuditLog:
    """
    Log an admin action to the audit log.

    Args:
        db: Database session
        admin_user_id: ID of the admin user performing the action
        action: Type of action ('edit', 'delete', 'deactivate',
                'activate', 'reset_password')
        target_user_id: ID of the user being acted upon (optional)
        details: Additional details about the action (optional)
        ip_address: IP address of the admin user (optional)

    Returns:
        AuditLog: The created audit log entry

    Example:
        >>> log_admin_action(
        ...     db=db,
        ...     admin_user_id=1,
        ...     action='edit',
        ...     target_user_id=5,
        ...     details={'changed_fields': ['email', 'role']},
        ...     ip_address='192.168.1.1'
        ... )
    """
    audit_entry = AuditLog(
        admin_user_id=admin_user_id,
        action=action,
        target_user_id=target_user_id,
        details=details,
        timestamp=datetime.utcnow(),
        ip_address=ip_address,
    )

    db.add(audit_entry)
    db.commit()
    db.refresh(audit_entry)

    return audit_entry
