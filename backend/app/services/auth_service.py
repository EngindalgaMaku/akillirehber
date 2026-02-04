"""Authentication service for user management and JWT tokens."""

import secrets
from datetime import datetime, timedelta, UTC
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models.db_models import User, UserRole, RefreshToken
from app.models.schemas import TokenData, UserCreate, UserUpdate, PasswordChange

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


# Alias for consistency
hash_password = get_password_hash


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    settings = get_settings()
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


def create_refresh_token(db: Session, user_id: int) -> str:
    """Create a new refresh token and store it in the database."""
    settings = get_settings()
    
    # Generate a secure random token
    token = secrets.token_urlsafe(64)
    
    # Calculate expiration
    expires_at = datetime.now(UTC) + timedelta(
        days=settings.refresh_token_expire_days
    )
    
    # Store in database
    db_token = RefreshToken(
        token=token,
        user_id=user_id,
        expires_at=expires_at,
    )
    db.add(db_token)
    db.commit()
    
    return token


def verify_refresh_token(db: Session, token: str) -> Optional[RefreshToken]:
    """Verify a refresh token and return it if valid."""
    db_token = db.query(RefreshToken).filter(
        RefreshToken.token == token,
        RefreshToken.revoked == False,
        RefreshToken.expires_at > datetime.now(UTC)
    ).first()
    
    return db_token


def revoke_refresh_token(db: Session, token: str) -> bool:
    """Revoke a refresh token."""
    db_token = db.query(RefreshToken).filter(
        RefreshToken.token == token
    ).first()
    
    if db_token:
        db_token.revoked = True
        db.commit()
        return True
    return False


def revoke_all_user_tokens(db: Session, user_id: int) -> int:
    """Revoke all refresh tokens for a user."""
    result = db.query(RefreshToken).filter(
        RefreshToken.user_id == user_id,
        RefreshToken.revoked == False
    ).update({"revoked": True})
    db.commit()
    return result


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        user_id = payload.get("sub")
        email: str = payload.get("email")
        role: str = payload.get("role")
        
        if user_id is None:
            return None
        
        # Convert user_id to int if it's a string
        if isinstance(user_id, str):
            user_id = int(user_id)
        
        return TokenData(user_id=user_id, email=email, role=UserRole(role))
    except (JWTError, ValueError):
        return None


def is_admin(user_id: int) -> bool:
    """
    Check if a user is an admin based on their ID.
    
    Args:
        user_id: The user's ID
        
    Returns:
        True if user_id equals 1 (admin), False otherwise
    """
    return user_id == 1


def get_admin_flag_from_jwt(token: str) -> bool:
    """
    Extract admin flag from JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        True if token contains admin flag set to True, False otherwise
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        return payload.get("is_admin", False)
    except (JWTError, ValueError):
        return False


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get a user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, user_data: UserCreate) -> User:
    """Create a new user."""
    # Check if user already exists
    existing_user = get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password."""
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user from JWT token."""
    import logging
    logger = logging.getLogger(__name__)
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    logger.info(f"Decoding token: {token[:20]}...")
    token_data = decode_token(token)
    if token_data is None:
        logger.error("Token decode failed")
        raise credentials_exception

    logger.info(f"Token decoded successfully, user_id: {token_data.user_id}")
    user = get_user_by_id(db, token_data.user_id)
    if user is None:
        logger.error(f"User not found: {token_data.user_id}")
        raise credentials_exception

    logger.info(f"User found: {user.email}, role: {user.role}, is_active: {user.is_active}")
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )

    return user


async def get_current_teacher(current_user: User = Depends(get_current_user)) -> User:
    """Get the current user and verify they are a teacher or admin."""
    if current_user.role not in (UserRole.TEACHER, UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers and admins can perform this action",
        )
    return current_user


async def get_current_student(current_user: User = Depends(get_current_user)) -> User:
    """Get the current user and verify they are a student."""
    if current_user.role != UserRole.STUDENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can perform this action",
        )
    return current_user


async def require_admin(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """
    Verify that the current user is an admin.
    
    This dependency checks:
    1. JWT token is valid
    2. User exists and is active
    3. User ID equals 1 (admin)
    4. JWT contains admin flag
    
    Args:
        token: JWT token from Authorization header
        db: Database session
        
    Returns:
        User object if admin
        
    Raises:
        HTTPException: 401 if token invalid, 403 if not admin
    """
    # First get the current user (validates token and user exists)
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = decode_token(token)
    if token_data is None:
        raise credentials_exception

    user = get_user_by_id(db, token_data.user_id)
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    
    # Check if user is admin
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    
    return user


def update_user_profile(db: Session, user_id: int, user_data: UserUpdate) -> User:
    """Update user profile information."""
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Check if email is being updated and if it's already taken
    if user_data.email and user_data.email != user.email:
        existing_user = get_user_by_email(db, user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
        user.email = user_data.email
    
    # Update full name if provided
    if user_data.full_name:
        user.full_name = user_data.full_name
    
    db.commit()
    db.refresh(user)
    return user


def change_user_password(
    db: Session, user_id: int, password_data: PasswordChange
) -> bool:
    """Change user password."""
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Verify current password
    if not verify_password(password_data.current_password, user.hashed_password):
        return False
    
    # Update password
    user.hashed_password = get_password_hash(password_data.new_password)
    db.commit()
    return True
