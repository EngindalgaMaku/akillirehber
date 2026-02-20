"""Authentication API endpoints."""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models.schemas import (
    Token,
    UserCreate,
    UserResponse,
    UserUpdate,
    PasswordChange,
    RefreshTokenRequest,
)
from app.services.auth_service import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    create_user,
    get_current_user,
    update_user_profile,
    change_user_password,
    verify_refresh_token,
    revoke_refresh_token,
    get_user_by_id,
)
from app.models.db_models import User

settings = get_settings()

router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.

    - **email**: User's email address (must be unique)
    - **password**: User's password (min 6 characters)
    - **full_name**: User's full name
    - **role**: User role (teacher or student)
    """
    user = create_user(db, user_data)
    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login and get access token + refresh token.

    - **username**: User's email address
    - **password**: User's password

    Returns JWT access token (15 min) and refresh token (30 days).
    
    If user logs in with a temporary password:
    - Temporary password is marked as used
    - Access token includes 'requires_password_change' flag
    """
    from datetime import datetime as dt
    from app.models.db_models import TemporaryPassword
    
    user = authenticate_user(db, form_data.username, form_data.password)
    
    # If regular authentication fails, check for temporary password
    temp_password_used = False
    if not user:
        # Try to find user by email
        user = db.query(User).filter(User.email == form_data.username).first()
        
        if user:
            # Check if there's a valid temporary password
            now = dt.utcnow()
            temp_password = db.query(TemporaryPassword).filter(
                TemporaryPassword.user_id == user.id,
                TemporaryPassword.used == False,  # noqa: E712
                TemporaryPassword.expires_at > now
            ).order_by(TemporaryPassword.created_at.desc()).first()
            
            if temp_password:
                # Verify the password against temporary password
                from app.services.auth_service import verify_password
                if verify_password(form_data.password, temp_password.hashed_password):
                    # Mark temporary password as used
                    temp_password.used = True
                    db.commit()
                    temp_password_used = True
                else:
                    user = None  # Wrong password
            else:
                user = None  # No valid temp password
        else:
            user = None  # User not found
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hesabınız devre dışı bırakılmış",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create short-lived access token
    access_token_expires = timedelta(
        minutes=settings.access_token_expire_minutes
    )
    
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role.value,
        "is_admin": user.is_admin,
    }
    
    # Add flag if temporary password was used
    if temp_password_used:
        token_data["requires_password_change"] = True
    
    access_token = create_access_token(
        data=token_data,
        expires_delta=access_token_expires,
    )
    
    # Create long-lived refresh token
    refresh_token = create_refresh_token(db, user.id)
    
    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """
    Get a new access token using a refresh token.

    - **refresh_token**: Valid refresh token from login

    Returns new JWT access token and the same refresh token.
    """
    # Verify refresh token
    db_token = verify_refresh_token(db, request.refresh_token)
    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user
    user = get_user_by_id(db, db_token.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create new access token
    access_token_expires = timedelta(
        minutes=settings.access_token_expire_minutes
    )
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value,
            "is_admin": user.is_admin,
        },
        expires_delta=access_token_expires,
    )
    
    return Token(
        access_token=access_token,
        refresh_token=request.refresh_token
    )


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """
    Logout and revoke refresh token.

    - **refresh_token**: Refresh token to revoke

    This invalidates the refresh token so it can't be used again.
    """
    revoked = revoke_refresh_token(db, request.refresh_token)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid refresh token",
        )
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user information.

    Requires valid JWT token in Authorization header.
    """
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_profile(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update current user's profile information.

    - **full_name**: New full name (optional)
    - **email**: New email address (optional, must be unique)

    Requires valid JWT token in Authorization header.
    """
    updated_user = update_user_profile(db, current_user.id, user_data)
    return updated_user


@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Change current user's password.

    - **current_password**: Current password for verification
    - **new_password**: New password (min 6 characters)

    Requires valid JWT token in Authorization header.
    """
    success = change_user_password(db, current_user.id, password_data)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    return {"message": "Password changed successfully"}
