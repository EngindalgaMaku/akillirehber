"""
Pytest configuration and fixtures
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.main import app
from app.database import Base, get_db
from app.models.db_models import User, UserRole
from app.services.auth_service import create_access_token


# Create test database engine
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    """Create a test database session."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Clean up tables
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    """Create a test client for the FastAPI app"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def hashed_password():
    """Fixture for hashed password."""
    # Pre-hashed version of "testpassword123"
    return "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYqVr/qviu."


@pytest.fixture
def admin_user(db_session: Session, hashed_password):
    """Create admin user (ID=1)."""
    # Ensure no user with ID=1 exists
    existing = db_session.query(User).filter(User.id == 1).first()
    if existing:
        db_session.delete(existing)
        db_session.commit()
    
    user = User(
        id=1,
        email="admin@test.com",
        hashed_password=hashed_password,
        full_name="Admin User",
        role=UserRole.TEACHER,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user):
    """Create admin JWT token."""
    return create_access_token(
        data={
            "sub": str(admin_user.id),
            "email": admin_user.email,
            "role": admin_user.role.value,
            "is_admin": True,
        }
    )


@pytest.fixture
def sample_users(db_session: Session, hashed_password, admin_user):
    """Create sample users for testing."""
    users = [
        User(
            email="teacher1@test.com",
            hashed_password=hashed_password,
            full_name="Teacher One",
            role=UserRole.TEACHER,
            is_active=True,
            created_at=datetime.utcnow() - timedelta(days=10),
            last_login=datetime.utcnow() - timedelta(days=1),
        ),
        User(
            email="teacher2@test.com",
            hashed_password=hashed_password,
            full_name="Teacher Two",
            role=UserRole.TEACHER,
            is_active=False,
            created_at=datetime.utcnow() - timedelta(days=20),
        ),
        User(
            email="student1@test.com",
            hashed_password=hashed_password,
            full_name="Student One",
            role=UserRole.STUDENT,
            is_active=True,
            created_at=datetime.utcnow() - timedelta(days=5),
            last_login=datetime.utcnow(),
        ),
        User(
            email="student2@test.com",
            hashed_password=hashed_password,
            full_name="Student Two",
            role=UserRole.STUDENT,
            is_active=True,
            created_at=datetime.utcnow() - timedelta(days=15),
        ),
    ]
    
    for user in users:
        db_session.add(user)
    db_session.commit()
    
    for user in users:
        db_session.refresh(user)
    
    return users
