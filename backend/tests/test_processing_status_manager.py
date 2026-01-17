"""Tests for ProcessingStatusManager service."""

import pytest
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.db_models import Document, ProcessingStatus, ProcessingStatusEnum, Course, User, UserRole
from app.services.processing_status_manager import ProcessingStatusManager


class TestProcessingStatusManager:
    """Test cases for ProcessingStatusManager."""

    def setup_method(self):
        """Set up test fixtures."""
        # This will be set up by the test database session
        pass

    def test_create_status(self, db_session: Session):
        """Test creating initial processing status."""
        # Create test user and course
        user = User(
            email="test@example.com",
            hashed_password="hashed",
            full_name="Test User",
            role=UserRole.TEACHER
        )
        db_session.add(user)
        db_session.commit()
        
        course = Course(
            name="Test Course",
            description="Test Description",
            teacher_id=user.id
        )
        db_session.add(course)
        db_session.commit()
        
        # Create test document
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_type="pdf",
            file_size=1000,
            char_count=500,
            content="Test content",
            course_id=course.id
        )
        db_session.add(document)
        db_session.commit()
        
        # Test creating status
        manager = ProcessingStatusManager(db_session)
        status = manager.create_status(document.id)
        
        assert status.document_id == document.id
        assert status.status == ProcessingStatusEnum.PENDING.value
        assert status.started_at is not None
        assert status.retry_count == 0

    def test_update_status(self, db_session: Session):
        """Test updating processing status."""
        # Create test data
        user = User(
            email="test@example.com",
            hashed_password="hashed",
            full_name="Test User",
            role=UserRole.TEACHER
        )
        db_session.add(user)
        db_session.commit()
        
        course = Course(
            name="Test Course",
            description="Test Description",
            teacher_id=user.id
        )
        db_session.add(course)
        db_session.commit()
        
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_type="pdf",
            file_size=1000,
            char_count=500,
            content="Test content",
            course_id=course.id
        )
        db_session.add(document)
        db_session.commit()
        
        # Create initial status
        manager = ProcessingStatusManager(db_session)
        manager.create_status(document.id)
        
        # Test updating status
        updated_status = manager.update_status(document.id, ProcessingStatusEnum.EXTRACTING)
        
        assert updated_status.status == ProcessingStatusEnum.EXTRACTING.value
        assert updated_status.updated_at is not None

    def test_status_transition_validation(self, db_session: Session):
        """Test that invalid status transitions are rejected."""
        # Create test data
        user = User(
            email="test@example.com",
            hashed_password="hashed",
            full_name="Test User",
            role=UserRole.TEACHER
        )
        db_session.add(user)
        db_session.commit()
        
        course = Course(
            name="Test Course",
            description="Test Description",
            teacher_id=user.id
        )
        db_session.add(course)
        db_session.commit()
        
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_type="pdf",
            file_size=1000,
            char_count=500,
            content="Test content",
            course_id=course.id
        )
        db_session.add(document)
        db_session.commit()
        
        # Create initial status
        manager = ProcessingStatusManager(db_session)
        manager.create_status(document.id)
        
        # Test invalid transition (PENDING -> COMPLETED should fail)
        with pytest.raises(ValueError, match="Invalid status transition"):
            manager.update_status(document.id, ProcessingStatusEnum.COMPLETED)

    def test_mark_for_retry(self, db_session: Session):
        """Test marking document for retry."""
        # Create test data
        user = User(
            email="test@example.com",
            hashed_password="hashed",
            full_name="Test User",
            role=UserRole.TEACHER
        )
        db_session.add(user)
        db_session.commit()
        
        course = Course(
            name="Test Course",
            description="Test Description",
            teacher_id=user.id
        )
        db_session.add(course)
        db_session.commit()
        
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_type="pdf",
            file_size=1000,
            char_count=500,
            content="Test content",
            course_id=course.id
        )
        db_session.add(document)
        db_session.commit()
        
        # Create status and set to error
        manager = ProcessingStatusManager(db_session)
        manager.create_status(document.id)
        manager.update_status(document.id, ProcessingStatusEnum.EXTRACTING)
        manager.update_status(document.id, ProcessingStatusEnum.ERROR, "Test error")
        
        # Test marking for retry
        retry_status = manager.mark_for_retry(document.id)
        
        assert retry_status.status == ProcessingStatusEnum.RETRYING.value
        assert retry_status.retry_count == 1
        assert retry_status.last_retry_at is not None

    def test_get_status(self, db_session: Session):
        """Test getting processing status."""
        # Create test data
        user = User(
            email="test@example.com",
            hashed_password="hashed",
            full_name="Test User",
            role=UserRole.TEACHER
        )
        db_session.add(user)
        db_session.commit()
        
        course = Course(
            name="Test Course",
            description="Test Description",
            teacher_id=user.id
        )
        db_session.add(course)
        db_session.commit()
        
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_type="pdf",
            file_size=1000,
            char_count=500,
            content="Test content",
            course_id=course.id
        )
        db_session.add(document)
        db_session.commit()
        
        # Create status
        manager = ProcessingStatusManager(db_session)
        manager.create_status(document.id)
        
        # Test getting status
        status_response = manager.get_status(document.id)
        
        assert status_response is not None
        assert status_response.document_id == document.id
        assert status_response.status == ProcessingStatusEnum.PENDING

    def test_get_processing_metrics(self, db_session: Session):
        """Test getting processing metrics."""
        # Create test data
        user = User(
            email="test@example.com",
            hashed_password="hashed",
            full_name="Test User",
            role=UserRole.TEACHER
        )
        db_session.add(user)
        db_session.commit()
        
        course = Course(
            name="Test Course",
            description="Test Description",
            teacher_id=user.id
        )
        db_session.add(course)
        db_session.commit()
        
        document = Document(
            filename="test.pdf",
            original_filename="test.pdf",
            file_type="pdf",
            file_size=1000,
            char_count=500,
            content="Test content",
            course_id=course.id
        )
        db_session.add(document)
        db_session.commit()
        
        # Create status
        manager = ProcessingStatusManager(db_session)
        manager.create_status(document.id)
        
        # Test getting metrics
        metrics = manager.get_processing_metrics()
        
        assert "status_distribution" in metrics
        assert "average_processing_time" in metrics
        assert "retry_statistics" in metrics
        assert metrics["status_distribution"]["pending"] == 1