"""
Integration tests for Alibaba embedding functionality.

These tests verify end-to-end workflows with Alibaba's text-embedding-v4 model.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import Session

from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.models.db_models import Course, CourseSettings, Document, User, UserRole


class TestAlibabaEndToEndEmbedding:
    """Test end-to-end Alibaba embedding workflow.
    
    Feature: alibaba-embedding-integration
    Requirements: 1.2, 1.3, 3.1, 4.3
    """
    
    @pytest.fixture
    def mock_alibaba_response(self):
        """Mock Alibaba API response."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1024)
        ]
        return mock_response
    
    @pytest.fixture
    def mock_alibaba_client(self, mock_alibaba_response):
        """Mock Alibaba client."""
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_alibaba_response
        return mock_client
    
    def test_end_to_end_alibaba_embedding(
        self, db_session: Session, mock_alibaba_client
    ):
        """
        Test complete workflow: create course with Alibaba model,
        upload document, and verify embeddings are generated.
        
        Requirements: 1.2, 1.3, 3.1, 4.3
        """
        # Create user
        user = User(
            email="teacher@test.com",
            hashed_password="hashed",
            full_name="Test Teacher",
            role=UserRole.TEACHER
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        # Create course
        course = Course(
            name="Test Course",
            description="Test course for Alibaba embeddings",
            teacher_id=user.id
        )
        db_session.add(course)
        db_session.commit()
        db_session.refresh(course)
        
        # Create course settings with Alibaba model
        settings = CourseSettings(
            course_id=course.id,
            default_embedding_model="alibaba/text-embedding-v4"
        )
        db_session.add(settings)
        db_session.commit()
        db_session.refresh(settings)
        
        # Verify settings were saved correctly (Requirement 1.2)
        assert settings.default_embedding_model == "alibaba/text-embedding-v4"
        
        # Create document
        document = Document(
            course_id=course.id,
            filename="test.txt",
            original_filename="test.txt",
            file_type="txt",
            file_size=100,
            char_count=35,
            content="Test document content for embedding",
            is_processed=True,
            embedding_model=None  # Will be set during embedding
        )
        db_session.add(document)
        db_session.commit()
        db_session.refresh(document)
        
        # Mock the Alibaba client
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            with patch.object(
                service, '_get_alibaba_client', return_value=mock_alibaba_client
            ):
                # Generate embedding (Requirement 1.3, 3.1)
                embedding = service.get_embedding(
                    document.content,
                    model="alibaba/text-embedding-v4"
                )
                
                # Verify embedding was generated
                assert embedding is not None
                assert len(embedding) == 1024  # Alibaba dimension
                
                # Update document with embedding model (Requirement 4.3)
                document.embedding_model = "alibaba/text-embedding-v4"
                db_session.commit()
                db_session.refresh(document)
                
                # Verify model is stored in document record
                assert document.embedding_model == "alibaba/text-embedding-v4"
                
                # Verify the correct API was called
                mock_alibaba_client.embeddings.create.assert_called_once()
                call_args = mock_alibaba_client.embeddings.create.call_args
                assert call_args[1]['model'] == "alibaba/text-embedding-v4"
                assert call_args[1]['input'] == document.content


class TestGracefulDegradation:
    """Test graceful degradation without Alibaba API key.
    
    Feature: alibaba-embedding-integration, Property 8
    Requirements: 2.2, 6.4
    """
    
    @pytest.fixture
    def mock_openrouter_response(self):
        """Mock OpenRouter API response."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]
        return mock_response
    
    @pytest.fixture
    def mock_openrouter_client(self, mock_openrouter_response):
        """Mock OpenRouter client."""
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_openrouter_response
        return mock_client
    
    def test_openrouter_works_without_alibaba_key(
        self, mock_openrouter_client
    ):
        """
        Test that OpenRouter models work when DASHSCOPE_API_KEY is not set.
        
        Property 8: Graceful Degradation Without Alibaba Key
        Requirements: 2.2, 6.4
        """
        # Ensure DASHSCOPE_API_KEY is not set
        env_without_alibaba = os.environ.copy()
        env_without_alibaba.pop('DASHSCOPE_API_KEY', None)
        env_without_alibaba['OPENROUTER_API_KEY'] = 'test-openrouter-key'
        
        with patch.dict(os.environ, env_without_alibaba, clear=True):
            service = EmbeddingService()
            
            with patch.object(
                service, '_get_client', return_value=mock_openrouter_client
            ):
                # Generate embedding with OpenRouter model
                embedding = service.get_embedding(
                    "Test text",
                    model="openai/text-embedding-3-small"
                )
                
                # Verify embedding was generated successfully
                assert embedding is not None
                assert len(embedding) == 1536
                
                # Verify OpenRouter client was used
                mock_openrouter_client.embeddings.create.assert_called_once()
    
    def test_alibaba_model_fails_without_key(self):
        """
        Test that Alibaba models fail gracefully when API key is missing.
        
        Requirements: 2.2
        """
        # Ensure DASHSCOPE_API_KEY is not set
        env_without_alibaba = os.environ.copy()
        env_without_alibaba.pop('DASHSCOPE_API_KEY', None)
        
        with patch.dict(os.environ, env_without_alibaba, clear=True):
            service = EmbeddingService()
            
            # Attempt to generate embedding with Alibaba model
            with pytest.raises(ValueError) as exc_info:
                service.get_embedding(
                    "Test text",
                    model="alibaba/text-embedding-v4"
                )
            
            # Verify error message is clear
            assert "DASHSCOPE_API_KEY" in str(exc_info.value)
            assert "required" in str(exc_info.value).lower()


class TestProviderSwitching:
    """Test switching between providers.
    
    Feature: alibaba-embedding-integration
    Requirements: 3.1, 6.2
    """
    
    @pytest.fixture
    def mock_openrouter_response(self):
        """Mock OpenRouter API response."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]
        return mock_response
    
    @pytest.fixture
    def mock_alibaba_response(self):
        """Mock Alibaba API response."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.2] * 1024)
        ]
        return mock_response
    
    @pytest.fixture
    def mock_openrouter_client(self, mock_openrouter_response):
        """Mock OpenRouter client."""
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_openrouter_response
        return mock_client
    
    @pytest.fixture
    def mock_alibaba_client(self, mock_alibaba_response):
        """Mock Alibaba client."""
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_alibaba_response
        return mock_client
    
    def test_provider_switching(
        self, db_session: Session, mock_openrouter_client, mock_alibaba_client
    ):
        """
        Test creating course with OpenRouter, generating embeddings,
        switching to Alibaba, and generating new embeddings.
        
        Requirements: 3.1, 6.2
        """
        # Create user
        user = User(
            email="teacher@test.com",
            hashed_password="hashed",
            full_name="Test Teacher",
            role=UserRole.TEACHER
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        # Create course
        course = Course(
            name="Test Course",
            description="Test course for provider switching",
            teacher_id=user.id
        )
        db_session.add(course)
        db_session.commit()
        db_session.refresh(course)
        
        # Create course settings with OpenRouter model
        settings = CourseSettings(
            course_id=course.id,
            default_embedding_model="openai/text-embedding-3-small"
        )
        db_session.add(settings)
        db_session.commit()
        db_session.refresh(settings)
        
        # Mock environment with both API keys
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "test-openrouter-key",
            "DASHSCOPE_API_KEY": "test-alibaba-key"
        }):
            service = EmbeddingService()
            
            # Generate embedding with OpenRouter
            with patch.object(
                service, '_get_client', return_value=mock_openrouter_client
            ):
                embedding1 = service.get_embedding(
                    "Test text 1",
                    model="openai/text-embedding-3-small"
                )
                
                assert embedding1 is not None
                assert len(embedding1) == 1536
                mock_openrouter_client.embeddings.create.assert_called_once()
            
            # Switch to Alibaba model
            settings.default_embedding_model = "alibaba/text-embedding-v4"
            db_session.commit()
            db_session.refresh(settings)
            
            assert settings.default_embedding_model == "alibaba/text-embedding-v4"
            
            # Generate embedding with Alibaba
            with patch.object(
                service, '_get_alibaba_client', return_value=mock_alibaba_client
            ):
                embedding2 = service.get_embedding(
                    "Test text 2",
                    model="alibaba/text-embedding-v4"
                )
                
                assert embedding2 is not None
                assert len(embedding2) == 1024
                mock_alibaba_client.embeddings.create.assert_called_once()
            
            # Verify both embeddings are different (different providers)
            assert len(embedding1) != len(embedding2)
            assert embedding1[0] != embedding2[0]
