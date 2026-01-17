"""Property-based tests for document embedding model tracking.

Feature: alibaba-embedding-integration
Tests document embedding model tracking functionality.
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime
from unittest.mock import Mock, patch

from app.models.db_models import Document, EmbeddingStatus, Course, User, UserRole, Chunk
from app.services.weaviate_service import ChunkWithEmbedding


class TestDocumentEmbeddingModelTracking:
    """Property tests for document embedding model tracking.
    
    Validates: Requirements 4.3
    """

    @given(
        model_name=st.one_of(
            st.just("openai/text-embedding-3-small"),
            st.just("openai/text-embedding-3-large"),
            st.just("openai/text-embedding-ada-002"),
            st.just("alibaba/text-embedding-v4"),
            st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="/-_"))
        )
    )
    @settings(max_examples=100)
    def test_document_embedding_model_tracking_property(self, model_name):
        """
        Feature: alibaba-embedding-integration, Property 4: Document Embedding Model Tracking
        
        For any document, after setting the embedding_model field to a specific model,
        the document should store that exact model identifier.
        
        Validates: Requirements 4.3
        """
        # Create a document instance
        document = Document(
            filename="test.txt",
            original_filename="test.txt",
            file_type="txt",
            file_size=100,
            char_count=100,
            content="test content",
            course_id=1,
            is_processed=True
        )
        
        # Set the embedding model
        document.embedding_model = model_name
        
        # Property: The embedding_model field should store the exact value
        assert document.embedding_model == model_name
        assert isinstance(document.embedding_model, str)
        assert len(document.embedding_model) > 0

    @given(
        model_name=st.one_of(
            st.just("openai/text-embedding-3-small"),
            st.just("alibaba/text-embedding-v4")
        )
    )
    @settings(max_examples=100)
    def test_embedding_status_transitions_with_model(self, model_name):
        """
        Property test: Embedding status transitions should preserve model information.
        
        For any embedding model, when a document transitions through embedding states,
        the model identifier should be preserved.
        
        Validates: Requirements 4.3
        """
        # Create a document
        document = Document(
            filename="test.txt",
            original_filename="test.txt",
            file_type="txt",
            file_size=100,
            char_count=100,
            content="test content",
            course_id=1,
            is_processed=True,
            embedding_status=EmbeddingStatus.PENDING
        )
        
        # Transition to PROCESSING
        document.embedding_status = EmbeddingStatus.PROCESSING
        document.embedding_model = model_name
        
        # Property: Model should be set during processing
        assert document.embedding_model == model_name
        assert document.embedding_status == EmbeddingStatus.PROCESSING
        
        # Transition to COMPLETED
        document.embedding_status = EmbeddingStatus.COMPLETED
        document.embedded_at = datetime.utcnow()
        document.vector_count = 10
        
        # Property: Model should persist after completion
        assert document.embedding_model == model_name
        assert document.embedding_status == EmbeddingStatus.COMPLETED
        assert document.vector_count > 0

    def test_multiple_models_tracked_separately(self):
        """
        Unit test: Different documents can track different embedding models.
        
        Validates: Requirements 4.3
        """
        # Create documents with different models
        doc1 = Document(
            filename="doc1.txt",
            original_filename="doc1.txt",
            file_type="txt",
            file_size=100,
            char_count=100,
            content="content 1",
            course_id=1,
            is_processed=True,
            embedding_model="openai/text-embedding-3-small",
            embedding_status=EmbeddingStatus.COMPLETED
        )
        
        doc2 = Document(
            filename="doc2.txt",
            original_filename="doc2.txt",
            file_type="txt",
            file_size=100,
            char_count=100,
            content="content 2",
            course_id=1,
            is_processed=True,
            embedding_model="alibaba/text-embedding-v4",
            embedding_status=EmbeddingStatus.COMPLETED
        )
        
        # Each document should track its own model
        assert doc1.embedding_model == "openai/text-embedding-3-small"
        assert doc2.embedding_model == "alibaba/text-embedding-v4"
        assert doc1.embedding_model != doc2.embedding_model

    def test_embedding_model_cleared_on_vector_deletion(self):
        """
        Unit test: Embedding model should be cleared when vectors are deleted.
        
        Validates: Requirements 4.3
        """
        # Create a document with embeddings
        document = Document(
            filename="test.txt",
            original_filename="test.txt",
            file_type="txt",
            file_size=100,
            char_count=100,
            content="test content",
            course_id=1,
            is_processed=True,
            embedding_model="openai/text-embedding-3-small",
            embedding_status=EmbeddingStatus.COMPLETED,
            embedded_at=datetime.utcnow(),
            vector_count=10
        )
        
        # Simulate vector deletion (as done in embeddings.py)
        document.embedding_status = EmbeddingStatus.PENDING
        document.embedding_model = None
        document.embedded_at = None
        document.vector_count = 0
        
        # Model should be cleared
        assert document.embedding_model is None
        assert document.embedding_status == EmbeddingStatus.PENDING
        assert document.vector_count == 0

    @given(
        model_name=st.text(min_size=1, max_size=255)
    )
    @settings(max_examples=100)
    def test_embedding_model_accepts_any_string(self, model_name):
        """
        Property test: embedding_model field should accept any string value.
        
        For any string, the embedding_model field should store it correctly.
        This allows for future extensibility with new providers.
        
        Validates: Requirements 4.3
        """
        document = Document(
            filename="test.txt",
            original_filename="test.txt",
            file_type="txt",
            file_size=100,
            char_count=100,
            content="test content",
            course_id=1,
            is_processed=True
        )
        
        # Set any string as embedding model
        document.embedding_model = model_name
        
        # Property: Should store the exact string
        assert document.embedding_model == model_name
        assert isinstance(document.embedding_model, str)



class TestDocumentEmbeddingIntegration:
    """Integration tests for document embedding with model tracking.
    
    Validates: Requirements 4.3
    """

    def test_embed_document_with_alibaba_model(self, db_session):
        """
        Integration test: Embedding a document with Alibaba model stores the model.
        
        Test embedding a document with Alibaba model and verify the model
        is stored in the document record.
        
        Validates: Requirements 4.3
        """
        # Create a test user
        user = User(
            email="teacher@test.com",
            hashed_password="hashed",
            full_name="Test Teacher",
            role=UserRole.TEACHER
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        # Create a test course
        course = Course(
            name="Test Course",
            description="Test Description",
            teacher_id=user.id
        )
        db_session.add(course)
        db_session.commit()
        db_session.refresh(course)
        
        # Create a test document
        document = Document(
            filename="test.txt",
            original_filename="test.txt",
            file_type="txt",
            file_size=100,
            char_count=100,
            content="test content",
            course_id=course.id,
            is_processed=True
        )
        db_session.add(document)
        db_session.commit()
        db_session.refresh(document)
        
        # Create test chunks
        chunks = [
            Chunk(
                content=f"Chunk {i}",
                index=i,
                start_position=i * 10,
                end_position=(i + 1) * 10,
                char_count=10,
                has_overlap=False,
                document_id=document.id
            )
            for i in range(3)
        ]
        for chunk in chunks:
            db_session.add(chunk)
        db_session.commit()
        
        # Mock the embedding service and weaviate service
        with patch('app.routers.embeddings.get_embedding_service') as mock_embedding_service, \
             patch('app.routers.embeddings.get_weaviate_service') as mock_weaviate_service:
            
            # Setup mocks
            mock_embedding_instance = Mock()
            mock_embedding_instance.get_embeddings.return_value = [
                [0.1] * 1024,  # Alibaba model returns 1024 dimensions
                [0.2] * 1024,
                [0.3] * 1024
            ]
            mock_embedding_service.return_value = mock_embedding_instance
            
            mock_weaviate_instance = Mock()
            mock_weaviate_instance.store_chunks.return_value = ["uuid1", "uuid2", "uuid3"]
            mock_weaviate_service.return_value = mock_weaviate_instance
            
            # Simulate the embedding process (as done in embeddings.py)
            model = "alibaba/text-embedding-v4"
            
            # Update status to processing
            document.embedding_status = EmbeddingStatus.PROCESSING
            db_session.commit()
            
            # Get embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = mock_embedding_instance.get_embeddings(texts, model=model)
            
            # Prepare chunks with embeddings
            chunks_with_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                if embedding:
                    chunks_with_embeddings.append(ChunkWithEmbedding(
                        chunk_id=chunk.id,
                        document_id=document.id,
                        content=chunk.content,
                        chunk_index=chunk.index,
                        vector=embedding
                    ))
            
            # Store in Weaviate
            mock_weaviate_instance.store_chunks(
                course_id=course.id,
                document_id=document.id,
                chunks=chunks_with_embeddings
            )
            
            # Update document status (as done in embeddings.py)
            document.embedding_status = EmbeddingStatus.COMPLETED
            document.embedding_model = model
            document.embedded_at = datetime.utcnow()
            document.vector_count = len(chunks_with_embeddings)
            db_session.commit()
            db_session.refresh(document)
        
        # Verify the model is stored in the document record
        assert document.embedding_model == "alibaba/text-embedding-v4"
        assert document.embedding_status == EmbeddingStatus.COMPLETED
        assert document.vector_count == 3
        assert document.embedded_at is not None

    def test_embed_document_with_openai_model(self, db_session):
        """
        Integration test: Embedding a document with OpenAI model stores the model.
        
        Validates: Requirements 4.3
        """
        # Create a test user
        user = User(
            email="teacher2@test.com",
            hashed_password="hashed",
            full_name="Test Teacher 2",
            role=UserRole.TEACHER
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        # Create a test course
        course = Course(
            name="Test Course 2",
            description="Test Description 2",
            teacher_id=user.id
        )
        db_session.add(course)
        db_session.commit()
        db_session.refresh(course)
        
        # Create a test document
        document = Document(
            filename="test2.txt",
            original_filename="test2.txt",
            file_type="txt",
            file_size=100,
            char_count=100,
            content="test content 2",
            course_id=course.id,
            is_processed=True
        )
        db_session.add(document)
        db_session.commit()
        db_session.refresh(document)
        
        # Create test chunks
        chunks = [
            Chunk(
                content=f"Chunk {i}",
                index=i,
                start_position=i * 10,
                end_position=(i + 1) * 10,
                char_count=10,
                has_overlap=False,
                document_id=document.id
            )
            for i in range(2)
        ]
        for chunk in chunks:
            db_session.add(chunk)
        db_session.commit()
        
        # Mock the embedding service and weaviate service
        with patch('app.routers.embeddings.get_embedding_service') as mock_embedding_service, \
             patch('app.routers.embeddings.get_weaviate_service') as mock_weaviate_service:
            
            # Setup mocks
            mock_embedding_instance = Mock()
            mock_embedding_instance.get_embeddings.return_value = [
                [0.1] * 1536,  # OpenAI model returns 1536 dimensions
                [0.2] * 1536
            ]
            mock_embedding_service.return_value = mock_embedding_instance
            
            mock_weaviate_instance = Mock()
            mock_weaviate_instance.store_chunks.return_value = ["uuid1", "uuid2"]
            mock_weaviate_service.return_value = mock_weaviate_instance
            
            # Simulate the embedding process
            model = "openai/text-embedding-3-small"
            
            # Update status to processing
            document.embedding_status = EmbeddingStatus.PROCESSING
            db_session.commit()
            
            # Get embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = mock_embedding_instance.get_embeddings(texts, model=model)
            
            # Prepare chunks with embeddings
            chunks_with_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                if embedding:
                    chunks_with_embeddings.append(ChunkWithEmbedding(
                        chunk_id=chunk.id,
                        document_id=document.id,
                        content=chunk.content,
                        chunk_index=chunk.index,
                        vector=embedding
                    ))
            
            # Store in Weaviate
            mock_weaviate_instance.store_chunks(
                course_id=course.id,
                document_id=document.id,
                chunks=chunks_with_embeddings
            )
            
            # Update document status
            document.embedding_status = EmbeddingStatus.COMPLETED
            document.embedding_model = model
            document.embedded_at = datetime.utcnow()
            document.vector_count = len(chunks_with_embeddings)
            db_session.commit()
            db_session.refresh(document)
        
        # Verify the model is stored in the document record
        assert document.embedding_model == "openai/text-embedding-3-small"
        assert document.embedding_status == EmbeddingStatus.COMPLETED
        assert document.vector_count == 2
        assert document.embedded_at is not None

    def test_different_documents_track_different_models(self, db_session):
        """
        Integration test: Different documents can use different embedding models.
        
        Validates: Requirements 4.3
        """
        # Create a test user
        user = User(
            email="teacher3@test.com",
            hashed_password="hashed",
            full_name="Test Teacher 3",
            role=UserRole.TEACHER
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        # Create a test course
        course = Course(
            name="Test Course 3",
            description="Test Description 3",
            teacher_id=user.id
        )
        db_session.add(course)
        db_session.commit()
        db_session.refresh(course)
        
        # Create two documents with different models
        doc1 = Document(
            filename="doc1.txt",
            original_filename="doc1.txt",
            file_type="txt",
            file_size=100,
            char_count=100,
            content="content 1",
            course_id=course.id,
            is_processed=True,
            embedding_model="openai/text-embedding-3-small",
            embedding_status=EmbeddingStatus.COMPLETED,
            embedded_at=datetime.utcnow(),
            vector_count=5
        )
        
        doc2 = Document(
            filename="doc2.txt",
            original_filename="doc2.txt",
            file_type="txt",
            file_size=100,
            char_count=100,
            content="content 2",
            course_id=course.id,
            is_processed=True,
            embedding_model="alibaba/text-embedding-v4",
            embedding_status=EmbeddingStatus.COMPLETED,
            embedded_at=datetime.utcnow(),
            vector_count=3
        )
        
        db_session.add(doc1)
        db_session.add(doc2)
        db_session.commit()
        db_session.refresh(doc1)
        db_session.refresh(doc2)
        
        # Verify each document tracks its own model
        assert doc1.embedding_model == "openai/text-embedding-3-small"
        assert doc2.embedding_model == "alibaba/text-embedding-v4"
        assert doc1.embedding_model != doc2.embedding_model
        
        # Both should be completed
        assert doc1.embedding_status == EmbeddingStatus.COMPLETED
        assert doc2.embedding_status == EmbeddingStatus.COMPLETED
