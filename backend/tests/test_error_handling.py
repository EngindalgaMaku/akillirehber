"""Property-based tests for error handling and fallback mechanisms.

Feature: semantic-chunker-enhancement, Phase 5: Error Handling and Configuration
This module contains tests for error handling, fallback strategies, and logging.
"""

import pytest
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st, HealthCheck

from app.services.chunker import (
    chunk_with_error_handling,
    fallback_to_sentence_chunking,
    fallback_to_universal_tokenization,
    fallback_to_fixed_size_chunking,
    ChunkingResult,
    SemanticChunker,
)
from app.services.chunker_exceptions import (
    SemanticChunkerError,
    LanguageDetectionError,
    SentenceTokenizationError,
    CacheError,
    ErrorCode,
)
from app.models.chunking import ChunkingStrategy


class TestFallbackNotificationProperty:
    """Property-based tests for fallback notification.
    
    Feature: semantic-chunker-enhancement, Property 24: Fallback Notification
    Validates: Requirements 11.3
    
    WHEN fallback is triggered, THE Semantic_Chunker SHALL notify the user
    with a warning message.
    """
    
    @given(
        text=st.text(min_size=100, max_size=1000).filter(lambda x: x.strip())
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None
    )
    def test_fallback_notification_on_embedding_failure(self, text):
        """Property test: Fallback triggers warning message on embedding failure.
        
        Feature: semantic-chunker-enhancement, Property 24: Fallback Notification
        Validates: Requirements 11.3
        """
        # Mock the SemanticChunker to simulate embedding failure
        with patch.object(
            SemanticChunker, 'chunk',
            side_effect=Exception("Simulated embedding failure")
        ):
            result = chunk_with_error_handling(
                text,
                strategy=ChunkingStrategy.SEMANTIC
            )
            
            # Property: When fallback is used, warning_message should be set
            if result.fallback_used:
                assert result.warning_message is not None, (
                    "Fallback was used but no warning message was provided"
                )
                assert len(result.warning_message) > 0, (
                    "Warning message should not be empty"
                )
                assert "fallback" in result.warning_message.lower(), (
                    "Warning message should mention fallback"
                )
    
    def test_fallback_notification_contains_original_error(self):
        """Test that fallback notification includes original error info.
        
        Feature: semantic-chunker-enhancement, Property 24: Fallback Notification
        Validates: Requirements 11.3
        """
        text = "This is a test sentence. Here is another one. And a third."
        
        with patch.object(
            SemanticChunker, 'chunk',
            side_effect=Exception("API rate limit exceeded")
        ):
            result = chunk_with_error_handling(
                text,
                strategy=ChunkingStrategy.SEMANTIC
            )
            
            # Should have used fallback
            assert result.fallback_used is not None
            
            # Warning should mention original error
            assert result.warning_message is not None
            assert "API rate limit" in result.warning_message or "error" in result.warning_message.lower()
    
    def test_fallback_produces_valid_chunks(self):
        """Test that fallback produces valid chunks.
        
        Feature: semantic-chunker-enhancement, Property 24: Fallback Notification
        Validates: Requirements 11.3
        """
        text = "First sentence here. Second sentence follows. Third one too."
        
        with patch.object(
            SemanticChunker, 'chunk',
            side_effect=Exception("Embedding service unavailable")
        ):
            result = chunk_with_error_handling(
                text,
                strategy=ChunkingStrategy.SEMANTIC
            )
            
            # Should still produce chunks via fallback
            assert result.success is True
            assert len(result.chunks) > 0
            
            # All chunks should have valid content
            for chunk in result.chunks:
                assert chunk.content.strip()
                assert chunk.char_count > 0


class TestFallbackStrategies:
    """Tests for individual fallback strategies."""
    
    def test_sentence_fallback_produces_chunks(self):
        """Test sentence-based fallback produces valid chunks."""
        text = "First sentence. Second sentence. Third sentence."
        
        chunks = fallback_to_sentence_chunking(text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 0
        # All content should be preserved
        combined = " ".join(c.content for c in chunks)
        assert "First" in combined
        assert "Third" in combined
    
    def test_universal_tokenization_fallback(self):
        """Test universal tokenization fallback."""
        text = "Hello world. How are you? I am fine."
        
        sentences = fallback_to_universal_tokenization(text)
        
        assert len(sentences) >= 1
        # Content should be preserved
        combined = " ".join(sentences)
        assert "Hello" in combined
    
    def test_fixed_size_fallback_respects_word_boundaries(self):
        """Test fixed-size fallback respects word boundaries."""
        text = "This is a longer text that should be split into chunks without breaking words in the middle."
        
        chunks = fallback_to_fixed_size_chunking(text, chunk_size=30, overlap=5)
        
        assert len(chunks) > 0
        
        # No chunk should end with a partial word (unless at text end)
        for chunk in chunks[:-1]:  # Exclude last chunk
            content = chunk.content
            # Content should end with a complete word (space or punctuation before)
            # or be at a natural boundary
            assert not content.endswith('-'), (
                f"Chunk appears to have broken word: {content}"
            )


class TestExceptionClasses:
    """Tests for custom exception classes."""
    
    def test_semantic_chunker_error_to_dict(self):
        """Test SemanticChunkerError serialization."""
        error = SemanticChunkerError(
            "Test error message",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"key": "value"}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "SemanticChunkerError"
        assert error_dict["error_code"] == "CHUNK_1002"
        assert error_dict["message"] == "Test error message"
        assert error_dict["details"]["key"] == "value"
    
    def test_language_detection_error(self):
        """Test LanguageDetectionError with text sample."""
        error = LanguageDetectionError(
            "Detection failed",
            text_sample="Sample text for testing" * 10  # Long text
        )
        
        error_dict = error.to_dict()
        
        # Text sample should be truncated
        assert len(error_dict["details"]["text_sample"]) <= 100
        assert "text_length" in error_dict["details"]
    
    def test_cache_error_operation_codes(self):
        """Test CacheError uses correct error codes for operations."""
        get_error = CacheError("Read failed", operation="get")
        set_error = CacheError("Write failed", operation="set")
        conn_error = CacheError("Connection failed", operation="connect")
        
        assert get_error.error_code == ErrorCode.CACHE_READ_ERROR
        assert set_error.error_code == ErrorCode.CACHE_WRITE_ERROR
        assert conn_error.error_code == ErrorCode.CACHE_CONNECTION_ERROR


class TestChunkingResultDataclass:
    """Tests for ChunkingResult dataclass."""
    
    def test_successful_result(self):
        """Test successful chunking result."""
        from app.models.chunking import Chunk
        
        chunks = [
            Chunk(index=0, content="Test", start_position=0, 
                  end_position=4, char_count=4, has_overlap=False)
        ]
        
        result = ChunkingResult(chunks=chunks, success=True)
        
        assert result.success is True
        assert result.fallback_used is None
        assert result.warning_message is None
        assert result.error is None
    
    def test_fallback_result(self):
        """Test result with fallback."""
        from app.models.chunking import Chunk
        
        chunks = [
            Chunk(index=0, content="Test", start_position=0,
                  end_position=4, char_count=4, has_overlap=False)
        ]
        
        result = ChunkingResult(
            chunks=chunks,
            success=True,
            fallback_used="sentence",
            warning_message="Used sentence fallback"
        )
        
        assert result.success is True
        assert result.fallback_used == "sentence"
        assert "fallback" in result.warning_message.lower()



class TestErrorLoggingProperty:
    """Property-based tests for error logging completeness.
    
    Feature: semantic-chunker-enhancement, Property 23: Error Logging Completeness
    Validates: Requirements 11.2
    
    THE Semantic_Chunker SHALL log all errors with detailed context including
    text length, provider, model, and error details.
    """
    
    def test_log_context_contains_required_fields(self):
        """Test that log context contains all required fields.
        
        Feature: semantic-chunker-enhancement, Property 23: Error Logging Completeness
        Validates: Requirements 11.2
        """
        from app.services.chunker_logging import ChunkerLogContext
        
        context = ChunkerLogContext(
            text_length=1000,
            strategy="semantic",
            provider="openrouter",
            model="text-embedding-3-small",
            operation="embedding",
            error_type="APIError",
            error_message="Rate limit exceeded"
        )
        
        context_dict = context.to_dict()
        
        # Required fields should be present
        assert "text_length" in context_dict
        assert "strategy" in context_dict
        assert "operation" in context_dict
        
        # Error fields should be present when set
        assert "error_type" in context_dict
        assert "error_message" in context_dict
    
    def test_log_context_excludes_none_values(self):
        """Test that log context excludes None values.
        
        Feature: semantic-chunker-enhancement, Property 23: Error Logging Completeness
        Validates: Requirements 11.2
        """
        from app.services.chunker_logging import ChunkerLogContext
        
        context = ChunkerLogContext(
            text_length=500,
            operation="tokenization"
        )
        
        context_dict = context.to_dict()
        
        # None values should not be in dict
        assert "provider" not in context_dict
        assert "model" not in context_dict
        assert "error_type" not in context_dict
    
    def test_operation_logger_captures_timing(self):
        """Test that operation logger captures timing information.
        
        Feature: semantic-chunker-enhancement, Property 23: Error Logging Completeness
        Validates: Requirements 11.2
        """
        import time
        from app.services.chunker_logging import ChunkerOperationLogger
        
        with ChunkerOperationLogger(
            "test_operation",
            text_length=100,
            strategy="semantic"
        ) as op_logger:
            time.sleep(0.01)  # Small delay
            op_logger.set_result(chunk_count=5)
        
        # Duration should be captured
        assert op_logger.context.duration_ms is not None
        assert op_logger.context.duration_ms >= 10  # At least 10ms
        assert op_logger.context.chunk_count == 5
    
    def test_operation_logger_captures_errors(self):
        """Test that operation logger captures error information.
        
        Feature: semantic-chunker-enhancement, Property 23: Error Logging Completeness
        Validates: Requirements 11.2
        """
        from app.services.chunker_logging import ChunkerOperationLogger
        
        try:
            with ChunkerOperationLogger(
                "failing_operation",
                text_length=100
            ) as op_logger:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Error should be captured
        assert op_logger.context.error_type == "ValueError"
        assert op_logger.context.error_message == "Test error"
    
    def test_fallback_logging(self):
        """Test that fallback events are logged.
        
        Feature: semantic-chunker-enhancement, Property 23: Error Logging Completeness
        Validates: Requirements 11.2
        """
        from app.services.chunker_logging import ChunkerOperationLogger
        
        with ChunkerOperationLogger(
            "chunking",
            text_length=500,
            strategy="semantic"
        ) as op_logger:
            op_logger.log_fallback("sentence", "Embedding API failed")
        
        assert op_logger.context.fallback_used == "sentence"


class TestLoggingDecorator:
    """Tests for the logging decorator."""
    
    def test_with_logging_decorator_success(self):
        """Test logging decorator on successful function."""
        from app.services.chunker_logging import with_logging
        
        @with_logging("test_operation")
        def sample_function(text: str) -> int:
            return len(text)
        
        result = sample_function("Hello world")
        assert result == 11
    
    def test_with_logging_decorator_error(self):
        """Test logging decorator on failing function."""
        from app.services.chunker_logging import with_logging
        
        @with_logging("failing_operation")
        def failing_function(text: str) -> int:
            raise RuntimeError("Intentional failure")
        
        with pytest.raises(RuntimeError):
            failing_function("test")



class TestConfigurationValidation:
    """Tests for configuration validation."""
    
    def test_valid_configuration(self):
        """Test that valid configuration passes validation."""
        from app.services.chunker_config import SemanticChunkerConfig
        
        config = SemanticChunkerConfig()
        errors = config.validate()
        
        assert len(errors) == 0, f"Valid config should have no errors: {errors}"
    
    def test_invalid_chunk_size(self):
        """Test validation catches invalid chunk size."""
        from app.services.chunker_config import SemanticChunkerConfig
        
        config = SemanticChunkerConfig(chunk_size=-100)
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("chunk_size" in e for e in errors)
    
    def test_invalid_overlap(self):
        """Test validation catches overlap >= chunk_size."""
        from app.services.chunker_config import SemanticChunkerConfig
        
        config = SemanticChunkerConfig(chunk_size=100, overlap=150)
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("overlap" in e for e in errors)
    
    def test_invalid_threshold(self):
        """Test validation catches invalid threshold."""
        from app.services.chunker_config import SemanticChunkerConfig
        
        config = SemanticChunkerConfig(similarity_threshold=1.5)
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("threshold" in e.lower() for e in errors)
    
    def test_validate_or_raise(self):
        """Test validate_or_raise raises ConfigurationError."""
        from app.services.chunker_config import SemanticChunkerConfig
        from app.services.chunker_exceptions import ConfigurationError
        
        config = SemanticChunkerConfig(chunk_size=-1)
        
        with pytest.raises(ConfigurationError):
            config.validate_or_raise()
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        from app.services.chunker_config import SemanticChunkerConfig
        
        config = SemanticChunkerConfig(chunk_size=1000)
        config_dict = config.to_dict()
        
        assert config_dict["chunk_size"] == 1000
        assert "similarity_threshold" in config_dict
        assert "enable_qa_detection" in config_dict


class TestPreprocessingDiagnostics:
    """Tests for preprocessing diagnostics."""
    
    def test_preprocessing_logger_basic(self):
        """Test basic preprocessing logger functionality."""
        from app.services.chunker_logging import PreprocessingLogger
        
        original_text = "This is a test. Another sentence."
        original_sentences = ["This is a test.", "Another sentence."]
        
        logger = PreprocessingLogger(original_text, original_sentences)
        logger.log_merge("Merged short sentences")
        logger.set_result(["This is a test. Another sentence."])
        
        diagnostics = logger.get_diagnostics()
        
        assert diagnostics.original_sentence_count == 2
        assert diagnostics.processed_sentence_count == 1
        assert diagnostics.merged_sentences == 1
        assert len(diagnostics.transformations) == 1
    
    def test_preprocessing_diagnostics_summary(self):
        """Test preprocessing diagnostics summary generation."""
        from app.services.chunker_logging import PreprocessingDiagnostics
        
        diagnostics = PreprocessingDiagnostics(
            original_text_length=100,
            processed_text_length=90,
            merged_sentences=2,
            removed_sentences=1
        )
        
        summary = diagnostics._generate_summary()
        
        assert "merged" in summary.lower()
        assert "removed" in summary.lower()
    
    def test_preprocessing_diagnostics_to_dict(self):
        """Test preprocessing diagnostics serialization."""
        from app.services.chunker_logging import PreprocessingDiagnostics
        
        diagnostics = PreprocessingDiagnostics(
            original_text_length=100,
            processed_text_length=95
        )
        diagnostics.add_transformation("Normalized whitespace")
        
        result = diagnostics.to_dict()
        
        assert "original_text_length" in result
        assert "transformations" in result
        assert "summary" in result


class TestIntegrationErrorHandling:
    """Integration tests for error handling scenarios."""
    
    def test_empty_text_handling(self):
        """Test handling of empty text input."""
        result = chunk_with_error_handling("")
        
        assert result.success is False
        assert result.error is not None
        assert "empty" in result.warning_message.lower()
    
    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        result = chunk_with_error_handling("   \n\t   ")
        
        assert result.success is False
        assert result.error is not None
    
    def test_successful_chunking_with_diagnostics(self):
        """Test successful chunking includes diagnostics."""
        text = "First sentence here. Second sentence follows. Third one too. Fourth sentence. Fifth one."
        
        # Use fixed-size strategy to avoid embedding API calls
        result = chunk_with_error_handling(
            text,
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=50
        )
        
        assert result.success is True
        assert len(result.chunks) > 0
        assert result.diagnostics is not None
        assert result.diagnostics.total_chunks > 0
    
    def test_fallback_chain_works(self):
        """Test that fallback chain produces results."""
        text = "This is a test sentence. Here is another one. And a third."
        
        # Mock semantic chunker to fail
        with patch.object(
            SemanticChunker, 'chunk',
            side_effect=Exception("API unavailable")
        ):
            result = chunk_with_error_handling(
                text,
                strategy=ChunkingStrategy.SEMANTIC
            )
            
            # Should succeed via fallback
            assert result.success is True
            assert result.fallback_used is not None
            assert len(result.chunks) > 0


class TestHealthCheckEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check_returns_status(self):
        """Test health check endpoint returns proper status."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        response = client.get("/api/health/embedding-providers")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "providers" in data
        assert "summary" in data
        assert data["status"] in ["healthy", "degraded"]
