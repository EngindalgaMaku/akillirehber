"""Integration tests for Cohere embedding provider.

Tests the complete integration of Cohere embedding models with the system,
including API client initialization, single/batch embedding generation,
error handling, and graceful degradation.
"""

import os
import pytest
from unittest.mock import Mock, patch
from app.services.embedding_service import EmbeddingService


class TestCohereIntegration:
    """Integration tests for Cohere embedding provider."""

    @pytest.fixture
    def mock_cohere_client(self):
        """Create a mock Cohere client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [[0.1] * 1024]  # Default 1024 dim
        mock_client.embed.return_value = mock_response
        return mock_client

    @pytest.fixture
    def service_with_cohere(self, mock_cohere_client):
        """Create embedding service with mocked Cohere client."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}):
            with patch("app.services.embedding_service.COHERE_AVAILABLE", True):
                with patch("app.services.embedding_service.cohere.Client") as mock_cohere:
                    mock_cohere.return_value = mock_cohere_client
                    service = EmbeddingService()
                    # Force the client to use our mock
                    service._cohere_client = mock_cohere_client
                    return service

    def test_cohere_client_initialization(self, service_with_cohere):
        """Test that Cohere client is properly initialized."""
        client = service_with_cohere._get_cohere_client()
        assert client is not None

    def test_cohere_client_missing_api_key(self):
        """Test error when COHERE_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("app.services.embedding_service.COHERE_AVAILABLE", True):
                service = EmbeddingService()
                with pytest.raises(ValueError, match="COHERE_API_KEY"):
                    service._get_cohere_client()

    def test_cohere_package_not_installed(self):
        """Test error when cohere package is not installed."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}):
            with patch("app.services.embedding_service.COHERE_AVAILABLE", False):
                service = EmbeddingService()
                with pytest.raises(ValueError, match="Cohere package is not installed"):
                    service._get_cohere_client()

    def test_cohere_provider_detection(self, service_with_cohere):
        """Test that Cohere models are correctly detected."""
        assert service_with_cohere._get_provider_for_model(
            "cohere/embed-multilingual-v3.0"
        ) == "cohere"
        assert service_with_cohere._get_provider_for_model(
            "cohere/embed-multilingual-light-v3.0"
        ) == "cohere"
        assert service_with_cohere._get_provider_for_model(
            "openai/text-embedding-3-small"
        ) == "openrouter"

    def test_cohere_single_embedding(self, service_with_cohere, mock_cohere_client):
        """Test generating a single embedding with Cohere."""
        # Setup mock response
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3] * 341 + [0.1]]  # 1024 dim
        mock_cohere_client.embed.return_value = mock_response

        embedding = service_with_cohere.get_embedding(
            "Test text",
            model="cohere/embed-multilingual-v3.0"
        )

        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)
        
        # Verify API call
        mock_cohere_client.embed.assert_called_once()
        call_args = mock_cohere_client.embed.call_args
        assert call_args[1]["texts"] == ["Test text"]
        assert call_args[1]["model"] == "embed-multilingual-v3.0"
        assert call_args[1]["input_type"] == "search_document"

    def test_cohere_batch_embeddings(self, service_with_cohere, mock_cohere_client):
        """Test generating batch embeddings with Cohere."""
        texts = [f"Text {i}" for i in range(5)]
        
        # Setup mock response
        mock_response = Mock()
        mock_response.embeddings = [
            [0.1, 0.2, 0.3] * 128  # 384 dim for light model
            for _ in range(5)
        ]
        mock_cohere_client.embed.return_value = mock_response

        embeddings = service_with_cohere.get_embeddings(
            texts,
            model="cohere/embed-multilingual-light-v3.0"
        )

        assert len(embeddings) == 5
        assert all(len(emb) == 384 for emb in embeddings)
        
        # Verify API call
        mock_cohere_client.embed.assert_called_once()
        call_args = mock_cohere_client.embed.call_args
        assert call_args[1]["texts"] == texts
        assert call_args[1]["model"] == "embed-multilingual-light-v3.0"

    def test_cohere_batch_size_limit(self, service_with_cohere, mock_cohere_client):
        """Test that Cohere respects batch size limit of 96."""
        # Create 200 texts to test batching
        texts = [f"Text {i}" for i in range(200)]
        
        # Setup mock response
        mock_response = Mock()
        mock_response.embeddings = [[0.1] * 1024]
        mock_cohere_client.embed.return_value = mock_response

        embeddings = service_with_cohere.get_embeddings(
            texts,
            model="cohere/embed-multilingual-v3.0"
        )

        assert len(embeddings) == 200
        
        # Should make 3 calls: 96 + 96 + 8
        assert mock_cohere_client.embed.call_count == 3
        
        # Verify batch sizes
        calls = mock_cohere_client.embed.call_args_list
        assert len(calls[0][1]["texts"]) == 96  # First batch
        assert len(calls[1][1]["texts"]) == 96  # Second batch
        assert len(calls[2][1]["texts"]) == 8   # Last batch

    def test_cohere_empty_text_handling(self, service_with_cohere):
        """Test handling of empty texts with Cohere."""
        embedding = service_with_cohere.get_embedding(
            "",
            model="cohere/embed-multilingual-v3.0"
        )
        assert embedding == []

    def test_cohere_mixed_empty_texts(self, service_with_cohere, mock_cohere_client):
        """Test handling of mixed empty and non-empty texts."""
        texts = ["Text 1", "", "Text 3", "   ", "Text 5"]
        
        # Setup mock response for 3 non-empty texts
        mock_response = Mock()
        mock_response.embeddings = [[0.1] * 1024 for _ in range(3)]
        mock_cohere_client.embed.return_value = mock_response

        embeddings = service_with_cohere.get_embeddings(
            texts,
            model="cohere/embed-multilingual-v3.0"
        )

        assert len(embeddings) == 5
        assert len(embeddings[0]) == 1024  # Text 1
        assert embeddings[1] == []          # Empty
        assert len(embeddings[2]) == 1024  # Text 3
        assert embeddings[3] == []          # Whitespace
        assert len(embeddings[4]) == 1024  # Text 5

    def test_cohere_model_dimensions(self, service_with_cohere):
        """Test that Cohere model dimensions are correctly configured."""
        assert service_with_cohere.get_embedding_dimension(
            "cohere/embed-multilingual-v3.0"
        ) == 1024
        assert service_with_cohere.get_embedding_dimension(
            "cohere/embed-multilingual-light-v3.0"
        ) == 384

    def test_cohere_error_handling(self, service_with_cohere, mock_cohere_client):
        """Test error handling for Cohere API errors."""
        mock_cohere_client.embed.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            service_with_cohere.get_embedding(
                "Test text",
                model="cohere/embed-multilingual-v3.0"
            )

    def test_cohere_authentication_error(self, mock_cohere_client):
        """Test handling of authentication errors."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "invalid-key"}):
            with patch("app.services.embedding_service.COHERE_AVAILABLE", True):
                with patch("app.services.embedding_service.cohere.Client") as mock_cohere:
                    mock_cohere.return_value = mock_cohere_client
                    
                    # Simulate authentication error
                    from openai import AuthenticationError
                    mock_cohere_client.embed.side_effect = AuthenticationError(
                        "Invalid API key",
                        response=Mock(status_code=401),
                        body=None
                    )
                    
                    service = EmbeddingService()
                    with pytest.raises(ValueError, match="Invalid API key"):
                        service.get_embedding(
                            "Test text",
                            model="cohere/embed-multilingual-v3.0"
                        )

    def test_cohere_graceful_degradation(self):
        """Test graceful degradation when Cohere is not available."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("app.services.embedding_service.COHERE_AVAILABLE", False):
                service = EmbeddingService()
                
                # Should raise clear error message
                with pytest.raises(ValueError, match="Cohere package is not installed"):
                    service.get_embedding(
                        "Test text",
                        model="cohere/embed-multilingual-v3.0"
                    )

    def test_cohere_model_prefix_stripping(self, service_with_cohere, mock_cohere_client):
        """Test that 'cohere/' prefix is stripped when calling API."""
        mock_response = Mock()
        mock_response.embeddings = [[0.1] * 1024]
        mock_cohere_client.embed.return_value = mock_response

        service_with_cohere.get_embedding(
            "Test text",
            model="cohere/embed-multilingual-v3.0"
        )

        # Verify that model name sent to API doesn't have prefix
        call_args = mock_cohere_client.embed.call_args
        assert call_args[1]["model"] == "embed-multilingual-v3.0"
        assert "cohere/" not in call_args[1]["model"]

    def test_cohere_multilingual_models_only(self, service_with_cohere):
        """Test that only multilingual Cohere models are configured."""
        # These should be configured
        assert service_with_cohere.get_embedding_dimension(
            "cohere/embed-multilingual-v3.0"
        ) == 1024
        assert service_with_cohere.get_embedding_dimension(
            "cohere/embed-multilingual-light-v3.0"
        ) == 384
        
        # English-only models should not be configured (return default)
        assert service_with_cohere.get_embedding_dimension(
            "cohere/embed-english-v3.0"
        ) == 1536  # Default dimension

    @pytest.mark.integration
    def test_cohere_real_api_call(self):
        """Test real API call to Cohere (requires COHERE_API_KEY).
        
        This test is skipped if COHERE_API_KEY is not set.
        """
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            pytest.skip("COHERE_API_KEY not set")

        service = EmbeddingService()
        
        # Test single embedding
        embedding = service.get_embedding(
            "Hello, world!",
            model="cohere/embed-multilingual-v3.0"
        )
        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)
        
        # Test batch embeddings
        embeddings = service.get_embeddings(
            ["Hello", "World", "Test"],
            model="cohere/embed-multilingual-light-v3.0"
        )
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
