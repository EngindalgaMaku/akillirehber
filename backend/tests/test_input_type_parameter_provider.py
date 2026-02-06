"""Tests for input_type parameter in embedding providers.

Feature: voyage-instruction-parameter
Tests for the input_type parameter support across all embedding providers.
"""

import pytest
import inspect
from typing import List, Optional
from unittest.mock import MagicMock

from app.services.embedding_provider import (
    EmbeddingProvider,
    EmbeddingProviderManager,
    VoyageProvider,
    CohereProvider,
    OpenAIProvider,
    AlibabaProvider,
    JinaProvider,
    OllamaProvider,
    OpenRouterProvider,
)


class TestInputTypeParameter:
    """Tests for input_type parameter in provider base class and implementations."""
    
    def test_base_class_has_input_type_parameter(self):
        """EmbeddingProvider abstract base class has input_type parameter."""
        # Get the signature of the abstract method
        sig = inspect.signature(EmbeddingProvider.get_embeddings)
        params = sig.parameters
        
        # Verify input_type parameter exists
        assert "input_type" in params
        # Verify it has a default value
        assert params["input_type"].default == "document"
    
    def test_voyage_provider_has_input_type_parameter(self):
        """VoyageProvider has input_type parameter."""
        sig = inspect.signature(VoyageProvider.get_embeddings)
        params = sig.parameters
        
        assert "input_type" in params
        assert params["input_type"].default == "document"
    
    def test_cohere_provider_has_input_type_parameter(self):
        """CohereProvider has input_type parameter."""
        sig = inspect.signature(CohereProvider.get_embeddings)
        params = sig.parameters
        
        assert "input_type" in params
        assert params["input_type"].default == "document"
    
    def test_openai_provider_has_input_type_parameter(self):
        """OpenAIProvider has input_type parameter."""
        sig = inspect.signature(OpenAIProvider.get_embeddings)
        params = sig.parameters
        
        assert "input_type" in params
        assert params["input_type"].default == "document"
    
    def test_openrouter_provider_has_input_type_parameter(self):
        """OpenRouterProvider has input_type parameter."""
        sig = inspect.signature(OpenRouterProvider.get_embeddings)
        params = sig.parameters
        
        assert "input_type" in params
        assert params["input_type"].default == "document"
    
    def test_alibaba_provider_has_input_type_parameter(self):
        """AlibabaProvider has input_type parameter."""
        sig = inspect.signature(AlibabaProvider.get_embeddings)
        params = sig.parameters
        
        assert "input_type" in params
        assert params["input_type"].default == "document"
    
    def test_jina_provider_has_input_type_parameter(self):
        """JinaProvider has input_type parameter."""
        sig = inspect.signature(JinaProvider.get_embeddings)
        params = sig.parameters
        
        assert "input_type" in params
        assert params["input_type"].default == "document"
    
    def test_ollama_provider_has_input_type_parameter(self):
        """OllamaProvider has input_type parameter."""
        sig = inspect.signature(OllamaProvider.get_embeddings)
        params = sig.parameters
        
        assert "input_type" in params
        assert params["input_type"].default == "document"
    
    def test_provider_manager_has_input_type_parameter(self):
        """EmbeddingProviderManager.get_embeddings has input_type parameter."""
        sig = inspect.signature(EmbeddingProviderManager.get_embeddings)
        params = sig.parameters
        
        assert "input_type" in params
        assert params["input_type"].default == "document"
    
    def test_provider_manager_get_embeddings_batch_has_input_type(self):
        """EmbeddingProviderManager.get_embeddings_batch has input_type parameter."""
        sig = inspect.signature(EmbeddingProviderManager.get_embeddings_batch)
        params = sig.parameters
        
        assert "input_type" in params
        assert params["input_type"].default == "document"
    
    def test_provider_manager_passes_input_type(self):
        """EmbeddingProviderManager passes input_type to providers."""
        # Create a mock provider that tracks calls
        mock_provider = MagicMock(spec=EmbeddingProvider)
        mock_provider.name = "mock"
        mock_provider.is_available.return_value = True
        mock_provider.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        
        manager = EmbeddingProviderManager(providers=[mock_provider])
        
        # Call with query input_type
        manager.get_embeddings(["test text"], input_type="query")
        
        # Verify provider was called with input_type
        mock_provider.get_embeddings.assert_called_once()
        call_args = mock_provider.get_embeddings.call_args
        # Check positional args
        assert call_args[0][0] == ["test text"]  # texts
        # Check that input_type was passed (either as positional or keyword arg)
        if len(call_args[0]) > 2:
            assert call_args[0][2] == "query"  # positional arg
        else:
            assert call_args[1].get("input_type") == "query"  # keyword arg
        
        # Reset and test with document input_type
        mock_provider.reset_mock()
        manager.get_embeddings(["test text"], input_type="document")
        
        # Verify provider was called with input_type
        mock_provider.get_embeddings.assert_called_once()
        call_args = mock_provider.get_embeddings.call_args
        if len(call_args[0]) > 2:
            assert call_args[0][2] == "document"  # positional arg
        else:
            assert call_args[1].get("input_type") == "document"  # keyword arg
    
    def test_default_input_type_is_document(self):
        """Default input_type is 'document' when not specified."""
        mock_provider = MagicMock(spec=EmbeddingProvider)
        mock_provider.name = "mock"
        mock_provider.is_available.return_value = True
        mock_provider.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        
        manager = EmbeddingProviderManager(providers=[mock_provider])
        
        # Call without input_type
        manager.get_embeddings(["test text"])
        
        # Verify provider was called with default "document"
        call_args = mock_provider.get_embeddings.call_args
        if len(call_args[0]) > 2:
            assert call_args[0][2] == "document"  # positional arg
        else:
            assert call_args[1].get("input_type") == "document"  # keyword arg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
