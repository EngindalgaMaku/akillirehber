"""Unit tests for input_type parameter in EmbeddingService.

Tests the new input_type parameter added to get_embedding() and get_embeddings()
to verify default values and parameter passing.
"""

import pytest
from unittest.mock import Mock, patch
from app.services.embedding_service import EmbeddingService


class TestInputTypeParameter:
    """Tests for input_type parameter functionality."""

    def test_get_embedding_defaults_to_query(self):
        """Test that get_embedding() defaults to 'query' when input_type is not provided."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_response
            service._client = mock_client
            
            # Call without input_type
            result = service.get_embedding("test text", model="openai/text-embedding-3-small")
            
            # Verify the result
            assert result == [0.1, 0.2, 0.3]
            
    def test_get_embeddings_defaults_to_document(self):
        """Test that get_embeddings() defaults to 'document' when input_type is not provided."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1, 0.2, 0.3]),
                Mock(embedding=[0.4, 0.5, 0.6])
            ]
            mock_client.embeddings.create.return_value = mock_response
            service._client = mock_client
            
            # Call without input_type
            result = service.get_embeddings(["text1", "text2"], model="openai/text-embedding-3-small")
            
            # Verify the result
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
            
    def test_get_embedding_accepts_explicit_input_type(self):
        """Test that get_embedding() accepts explicit input_type parameter."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create.return_value = mock_response
            service._client = mock_client
            
            # Call with explicit input_type
            result = service.get_embedding(
                "test text", 
                model="openai/text-embedding-3-small",
                input_type="document"
            )
            
            # Verify the result
            assert result == [0.1, 0.2, 0.3]
            
    def test_get_embeddings_accepts_explicit_input_type(self):
        """Test that get_embeddings() accepts explicit input_type parameter."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1, 0.2, 0.3]),
                Mock(embedding=[0.4, 0.5, 0.6])
            ]
            mock_client.embeddings.create.return_value = mock_response
            service._client = mock_client
            
            # Call with explicit input_type
            result = service.get_embeddings(
                ["text1", "text2"], 
                model="openai/text-embedding-3-small",
                input_type="query"
            )
            
            # Verify the result
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]


class TestVoyageInputTypeParameter:
    """Tests for Voyage AI input_type parameter functionality."""

    @patch('requests.post')
    def test_voyage_get_embedding_uses_input_type_parameter(
        self, mock_post
    ):
        """Test that Voyage get_embedding() uses input_type parameter."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            }
            mock_post.return_value = mock_response
            
            # Call with explicit input_type
            result = service.get_embedding(
                "test text",
                model="voyage/voyage-3",
                input_type="document"
            )
            
            # Verify the result
            assert result == [0.1, 0.2, 0.3]
            
            # Verify the API was called with correct input_type
            assert mock_post.called
            call_args = mock_post.call_args
            request_data = call_args[1]['json']
            assert request_data['input_type'] == "document"
            
    @patch('requests.post')
    def test_voyage_get_embedding_defaults_to_query(self, mock_post):
        """Test that Voyage get_embedding() defaults to 'query'."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            }
            mock_post.return_value = mock_response
            
            # Call without input_type
            result = service.get_embedding(
                "test text",
                model="voyage/voyage-3"
            )
            
            # Verify the result
            assert result == [0.1, 0.2, 0.3]
            
            # Verify the API was called with default 'query'
            assert mock_post.called
            call_args = mock_post.call_args
            request_data = call_args[1]['json']
            assert request_data['input_type'] == "query"
            
    @patch('requests.post')
    def test_voyage_get_embeddings_uses_input_type_parameter(
        self, mock_post
    ):
        """Test that Voyage get_embeddings() uses input_type parameter."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]}
                ]
            }
            mock_post.return_value = mock_response
            
            # Call with explicit input_type
            result = service.get_embeddings(
                ["text1", "text2"],
                model="voyage/voyage-3",
                input_type="query"
            )
            
            # Verify the result
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
            
            # Verify the API was called with correct input_type
            assert mock_post.called
            call_args = mock_post.call_args
            request_data = call_args[1]['json']
            assert request_data['input_type'] == "query"
            
    @patch('requests.post')
    def test_voyage_get_embeddings_defaults_to_document(self, mock_post):
        """Test that Voyage get_embeddings() defaults to 'document'."""
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]}
                ]
            }
            mock_post.return_value = mock_response
            
            # Call without input_type
            result = service.get_embeddings(
                ["text1", "text2"],
                model="voyage/voyage-3"
            )
            
            # Verify the result
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
            
            # Verify the API was called with default 'document'
            assert mock_post.called
            call_args = mock_post.call_args
            request_data = call_args[1]['json']
            assert request_data['input_type'] == "document"



class TestCohereInputTypeMapping:
    """Tests for Cohere input_type mapping functionality."""

    def test_cohere_get_embedding_maps_query_to_search_query(self):
        """Test that Cohere get_embedding() maps 'query' to 'search_query'."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the Cohere client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.embeddings = [[0.1, 0.2, 0.3]]
            mock_client.embed.return_value = mock_response
            service._cohere_client = mock_client
            
            # Call with input_type="query"
            result = service.get_embedding(
                "test text",
                model="cohere/embed-multilingual-v3.0",
                input_type="query"
            )
            
            # Verify the result
            assert result == [0.1, 0.2, 0.3]
            
            # Verify the API was called with 'search_query'
            assert mock_client.embed.called
            call_args = mock_client.embed.call_args
            assert call_args[1]['input_type'] == "search_query"
            
    def test_cohere_get_embedding_maps_document_to_search_document(self):
        """Test Cohere get_embedding() maps 'document' to 'search_document'."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the Cohere client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.embeddings = [[0.1, 0.2, 0.3]]
            mock_client.embed.return_value = mock_response
            service._cohere_client = mock_client
            
            # Call with input_type="document"
            result = service.get_embedding(
                "test text",
                model="cohere/embed-multilingual-v3.0",
                input_type="document"
            )
            
            # Verify the result
            assert result == [0.1, 0.2, 0.3]
            
            # Verify the API was called with 'search_document'
            assert mock_client.embed.called
            call_args = mock_client.embed.call_args
            assert call_args[1]['input_type'] == "search_document"
            
    def test_cohere_get_embedding_defaults_to_query(self):
        """Test that Cohere get_embedding() defaults to 'query' (search_query)."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the Cohere client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.embeddings = [[0.1, 0.2, 0.3]]
            mock_client.embed.return_value = mock_response
            service._cohere_client = mock_client
            
            # Call without input_type
            result = service.get_embedding(
                "test text",
                model="cohere/embed-multilingual-v3.0"
            )
            
            # Verify the result
            assert result == [0.1, 0.2, 0.3]
            
            # Verify the API was called with default 'search_query'
            assert mock_client.embed.called
            call_args = mock_client.embed.call_args
            assert call_args[1]['input_type'] == "search_query"
            
    def test_cohere_get_embeddings_maps_query_to_search_query(self):
        """Test Cohere get_embeddings() maps 'query' to 'search_query'."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the Cohere client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_client.embed.return_value = mock_response
            service._cohere_client = mock_client
            
            # Call with input_type="query"
            result = service.get_embeddings(
                ["text1", "text2"],
                model="cohere/embed-multilingual-v3.0",
                input_type="query"
            )
            
            # Verify the result
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
            
            # Verify the API was called with 'search_query'
            assert mock_client.embed.called
            call_args = mock_client.embed.call_args
            assert call_args[1]['input_type'] == "search_query"
            
    def test_cohere_get_embeddings_maps_document_to_search_document(self):
        """Test Cohere get_embeddings() maps 'document' to 'search_document'."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the Cohere client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_client.embed.return_value = mock_response
            service._cohere_client = mock_client
            
            # Call with input_type="document"
            result = service.get_embeddings(
                ["text1", "text2"],
                model="cohere/embed-multilingual-v3.0",
                input_type="document"
            )
            
            # Verify the result
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
            
            # Verify the API was called with 'search_document'
            assert mock_client.embed.called
            call_args = mock_client.embed.call_args
            assert call_args[1]['input_type'] == "search_document"
            
    def test_cohere_get_embeddings_defaults_to_document(self):
        """Test Cohere get_embeddings() defaults to 'document' (search_document)."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            service = EmbeddingService()
            
            # Mock the Cohere client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_client.embed.return_value = mock_response
            service._cohere_client = mock_client
            
            # Call without input_type
            result = service.get_embeddings(
                ["text1", "text2"],
                model="cohere/embed-multilingual-v3.0"
            )
            
            # Verify the result
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
            
            # Verify the API was called with default 'search_document'
            assert mock_client.embed.called
            call_args = mock_client.embed.call_args
            assert call_args[1]['input_type'] == "search_document"
