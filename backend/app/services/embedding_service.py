"""Embedding service using OpenRouter API."""

import logging
import os
from typing import List, Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings via OpenRouter API.
    
    Uses OpenRouter to access various embedding models including
    OpenAI's text-embedding-3-small.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    ALIBABA_MODEL_PREFIX = "alibaba/"
    OPENAI_MODEL_PREFIX = "openai/"
    COHERE_MODEL_PREFIX = "cohere/"
    DEFAULT_MODEL = "openai/text-embedding-3-small"
    BATCH_SIZE = 100  # Max texts per API call for OpenRouter
    ALIBABA_BATCH_SIZE = 10  # Max texts per API call for Alibaba
    COHERE_BATCH_SIZE = 96  # Max texts per API call for Cohere

    def __init__(self, api_key: str = None):
        """Initialize embedding service.
        
        Args:
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
        """
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
        self._cohere_api_key = os.environ.get("COHERE_API_KEY")
        self._client = None
        self._alibaba_client = None
        self._cohere_client = None

    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client for OpenRouter."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable is required "
                    "for embedding generation."
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self.OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "RAG System"
                }
            )
        return self._client

    def _get_alibaba_client(self) -> OpenAI:
        """Get or create OpenAI client for Alibaba DashScope.
        
        Returns:
            OpenAI client configured for DashScope
            
        Raises:
            ValueError: If DASHSCOPE_API_KEY is not set
        """
        if self._alibaba_client is None:
            if not self._dashscope_api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY environment variable is required "
                    "for Alibaba embedding models. Please configure the API key "
                    "or use an OpenRouter model instead."
                )
            self._alibaba_client = OpenAI(
                api_key=self._dashscope_api_key,
                base_url=self.DASHSCOPE_BASE_URL
            )
        return self._alibaba_client

    def _get_cohere_client(self):
        """Get or create Cohere client.
        
        Returns:
            Cohere client
            
        Raises:
            ValueError: If COHERE_API_KEY is not set or cohere package not installed
        """
        if not COHERE_AVAILABLE:
            raise ValueError(
                "Cohere package is not installed. "
                "Please install it with: pip install cohere"
            )
        
        if self._cohere_client is None:
            if not self._cohere_api_key:
                raise ValueError(
                    "COHERE_API_KEY environment variable is required "
                    "for Cohere embedding models. Please configure the API key "
                    "or use an OpenRouter model instead."
                )
            self._cohere_client = cohere.Client(api_key=self._cohere_api_key)
        return self._cohere_client

    def _get_provider_for_model(self, model: str) -> str:
        """Determine which provider to use based on model name.
        
        Args:
            model: Model identifier (e.g., "alibaba/text-embedding-v4")
            
        Returns:
            Provider name: "alibaba", "cohere", or "openrouter"
        """
        if model.startswith(self.ALIBABA_MODEL_PREFIX):
            return "alibaba"
        elif model.startswith(self.COHERE_MODEL_PREFIX):
            return "cohere"
        return "openrouter"

    def _get_client_for_model(self, model: str) -> OpenAI:
        """Get appropriate client based on model.
        
        Args:
            model: Model identifier
            
        Returns:
            Configured OpenAI client for the provider (or Cohere client for Cohere models)
        """
        provider = self._get_provider_for_model(model)
        if provider == "alibaba":
            return self._get_alibaba_client()
        elif provider == "cohere":
            return self._get_cohere_client()
        return self._get_client()

    def get_embedding(
        self, text: str, model: str = None
    ) -> List[float]:
        """Get embedding for a single text.
        
        Now supports multiple providers based on model prefix.
        Routes to OpenRouter for "openai/*" models.
        Routes to DashScope for "alibaba/*" models.
        Routes to Cohere for "cohere/*" models.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            Embedding vector
            
        Raises:
            ValueError: If API key is not configured
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit is exceeded
            APIConnectionError: If network error occurs
            APIError: If API returns an error
        """
        if not text or not text.strip():
            return []
            
        model = model or self.DEFAULT_MODEL
        provider = self._get_provider_for_model(model)
        
        try:
            # Cohere uses different API
            if provider == "cohere":
                client = self._get_cohere_client()
                # Extract model name without prefix
                cohere_model = model.replace(self.COHERE_MODEL_PREFIX, "")
                
                response = client.embed(
                    texts=[text.strip()],
                    model=cohere_model,
                    input_type="search_document"
                )
                
                embedding = response.embeddings[0]
                dimension = len(embedding)
                
                logger.info(
                    f"Generated embedding using {provider}",
                    extra={
                        "model": model,
                        "provider": provider,
                        "text_length": len(text),
                        "dimension": dimension
                    }
                )
                
                return embedding
            
            # OpenAI-compatible providers (OpenRouter, Alibaba)
            client = self._get_client_for_model(model)
            
            response = client.embeddings.create(
                model=model,
                input=text.strip()
            )
            
            embedding = response.data[0].embedding
            dimension = len(embedding)
            
            logger.info(
                f"Generated embedding using {provider}",
                extra={
                    "model": model,
                    "provider": provider,
                    "text_length": len(text),
                    "dimension": dimension
                }
            )
            
            return embedding
            
        except ValueError as e:
            # Configuration error (missing API key)
            logger.error(
                f"Configuration error for {provider} provider: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "configuration"
                }
            )
            raise
            
        except AuthenticationError as e:
            # Invalid API key
            logger.error(
                f"Authentication failed for {provider} provider: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "authentication"
                }
            )
            api_key_name = {
                "alibaba": "DASHSCOPE_API_KEY",
                "cohere": "COHERE_API_KEY",
                "openrouter": "OPENROUTER_API_KEY"
            }.get(provider, "OPENROUTER_API_KEY")
            
            raise ValueError(
                f"Invalid API key for {provider} provider. "
                f"Please check your {api_key_name} environment variable."
            ) from e
            
        except RateLimitError as e:
            # Rate limit exceeded
            logger.error(
                f"Rate limit exceeded for {provider} provider: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "rate_limit"
                }
            )
            raise ValueError(
                f"Rate limit exceeded for {provider} provider. "
                f"Please try again later or reduce request frequency."
            ) from e
            
        except APIConnectionError as e:
            # Network error
            logger.error(
                f"Network error connecting to {provider} provider: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "network"
                }
            )
            raise ValueError(
                f"Failed to connect to {provider} provider. "
                f"Please check your network connection and try again."
            ) from e
            
        except APIError as e:
            # General API error
            logger.error(
                f"API error from {provider} provider: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "api",
                    "status_code": getattr(e, 'status_code', None)
                }
            )
            raise ValueError(
                f"API error from {provider} provider: {str(e)}"
            ) from e
            
        except Exception as e:
            # Unexpected error
            logger.error(
                f"Unexpected error generating embedding with {provider} provider: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "unexpected",
                    "exception_type": type(e).__name__
                }
            )
            raise

    def get_embeddings(
        self, texts: List[str], model: str = None
    ) -> List[List[float]]:
        """Get embeddings for multiple texts.
        
        Now supports multiple providers with appropriate batch sizes.
        Routes to OpenRouter for "openai/*" models (batch size: 100).
        Routes to DashScope for "alibaba/*" models (batch size: 10).
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If API key is not configured or API error occurs
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit is exceeded
            APIConnectionError: If network error occurs
            APIError: If API returns an error
        """
        if not texts:
            return []
            
        # Filter empty texts but track indices
        non_empty = [(i, t.strip()) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            return [[] for _ in texts]
            
        model = model or self.DEFAULT_MODEL
        provider = self._get_provider_for_model(model)
        
        # Determine batch size based on provider
        batch_size_map = {
            "alibaba": self.ALIBABA_BATCH_SIZE,
            "cohere": self.COHERE_BATCH_SIZE,
            "openrouter": self.BATCH_SIZE
        }
        batch_size = batch_size_map.get(provider, self.BATCH_SIZE)
        
        try:
            # Cohere uses different API
            if provider == "cohere":
                client = self._get_cohere_client()
                cohere_model = model.replace(self.COHERE_MODEL_PREFIX, "")
                
                # Process in batches
                all_embeddings = {}
                total_batches = (len(non_empty) + batch_size - 1) // batch_size
                
                logger.info(
                    f"Starting batch embedding generation with {provider}",
                    extra={
                        "model": model,
                        "provider": provider,
                        "total_texts": len(texts),
                        "non_empty_texts": len(non_empty),
                        "batch_size": batch_size,
                        "total_batches": total_batches
                    }
                )
                
                for batch_num, batch_start in enumerate(range(0, len(non_empty), batch_size), 1):
                    batch = non_empty[batch_start:batch_start + batch_size]
                    batch_texts = [t for _, t in batch]
                    batch_indices = [i for i, _ in batch]
                    
                    try:
                        response = client.embed(
                            texts=batch_texts,
                            model=cohere_model,
                            input_type="search_document"
                        )
                        
                        for j, embedding in enumerate(response.embeddings):
                            original_idx = batch_indices[j]
                            all_embeddings[original_idx] = embedding
                        
                        logger.debug(
                            f"Completed batch {batch_num}/{total_batches} for {provider}",
                            extra={
                                "model": model,
                                "provider": provider,
                                "batch_num": batch_num,
                                "batch_size": len(batch_texts)
                            }
                        )
                        
                    except Exception as e:
                        logger.error(
                            f"Failed to process batch {batch_num}/{total_batches} for {provider}: {e}",
                            extra={
                                "model": model,
                                "provider": provider,
                                "batch_num": batch_num,
                                "batch_start": batch_start,
                                "batch_size": len(batch_texts),
                                "error_type": type(e).__name__
                            }
                        )
                        raise
                
                # Reconstruct result with empty vectors for empty texts
                result = []
                for i in range(len(texts)):
                    if i in all_embeddings:
                        result.append(all_embeddings[i])
                    else:
                        result.append([])
                
                dimension = len(result[0]) if result and result[0] else 0
                logger.info(
                    f"Completed batch embedding generation with {provider}",
                    extra={
                        "model": model,
                        "provider": provider,
                        "total_texts": len(texts),
                        "successful_embeddings": len(all_embeddings),
                        "dimension": dimension
                    }
                )
                        
                return result
            
            # OpenAI-compatible providers (OpenRouter, Alibaba)
            client = self._get_client_for_model(model)
            
            # Process in batches
            all_embeddings = {}
            total_batches = (len(non_empty) + batch_size - 1) // batch_size
            
            logger.info(
                f"Starting batch embedding generation with {provider}",
                extra={
                    "model": model,
                    "provider": provider,
                    "total_texts": len(texts),
                    "non_empty_texts": len(non_empty),
                    "batch_size": batch_size,
                    "total_batches": total_batches
                }
            )
            
            for batch_num, batch_start in enumerate(range(0, len(non_empty), batch_size), 1):
                batch = non_empty[batch_start:batch_start + batch_size]
                batch_texts = [t for _, t in batch]
                batch_indices = [i for i, _ in batch]
                
                try:
                    response = client.embeddings.create(
                        model=model,
                        input=batch_texts
                    )
                    
                    for j, item in enumerate(response.data):
                        original_idx = batch_indices[j]
                        all_embeddings[original_idx] = item.embedding
                    
                    logger.debug(
                        f"Completed batch {batch_num}/{total_batches} for {provider}",
                        extra={
                            "model": model,
                            "provider": provider,
                            "batch_num": batch_num,
                            "batch_size": len(batch_texts)
                        }
                    )
                    
                except Exception as e:
                    # Log which batch failed
                    logger.error(
                        f"Failed to process batch {batch_num}/{total_batches} for {provider}: {e}",
                        extra={
                            "model": model,
                            "provider": provider,
                            "batch_num": batch_num,
                            "batch_start": batch_start,
                            "batch_size": len(batch_texts),
                            "error_type": type(e).__name__
                        }
                    )
                    raise
            
            # Reconstruct result with empty vectors for empty texts
            result = []
            for i in range(len(texts)):
                if i in all_embeddings:
                    result.append(all_embeddings[i])
                else:
                    result.append([])
            
            dimension = len(result[0]) if result and result[0] else 0
            logger.info(
                f"Completed batch embedding generation with {provider}",
                extra={
                    "model": model,
                    "provider": provider,
                    "total_texts": len(texts),
                    "successful_embeddings": len(all_embeddings),
                    "dimension": dimension
                }
            )
                    
            return result
            
        except ValueError as e:
            # Configuration error (missing API key)
            logger.error(
                f"Configuration error for {provider} provider during batch processing: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "configuration",
                    "total_texts": len(texts)
                }
            )
            raise
            
        except AuthenticationError as e:
            # Invalid API key
            api_key_name = {
                "alibaba": "DASHSCOPE_API_KEY",
                "cohere": "COHERE_API_KEY",
                "openrouter": "OPENROUTER_API_KEY"
            }.get(provider, "OPENROUTER_API_KEY")
            
            logger.error(
                f"Authentication failed for {provider} provider during batch processing: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "authentication",
                    "total_texts": len(texts)
                }
            )
            raise ValueError(
                f"Invalid API key for {provider} provider. "
                f"Please check your {api_key_name} environment variable."
            ) from e
            
        except RateLimitError as e:
            # Rate limit exceeded
            logger.error(
                f"Rate limit exceeded for {provider} provider during batch processing: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "rate_limit",
                    "total_texts": len(texts)
                }
            )
            raise ValueError(
                f"Rate limit exceeded for {provider} provider. "
                f"Please try again later or reduce batch size."
            ) from e
            
        except APIConnectionError as e:
            # Network error
            logger.error(
                f"Network error connecting to {provider} provider during batch processing: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "network",
                    "total_texts": len(texts)
                }
            )
            raise ValueError(
                f"Failed to connect to {provider} provider. "
                f"Please check your network connection and try again."
            ) from e
            
        except APIError as e:
            # General API error
            logger.error(
                f"API error from {provider} provider during batch processing: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "api",
                    "status_code": getattr(e, 'status_code', None),
                    "total_texts": len(texts)
                }
            )
            raise ValueError(
                f"API error from {provider} provider: {str(e)}"
            ) from e
            
        except Exception as e:
            # Unexpected error
            logger.error(
                f"Unexpected error during batch embedding generation with {provider} provider: {e}",
                extra={
                    "model": model,
                    "provider": provider,
                    "error_type": "unexpected",
                    "exception_type": type(e).__name__,
                    "total_texts": len(texts)
                }
            )
            raise

    def get_embedding_dimension(self, model: str = None) -> int:
        """Get embedding dimension for a model.
        
        Args:
            model: Embedding model
            
        Returns:
            Embedding dimension
        """
        model = model or self.DEFAULT_MODEL
        
        # Known dimensions
        dimensions = {
            "openai/text-embedding-3-small": 1536,
            "openai/text-embedding-3-large": 3072,
            "openai/text-embedding-ada-002": 1536,
            "alibaba/text-embedding-v4": 1024,
            "cohere/embed-multilingual-v3.0": 1024,
            "cohere/embed-multilingual-light-v3.0": 384,
        }
        
        return dimensions.get(model, 1536)


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get singleton EmbeddingService instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
