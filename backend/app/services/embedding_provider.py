"""Embedding provider abstraction layer for semantic chunking.

This module provides a unified interface for multiple embedding providers
with fallback support, caching, and batch processing.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingProviderConfig:
    """Configuration for embedding providers."""
    primary_provider: str = "openrouter"  # "openrouter" or "openai"
    fallback_providers: List[str] = field(default_factory=list)
    openrouter_model: str = "openai/text-embedding-3-small"
    openai_model: str = "text-embedding-3-small"
    batch_size: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.
    
    All embedding providers must implement this interface to ensure
    consistent behavior across different backends.
    """
    
    @abstractmethod
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            model: Optional model override
            input_type: Context type - "query" for search queries,
                       "document" for indexing. Defaults to "document".
            
        Returns:
            List of embedding vectors (list of floats)
            
        Raises:
            EmbeddingProviderError: If embedding fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured.
        
        Returns:
            True if provider can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def get_cost_estimate(self, text_count: int, avg_tokens: int = 100) -> float:
        """Estimate cost for embedding texts.
        
        Args:
            text_count: Number of texts to embed
            avg_tokens: Average tokens per text
            
        Returns:
            Estimated cost in USD
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and identification."""
        pass


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider errors."""
    
    def __init__(self, message: str, provider: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class OpenRouterProvider(EmbeddingProvider):
    """OpenRouter API embedding provider.
    
    Uses OpenRouter's unified API to access various embedding models
    including OpenAI's text-embedding models.
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openai/text-embedding-3-small"
    
    # Pricing per 1M tokens (approximate)
    PRICING = {
        "openai/text-embedding-3-small": 0.02,
        "openai/text-embedding-3-large": 0.13,
        "openai/text-embedding-ada-002": 0.10,
        "qwen/qwen3-embedding-8b": 0.03,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key. If not provided, reads from
                     OPENROUTER_API_KEY environment variable.
        """
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = None
    
    def _ensure_client(self):
        """Initialize the OpenAI client for OpenRouter."""
        if self._client is None:
            if not self._api_key:
                raise EmbeddingProviderError(
                    "OPENROUTER_API_KEY not configured",
                    provider=self.name
                )
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self.BASE_URL
                )
            except ImportError as exc:
                raise EmbeddingProviderError(
                    "openai package required. Install with: pip install openai",
                    provider=self.name
                ) from exc
    
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings via OpenRouter API.
        
        Args:
            texts: List of texts to embed
            model: Model identifier
            input_type: Context type - "query" or "document" (ignored by OpenRouter)
        
        Returns:
            List of embedding vectors
        """
        self._ensure_client()
        
        model = model or self.DEFAULT_MODEL
        
        # Filter empty texts
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            return []
        
        try:
            response = self._client.embeddings.create(
                model=model,
                input=non_empty_texts
            )
            if not getattr(response, "data", None):
                raise EmbeddingProviderError(
                    "No embedding data received",
                    provider=self.name,
                    details={"model": model, "text_count": len(non_empty_texts)},
                )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingProviderError(
                f"OpenRouter embedding failed: {str(e)}",
                provider=self.name,
                details={"model": model, "text_count": len(non_empty_texts)}
            ) from e
    
    def is_available(self) -> bool:
        """Check if OpenRouter is configured."""
        return bool(self._api_key or os.environ.get("OPENROUTER_API_KEY"))
    
    def get_cost_estimate(self, text_count: int, avg_tokens: int = 100) -> float:
        """Estimate cost for OpenRouter embeddings."""
        total_tokens = text_count * avg_tokens
        price_per_token = self.PRICING.get(self.DEFAULT_MODEL, 0.02) / 1_000_000
        return total_tokens * price_per_token
    
    @property
    def name(self) -> str:
        return "openrouter"


class OpenAIProvider(EmbeddingProvider):
    """Direct OpenAI API embedding provider.
    
    Uses OpenAI's official API for embeddings. Supports text-embedding-3-small
    and text-embedding-3-large models.
    """
    
    DEFAULT_MODEL = "text-embedding-3-small"
    
    # Pricing per 1M tokens
    PRICING = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key. If not provided, reads from
                     OPENAI_API_KEY environment variable.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
    
    def _ensure_client(self):
        """Initialize the OpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise EmbeddingProviderError(
                    "OPENAI_API_KEY not configured",
                    provider=self.name
                )
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError as exc:
                raise EmbeddingProviderError(
                    "openai package required. Install with: pip install openai",
                    provider=self.name
                ) from exc
    
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings via OpenAI API.
        
        Args:
            texts: List of texts to embed
            model: Model identifier
            input_type: Context type - "query" or "document" (ignored by OpenAI)
        
        Returns:
            List of embedding vectors
        """
        self._ensure_client()
        
        model = model or self.DEFAULT_MODEL
        if model and "/" in model:
            model = model.split("/", 1)[1]
        
        # Filter empty texts
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            return []
        
        try:
            response = self._client.embeddings.create(
                model=model,
                input=non_empty_texts
            )
            if not getattr(response, "data", None):
                raise EmbeddingProviderError(
                    "No embedding data received",
                    provider=self.name,
                    details={"model": model, "text_count": len(non_empty_texts)},
                )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingProviderError(
                f"OpenAI embedding failed: {str(e)}",
                provider=self.name,
                details={"model": model, "text_count": len(non_empty_texts)}
            ) from e
    
    def is_available(self) -> bool:
        """Check if OpenAI is configured."""
        return bool(self._api_key or os.environ.get("OPENAI_API_KEY"))
    
    def get_cost_estimate(self, text_count: int, avg_tokens: int = 100) -> float:
        """Estimate cost for OpenAI embeddings."""
        total_tokens = text_count * avg_tokens
        price_per_token = self.PRICING.get(self.DEFAULT_MODEL, 0.02) / 1_000_000
        return total_tokens * price_per_token
    
    @property
    def name(self) -> str:
        return "openai"


class CohereProvider(EmbeddingProvider):
    """Cohere embedding provider.
    
    Uses Cohere's API for embeddings.
    """
    
    DEFAULT_MODEL = "embed-english-v3.0"
    BATCH_SIZE = 96  # Cohere supports up to 96 texts per batch
    
    # Pricing per 1M tokens (approximate)
    PRICING = {
        "embed-english-v3.0": 0.10,
        "embed-multilingual-v3.0": 0.10,
        "embed-english-light-v3.0": 0.10,
        "embed-multilingual-light-v3.0": 0.10,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Cohere provider.
        
        Args:
            api_key: Cohere API key. If not provided, reads from
                     COHERE_API_KEY environment variable.
        """
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        self._client = None
    
    def _ensure_client(self):
        """Initialize the Cohere client."""
        if self._client is None:
            if not self._api_key:
                raise EmbeddingProviderError(
                    "COHERE_API_KEY not configured",
                    provider=self.name
                )
            try:
                import cohere
                self._client = cohere.Client(api_key=self._api_key)
            except ImportError as exc:
                raise EmbeddingProviderError(
                    "cohere package required. Install with: pip install cohere",
                    provider=self.name
                ) from exc
    
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings via Cohere API.
        
        Args:
            texts: List of texts to embed
            model: Model identifier
            input_type: Context type - "query" for search queries,
                       "document" for indexing. Defaults to "document".
        
        Returns:
            List of embedding vectors
        """
        self._ensure_client()
        
        model = model or self.DEFAULT_MODEL
        # Remove cohere/ prefix if present
        if model.startswith("cohere/"):
            model = model.replace("cohere/", "")
        
        # Filter empty texts
        non_empty_texts = [t.strip() for t in texts if t and t.strip()]
        if not non_empty_texts:
            return []
        
        # Map input_type to Cohere's format
        cohere_input_type = "search_query" if input_type == "query" else "search_document"
        
        try:
            response = self._client.embed(
                texts=non_empty_texts,
                model=model,
                input_type=cohere_input_type
            )
            
            if not hasattr(response, 'embeddings') or not response.embeddings:
                raise EmbeddingProviderError(
                    "No embedding data received from Cohere API",
                    provider=self.name,
                    details={"model": model, "text_count": len(non_empty_texts)}
                )
            
            return response.embeddings
            
        except Exception as e:
            if isinstance(e, EmbeddingProviderError):
                raise
            raise EmbeddingProviderError(
                f"Cohere embedding failed: {str(e)}",
                provider=self.name,
                details={"model": model, "text_count": len(non_empty_texts)}
            ) from e
    
    def is_available(self) -> bool:
        """Check if Cohere is configured."""
        return bool(self._api_key or os.environ.get("COHERE_API_KEY"))
    
    def get_cost_estimate(self, text_count: int, avg_tokens: int = 100) -> float:
        """Estimate cost for Cohere embeddings."""
        total_tokens = text_count * avg_tokens
        price_per_token = self.PRICING.get(self.DEFAULT_MODEL, 0.10) / 1_000_000
        return total_tokens * price_per_token
    
    @property
    def name(self) -> str:
        return "cohere"


class JinaProvider(EmbeddingProvider):
    """Jina AI embedding provider.
    
    Uses Jina AI's API for embeddings.
    """
    
    BASE_URL = "https://api.jina.ai/v1"
    DEFAULT_MODEL = "jina-embeddings-v3"
    BATCH_SIZE = 100  # Jina supports up to 100 texts per batch
    
    # Pricing per 1M tokens (approximate)
    PRICING = {
        "jina-embeddings-v3": 0.02,
        "jina-embeddings-v2-base-en": 0.02,
        "jina-clip-v1": 0.02,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Jina provider.
        
        Args:
            api_key: Jina API key. If not provided, reads from
                     JINA_AI_API_KEY environment variable.
        """
        self._api_key = api_key or os.environ.get("JINA_AI_API_KEY")
    
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings via Jina AI API.
        
        Args:
            texts: List of texts to embed
            model: Model identifier
            input_type: Context type - "query" or "document" (ignored by Jina)
        
        Returns:
            List of embedding vectors
        """
        if not self._api_key:
            raise EmbeddingProviderError(
                "JINA_AI_API_KEY not configured",
                provider=self.name
            )
        
        model = model or self.DEFAULT_MODEL
        # Remove jina/ prefix if present
        if model.startswith("jina/"):
            model = model.replace("jina/", "")
        
        # Filter empty texts
        non_empty_texts = [t.strip() for t in texts if t and t.strip()]
        if not non_empty_texts:
            return []
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "input": non_empty_texts,
                "model": model,
            }
            
            response = requests.post(
                f"{self.BASE_URL}/embeddings",
                headers=headers,
                json=data,
                timeout=30,
            )
            
            response.raise_for_status()
            result = response.json()
            embeddings = result.get("data") or []
            
            if not embeddings:
                raise EmbeddingProviderError(
                    "No embedding data received from Jina AI API",
                    provider=self.name,
                    details={"model": model, "text_count": len(non_empty_texts)}
                )
            
            return [item.get("embedding") for item in embeddings]
            
        except ImportError as exc:
            raise EmbeddingProviderError(
                "requests package required. Install with: pip install requests",
                provider=self.name
            ) from exc
        except Exception as e:
            if isinstance(e, EmbeddingProviderError):
                raise
            raise EmbeddingProviderError(
                f"Jina embedding failed: {str(e)}",
                provider=self.name,
                details={"model": model, "text_count": len(non_empty_texts)}
            ) from e
    
    def is_available(self) -> bool:
        """Check if Jina is configured."""
        return bool(self._api_key or os.environ.get("JINA_AI_API_KEY"))
    
    def get_cost_estimate(self, text_count: int, avg_tokens: int = 100) -> float:
        """Estimate cost for Jina embeddings."""
        total_tokens = text_count * avg_tokens
        price_per_token = self.PRICING.get(self.DEFAULT_MODEL, 0.02) / 1_000_000
        return total_tokens * price_per_token
    
    @property
    def name(self) -> str:
        return "jina"


class AlibabaProvider(EmbeddingProvider):
    """Alibaba DashScope embedding provider.
    
    Uses Alibaba Cloud's DashScope API for embeddings.
    """
    
    BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "text-embedding-v3"
    BATCH_SIZE = 10  # Conservative batch size
    
    # Pricing per 1M tokens (approximate)
    PRICING = {
        "text-embedding-v3": 0.07,
        "text-embedding-v2": 0.07,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Alibaba provider.
        
        Args:
            api_key: DashScope API key. If not provided, reads from
                     DASHSCOPE_API_KEY environment variable.
        """
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self._client = None
    
    def _ensure_client(self):
        """Initialize the OpenAI client for DashScope."""
        if self._client is None:
            if not self._api_key:
                raise EmbeddingProviderError(
                    "DASHSCOPE_API_KEY not configured",
                    provider=self.name
                )
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self.BASE_URL
                )
            except ImportError as exc:
                raise EmbeddingProviderError(
                    "openai package required. Install with: pip install openai",
                    provider=self.name
                ) from exc
    
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings via Alibaba DashScope API.
        
        Args:
            texts: List of texts to embed
            model: Model identifier
            input_type: Context type - "query" or "document" (ignored by Alibaba)
        
        Returns:
            List of embedding vectors
        """
        self._ensure_client()
        
        model = model or self.DEFAULT_MODEL
        # Remove alibaba/ prefix if present
        if model.startswith("alibaba/"):
            model = model.replace("alibaba/", "")
        
        # Filter empty texts
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            return []
        
        try:
            response = self._client.embeddings.create(
                model=model,
                input=non_empty_texts
            )
            if not getattr(response, "data", None):
                raise EmbeddingProviderError(
                    "No embedding data received",
                    provider=self.name,
                    details={"model": model, "text_count": len(non_empty_texts)},
                )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingProviderError(
                f"Alibaba embedding failed: {str(e)}",
                provider=self.name,
                details={"model": model, "text_count": len(non_empty_texts)}
            ) from e
    
    def is_available(self) -> bool:
        """Check if Alibaba is configured."""
        return bool(self._api_key or os.environ.get("DASHSCOPE_API_KEY"))
    
    def get_cost_estimate(self, text_count: int, avg_tokens: int = 100) -> float:
        """Estimate cost for Alibaba embeddings."""
        total_tokens = text_count * avg_tokens
        price_per_token = self.PRICING.get(self.DEFAULT_MODEL, 0.07) / 1_000_000
        return total_tokens * price_per_token
    
    @property
    def name(self) -> str:
        return "alibaba"


class OllamaProvider(EmbeddingProvider):
    """Ollama local embedding provider.
    
    Uses local Ollama instance for embeddings.
    """
    
    BASE_URL = "http://host.docker.internal:11434"
    DEFAULT_MODEL = "nomic-embed-text"
    BATCH_SIZE = 10  # Conservative batch size
    
    # Model dimensions for consistency
    MODEL_DIMENSIONS = {
        "bge-m3": 1024,
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }
    
    def __init__(self, base_url: Optional[str] = None):
        """Initialize Ollama provider.
        
        Args:
            base_url: Ollama API base URL. If not provided, uses default.
        """
        self._base_url = base_url or os.environ.get("OLLAMA_BASE_URL", self.BASE_URL)
    
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings via Ollama API.
        
        Args:
            texts: List of texts to embed
            model: Model identifier
            input_type: Context type - "query" or "document" (ignored by Ollama)
        
        Returns:
            List of embedding vectors
        """
        model = model or self.DEFAULT_MODEL
        # Remove ollama/ prefix if present
        if model.startswith("ollama/"):
            model = model.replace("ollama/", "")
        
        # Filter empty texts
        non_empty_texts = [t.strip() for t in texts if t and t.strip()]
        if not non_empty_texts:
            return []
        
        try:
            import requests
            
            headers = {
                "Content-Type": "application/json",
            }
            
            embeddings = []
            max_retries = 3
            base_delay = 2
            
            for text in non_empty_texts:
                # Truncate text to avoid issues
                max_chars = int(os.environ.get("OLLAMA_EMBED_MAX_CHARS", "8000"))
                truncated_text = text[:max_chars]
                
                data = {
                    "model": model,
                    "prompt": truncated_text,
                }
                
                for attempt in range(max_retries):
                    try:
                        response = requests.post(
                            f"{self._base_url}/api/embeddings",
                            headers=headers,
                            json=data,
                            timeout=60,
                        )
                        
                        if response.status_code == 429:
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)
                                logger.warning(
                                    f"Ollama rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                                )
                                time.sleep(delay)
                                continue
                            else:
                                raise EmbeddingProviderError(
                                    "Ollama rate limit exceeded. Please try again later.",
                                    provider=self.name,
                                    details={"model": model, "text_count": len(non_empty_texts)}
                                )
                        
                        response.raise_for_status()
                        result = response.json()
                        
                        if "embedding" not in result:
                            raise EmbeddingProviderError(
                                "No embedding data received from Ollama API",
                                provider=self.name,
                                details={"model": model}
                            )
                        
                        embeddings.append(result["embedding"])
                        break
                        
                    except requests.exceptions.RequestException as e:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(
                                f"Ollama request failed, retrying in {delay}s (attempt {attempt + 1}/{max_retries}): {e}"
                            )
                            time.sleep(delay)
                            continue
                        else:
                            raise EmbeddingProviderError(
                                f"Ollama embedding failed: {str(e)}",
                                provider=self.name,
                                details={"model": model, "text_count": len(non_empty_texts)}
                            ) from e
            
            return embeddings
                        
        except ImportError as exc:
            raise EmbeddingProviderError(
                "requests package required. Install with: pip install requests",
                provider=self.name
            ) from exc
        except Exception as e:
            if isinstance(e, EmbeddingProviderError):
                raise
            raise EmbeddingProviderError(
                f"Ollama embedding failed: {str(e)}",
                provider=self.name,
                details={"model": model, "text_count": len(non_empty_texts)}
            ) from e
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import requests
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_cost_estimate(self, text_count: int, avg_tokens: int = 100) -> float:
        """Estimate cost for Ollama embeddings (free for local)."""
        return 0.0
    
    @property
    def name(self) -> str:
        return "ollama"


class VoyageProvider(EmbeddingProvider):
    """Voyage AI embedding provider.
    
    Uses Voyage AI's API for high-quality embeddings.
    """
    
    BASE_URL = "https://api.voyageai.com/v1"
    DEFAULT_MODEL = "voyage-3-large"
    BATCH_SIZE = 10  # Conservative batch size to avoid rate limits
    
    # Pricing per 1M tokens (approximate)
    PRICING = {
        "voyage-4-large": 0.12,
        "voyage-3-large": 0.12,
        "voyage-3-lite": 0.06,
        "voyage-3": 0.12,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Voyage provider.
        
        Args:
            api_key: Voyage API key. If not provided, reads from
                     VOYAGE_API_KEY environment variable.
        """
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
    
    # VoyageAI token limits per single text input by model
    MAX_TOKENS_PER_TEXT = {
        "voyage-3-large": 32000,
        "voyage-4-large": 32000,
        "voyage-3-lite": 16000,
        "voyage-3": 32000,
    }
    # VoyageAI max tokens per batch by model
    MAX_TOKENS_PER_BATCH = {
        "voyage-4-large": 120000,
        "voyage-3-large": 120000,
        "voyage-3-lite": 120000,
        "voyage-3": 120000,
    }
    # Approximate chars-per-token ratio (conservative — overestimates tokens)
    CHARS_PER_TOKEN_ESTIMATE = 3

    def _split_and_average_embedding(
        self, text: str, model: str, headers: dict, input_type: str
    ) -> List[float]:
        """Split a long text into smaller segments, embed each, and average.
        
        This avoids truncation-based data loss: every part of the text
        contributes to the final embedding vector.
        """
        import requests
        try:
            import numpy as np
        except ImportError:
            # Fallback: simple average without numpy
            np = None

        max_tokens = self.MAX_TOKENS_PER_TEXT.get(model, 32000)
        max_chars = max_tokens * self.CHARS_PER_TOKEN_ESTIMATE
        # Use 90% of limit to leave safety margin
        segment_max = int(max_chars * 0.9)

        # Split at sentence/word boundaries
        segments = []
        remaining = text
        while remaining:
            if len(remaining) <= segment_max:
                segments.append(remaining)
                break
            # Find a good split point
            cut = remaining[:segment_max]
            # Prefer sentence boundary
            for sep in [". ", ".\n", "? ", "! ", "\n\n", "\n", " "]:
                pos = cut.rfind(sep)
                if pos > segment_max * 0.5:
                    cut = remaining[:pos + len(sep)]
                    break
            else:
                cut = remaining[:segment_max]
            segments.append(cut.strip())
            remaining = remaining[len(cut):].strip()

        logger.info(
            f"VoyageAI: Text too long ({len(text)} chars), split into "
            f"{len(segments)} segments for averaged embedding"
        )

        # Embed each segment
        all_segment_embeddings = []
        for seg in segments:
            if not seg.strip():
                continue
            data = {
                "input": [seg],
                "model": model,
                "input_type": input_type,
            }
            response = requests.post(
                f"{self.BASE_URL}/embeddings",
                headers=headers,
                json=data,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            emb_data = result.get("data")
            if emb_data:
                all_segment_embeddings.append(emb_data[0].get("embedding"))

        if not all_segment_embeddings:
            return []

        # Weighted average by segment length (longer segments contribute more)
        weights = [len(s) for s in segments if s.strip()]
        if np is not None:
            arr = np.array(all_segment_embeddings)
            w = np.array(weights[:len(arr)], dtype=float)
            w /= w.sum()
            averaged = np.average(arr, axis=0, weights=w).tolist()
        else:
            # Simple average fallback without numpy
            dim = len(all_segment_embeddings[0])
            total_weight = sum(weights[:len(all_segment_embeddings)])
            averaged = [0.0] * dim
            for emb, weight in zip(all_segment_embeddings, weights):
                for j in range(dim):
                    averaged[j] += emb[j] * weight / total_weight
        return averaged

    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings via Voyage API.
        
        Args:
            texts: List of texts to embed
            model: Model identifier
            input_type: Context type - "query" for search queries,
                       "document" for indexing. Defaults to "document".
        
        Returns:
            List of embedding vectors
        """
        if not self._api_key:
            raise EmbeddingProviderError(
                "VOYAGE_API_KEY not configured",
                provider=self.name
            )
        
        model = model or self.DEFAULT_MODEL
        # Remove voyage/ prefix if present
        if model.startswith("voyage/"):
            model = model.replace("voyage/", "")
        
        # Filter empty texts
        non_empty_texts = [t.strip() for t in texts if t and t.strip()]
        if not non_empty_texts:
            return []
        
        try:
            import requests
            import random
            
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            
            # Check for texts that exceed token limits — handle them individually
            max_tokens = self.MAX_TOKENS_PER_TEXT.get(model, 32000)
            max_chars = max_tokens * self.CHARS_PER_TOKEN_ESTIMATE
            
            short_texts = []
            short_indices = []
            long_texts_map = {}  # index -> original text
            
            for i, text in enumerate(non_empty_texts):
                if len(text) > max_chars:
                    long_texts_map[i] = text
                else:
                    short_texts.append(text)
                    short_indices.append(i)
            
            # Pre-allocate results
            results = [None] * len(non_empty_texts)
            
            # Handle long texts via split-and-average (no data loss)
            for idx, long_text in long_texts_map.items():
                try:
                    averaged = self._split_and_average_embedding(
                        long_text, model, headers, input_type
                    )
                    results[idx] = averaged
                except Exception as e:
                    logger.error(
                        f"VoyageAI: Failed to embed long text (index {idx}, "
                        f"{len(long_text)} chars) via split-and-average: {e}"
                    )
                    raise EmbeddingProviderError(
                        f"Failed to embed long text ({len(long_text)} chars): {str(e)}",
                        provider=self.name,
                        details={"model": model, "text_length": len(long_text)}
                    ) from e
            
            # Handle normal-length texts — split into sub-batches by token limit
            if short_texts:
                max_batch_tokens = self.MAX_TOKENS_PER_BATCH.get(model, 120000)
                # Use 90% of limit as safety margin
                max_batch_chars = int(max_batch_tokens * self.CHARS_PER_TOKEN_ESTIMATE * 0.9)
                
                # Build sub-batches that fit within the token budget
                sub_batches = []  # list of (batch_texts, batch_original_indices)
                current_batch_texts = []
                current_batch_indices = []
                current_batch_chars = 0
                
                for text, orig_idx in zip(short_texts, short_indices):
                    text_chars = len(text)
                    # If adding this text would exceed the batch limit, flush
                    if current_batch_texts and (current_batch_chars + text_chars) > max_batch_chars:
                        sub_batches.append((current_batch_texts, current_batch_indices))
                        current_batch_texts = []
                        current_batch_indices = []
                        current_batch_chars = 0
                    current_batch_texts.append(text)
                    current_batch_indices.append(orig_idx)
                    current_batch_chars += text_chars
                
                if current_batch_texts:
                    sub_batches.append((current_batch_texts, current_batch_indices))
                
                if len(sub_batches) > 1:
                    logger.info(
                        f"VoyageAI: Split {len(short_texts)} texts into "
                        f"{len(sub_batches)} sub-batches to stay within "
                        f"{max_batch_tokens} token batch limit"
                    )
                
                # Send each sub-batch
                for batch_texts, batch_indices in sub_batches:
                    data = {
                        "input": batch_texts,
                        "model": model,
                        "input_type": input_type,
                    }
                    
                    # Retry logic for transient errors (429, 5xx, network)
                    # 400 errors are NOT retried — they indicate bad input data
                    max_retries = 8
                    base_delay = 5
                    
                    for attempt in range(max_retries):
                        try:
                            response = requests.post(
                                f"{self.BASE_URL}/embeddings",
                                headers=headers,
                                json=data,
                                timeout=60,
                            )
                            
                            if response.status_code == 429:
                                if attempt < max_retries - 1:
                                    delay = base_delay * (2 ** attempt) + random.uniform(1.0, 3.0)
                                    logger.warning(
                                        f"VoyageAI rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                                    )
                                    time.sleep(delay)
                                    continue
                                else:
                                    raise EmbeddingProviderError(
                                        "VoyageAI rate limit exceeded. Please try again later.",
                                        provider=self.name,
                                        details={"model": model, "text_count": len(batch_texts)}
                                    )
                            
                            # 400 Bad Request — do NOT retry, log the response body
                            if response.status_code == 400:
                                error_body = ""
                                try:
                                    error_body = response.text
                                except Exception:
                                    error_body = "(could not read response body)"
                                
                                text_lengths = [len(t) for t in batch_texts]
                                logger.error(
                                    f"VoyageAI 400 Bad Request (not retrying). "
                                    f"Model: {model}, text_count: {len(batch_texts)}, "
                                    f"text_lengths: min={min(text_lengths)}, max={max(text_lengths)}, "
                                    f"total_chars: {sum(text_lengths)}, "
                                    f"response: {error_body[:500]}"
                                )
                                raise EmbeddingProviderError(
                                    f"VoyageAI rejected the request (400 Bad Request): {error_body[:300]}. "
                                    f"Text count: {len(batch_texts)}, max text length: {max(text_lengths)} chars.",
                                    provider=self.name,
                                    details={
                                        "model": model,
                                        "text_count": len(batch_texts),
                                        "text_lengths": text_lengths,
                                        "response_body": error_body[:500],
                                    }
                                )
                            
                            response.raise_for_status()
                            result = response.json()
                            embeddings = result.get("data") or []
                            
                            if not embeddings:
                                raise EmbeddingProviderError(
                                    "No embedding data received from Voyage API",
                                    provider=self.name,
                                    details={"model": model, "text_count": len(batch_texts)}
                                )
                            
                            # Place batch results into correct positions
                            batch_embeddings = [item.get("embedding") for item in embeddings]
                            for i, emb in zip(batch_indices, batch_embeddings):
                                results[i] = emb
                            break  # Success, exit retry loop
                            
                        except requests.exceptions.RequestException as e:
                            # Don't retry client errors (4xx) other than 429
                            status_code = getattr(getattr(e, 'response', None), 'status_code', None)
                            if status_code and 400 <= status_code < 500 and status_code != 429:
                                logger.error(
                                    f"VoyageAI client error {status_code} (not retrying): {e}"
                                )
                                raise EmbeddingProviderError(
                                    f"Voyage embedding failed with client error {status_code}: {str(e)}",
                                    provider=self.name,
                                    details={"model": model, "text_count": len(batch_texts)}
                                ) from e
                            
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt) + random.uniform(1.0, 3.0)
                                logger.warning(
                                    f"VoyageAI request failed, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                                )
                                time.sleep(delay)
                                continue
                            else:
                                raise EmbeddingProviderError(
                                    f"Voyage embedding failed: {str(e)}",
                                    provider=self.name,
                                    details={"model": model, "text_count": len(batch_texts)}
                                ) from e
            
            return results
                        
        except ImportError as exc:
            raise EmbeddingProviderError(
                "requests package required. Install with: pip install requests",
                provider=self.name
            ) from exc
        except Exception as e:
            if isinstance(e, EmbeddingProviderError):
                raise
            raise EmbeddingProviderError(
                f"Voyage embedding failed: {str(e)}",
                provider=self.name,
                details={"model": model, "text_count": len(non_empty_texts)}
            ) from e
    
    def is_available(self) -> bool:
        """Check if Voyage is configured."""
        return bool(self._api_key or os.environ.get("VOYAGE_API_KEY"))
    
    def get_cost_estimate(self, text_count: int, avg_tokens: int = 100) -> float:
        """Estimate cost for Voyage embeddings."""
        total_tokens = text_count * avg_tokens
        price_per_token = self.PRICING.get(self.DEFAULT_MODEL, 0.12) / 1_000_000
        return total_tokens * price_per_token
    
    @property
    def name(self) -> str:
        return "voyage"


class EmbeddingProviderManager:
    """Manage multiple embedding providers with fallback and retry logic.
    
    Features:
    - Automatic fallback to secondary providers on failure
    - Exponential backoff retry logic
    - Provider health tracking
    - Batch processing support
    """
    
    def __init__(
        self,
        providers: Optional[List[EmbeddingProvider]] = None,
        config: Optional[EmbeddingProviderConfig] = None
    ):
        """Initialize the provider manager.
        
        Args:
            providers: List of embedding providers in priority order.
                      If not provided, creates default providers based on config.
            config: Configuration for provider behavior.
        """
        self.config = config or EmbeddingProviderConfig()
        self._providers = providers or self._create_default_providers()
        self._provider_health: Dict[str, bool] = {}
        self._last_failure_time: Dict[str, float] = {}
    
    def _create_default_providers(self) -> List[EmbeddingProvider]:
        """Create default providers based on configuration."""
        providers = []
        
        # Add primary provider
        if self.config.primary_provider == "openrouter":
            providers.append(OpenRouterProvider())
        elif self.config.primary_provider == "openai":
            providers.append(OpenAIProvider())
        elif self.config.primary_provider == "voyage":
            providers.append(VoyageProvider())
        elif self.config.primary_provider == "ollama":
            providers.append(OllamaProvider())
        elif self.config.primary_provider == "cohere":
            providers.append(CohereProvider())
        elif self.config.primary_provider == "jina":
            providers.append(JinaProvider())
        elif self.config.primary_provider == "alibaba":
            providers.append(AlibabaProvider())
        
        # Add fallback providers
        for fallback in self.config.fallback_providers:
            if fallback == "openrouter" and not any(
                isinstance(p, OpenRouterProvider) for p in providers
            ):
                providers.append(OpenRouterProvider())
            elif fallback == "openai" and not any(
                isinstance(p, OpenAIProvider) for p in providers
            ):
                providers.append(OpenAIProvider())
            elif fallback == "voyage" and not any(
                isinstance(p, VoyageProvider) for p in providers
            ):
                providers.append(VoyageProvider())
            elif fallback == "ollama" and not any(
                isinstance(p, OllamaProvider) for p in providers
            ):
                providers.append(OllamaProvider())
            elif fallback == "cohere" and not any(
                isinstance(p, CohereProvider) for p in providers
            ):
                providers.append(CohereProvider())
            elif fallback == "jina" and not any(
                isinstance(p, JinaProvider) for p in providers
            ):
                providers.append(JinaProvider())
            elif fallback == "alibaba" and not any(
                isinstance(p, AlibabaProvider) for p in providers
            ):
                providers.append(AlibabaProvider())
        
        return providers
    
    @property
    def providers(self) -> List[EmbeddingProvider]:
        """Get list of configured providers."""
        return self._providers
    
    def get_available_providers(self) -> List[EmbeddingProvider]:
        """Get list of currently available providers."""
        return [p for p in self._providers if p.is_available()]
    
    def _get_providers_for_model(self, model: Optional[str]) -> List[EmbeddingProvider]:
        """Get providers in optimal order for the given model.
        
        Args:
            model: Model identifier (e.g., "voyage/voyage-4-large")
            
        Returns:
            List of providers to try, in order of preference
        """
        if not model:
            return self._providers
        
        # Determine the best provider for this model
        preferred_provider = None
        
        if model.startswith("voyage/") or model.startswith("voyage-"):
            preferred_provider = VoyageProvider
        elif model.startswith("cohere/"):
            preferred_provider = CohereProvider
        elif model.startswith("jina/"):
            preferred_provider = JinaProvider
        elif model.startswith("alibaba/"):
            preferred_provider = AlibabaProvider
        elif model.startswith("ollama/"):
            preferred_provider = OllamaProvider
        elif model.startswith("openai/"):
            # OpenAI models can be served by both OpenAI and OpenRouter
            # Prefer OpenAI if available, fallback to OpenRouter
            preferred_provider = OpenAIProvider
        
        if not preferred_provider:
            return self._providers
        
        # Reorder providers: preferred first, then others
        preferred = [p for p in self._providers if isinstance(p, preferred_provider)]
        others = [p for p in self._providers if not isinstance(p, preferred_provider)]
        
        return preferred + others
    
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings with automatic fallback and retry.
        
        Args:
            texts: List of texts to embed
            model: Optional model override
            input_type: Context type - "query" for search queries,
                       "document" for indexing. Defaults to "document".
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingProviderError: If all providers fail
        """
        if not texts:
            return []
        
        # Filter empty texts but track indices for result mapping
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]
        
        if not non_empty_texts:
            return []
        
        # Smart provider selection based on model prefix
        providers_to_try = self._get_providers_for_model(model)
        
        logger.info(f"[EMBEDDING DEBUG] Requested model: {model}")
        logger.info(f"[EMBEDDING DEBUG] Providers to try: {[p.name for p in providers_to_try]}")
        
        last_error = None
        
        for provider in providers_to_try:
            if not provider.is_available():
                logger.debug(f"Provider {provider.name} not available, skipping")
                continue
            
            # Check if provider recently failed (circuit breaker pattern)
            if self._is_provider_in_cooldown(provider.name):
                logger.debug(
                    f"Provider {provider.name} in cooldown, skipping"
                )
                continue
            
            try:
                normalized_model = self._normalize_model_for_provider(provider, model)
                logger.info(f"[EMBEDDING DEBUG] Trying provider: {provider.name} with model: {normalized_model}")
                embeddings = self._get_embeddings_with_retry(
                    provider, non_empty_texts, normalized_model, input_type
                )
                
                # Mark provider as healthy
                self._provider_health[provider.name] = True
                
                logger.debug(
                    f"Successfully got {len(embeddings)} embeddings "
                    f"from {provider.name}"
                )
                return embeddings
                
            except EmbeddingProviderError as e:
                last_error = e
                self._mark_provider_failed(provider.name)
                logger.warning(
                    f"Provider {provider.name} failed: {e}, "
                    f"trying next provider"
                )
                continue
        
        # All providers failed
        raise EmbeddingProviderError(
            f"All embedding providers failed. Last error: {last_error}",
            provider="manager",
            details={"providers_tried": [p.name for p in self._providers]}
        )

    def _normalize_model_for_provider(
        self, provider: EmbeddingProvider, model: Optional[str]
    ) -> Optional[str]:
        if not model:
            return model

        if isinstance(provider, OpenAIProvider):
            if "/" in model:
                return model.split("/", 1)[1]
            return model

        if isinstance(provider, OpenRouterProvider):
            if "/" not in model:
                return f"openai/{model}"
            return model

        if isinstance(provider, VoyageProvider):
            # Remove voyage/ prefix if present
            if model.startswith("voyage/"):
                return model.replace("voyage/", "")
            return model

        if isinstance(provider, CohereProvider):
            # Remove cohere/ prefix if present
            if model.startswith("cohere/"):
                return model.replace("cohere/", "")
            return model

        if isinstance(provider, JinaProvider):
            # Remove jina/ prefix if present
            if model.startswith("jina/"):
                return model.replace("jina/", "")
            return model

        if isinstance(provider, AlibabaProvider):
            # Remove alibaba/ prefix if present
            if model.startswith("alibaba/"):
                return model.replace("alibaba/", "")
            return model

        if isinstance(provider, OllamaProvider):
            # Remove ollama/ prefix if present
            if model.startswith("ollama/"):
                return model.replace("ollama/", "")
            return model

        return model
    
    def _get_embeddings_with_retry(
        self,
        provider: EmbeddingProvider,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings with exponential backoff retry.
        
        Args:
            provider: Provider to use
            texts: List of texts to embed
            model: Optional model override
            input_type: Context type - "query" or "document"
        
        Returns:
            List of embedding vectors
        """
        last_error = None
        delay = self.config.retry_delay
        
        for attempt in range(self.config.max_retries):
            try:
                return provider.get_embeddings(texts, model, input_type)
            except EmbeddingProviderError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    logger.debug(
                        f"Retry {attempt + 1}/{self.config.max_retries} "
                        f"for {provider.name}, waiting {delay}s"
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
        
        raise last_error
    
    def _is_provider_in_cooldown(self, provider_name: str) -> bool:
        """Check if provider is in cooldown after failure."""
        if provider_name not in self._last_failure_time:
            return False
        
        # 30 second cooldown after failure
        cooldown_period = 30.0
        elapsed = time.time() - self._last_failure_time[provider_name]
        return elapsed < cooldown_period
    
    def _mark_provider_failed(self, provider_name: str):
        """Mark a provider as failed."""
        self._provider_health[provider_name] = False
        self._last_failure_time[provider_name] = time.time()
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings in batches to avoid API limits.
        
        Args:
            texts: List of texts to embed
            model: Optional model override
            input_type: Context type - "query" or "document"
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        batch_size = self.config.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch, model, input_type)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers.
        
        Returns:
            Dict mapping provider name to status info
        """
        status = {}
        for provider in self._providers:
            status[provider.name] = {
                "available": provider.is_available(),
                "healthy": self._provider_health.get(provider.name, True),
                "in_cooldown": self._is_provider_in_cooldown(provider.name),
            }
        return status
    
    def reset_provider_health(self):
        """Reset health status for all providers."""
        self._provider_health.clear()
        self._last_failure_time.clear()
