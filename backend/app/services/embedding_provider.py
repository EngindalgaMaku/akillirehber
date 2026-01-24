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
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            model: Optional model override
            
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
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings via OpenRouter API."""
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
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings via OpenAI API."""
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
        
        return providers
    
    @property
    def providers(self) -> List[EmbeddingProvider]:
        """Get list of configured providers."""
        return self._providers
    
    def get_available_providers(self) -> List[EmbeddingProvider]:
        """Get list of currently available providers."""
        return [p for p in self._providers if p.is_available()]
    
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings with automatic fallback and retry.
        
        Args:
            texts: List of texts to embed
            model: Optional model override
            
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
        
        last_error = None
        
        for provider in self._providers:
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
                embeddings = self._get_embeddings_with_retry(
                    provider, non_empty_texts, normalized_model
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

        return model
    
    def _get_embeddings_with_retry(
        self,
        provider: EmbeddingProvider,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings with exponential backoff retry."""
        last_error = None
        delay = self.config.retry_delay
        
        for attempt in range(self.config.max_retries):
            try:
                return provider.get_embeddings(texts, model)
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
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings in batches to avoid API limits.
        
        Args:
            texts: List of texts to embed
            model: Optional model override
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        batch_size = self.config.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch, model)
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
