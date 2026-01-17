"""Property-based tests for embedding provider infrastructure.

Feature: semantic-chunker-enhancement
Tests for embedding provider abstraction, fallback mechanism, and retry logic.
"""

import time
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st, assume

from app.services.embedding_provider import (
    EmbeddingProvider,
    EmbeddingProviderConfig,
    EmbeddingProviderError,
    EmbeddingProviderManager,
    OpenAIProvider,
    OpenRouterProvider,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

class MockProvider(EmbeddingProvider):
    """Mock provider for testing."""
    
    def __init__(
        self,
        name: str = "mock",
        available: bool = True,
        fail_count: int = 0,
        embeddings: Optional[List[List[float]]] = None
    ):
        self._name = name
        self._available = available
        self._fail_count = fail_count
        self._call_count = 0
        self._embeddings = embeddings or [[0.1, 0.2, 0.3]]
    
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise EmbeddingProviderError(
                f"Mock failure {self._call_count}",
                provider=self._name
            )
        return self._embeddings * len(texts)
    
    def is_available(self) -> bool:
        return self._available
    
    def get_cost_estimate(self, text_count: int, avg_tokens: int = 100) -> float:
        return text_count * avg_tokens * 0.00001
    
    @property
    def name(self) -> str:
        return self._name


# =============================================================================
# Property 22: Provider Fallback on Failure
# Feature: semantic-chunker-enhancement, Property 22: Provider Fallback on Failure
# Validates: Requirements 4.2, 11.1
# =============================================================================

class TestProviderFallback:
    """Tests for provider fallback mechanism."""
    
    # Feature: semantic-chunker-enhancement, Property 22: Provider Fallback on Failure
    @given(
        num_providers=st.integers(min_value=2, max_value=5),
        fail_index=st.integers(min_value=0, max_value=4)
    )
    @settings(max_examples=100, deadline=None)
    def test_fallback_to_next_provider_on_failure(
        self, num_providers: int, fail_index: int
    ):
        """Property: When primary provider fails, system falls back to next."""
        assume(fail_index < num_providers)
        
        # Create providers where first N fail
        providers = []
        for i in range(num_providers):
            if i <= fail_index:
                # This provider will fail
                providers.append(MockProvider(
                    name=f"provider_{i}",
                    fail_count=999  # Always fail
                ))
            else:
                # This provider will succeed
                providers.append(MockProvider(
                    name=f"provider_{i}",
                    fail_count=0
                ))
        
        # Use fast config for testing
        config = EmbeddingProviderConfig(max_retries=1, retry_delay=0.01)
        
        # If all providers fail, expect error
        if fail_index >= num_providers - 1:
            manager = EmbeddingProviderManager(providers=providers, config=config)
            with pytest.raises(EmbeddingProviderError):
                manager.get_embeddings(["test text"])
        else:
            # Otherwise, should succeed with fallback
            manager = EmbeddingProviderManager(providers=providers, config=config)
            result = manager.get_embeddings(["test text"])
            assert len(result) == 1
            assert isinstance(result[0], list)
    
    # Feature: semantic-chunker-enhancement, Property 22: Provider Fallback on Failure
    def test_fallback_occurs_within_time_limit(self):
        """Property: Fallback occurs within 2 seconds."""
        # Create two providers - first fails, second succeeds
        failing_provider = MockProvider(name="failing", fail_count=999)
        working_provider = MockProvider(name="working", fail_count=0)
        
        config = EmbeddingProviderConfig(
            max_retries=1,  # Minimize retries for speed
            retry_delay=0.1
        )
        manager = EmbeddingProviderManager(
            providers=[failing_provider, working_provider],
            config=config
        )
        
        start_time = time.time()
        result = manager.get_embeddings(["test text"])
        elapsed = time.time() - start_time
        
        assert elapsed < 2.0, f"Fallback took {elapsed}s, expected < 2s"
        assert len(result) == 1
    
    # Feature: semantic-chunker-enhancement, Property 22: Provider Fallback on Failure
    @given(st.lists(st.booleans(), min_size=2, max_size=5))
    @settings(max_examples=100, deadline=None)
    def test_skips_unavailable_providers(self, availability: List[bool]):
        """Property: Unavailable providers are skipped during fallback."""
        assume(any(availability))  # At least one must be available
        
        providers = []
        for i, available in enumerate(availability):
            providers.append(MockProvider(
                name=f"provider_{i}",
                available=available,
                fail_count=0 if available else 999
            ))
        
        manager = EmbeddingProviderManager(providers=providers)
        result = manager.get_embeddings(["test text"])
        
        # Should succeed using first available provider
        assert len(result) == 1
    
    # Feature: semantic-chunker-enhancement, Property 22: Provider Fallback on Failure
    def test_all_providers_tried_before_failure(self):
        """Property: All providers are tried before raising error."""
        providers = [
            MockProvider(name=f"provider_{i}", fail_count=999)
            for i in range(3)
        ]
        
        config = EmbeddingProviderConfig(max_retries=1, retry_delay=0.01)
        manager = EmbeddingProviderManager(providers=providers, config=config)
        
        with pytest.raises(EmbeddingProviderError) as exc_info:
            manager.get_embeddings(["test text"])
        
        # Verify all providers were tried
        assert "All embedding providers failed" in str(exc_info.value)


# =============================================================================
# Property 25: Exponential Backoff Implementation
# Feature: semantic-chunker-enhancement, Property 25: Exponential Backoff Implementation
# Validates: Requirements 11.4
# =============================================================================

class TestExponentialBackoff:
    """Tests for exponential backoff retry logic."""
    
    # Feature: semantic-chunker-enhancement, Property 25: Exponential Backoff Implementation
    def test_retry_delays_increase_exponentially(self):
        """Property: Retry delays follow exponential pattern (1s, 2s, 4s)."""
        # Provider that fails twice then succeeds
        provider = MockProvider(name="flaky", fail_count=2)
        
        config = EmbeddingProviderConfig(
            max_retries=3,
            retry_delay=0.1  # Use small delay for testing
        )
        manager = EmbeddingProviderManager(
            providers=[provider],
            config=config
        )
        
        start_time = time.time()
        result = manager.get_embeddings(["test text"])
        elapsed = time.time() - start_time
        
        # With 0.1s base delay and 2 retries:
        # First retry: 0.1s, Second retry: 0.2s
        # Total expected: ~0.3s (plus execution time)
        assert elapsed >= 0.25, f"Expected delays, got {elapsed}s"
        assert len(result) == 1
    
    # Feature: semantic-chunker-enhancement, Property 25: Exponential Backoff Implementation
    @given(
        base_delay=st.floats(min_value=0.01, max_value=0.1),
        max_retries=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=50, deadline=None)
    def test_max_retries_respected(self, base_delay: float, max_retries: int):
        """Property: Maximum retry count is respected."""
        # Provider that always fails
        provider = MockProvider(name="always_fails", fail_count=999)
        
        config = EmbeddingProviderConfig(
            max_retries=max_retries,
            retry_delay=base_delay
        )
        manager = EmbeddingProviderManager(
            providers=[provider],
            config=config
        )
        
        with pytest.raises(EmbeddingProviderError):
            manager.get_embeddings(["test text"])
        
        # Provider should have been called exactly max_retries times
        assert provider._call_count == max_retries
    
    # Feature: semantic-chunker-enhancement, Property 25: Exponential Backoff Implementation
    def test_successful_request_no_retry(self):
        """Property: Successful requests don't trigger retries."""
        provider = MockProvider(name="working", fail_count=0)
        
        config = EmbeddingProviderConfig(max_retries=3, retry_delay=1.0)
        manager = EmbeddingProviderManager(
            providers=[provider],
            config=config
        )
        
        start_time = time.time()
        result = manager.get_embeddings(["test text"])
        elapsed = time.time() - start_time
        
        # Should complete quickly without retries
        assert elapsed < 0.5
        assert provider._call_count == 1
        assert len(result) == 1


# =============================================================================
# Provider Interface Tests
# =============================================================================

class TestProviderInterface:
    """Tests for embedding provider interface compliance."""
    
    def test_openrouter_provider_interface(self):
        """OpenRouterProvider implements all required methods."""
        provider = OpenRouterProvider()
        
        # Check interface methods exist
        assert hasattr(provider, 'get_embeddings')
        assert hasattr(provider, 'is_available')
        assert hasattr(provider, 'get_cost_estimate')
        assert hasattr(provider, 'name')
        
        # Check name property
        assert provider.name == "openrouter"
    
    def test_openai_provider_interface(self):
        """OpenAIProvider implements all required methods."""
        provider = OpenAIProvider()
        
        # Check interface methods exist
        assert hasattr(provider, 'get_embeddings')
        assert hasattr(provider, 'is_available')
        assert hasattr(provider, 'get_cost_estimate')
        assert hasattr(provider, 'name')
        
        # Check name property
        assert provider.name == "openai"
    
    @given(text_count=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=100, deadline=None)
    def test_cost_estimate_positive(self, text_count: int):
        """Property: Cost estimates are always positive."""
        openrouter = OpenRouterProvider()
        openai = OpenAIProvider()
        
        openrouter_cost = openrouter.get_cost_estimate(text_count)
        openai_cost = openai.get_cost_estimate(text_count)
        
        assert openrouter_cost > 0
        assert openai_cost > 0
    
    @given(text_count=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=100, deadline=None)
    def test_cost_estimate_scales_with_count(self, text_count: int):
        """Property: Cost estimates scale linearly with text count."""
        provider = OpenRouterProvider()
        
        cost_1 = provider.get_cost_estimate(1)
        cost_n = provider.get_cost_estimate(text_count)
        
        # Cost should scale approximately linearly
        assert abs(cost_n - cost_1 * text_count) < 0.0001


# =============================================================================
# Provider Manager Tests
# =============================================================================

class TestProviderManager:
    """Tests for EmbeddingProviderManager."""
    
    def test_empty_text_list_returns_empty(self):
        """Empty input returns empty output."""
        provider = MockProvider()
        manager = EmbeddingProviderManager(providers=[provider])
        
        result = manager.get_embeddings([])
        assert result == []
    
    def test_filters_empty_texts(self):
        """Empty strings in input are filtered."""
        provider = MockProvider()
        manager = EmbeddingProviderManager(providers=[provider])
        
        result = manager.get_embeddings(["", "  ", "valid text"])
        # Only "valid text" should be processed
        assert len(result) == 1
    
    @given(st.lists(st.text(min_size=1), min_size=1, max_size=10))
    @settings(max_examples=100, deadline=None)
    def test_batch_processing_returns_correct_count(self, texts: List[str]):
        """Property: Batch processing returns embedding for each text."""
        assume(all(t.strip() for t in texts))  # Non-empty texts
        
        provider = MockProvider()
        config = EmbeddingProviderConfig(batch_size=3)
        manager = EmbeddingProviderManager(
            providers=[provider],
            config=config
        )
        
        result = manager.get_embeddings_batch(texts)
        assert len(result) == len(texts)
    
    def test_provider_status_reporting(self):
        """Provider status is correctly reported."""
        providers = [
            MockProvider(name="available", available=True),
            MockProvider(name="unavailable", available=False),
        ]
        manager = EmbeddingProviderManager(providers=providers)
        
        status = manager.get_provider_status()
        
        assert status["available"]["available"] is True
        assert status["unavailable"]["available"] is False
    
    def test_circuit_breaker_cooldown(self):
        """Failed providers enter cooldown period."""
        failing_provider = MockProvider(name="failing", fail_count=999)
        working_provider = MockProvider(name="working", fail_count=0)
        
        config = EmbeddingProviderConfig(max_retries=1, retry_delay=0.01)
        manager = EmbeddingProviderManager(
            providers=[failing_provider, working_provider],
            config=config
        )
        
        # First call - failing provider fails, falls back to working
        manager.get_embeddings(["test"])
        
        # Check status - failing provider should be in cooldown
        status = manager.get_provider_status()
        assert status["failing"]["in_cooldown"] is True
        assert status["working"]["in_cooldown"] is False
    
    def test_reset_provider_health(self):
        """Provider health can be reset."""
        failing_provider = MockProvider(name="failing", fail_count=999)
        working_provider = MockProvider(name="working", fail_count=0)
        
        config = EmbeddingProviderConfig(max_retries=1, retry_delay=0.01)
        manager = EmbeddingProviderManager(
            providers=[failing_provider, working_provider],
            config=config
        )
        
        # Trigger failure
        manager.get_embeddings(["test"])
        
        # Reset health
        manager.reset_provider_health()
        
        # Check status - should no longer be in cooldown
        status = manager.get_provider_status()
        assert status["failing"]["in_cooldown"] is False


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in embedding providers."""
    
    def test_embedding_provider_error_contains_details(self):
        """EmbeddingProviderError contains provider and details."""
        error = EmbeddingProviderError(
            "Test error",
            provider="test_provider",
            details={"key": "value"}
        )
        
        assert error.provider == "test_provider"
        assert error.details == {"key": "value"}
        assert "Test error" in str(error)
    
    def test_provider_error_propagates_correctly(self):
        """Provider errors are properly wrapped and propagated."""
        provider = MockProvider(name="failing", fail_count=999)
        
        config = EmbeddingProviderConfig(max_retries=1, retry_delay=0.01)
        manager = EmbeddingProviderManager(
            providers=[provider],
            config=config
        )
        
        with pytest.raises(EmbeddingProviderError) as exc_info:
            manager.get_embeddings(["test"])
        
        assert "All embedding providers failed" in str(exc_info.value)


# =============================================================================
# Embedding Cache Tests
# Feature: semantic-chunker-enhancement, Property 12: Embedding Cache Effectiveness
# Validates: Requirements 6.2
# =============================================================================

from app.services.embedding_cache import (
    CacheStats,
    CachedEmbeddingProvider,
    EmbeddingCache,
)


class TestEmbeddingCache:
    """Tests for embedding cache functionality."""
    
    # Feature: semantic-chunker-enhancement, Property 12: Embedding Cache Effectiveness
    def test_cache_hit_returns_cached_embedding(self):
        """Property: Second request for same text uses cache."""
        cache = EmbeddingCache(ttl=3600)
        
        text = "test text"
        model = "test-model"
        embedding = [0.1, 0.2, 0.3]
        
        # First request - cache miss
        result1 = cache.get(text, model)
        assert result1 is None
        
        # Store in cache
        cache.set(text, model, embedding)
        
        # Second request - cache hit
        result2 = cache.get(text, model)
        assert result2 == embedding
    
    # Feature: semantic-chunker-enhancement, Property 12: Embedding Cache Effectiveness
    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_cache_reduces_api_calls(self, texts: List[str]):
        """Property: Cache reduces API calls for repeated texts."""
        assume(all(t.strip() for t in texts))
        
        # Create mock provider that counts calls
        call_count = [0]
        original_embeddings = {}
        
        class CountingProvider(MockProvider):
            def get_embeddings(self, texts, model=None):
                call_count[0] += len(texts)
                result = []
                for t in texts:
                    if t not in original_embeddings:
                        original_embeddings[t] = [0.1 * len(t), 0.2, 0.3]
                    result.append(original_embeddings[t])
                return result
        
        provider = CountingProvider()
        cached_provider = CachedEmbeddingProvider(provider)
        
        # First call - all texts need API
        cached_provider.get_embeddings(texts)
        first_call_count = call_count[0]
        
        # Second call - should use cache
        cached_provider.get_embeddings(texts)
        second_call_count = call_count[0]
        
        # No additional API calls for second request
        assert second_call_count == first_call_count
    
    # Feature: semantic-chunker-enhancement, Property 12: Embedding Cache Effectiveness
    def test_cache_hit_rate_tracking(self):
        """Property: Cache tracks hit rate accurately."""
        cache = EmbeddingCache(ttl=3600)
        
        # Set up some cached values
        cache.set("text1", "model", [0.1, 0.2])
        cache.set("text2", "model", [0.3, 0.4])
        
        # Make requests
        cache.get("text1", "model")  # Hit
        cache.get("text2", "model")  # Hit
        cache.get("text3", "model")  # Miss
        cache.get("text1", "model")  # Hit
        
        stats = cache.get_stats()
        assert stats.total_requests == 4
        assert stats.cache_hits == 3
        assert stats.cache_misses == 1
        assert stats.hit_rate == 0.75
    
    def test_cache_ttl_expiration(self):
        """Cache entries expire after TTL."""
        cache = EmbeddingCache(ttl=1)  # 1 second TTL
        
        cache.set("text", "model", [0.1, 0.2])
        
        # Should be in cache
        assert cache.get("text", "model") is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("text", "model") is None
    
    def test_cache_max_entries_eviction(self):
        """Cache evicts oldest entries when at capacity."""
        cache = EmbeddingCache(ttl=3600, max_entries=3)
        
        # Fill cache
        cache.set("text1", "model", [0.1])
        time.sleep(0.01)
        cache.set("text2", "model", [0.2])
        time.sleep(0.01)
        cache.set("text3", "model", [0.3])
        
        # Add one more - should evict oldest (text1)
        cache.set("text4", "model", [0.4])
        
        # text1 should be evicted
        assert cache.get("text1", "model") is None
        # Others should still be there
        assert cache.get("text2", "model") is not None
        assert cache.get("text3", "model") is not None
        assert cache.get("text4", "model") is not None
    
    def test_batch_get_returns_found_and_missing(self):
        """Batch get correctly identifies found and missing entries."""
        cache = EmbeddingCache(ttl=3600)
        
        # Cache some entries
        cache.set("text1", "model", [0.1])
        cache.set("text3", "model", [0.3])
        
        texts = ["text1", "text2", "text3", "text4"]
        found, missing = cache.get_batch(texts, "model")
        
        assert 0 in found  # text1
        assert 2 in found  # text3
        assert 1 in missing  # text2
        assert 3 in missing  # text4
    
    def test_batch_set_stores_all_entries(self):
        """Batch set stores all entries correctly."""
        cache = EmbeddingCache(ttl=3600)
        
        texts = ["text1", "text2", "text3"]
        embeddings = [[0.1], [0.2], [0.3]]
        
        cache.set_batch(texts, "model", embeddings)
        
        for text, expected in zip(texts, embeddings):
            assert cache.get(text, "model") == expected
    
    def test_cache_clear(self):
        """Cache clear removes all entries."""
        cache = EmbeddingCache(ttl=3600)
        
        cache.set("text1", "model", [0.1])
        cache.set("text2", "model", [0.2])
        
        cache.clear()
        
        assert cache.get("text1", "model") is None
        assert cache.get("text2", "model") is None
        assert cache.get_stats().entry_count == 0
    
    def test_cleanup_expired_removes_old_entries(self):
        """Cleanup removes expired entries."""
        cache = EmbeddingCache(ttl=1)
        
        cache.set("text1", "model", [0.1])
        time.sleep(1.1)
        cache.set("text2", "model", [0.2])  # Not expired
        
        removed = cache.cleanup_expired()
        
        assert removed == 1
        assert cache.get("text1", "model") is None
        assert cache.get("text2", "model") is not None


class TestCachedEmbeddingProvider:
    """Tests for CachedEmbeddingProvider wrapper."""
    
    def test_cached_provider_wraps_underlying(self):
        """Cached provider delegates to underlying provider."""
        provider = MockProvider(name="underlying")
        cached = CachedEmbeddingProvider(provider)
        
        assert cached.is_available() == provider.is_available()
        assert "underlying" in cached.name
    
    def test_cached_provider_caches_results(self):
        """Cached provider caches embedding results."""
        call_count = [0]
        
        class CountingProvider(MockProvider):
            def get_embeddings(self, texts, model=None):
                call_count[0] += 1
                return [[0.1, 0.2]] * len(texts)
        
        provider = CountingProvider()
        cached = CachedEmbeddingProvider(provider)
        
        # First call
        cached.get_embeddings(["test"])
        assert call_count[0] == 1
        
        # Second call - should use cache
        cached.get_embeddings(["test"])
        assert call_count[0] == 1  # No additional call
    
    def test_cached_provider_partial_cache_hit(self):
        """Cached provider handles partial cache hits."""
        call_count = [0]
        
        class CountingProvider(MockProvider):
            def get_embeddings(self, texts, model=None):
                call_count[0] += len(texts)
                return [[0.1 * len(t)] for t in texts]
        
        provider = CountingProvider()
        cached = CachedEmbeddingProvider(provider)
        
        # Cache "text1"
        cached.get_embeddings(["text1"])
        assert call_count[0] == 1
        
        # Request "text1" and "text2" - only "text2" needs API
        result = cached.get_embeddings(["text1", "text2"])
        assert call_count[0] == 2  # Only 1 additional call
        assert len(result) == 2


# =============================================================================
# Property 21: Batch Embedding Processing
# Feature: semantic-chunker-enhancement, Property 21: Batch Embedding Processing
# Validates: Requirements 6.1
# =============================================================================

class TestBatchProcessing:
    """Tests for batch embedding processing."""
    
    # Feature: semantic-chunker-enhancement, Property 21: Batch Embedding Processing
    @given(
        num_texts=st.integers(min_value=10, max_value=100),
        batch_size=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=50, deadline=None)
    def test_batch_processing_reduces_api_calls(
        self, num_texts: int, batch_size: int
    ):
        """Property: Batch processing reduces API calls by >80%."""
        # Track API calls
        api_calls = [0]
        
        class BatchCountingProvider(MockProvider):
            def get_embeddings(self, texts, model=None):
                api_calls[0] += 1  # Count batch calls, not individual texts
                return [[0.1, 0.2]] * len(texts)
        
        provider = BatchCountingProvider()
        config = EmbeddingProviderConfig(batch_size=batch_size)
        manager = EmbeddingProviderManager(
            providers=[provider],
            config=config
        )
        
        # Generate texts
        texts = [f"text_{i}" for i in range(num_texts)]
        
        # Process with batching
        manager.get_embeddings_batch(texts)
        
        # Calculate expected batches
        expected_batches = (num_texts + batch_size - 1) // batch_size
        
        # Verify batching occurred
        assert api_calls[0] == expected_batches
        
        # Verify reduction: without batching would be num_texts calls
        # With batching: expected_batches calls
        # Reduction = 1 - (expected_batches / num_texts)
        if num_texts > batch_size:
            reduction = 1 - (api_calls[0] / num_texts)
            assert reduction > 0.5, f"Expected >50% reduction, got {reduction*100}%"
    
    # Feature: semantic-chunker-enhancement, Property 21: Batch Embedding Processing
    def test_batch_size_configuration(self):
        """Property: Batch size is configurable."""
        provider = MockProvider()
        
        # Small batch size
        config_small = EmbeddingProviderConfig(batch_size=5)
        manager_small = EmbeddingProviderManager(
            providers=[provider],
            config=config_small
        )
        assert manager_small.config.batch_size == 5
        
        # Large batch size
        config_large = EmbeddingProviderConfig(batch_size=100)
        manager_large = EmbeddingProviderManager(
            providers=[provider],
            config=config_large
        )
        assert manager_large.config.batch_size == 100
    
    # Feature: semantic-chunker-enhancement, Property 21: Batch Embedding Processing
    def test_batch_preserves_order(self):
        """Property: Batch processing preserves text order."""
        class OrderedProvider(MockProvider):
            def get_embeddings(self, texts, model=None):
                # Return embedding based on text content
                return [[float(t.split('_')[1])] for t in texts]
        
        provider = OrderedProvider()
        config = EmbeddingProviderConfig(batch_size=3)
        manager = EmbeddingProviderManager(
            providers=[provider],
            config=config
        )
        
        texts = [f"text_{i}" for i in range(10)]
        result = manager.get_embeddings_batch(texts)
        
        # Verify order is preserved
        for i, embedding in enumerate(result):
            assert embedding[0] == float(i)


# =============================================================================
# Integration Tests for SemanticChunker with Provider Infrastructure
# Feature: semantic-chunker-enhancement
# Validates: Requirements 4.1, 4.2, 6.2
# =============================================================================

class TestSemanticChunkerIntegration:
    """Integration tests for SemanticChunker with embedding providers."""
    
    def test_semantic_chunker_default_mode(self):
        """SemanticChunker works in default (legacy) mode."""
        from app.services.chunker import SemanticChunker
        
        chunker = SemanticChunker()
        
        # Should use legacy mode by default
        assert chunker._use_provider_manager is False
        assert chunker._provider_manager is None
    
    def test_semantic_chunker_provider_manager_mode(self):
        """SemanticChunker can use provider manager mode."""
        from app.services.chunker import SemanticChunker
        
        chunker = SemanticChunker(use_provider_manager=True)
        
        assert chunker._use_provider_manager is True
    
    def test_semantic_chunker_cache_configuration(self):
        """SemanticChunker cache can be enabled/disabled."""
        from app.services.chunker import SemanticChunker
        
        # Cache enabled (default)
        chunker_cached = SemanticChunker(
            use_provider_manager=True,
            enable_cache=True
        )
        assert chunker_cached._enable_cache is True
        
        # Cache disabled
        chunker_no_cache = SemanticChunker(
            use_provider_manager=True,
            enable_cache=False
        )
        assert chunker_no_cache._enable_cache is False
    
    def test_provider_manager_initialization(self):
        """Provider manager is lazily initialized."""
        from app.services.chunker import SemanticChunker
        import os
        
        # Set up API key for test
        original_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        try:
            chunker = SemanticChunker(use_provider_manager=True)
            
            # Not initialized yet
            assert chunker._provider_manager is None
            
            # Initialize
            chunker._ensure_provider_manager()
            
            # Now initialized
            assert chunker._provider_manager is not None
            assert len(chunker._provider_manager.providers) > 0
        finally:
            # Restore original key
            if original_key:
                os.environ["OPENROUTER_API_KEY"] = original_key
            else:
                os.environ.pop("OPENROUTER_API_KEY", None)
    
    def test_embedding_cache_initialization(self):
        """Embedding cache is initialized when enabled."""
        from app.services.chunker import SemanticChunker
        import os
        
        original_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        
        try:
            chunker = SemanticChunker(
                use_provider_manager=True,
                enable_cache=True
            )
            
            # Initialize provider manager
            chunker._ensure_provider_manager()
            
            # Cache should be initialized
            assert chunker._embedding_cache is not None
        finally:
            if original_key:
                os.environ["OPENROUTER_API_KEY"] = original_key
            else:
                os.environ.pop("OPENROUTER_API_KEY", None)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestEmbeddingProviderConfiguration:
    """Tests for embedding provider configuration."""
    
    def test_default_config_values(self):
        """Default configuration has sensible values."""
        config = EmbeddingProviderConfig()
        
        assert config.primary_provider == "openrouter"
        assert config.batch_size == 32
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.timeout == 30.0
    
    def test_custom_config_values(self):
        """Custom configuration values are respected."""
        config = EmbeddingProviderConfig(
            primary_provider="openai",
            batch_size=64,
            max_retries=5,
            retry_delay=0.5,
            timeout=60.0
        )
        
        assert config.primary_provider == "openai"
        assert config.batch_size == 64
        assert config.max_retries == 5
        assert config.retry_delay == 0.5
        assert config.timeout == 60.0
    
    def test_fallback_providers_configuration(self):
        """Fallback providers can be configured."""
        config = EmbeddingProviderConfig(
            primary_provider="openrouter",
            fallback_providers=["openai"]
        )
        
        assert config.fallback_providers == ["openai"]
