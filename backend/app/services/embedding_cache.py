"""Embedding cache for reducing API calls and improving performance.

This module provides a caching layer for embedding vectors with
memory-based storage and TTL support.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    entry_count: int = 0
    total_size_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


@dataclass
class CacheEntry:
    """Single cache entry with TTL support."""
    embedding: List[float]
    created_at: float
    ttl: int
    size_bytes: int = 0
    
    def __post_init__(self):
        # Estimate size: 8 bytes per float + overhead
        self.size_bytes = len(self.embedding) * 8 + 100
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl <= 0:
            return False  # No expiration
        return time.time() - self.created_at > self.ttl


class EmbeddingCache:
    """Cache for embedding vectors with memory backend.
    
    Features:
    - Hash-based cache keys for efficient lookup
    - TTL support for automatic expiration
    - Batch get/set operations
    - Thread-safe operations
    - Statistics tracking
    """
    
    def __init__(
        self,
        ttl: int = 3600,
        max_entries: int = 10000
    ):
        """Initialize the embedding cache.
        
        Args:
            ttl: Time-to-live in seconds (default 1 hour). Set to 0 for no expiration.
            max_entries: Maximum number of entries to store.
        """
        self.ttl = ttl
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self._stats = CacheStats()
    
    def _generate_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model combination.
        
        Args:
            text: Text that was embedded
            model: Model used for embedding
            
        Returns:
            Hash-based cache key
        """
        combined = f"{model}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    def get(
        self,
        text: str,
        model: str
    ) -> Optional[List[float]]:
        """Get cached embedding for text.
        
        Args:
            text: Text to look up
            model: Model used for embedding
            
        Returns:
            Cached embedding or None if not found/expired
        """
        key = self._generate_key(text, model)
        
        with self._lock:
            self._stats.total_requests += 1
            
            entry = self._cache.get(key)
            if entry is None:
                self._stats.cache_misses += 1
                return None
            
            if entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                self._stats.cache_misses += 1
                self._update_stats()
                return None
            
            self._stats.cache_hits += 1
            return entry.embedding
    
    def set(
        self,
        text: str,
        model: str,
        embedding: List[float]
    ) -> None:
        """Cache embedding for text.
        
        Args:
            text: Text that was embedded
            model: Model used for embedding
            embedding: Embedding vector to cache
        """
        key = self._generate_key(text, model)
        
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_entries:
                self._evict_oldest()
            
            self._cache[key] = CacheEntry(
                embedding=embedding,
                created_at=time.time(),
                ttl=self.ttl
            )
            self._update_stats()
    
    def get_batch(
        self,
        texts: List[str],
        model: str
    ) -> Tuple[Dict[int, List[float]], List[int]]:
        """Get multiple cached embeddings.
        
        Args:
            texts: List of texts to look up
            model: Model used for embedding
            
        Returns:
            Tuple of (found embeddings dict, missing indices list)
            - found: Dict mapping index to embedding
            - missing: List of indices not found in cache
        """
        found: Dict[int, List[float]] = {}
        missing: List[int] = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text, model)
            if embedding is not None:
                found[i] = embedding
            else:
                missing.append(i)
        
        return found, missing
    
    def set_batch(
        self,
        texts: List[str],
        model: str,
        embeddings: List[List[float]]
    ) -> None:
        """Cache multiple embeddings.
        
        Args:
            texts: List of texts that were embedded
            model: Model used for embedding
            embeddings: List of embedding vectors
        """
        if len(texts) != len(embeddings):
            raise ValueError(
                f"texts and embeddings must have same length: "
                f"{len(texts)} vs {len(embeddings)}"
            )
        
        for text, embedding in zip(texts, embeddings):
            self.set(text, model, embedding)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics.
        
        Returns:
            CacheStats with hit rate, size, etc.
        """
        with self._lock:
            self._update_stats()
            return CacheStats(
                total_requests=self._stats.total_requests,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                entry_count=self._stats.entry_count,
                total_size_bytes=self._stats.total_size_bytes
            )
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._update_stats()
    
    def _update_stats(self) -> None:
        """Update cache statistics (must be called with lock held)."""
        self._stats.entry_count = len(self._cache)
        self._stats.total_size_bytes = sum(
            entry.size_bytes for entry in self._cache.values()
        )
    
    def _evict_oldest(self) -> None:
        """Evict oldest entry (must be called with lock held)."""
        if not self._cache:
            return
        
        # Find oldest entry
        oldest_key = None
        oldest_time = float('inf')
        
        for key, entry in self._cache.items():
            if entry.created_at < oldest_time:
                oldest_time = entry.created_at
                oldest_key = key
        
        if oldest_key:
            del self._cache[oldest_key]
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            self._update_stats()
            return len(expired_keys)


class CachedEmbeddingProvider:
    """Wrapper that adds caching to any embedding provider.
    
    This class wraps an EmbeddingProvider and adds transparent caching
    of embedding results.
    """
    
    def __init__(
        self,
        provider,  # EmbeddingProvider
        cache: Optional[EmbeddingCache] = None
    ):
        """Initialize cached provider.
        
        Args:
            provider: Underlying embedding provider
            cache: Cache instance (creates new one if not provided)
        """
        self.provider = provider
        self.cache = cache or EmbeddingCache()
    
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "document"
    ) -> List[List[float]]:
        """Get embeddings with caching.
        
        Args:
            texts: List of texts to embed
            model: Model to use
            input_type: Context type - "query" or "document"
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        model = model or getattr(self.provider, 'DEFAULT_MODEL', 'default')
        
        # Check cache for existing embeddings
        found, missing = self.cache.get_batch(texts, model)
        
        # If all found in cache, return immediately
        if not missing:
            return [found[i] for i in range(len(texts))]
        
        # Get embeddings for missing texts
        missing_texts = [texts[i] for i in missing]
        new_embeddings = self.provider.get_embeddings(missing_texts, model, input_type)
        
        # Cache new embeddings
        self.cache.set_batch(missing_texts, model, new_embeddings)
        
        # Combine cached and new embeddings in correct order
        result = []
        new_idx = 0
        for i in range(len(texts)):
            if i in found:
                result.append(found[i])
            else:
                result.append(new_embeddings[new_idx])
                new_idx += 1
        
        return result
    
    def is_available(self) -> bool:
        """Check if underlying provider is available."""
        return self.provider.is_available()
    
    def get_cost_estimate(self, text_count: int, avg_tokens: int = 100) -> float:
        """Estimate cost (may be reduced by cache hits)."""
        return self.provider.get_cost_estimate(text_count, avg_tokens)
    
    @property
    def name(self) -> str:
        """Provider name."""
        return f"cached_{self.provider.name}"
    
    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.cache.get_stats()
