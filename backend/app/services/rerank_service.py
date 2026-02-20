"""Reranker service for improving search result relevance.

This service provides a unified interface for reranking search results
using various providers (Cohere, Alibaba).
"""

import hashlib
import logging
import os
import time
import requests
from typing import Dict, List, Optional
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from openai import OpenAI

logger = logging.getLogger(__name__)


# Provider base URLs
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
VOYAGE_BASE_URL = "https://api.voyageai.com/v1"


# Reranker model configurations
RERANKER_MODELS = {
    "cohere": {
        "rerank-english-v3.0": {
            "name": "Cohere Rerank English v3.0",
            "languages": ["en"],
            "max_documents": 1000,
            "max_query_length": 2048,
            "description": "Optimized for English text reranking"
        },
        "rerank-multilingual-v3.0": {
            "name": "Cohere Rerank Multilingual v3.0",
            "languages": ["100+"],
            "max_documents": 1000,
            "max_query_length": 2048,
            "description": "Supports 100+ languages"
        }
    },
    "alibaba": {
        "gte-rerank-v2": {
            "name": "Alibaba GTE Rerank v2",
            "languages": ["100+"],
            "max_documents": 500,
            "max_query_length": 1024,
            "description": "Latest multilingual reranking model"
        }
    },
    "jina": {
        "jina-reranker-v1-base-en": {
            "name": "Jina Reranker v1 Base English",
            "languages": ["en"],
            "max_documents": 1000,
            "max_query_length": 2048,
            "description": "English reranking model"
        },
        "jina-reranker-v2-base-multilingual": {
            "name": "Jina Reranker v2 Base Multilingual",
            "languages": ["100+"],
            "max_documents": 1000,
            "max_query_length": 2048,
            "description": "Multilingual reranking model supporting 100+ languages"
        }
    },
    "bge": {
        "bge-reranker-v2-m3": {
            "name": "BGE Reranker v2-M3",
            "languages": ["100+"],
            "max_documents": 1000,
            "max_query_length": 2048,
            "description": "Multilingual reranking model from BAAI"
        }
    },
    "zeroentropy": {
        "zerank-2": {
            "name": "ZeroEntropy ZeRank-2",
            "languages": ["100+"],
            "max_documents": 2048,
            "max_query_length": 4096,
            "description": "ZeroEntropy hosted reranker (models/rerank)"
        }
    },
    "voyage": {
        "rerank-2": {
            "name": "Voyage Rerank-2",
            "languages": ["100+"],
            "max_documents": 1000,
            "max_query_length": 4096,
            "description": "Voyage hosted reranker (v1/rerank)"
        },
        "rerank-2.5": {
            "name": "Voyage Rerank-2.5",
            "languages": ["100+"],
            "max_documents": 1000,
            "max_query_length": 4096,
            "description": "Voyage hosted reranker (v1/rerank)"
        },
        "rerank-2.5-lite": {
            "name": "Voyage Rerank-2.5-Lite",
            "languages": ["100+"],
            "max_documents": 1000,
            "max_query_length": 4096,
            "description": "Voyage hosted reranker (v1/rerank)"
        }
    }
}

# Default models for each provider
DEFAULT_MODELS = {
    "cohere": "rerank-multilingual-v3.0",
    "alibaba": "gte-rerank-v2",
    "jina": "jina-reranker-v2-base-multilingual",
    "bge": "ollama-bge-reranker-v2-m3",
    "zeroentropy": "zerank-2",
    "voyage": "rerank-2"
}

# Reranker configuration from environment
RERANKER_CACHE_TTL = int(os.environ.get("RERANKER_CACHE_TTL", "300"))  # 5 minutes
RERANKER_TIMEOUT = int(os.environ.get("RERANKER_TIMEOUT", "5"))  # 5 seconds
RERANKER_MAX_DOCUMENTS = int(os.environ.get("RERANKER_MAX_DOCUMENTS", "1000"))


class RerankService:
    """Service for reranking search results using various providers.
    
    Supports multiple reranker providers:
    - Cohere: High-quality multilingual reranking
    - Alibaba: Chinese-optimized reranking
    - Jina: Open-source multilingual reranking
    
    Each provider has different models and capabilities.
    """
    
    def __init__(self):
        """Initialize reranker service."""
        self._cohere_api_key = os.environ.get("COHERE_API_KEY")
        self._dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
        self._zeroentropy_api_key = os.environ.get("ZEROENTROPY_API_KEY")
        self._voyage_api_key = os.environ.get("VOYAGE_API_KEY")
        self._zeroentropy_base_url = os.environ.get(
            "ZEROENTROPY_BASE_URL",
            "https://api.zeroentropy.dev/v1",
        )
        self._cohere_client = None
        self._alibaba_client = None
        
        # Initialize cache with TTL
        self._cache = TTLCache(maxsize=1000, ttl=RERANKER_CACHE_TTL)
        
        # Performance metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_rerank_time = 0.0
        self._rerank_count = 0
        
        logger.info(
            "RerankService initialized",
            extra={
                "cache_ttl": RERANKER_CACHE_TTL,
                "timeout": RERANKER_TIMEOUT,
                "max_documents": RERANKER_MAX_DOCUMENTS,
                "cohere_available": bool(self._cohere_api_key and COHERE_AVAILABLE),
                "alibaba_available": bool(self._dashscope_api_key),
                "bge_available": True,  # Ollama is always available if running
                "zeroentropy_available": bool(self._zeroentropy_api_key),
                "voyage_available": bool(self._voyage_api_key)
            }
        )

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
                    "for Cohere reranker. Please configure the API key "
                    "or use a different reranker provider."
                )
            self._cohere_client = cohere.ClientV2(api_key=self._cohere_api_key)
            logger.info("Cohere reranker client initialized")
        
        return self._cohere_client

    def _get_alibaba_client(self) -> OpenAI:
        """Get or create Alibaba DashScope client.
        
        Returns:
            OpenAI client configured for DashScope
            
        Raises:
            ValueError: If DASHSCOPE_API_KEY is not set
        """
        if self._alibaba_client is None:
            if not self._dashscope_api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY environment variable is required "
                    "for Alibaba reranker. Please configure the API key "
                    "or use a different reranker provider."
                )
            # Alibaba DashScope uses OpenAI-compatible API
            self._alibaba_client = OpenAI(
                api_key=self._dashscope_api_key,
                base_url=DASHSCOPE_BASE_URL
            )
            logger.info("Alibaba reranker client initialized")
        
        return self._alibaba_client

    def get_cache_stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            self._cache_hits / total_requests
            if total_requests > 0
            else 0.0
        )
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "cache_maxsize": self._cache.maxsize,
            "cache_ttl": self._cache.ttl
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        avg_latency = (
            self._total_rerank_time / self._rerank_count
            if self._rerank_count > 0
            else 0.0
        )
        
        cache_stats = self.get_cache_stats()
        
        return {
            "total_reranks": self._rerank_count,
            "total_time_seconds": self._total_rerank_time,
            "average_latency_seconds": avg_latency,
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_hits": cache_stats["cache_hits"],
            "cache_misses": cache_stats["cache_misses"]
        }

    def _generate_cache_key(
        self,
        query: str,
        documents: List[Dict],
        provider: str,
        model: str
    ) -> str:
        """Generate unique cache key for reranking request.
        
        Args:
            query: Search query
            documents: List of documents
            provider: Reranker provider
            model: Reranker model
            
        Returns:
            Cache key (SHA256 hash)
        """
        # Create a unique string from query, document IDs, provider, and model
        doc_ids = [str(doc.get('id', '')) for doc in documents]
        cache_string = f"{query}|{','.join(doc_ids)}|{provider}|{model}"
        
        # Generate SHA256 hash
        cache_key = hashlib.sha256(cache_string.encode()).hexdigest()
        
        return cache_key

    def get_available_models(self, provider: str) -> List[str]:
        """Get available models for a provider.
        
        Args:
            provider: Reranker provider
            
        Returns:
            List of available model names
        """
        if provider not in RERANKER_MODELS:
            return []
        
        return list(RERANKER_MODELS[provider].keys())

    def get_model_info(self, provider: str, model: str) -> Optional[Dict]:
        """Get information about a specific model.
        
        Args:
            provider: Reranker provider
            model: Model name
            
        Returns:
            Model information dict or None if not found
        """
        if provider not in RERANKER_MODELS:
            return None
        
        return RERANKER_MODELS[provider].get(model)

    def validate_provider(self, provider: str) -> bool:
        """Validate if provider is supported.
        
        Args:
            provider: Reranker provider
            
        Returns:
            True if provider is valid
        """
        return provider in RERANKER_MODELS

    def validate_model(self, provider: str, model: str) -> bool:
        """Validate if model is available for provider.
        
        Args:
            provider: Reranker provider
            model: Model name
            
        Returns:
            True if model is valid for provider
        """
        if not self.validate_provider(provider):
            return False
        
        return model in RERANKER_MODELS[provider]

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        provider: str,
        model: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of documents with 'content' field
            provider: Reranker provider (cohere/alibaba)
            model: Provider-specific model name (uses default if None)
            top_k: Number of top results to return (returns all if None)
            
        Returns:
            Reranked documents with relevance scores
            
        Raises:
            ValueError: If provider is invalid or not configured
        """
        start_time = time.time()
        
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if documents is None:
            raise ValueError("Documents list cannot be None")
        
        # Return empty list for empty documents
        if not documents:
            return []
        
        # Validate provider
        if not self.validate_provider(provider):
            raise ValueError(
                f"Invalid reranker provider: {provider}. "
                f"Supported providers: {list(RERANKER_MODELS.keys())}"
            )
        
        # Use default model if not specified
        if model is None:
            model = DEFAULT_MODELS[provider]
        
        # Validate model
        if not self.validate_model(provider, model):
            logger.warning(
                "Invalid model %s for provider %s, using default: %s",
                model,
                provider,
                DEFAULT_MODELS[provider]
            )
            model = DEFAULT_MODELS[provider]
        
        # Check cache
        cache_key = self._generate_cache_key(query, documents, provider, model)
        if cache_key in self._cache:
            self._cache_hits += 1
            elapsed_time = time.time() - start_time
            logger.info(
                "Cache hit for reranking request",
                extra={
                    "provider": provider,
                    "model": model,
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses,
                    "elapsed_time": elapsed_time
                }
            )
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        # Route to appropriate provider with timeout
        try:
            # Use ThreadPoolExecutor for timeout handling
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._rerank_with_provider,
                    query,
                    documents,
                    provider,
                    model,
                    top_k
                )
                
                try:
                    reranked = future.result(timeout=RERANKER_TIMEOUT)
                except FuturesTimeoutError:
                    logger.warning(
                        "Reranking timeout after %s seconds",
                        RERANKER_TIMEOUT,
                        extra={
                            "provider": provider,
                            "model": model,
                            "timeout": RERANKER_TIMEOUT,
                            "document_count": len(documents)
                        }
                    )
                    # Return original documents on timeout
                    return documents[:top_k] if top_k else documents
            
            # Store in cache
            self._cache[cache_key] = reranked
            
            # Track performance metrics
            elapsed_time = time.time() - start_time
            self._total_rerank_time += elapsed_time
            self._rerank_count += 1
            
            # Calculate score improvement if possible
            score_improvement = self._calculate_score_improvement(
                documents,
                reranked
            )
            
            logger.info(
                "Reranked %d documents using %s/%s in %.3f seconds",
                len(documents),
                provider,
                model,
                elapsed_time,
                extra={
                    "provider": provider,
                    "model": model,
                    "input_count": len(documents),
                    "output_count": len(reranked),
                    "elapsed_time": elapsed_time,
                    "score_improvement": score_improvement,
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses
                }
            )
            
            return reranked
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                "Reranking failed with %s/%s: %s",
                provider,
                model,
                str(e),
                exc_info=True,
                extra={
                    "provider": provider,
                    "model": model,
                    "query_length": len(query),
                    "document_count": len(documents),
                    "error_type": type(e).__name__,
                    "elapsed_time": elapsed_time
                }
            )
            # Fallback: return original documents unchanged
            logger.warning(
                "Falling back to original results due to reranking failure",
                extra={
                    "provider": provider,
                    "model": model
                }
            )
            return documents[:top_k] if top_k else documents
    
    def _rerank_with_provider(
        self,
        query: str,
        documents: List[Dict],
        provider: str,
        model: str,
        top_k: Optional[int]
    ) -> List[Dict]:
        """Internal method to route reranking to appropriate provider.
        
        This method is called by rerank() within a timeout context.
        
        Args:
            query: Search query
            documents: List of documents
            provider: Reranker provider
            model: Model name
            top_k: Number of top results
            
        Returns:
            Reranked documents
        """
        if provider == "cohere":
            return self._rerank_cohere(query, documents, model, top_k)
        elif provider == "alibaba":
            return self._rerank_alibaba(query, documents, model, top_k)
        elif provider == "jina":
            return self._rerank_jina(query, documents, model, top_k)
        elif provider == "bge":
            return self._rerank_bge(query, documents, model, top_k)
        elif provider == "zeroentropy":
            return self._rerank_zeroentropy(query, documents, model, top_k)
        elif provider == "voyage":
            return self._rerank_voyage(query, documents, model, top_k)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _rerank_voyage(
        self,
        query: str,
        documents: List[Dict],
        model: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        if not self._voyage_api_key:
            raise ValueError(
                "VOYAGE_API_KEY environment variable is required "
                "for Voyage reranker. Please configure the API key "
                "or use a different reranker provider."
            )

        doc_texts = [doc.get("content", "") for doc in documents]
        payload: Dict = {
            "query": query,
            "documents": doc_texts,
            "model": model,
        }

        if top_k is not None:
            payload["top_k"] = top_k

        headers = {
            "Authorization": f"Bearer {self._voyage_api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{VOYAGE_BASE_URL}/rerank",
            headers=headers,
            json=payload,
            timeout=RERANKER_TIMEOUT,
        )
        response.raise_for_status()
        result_data = response.json()

        raw_results = result_data.get("results") or result_data.get("data") or []
        reranked: List[Dict] = []
        for r in raw_results:
            idx = r.get("index")
            if idx is None:
                continue
            if not (0 <= int(idx) < len(documents)):
                continue

            doc = documents[int(idx)].copy()
            doc["relevance_score"] = r.get("relevance_score", 0)
            doc["rerank_index"] = int(idx)

            if "score" in doc and "original_score" not in doc:
                doc["original_score"] = doc["score"]

            reranked.append(doc)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def _rerank_zeroentropy(
        self,
        query: str,
        documents: List[Dict],
        model: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Rerank using ZeroEntropy API.

        Args:
            query: Search query
            documents: List of documents
            model: ZeroEntropy model name (e.g., zerank-2)
            top_k: Number of top results to return

        Returns:
            Reranked documents with relevance scores
        """
        if not self._zeroentropy_api_key:
            raise ValueError(
                "ZEROENTROPY_API_KEY environment variable is required "
                "for ZeroEntropy reranker. Please configure the API key "
                "or use a different reranker provider."
            )

        doc_texts = [doc.get("content", "") for doc in documents]

        url = f"{self._zeroentropy_base_url}/models/rerank"
        headers = {
            "Authorization": f"Bearer {self._zeroentropy_api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict = {
            "model": model,
            "query": query,
            "documents": doc_texts,
        }

        if top_k is not None:
            payload["top_n"] = top_k

        latency_mode = os.environ.get("ZEROENTROPY_LATENCY")
        if latency_mode in {"fast", "slow"}:
            payload["latency"] = latency_mode

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=RERANKER_TIMEOUT,
        )

        if response.status_code != 200:
            logger.error(
                "ZeroEntropy reranker error: %s",
                response.status_code,
                extra={
                    "status_code": response.status_code,
                    "response_body": response.text[:500],
                    "model": model,
                },
            )

        response.raise_for_status()
        result_data = response.json()

        reranked: List[Dict] = []
        for r in result_data.get("results", []):
            idx = r.get("index")
            if idx is None:
                continue
            if not (0 <= int(idx) < len(documents)):
                continue

            doc = documents[int(idx)].copy()
            doc["relevance_score"] = r.get("relevance_score", 0)
            doc["rerank_index"] = int(idx)

            if "score" in doc and "original_score" not in doc:
                doc["original_score"] = doc["score"]

            reranked.append(doc)

        if top_k is not None:
            reranked = reranked[:top_k]

        logger.info(
            "ZeroEntropy reranked %d documents to %d",
            len(documents),
            len(reranked),
            extra={
                "model": model,
                "input_count": len(documents),
                "output_count": len(reranked),
                "top_k": top_k,
                "actual_latency_mode": result_data.get("actual_latency_mode"),
                "e2e_latency": result_data.get("e2e_latency"),
                "inference_latency": result_data.get("inference_latency"),
                "total_tokens": result_data.get("total_tokens"),
                "total_bytes": result_data.get("total_bytes"),
            },
        )

        return reranked
    
    def _calculate_score_improvement(
        self,
        original_docs: List[Dict],
        reranked_docs: List[Dict]
    ) -> Optional[float]:
        """Calculate average score improvement from reranking.
        
        Args:
            original_docs: Original documents with scores
            reranked_docs: Reranked documents with relevance_score
            
        Returns:
            Average score improvement or None if not calculable
        """
        try:
            # Get original scores
            original_scores = [
                doc.get('score', doc.get('original_score', 0))
                for doc in original_docs
            ]
            
            # Get reranked scores
            reranked_scores = [
                doc.get('relevance_score', 0)
                for doc in reranked_docs
            ]
            
            if not original_scores or not reranked_scores:
                return None
            
            # Calculate average improvement
            avg_original = sum(original_scores) / len(original_scores)
            avg_reranked = sum(reranked_scores) / len(reranked_scores)
            
            improvement = avg_reranked - avg_original
            
            return improvement
            
        except Exception as e:
            logger.debug(
                "Could not calculate score improvement: %s",
                str(e)
            )
            return None

    def _rerank_cohere(
        self,
        query: str,
        documents: List[Dict],
        model: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Rerank using Cohere API.
        
        Args:
            query: Search query
            documents: List of documents
            model: Cohere model name
            top_k: Number of top results to return
            
        Returns:
            Reranked documents with relevance scores
        """
        client = self._get_cohere_client()
        
        # Extract text content from documents
        doc_texts = [doc.get('content', '') for doc in documents]
        
        try:
            # Call Cohere rerank API
            response = client.rerank(
                model=model,
                query=query,
                documents=doc_texts,
                top_n=top_k,
                return_documents=True
            )
            
            # Map results back to original documents
            reranked = []
            for result in response.results:
                # Get the original document
                doc = documents[result.index].copy()
                
                # Add reranking metadata
                doc['relevance_score'] = result.relevance_score
                doc['rerank_index'] = result.index
                
                # Preserve original score if it exists
                if 'score' in doc and 'original_score' not in doc:
                    doc['original_score'] = doc['score']
                
                reranked.append(doc)
            
            logger.info(
                f"Cohere reranked {len(documents)} documents to {len(reranked)}",
                extra={
                    "model": model,
                    "input_count": len(documents),
                    "output_count": len(reranked),
                    "top_k": top_k
                }
            )
            
            return reranked
            
        except Exception as e:
            logger.error(
                f"Cohere reranking failed: {e}",
                exc_info=True,
                extra={
                    "model": model,
                    "query_length": len(query),
                    "document_count": len(documents)
                }
            )
            raise

    def _rerank_alibaba(
        self,
        query: str,
        documents: List[Dict],
        model: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Rerank using Alibaba DashScope API.
        
        Args:
            query: Search query
            documents: List of documents
            model: Alibaba model name
            top_k: Number of top results to return
            
        Returns:
            Reranked documents with relevance scores
        """
        # Extract text content from documents
        doc_texts = [doc.get('content', '') for doc in documents]
        
        try:
            # Alibaba DashScope rerank API uses a different structure
            # The API endpoint is different from the OpenAI-compatible one
            api_key = self._dashscope_api_key
            url = DASHSCOPE_BASE_URL + "/text-rerank/text-rerank"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Alibaba API expects model name without provider prefix
            # Remove "alibaba/" prefix if present
            alibaba_model = model.replace("alibaba/", "") if model.startswith("alibaba/") else model
            
            payload = {
                "model": alibaba_model,
                "input": {
                    "query": query,
                    "documents": doc_texts
                },
                "parameters": {
                    "return_documents": True
                }
            }
            
            if top_k is not None:
                payload["parameters"]["top_n"] = top_k
            
            # DEBUG: Log the request
            logger.info(
                f"[ALIBABA RERANK DEBUG] Sending request to Alibaba API\n"
                f"Original model: {model}\n"
                f"Cleaned model: {alibaba_model}\n"
                f"Query length: {len(query)}\n"
                f"Documents count: {len(doc_texts)}\n"
                f"Top K: {top_k}\n"
                f"Payload: {payload}"
            )
            
            response = requests.post(url, headers=headers, json=payload, timeout=RERANKER_TIMEOUT)
            
            # DEBUG: Log the response
            logger.info(
                f"[ALIBABA RERANK DEBUG] Response status: {response.status_code}\n"
                f"Response body: {response.text[:500]}"
            )
            
            if response.status_code != 200:
                logger.error(
                    f"[ALIBABA RERANK ERROR] Bad response from Alibaba API\n"
                    f"Status: {response.status_code}\n"
                    f"Body: {response.text}"
                )
            
            response.raise_for_status()
            
            result_data = response.json()
            
            # Check for errors
            if "code" in result_data and result_data["code"]:
                raise ValueError(f"Alibaba API error: {result_data.get('message', 'Unknown error')}")
            
            # Map results back to original documents
            reranked = []
            for result in result_data["output"]["results"]:
                # Get the original document
                doc = documents[result["index"]].copy()
                
                # Add reranking metadata
                doc['relevance_score'] = result["relevance_score"]
                doc['rerank_index'] = result["index"]
                
                # Preserve original score if it exists
                if 'score' in doc and 'original_score' not in doc:
                    doc['original_score'] = doc['score']
                
                reranked.append(doc)
            
            logger.info(
                f"Alibaba reranked {len(documents)} documents to {len(reranked)}",
                extra={
                    "model": model,
                    "input_count": len(documents),
                    "output_count": len(reranked),
                    "top_k": top_k
                }
            )
            
            return reranked
            
        except Exception as e:
            logger.error(
                f"Alibaba reranking failed: {e}",
                exc_info=True,
                extra={
                    "model": model,
                    "query_length": len(query),
                    "document_count": len(documents)
                }
            )
            raise

    def _rerank_jina(
        self,
        query: str,
        documents: List[Dict],
        model: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Rerank using Jina Reranker API.
        
        Args:
            query: Search query
            documents: List of documents
            model: Jina model name
            top_k: Number of top results to return
            
        Returns:
            Reranked documents with relevance scores
        """
        # Extract text content from documents
        doc_texts = [doc.get('content', '') for doc in documents]
        
        try:
            # Jina Reranker API endpoint
            url = "https://api.jina.ai/v1/rerank"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('JINA_API_KEY', '')}"
            }
            
            # Clean model name (remove provider prefix if present)
            jina_model = model.replace("jina/", "") if model.startswith("jina/") else model
            
            payload = {
                "model": jina_model,
                "query": query,
                "documents": doc_texts,
                "top_k": top_k if top_k is not None else len(documents)
            }
            
            # DEBUG: Log the request
            logger.info(
                f"[JINA RERANK DEBUG] Sending request to Jina API\n"
                f"Original model: {model}\n"
                f"Cleaned model: {jina_model}\n"
                f"Query length: {len(query)}\n"
                f"Documents count: {len(doc_texts)}\n"
                f"Top K: {top_k}"
            )
            
            response = requests.post(url, headers=headers, json=payload, timeout=RERANKER_TIMEOUT)
            
            # DEBUG: Log the response
            logger.info(
                f"[JINA RERANK DEBUG] Response status: {response.status_code}\n"
                f"Response body: {response.text[:500]}"
            )
            
            if response.status_code != 200:
                logger.error(
                    f"[JINA RERANK ERROR] Bad response from Jina API\n"
                    f"Status: {response.status_code}\n"
                    f"Body: {response.text}"
                )
            
            response.raise_for_status()
            
            result_data = response.json()
            
            # Map results back to original documents
            reranked = []
            for result in result_data.get("results", []):
                # Get the original document
                doc = documents[result["index"]].copy()
                
                # Add reranking metadata
                doc['relevance_score'] = result["relevance_score"]
                doc['rerank_index'] = result["index"]
                
                # Preserve original score if it exists
                if 'score' in doc and 'original_score' not in doc:
                    doc['original_score'] = doc['score']
                
                reranked.append(doc)
            
            logger.info(
                f"Jina reranked {len(documents)} documents to {len(reranked)}",
                extra={
                    "model": model,
                    "input_count": len(documents),
                    "output_count": len(reranked),
                    "top_k": top_k
                }
            )
            
            return reranked
            
        except Exception as e:
            logger.error(
                f"Jina reranking failed: {e}",
                exc_info=True,
                extra={
                    "model": model,
                    "query_length": len(query),
                    "document_count": len(documents)
                }
            )
            raise

    def _rerank_bge(
        self,
        query: str,
        documents: List[Dict],
        model: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Rerank using Ollama BGE Reranker v2-M3.
        
        Args:
            query: Search query
            documents: List of documents
            model: BGE model name
            top_k: Number of top results to return
            
        Returns:
            Reranked documents with relevance scores
        """
        # Extract text content from documents
        doc_texts = [doc.get('content', '') for doc in documents]
        
        try:
            # Ollama API endpoint for reranking
            url = f"{OLLAMA_BASE_URL}/api/generate"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # BGE Reranker v2-M3 expects specific format
            # We need to create a prompt that asks the model to rerank
            prompt = f"""Please rerank the following documents based on their relevance to the query.

Query: {query}

Documents to rerank:
"""
            
            for i, doc_text in enumerate(doc_texts):
                prompt += f"{i+1}. {doc_text}\n"
            
            prompt += f"""
Please return the reranked results in the following format:
- Each line should contain: document_number,relevance_score
- Relevance scores should be between 0.0 and 1.0
- Higher scores indicate better relevance
- Sort by relevance score (highest first)
- Only return the top {top_k if top_k else len(doc_texts)} results

Example format:
3,0.95
1,0.87
2,0.76
"""
            
            payload = {
                "model": "bge-reranker-v2-m3",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=RERANKER_TIMEOUT)
            
            if response.status_code != 200:
                logger.error(
                    f"Ollama BGE reranker error: {response.status_code}",
                    extra={
                        "status_code": response.status_code,
                        "response_body": response.text[:500]
                    }
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Parse the response to extract reranking results
            response_text = result.get('response', '')
            
            # Parse the reranked results
            reranked = []
            lines = response_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                
                try:
                    doc_num_str, score_str = line.split(',', 1)
                    doc_num = int(doc_num_str.strip()) - 1  # Convert to 0-based index
                    score = float(score_str.strip())
                    
                    if 0 <= doc_num < len(documents):
                        doc = documents[doc_num].copy()
                        doc['relevance_score'] = score
                        doc['rerank_index'] = doc_num
                        
                        # Preserve original score if it exists
                        if 'score' in doc and 'original_score' not in doc:
                            doc['original_score'] = doc['score']
                        
                        reranked.append(doc)
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse rerank line: {line}, error: {e}")
                    continue
            
            # Sort by relevance score
            reranked.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Apply top_k limit
            if top_k is not None:
                reranked = reranked[:top_k]
            
            logger.info(
                f"Ollama BGE reranked {len(documents)} documents to {len(reranked)}",
                extra={
                    "model": model,
                    "input_count": len(documents),
                    "output_count": len(reranked),
                    "top_k": top_k
                }
            )
            
            return reranked
            
        except Exception as e:
            logger.error(
                f"Ollama BGE reranking failed: {e}",
                exc_info=True,
                extra={
                    "model": model,
                    "query_length": len(query),
                    "document_count": len(documents)
                }
            )
            raise


# Singleton instance
_rerank_service: Optional[RerankService] = None

def get_rerank_service() -> RerankService:
    """Get singleton RerankService instance."""
    global _rerank_service
    if _rerank_service is None:
        _rerank_service = RerankService()
    return _rerank_service
