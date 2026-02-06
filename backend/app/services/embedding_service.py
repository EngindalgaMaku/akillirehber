"""Embedding service using OpenRouter API."""

import logging
import os
import random
import time
import threading
from typing import List, Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings via OpenRouter API.
    
    Uses OpenRouter to access various embedding models including
    OpenAI's text-embedding-3-small.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    JINA_AI_BASE_URL = "https://api.jina.ai/v1"
    HUGGINGFACE_BASE_URL = "https://router.huggingface.co"
    VOYAGE_BASE_URL = "https://api.voyageai.com/v1"
    OLLAMA_BASE_URL = "http://host.docker.internal:11434"
    ALIBABA_MODEL_PREFIX = "alibaba/"
    OPENAI_MODEL_PREFIX = "openai/"
    COHERE_MODEL_PREFIX = "cohere/"
    JINA_AI_MODEL_PREFIX = "jina/"
    BGE_MODEL_PREFIX = "bge/"
    OLLAMA_MODEL_PREFIX = "ollama/"
    VOYAGE_MODEL_PREFIX = "voyage/"
    DEFAULT_MODEL = "openai/text-embedding-3-small"
    BATCH_SIZE = 50  # Max texts per API call for OpenRouter (reduced from 100 for large docs)
    ALIBABA_BATCH_SIZE = 10  # Max texts per API call for Alibaba
    COHERE_BATCH_SIZE = 96  # Max texts per API call for Cohere
    JINA_AI_BATCH_SIZE = 100  # Max texts per API call for Jina AI
    OLLAMA_BATCH_SIZE = 10  # Max texts per API call for Ollama
    VOYAGE_BATCH_SIZE = 10  # Further reduced to prevent rate limiting (was 32)

    _ollama_lock = threading.Lock()

    def _prepare_ollama_prompt(self, text: str) -> str:
        max_chars = int(os.getenv("OLLAMA_EMBED_MAX_CHARS", "8000"))
        return (text or "").strip()[:max_chars]

    def _get_ollama_fallback_model(self) -> str:
        return os.getenv("OLLAMA_EMBED_FALLBACK_MODEL", "nomic-embed-text").strip()

    def _get_expected_ollama_dim(self, ollama_model: str) -> Optional[int]:
        # Keep dimensions consistent within a course/index.
        # Add more mappings here if you use different Ollama embedding models.
        dims = {
            "bge-m3": 1024,
            "nomic-embed-text": 768,
        }
        return dims.get((ollama_model or "").strip())

    def _zero_vector(self, dim: int) -> List[float]:
        return [0.0] * dim

    def _is_ollama_nan_error(self, body: str) -> bool:
        return "unsupported value: NaN" in (body or "")

    def __init__(self, api_key: str = None):
        """Initialize embedding service.
        
        Args:
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
        """
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
        self._cohere_api_key = os.environ.get("COHERE_API_KEY")
        self._jina_ai_api_key = os.environ.get("JINA_AI_API_KEY")
        self._huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
        self._voyage_api_key = os.environ.get("VOYAGE_API_KEY")
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

    def _get_jina_ai_client(self):
        """Get or create Jina AI client.
        
        Returns:
            Jina AI client (requests session)
            
        Raises:
            ValueError: If JINA_AI_API_KEY is not set or requests package not installed
        """
        if not REQUESTS_AVAILABLE:
            raise ValueError(
                "requests package is not installed. "
                "Please install it with: pip install requests"
            )
        
        if not self._jina_ai_api_key:
            raise ValueError(
                "JINA_AI_API_KEY environment variable is required "
                "for Jina AI embedding models. Please configure the API key "
                "or use an OpenRouter model instead."
            )
        
        # Jina AI uses simple HTTP requests, so we return the API key
        # and use requests directly in the embedding methods
        return self._jina_ai_api_key

    def _get_huggingface_client(self):
        """Get or create HuggingFace client.
        
        Returns:
            HuggingFace API key (for requests)
            
        Raises:
            ValueError: If HUGGINGFACE_API_KEY is not set or requests package not installed
        """
        if not REQUESTS_AVAILABLE:
            raise ValueError(
                "requests package is not installed. "
                "Please install it with: pip install requests"
            )
        
        if not self._huggingface_api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY environment variable is required "
                "for BGE embedding models. Please configure the API key "
                "or use an OpenRouter model instead."
            )
        
        # HuggingFace uses simple HTTP requests, so we return the API key
        # and use requests directly in the embedding methods
        return self._huggingface_api_key

    def _get_provider_for_model(self, model: str) -> str:
        """Determine which provider to use based on model name.
        
        Args:
            model: Model identifier (e.g., "alibaba/text-embedding-v4")
            
        Returns:
            Provider name: "alibaba", "cohere", "jina", or "openrouter"
        """
        if model.startswith(self.ALIBABA_MODEL_PREFIX):
            return "alibaba"
        elif model.startswith(self.COHERE_MODEL_PREFIX):
            return "cohere"
        elif model.startswith(self.JINA_AI_MODEL_PREFIX):
            return "jina"
        elif model.startswith(self.VOYAGE_MODEL_PREFIX) or model.startswith("voyage-"):
            return "voyage"
        elif model.startswith(self.OLLAMA_MODEL_PREFIX):
            return "ollama"
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
        elif provider == "jina":
            return self._get_jina_ai_client()
        elif provider == "voyage":
            if not REQUESTS_AVAILABLE:
                raise ValueError(
                    "requests package is not installed. "
                    "Please install it with: pip install requests"
                )
            if not self._voyage_api_key:
                raise ValueError(
                    "VOYAGE_API_KEY environment variable is required "
                    "for Voyage embedding models."
                )
            return self._voyage_api_key
        elif provider == "ollama":
            return None  # Ollama doesn't need a client
        return self._get_client()

    def get_embedding(
        self, text: str, model: str = None, input_type: str = None
    ) -> List[float]:
        """Get embedding for a single text.
        
        Now supports multiple providers based on model prefix.
        Routes to OpenRouter for "openai/*" models.
        Routes to DashScope for "alibaba/*" models.
        Routes to Cohere for "cohere/*" models.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            input_type: Context type - "query" for search queries, 
                       "document" for indexing. Defaults to "query".
                       Supported by Voyage AI and Cohere providers.
            
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
        
        # Default to "query" for backward compatibility
        if input_type is None:
            input_type = "query"
            
        model = model or self.DEFAULT_MODEL
        
        # DEBUG: Log everything
        logger.debug("=== EMBEDDING DEBUG ===")
        logger.debug("Model: %s", model)
        logger.debug("Text: %s...", text[:100])
        
        provider = self._get_provider_for_model(model)
        logger.debug("Provider: %s", provider)
        
        # DEBUG: Check Ollama prefix
        ollama_prefix = self.OLLAMA_MODEL_PREFIX
        logger.debug("OLLAMA_MODEL_PREFIX: %s", ollama_prefix)
        logger.debug("Model starts with ollama/: %s", model.startswith(ollama_prefix))
        
        try:
            if provider == "voyage":
                api_key = self._get_client_for_model(model)
                voyage_model = model.replace(self.VOYAGE_MODEL_PREFIX, "")

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                data = {
                    "input": text.strip(),
                    "model": voyage_model,
                    "input_type": input_type,  # Use parameter instead of hardcoded "query"
                }

                # Enhanced retry logic for rate limiting
                max_retries = 8  # Increased from 5
                base_delay = 5  # Increased from 3 seconds
                
                for attempt in range(max_retries):
                    try:
                        response = requests.post(
                            f"{self.VOYAGE_BASE_URL}/embeddings",
                            headers=headers,
                            json=data,
                            timeout=30,
                        )
                        
                        if response.status_code == 429:
                            if attempt < max_retries - 1:
                                # More aggressive exponential backoff with jitter
                                delay = base_delay * (2 ** attempt) + random.uniform(0.5, 2.0)
                                logger.warning(
                                    "VoyageAI rate limit hit, retrying in %.1f seconds (attempt %d/%d)",
                                    delay, attempt + 1, max_retries
                                )
                                time.sleep(delay)
                                continue
                            else:
                                logger.error(
                                    "VoyageAI rate limit exceeded after %d retries. "
                                    "Consider reducing batch size or waiting longer before retrying.",
                                    max_retries
                                )
                                raise ValueError("VoyageAI rate limit exceeded. Please try again later.")
                        
                        response.raise_for_status()
                        result = response.json()
                        embeddings = result.get("data") or []
                        if not embeddings:
                            raise ValueError("No embedding data received from Voyage API")

                        embedding = (embeddings[0] or {}).get("embedding")
                        if not embedding:
                            raise ValueError("Invalid response from Voyage embeddings API")

                        return embedding
                        
                    except requests.exceptions.RequestException as e:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0.5, 2.0)
                            logger.warning(
                                "VoyageAI request failed, retrying in %.1f seconds (attempt %d/%d): %s",
                                delay, attempt + 1, max_retries, str(e)
                            )
                            time.sleep(delay)
                            continue
                        else:
                            raise

            if provider == "cohere":
                client = self._get_cohere_client()
                cohere_model = model.replace(self.COHERE_MODEL_PREFIX, "")
                
                # Map input_type to Cohere's format
                cohere_input_type = "search_query" if input_type == "query" else "search_document"
                
                response = client.embed(
                    texts=[text.strip()],
                    model=cohere_model,
                    input_type=cohere_input_type,
                )
                embeddings = getattr(response, "embeddings", None)
                if not embeddings:
                    raise ValueError("No embedding data received from Cohere")
                return embeddings[0]

            if provider == "jina":
                api_key = self._get_jina_ai_client()
                jina_model = model.replace(self.JINA_AI_MODEL_PREFIX, "")
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                data = {
                    "model": jina_model,
                    "input": [text.strip()],
                }
                response = requests.post(
                    f"{self.JINA_AI_BASE_URL}/embeddings",
                    headers=headers,
                    json=data,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                items = result.get("data") or []
                if not items:
                    raise ValueError("No embedding data received from Jina AI API")
                embedding = (items[0] or {}).get("embedding")
                if not embedding:
                    raise ValueError("Invalid response from Jina AI embeddings API")
                return embedding
            
            # Ollama local models use HTTP requests
            if provider == "ollama":
                logger.debug("ENTERING OLLAMA BRANCH")
                ollama_model = model.replace(self.OLLAMA_MODEL_PREFIX, "")
                logger.debug("Ollama model: %s", ollama_model)
                
                headers = {
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": ollama_model,
                    "prompt": self._prepare_ollama_prompt(text)
                }
                
                import time
                max_retries = 3
                retry_delay = 2  # seconds
                used_fallback = False
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Ollama attempt {attempt + 1}/{max_retries}")
                        with self._ollama_lock:
                            response = requests.post(
                                f"{self.OLLAMA_BASE_URL}/api/embeddings",
                                headers=headers,
                                json=data,
                                timeout=60  # Increased from 30 to 60 seconds
                            )
                        
                        logger.debug("Ollama response status: %s", response.status_code)
                        
                        if response.status_code == 429:
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Ollama rate limit hit, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                                )
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            else:
                                raise ValueError("Ollama rate limit exceeded. Please try again later.")
                        
                        response.raise_for_status()
                        result = response.json()
                        
                        logger.debug("Ollama response keys: %s", list(result.keys()))
                        
                        if "embedding" not in result:
                            raise ValueError("Invalid response from Ollama API")
                        
                        embedding = result["embedding"]
                        dimension = len(embedding)
                        
                        logger.info("Ollama embedding generated successfully")
                        logger.info("Model: %s, Dimension: %s", ollama_model, dimension)
                        
                        return embedding
                        
                    except requests.exceptions.RequestException as e:
                        resp = getattr(e, "response", None)
                        if resp is not None:
                            # Ollama sometimes returns 500 when the embedding contains NaN and it can't JSON encode.
                            # In that case, fall back to a secondary local Ollama embedding model (no OpenAI fallback).
                            if resp.status_code == 500 and self._is_ollama_nan_error(resp.text) and not used_fallback:
                                fallback_model = self._get_ollama_fallback_model()
                                requested_model = data.get("model")
                                expected_dim = self._get_expected_ollama_dim(requested_model)

                                if fallback_model and fallback_model != requested_model:
                                    logger.warning(
                                        "Ollama returned NaN embedding for model '%s'; falling back to '%s'",
                                        requested_model,
                                        fallback_model,
                                    )
                                    used_fallback = True
                                    fallback_data = {
                                        "model": fallback_model,
                                        "prompt": data.get("prompt", ""),
                                    }
                                    with self._ollama_lock:
                                        fallback_resp = requests.post(
                                            f"{self.OLLAMA_BASE_URL}/api/embeddings",
                                            headers=headers,
                                            json=fallback_data,
                                            timeout=60,
                                        )
                                    fallback_resp.raise_for_status()
                                    fallback_json = fallback_resp.json()
                                    if "embedding" not in fallback_json:
                                        raise ValueError("Invalid response from Ollama API (fallback model)")
                                    embedding = fallback_json["embedding"]
                                    if expected_dim is not None and len(embedding) != expected_dim:
                                        logger.warning(
                                            "Ollama fallback embedding dimension mismatch: requested_dim=%s fallback_dim=%s. Returning zero-vector.",
                                            expected_dim,
                                            len(embedding),
                                        )
                                        return self._zero_vector(expected_dim)

                                    return embedding

                                if expected_dim is not None:
                                    logger.warning(
                                        "Ollama returned NaN embedding and no compatible fallback available. Returning zero-vector dim=%s",
                                        expected_dim,
                                    )
                                    return self._zero_vector(expected_dim)

                            logger.error(
                                "Ollama request failed: %s (status=%s, body=%s)",
                                e,
                                resp.status_code,
                                (resp.text or "")[:500],
                            )
                        else:
                            logger.error("Ollama request failed: %s", e)
                        if attempt < max_retries - 1:
                            # Check if it's a timeout
                            if "timeout" in str(e).lower():
                                logger.warning(f"Ollama timeout, retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                            else:
                                logger.warning(f"Ollama request failed, retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                        raise
            
            # OpenAI-compatible providers (OpenRouter, Alibaba)
            logger.debug("ENTERING OPENAI-COMPATIBLE BRANCH")
            logger.info(f"Getting client for model: {model}, provider: {provider}")
            client = self._get_client_for_model(model)
            
            logger.debug("Client returned: %s", client)
            
            # Check if client is None (shouldn't happen for OpenAI-compatible providers)
            if client is None:
                raise ValueError(f"Client is None for provider: {provider}")
            
            logger.debug("Calling embeddings.create with model: %s", model)
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
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def get_embeddings(
        self, texts: List[str], model: str = None, input_type: str = None
    ) -> List[List[float]]:
        """Get embeddings for multiple texts.
        
        Now supports multiple providers with appropriate batch sizes.
        Routes to OpenRouter for "openai/*" models (batch size: 100).
        Routes to DashScope for "alibaba/*" models (batch size: 10).
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            input_type: Context type - "query" for search queries,
                       "document" for indexing. Defaults to "document".
                       Supported by Voyage AI and Cohere providers.
            
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
        
        # Default to "document" for backward compatibility
        if input_type is None:
            input_type = "document"
            
        # Filter empty texts but track indices
        non_empty = [(i, t.strip()) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            return [[] for _ in texts]
            
        model = model or self.DEFAULT_MODEL
        provider = self._get_provider_for_model(model)
        
        # Determine batch size based on provider
        batch_size_map = {
            "ollama": self.OLLAMA_BATCH_SIZE,
            "cohere": self.COHERE_BATCH_SIZE,
            "jina": self.JINA_AI_BATCH_SIZE,
            "voyage": self.VOYAGE_BATCH_SIZE,
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
                        # Map input_type to Cohere's format
                        cohere_input_type = "search_query" if input_type == "query" else "search_document"
                        
                        response = client.embed(
                            texts=batch_texts,
                            model=cohere_model,
                            input_type=cohere_input_type
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
            
            # Jina AI uses HTTP requests
            elif provider == "jina":
                api_key = self._get_jina_ai_client()
                jina_model = model.replace(self.JINA_AI_MODEL_PREFIX, "")
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
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
                        data = {
                            "model": jina_model,
                            "input": batch_texts
                        }
                        
                        # Add retry logic for batch processing
                        batch_max_retries = 3
                        batch_retry_delay = 2
                        
                        for batch_attempt in range(batch_max_retries):
                            try:
                                response = requests.post(
                                    f"{self.JINA_AI_BASE_URL}/embeddings",
                                    headers=headers,
                                    json=data,
                                    timeout=30
                                )
                                
                                if response.status_code == 429:
                                    if batch_attempt < batch_max_retries - 1:
                                        logger.warning(
                                            f"Jina AI batch rate limit hit, retrying in {batch_retry_delay}s... (batch {batch_num}, attempt {batch_attempt + 1}/{batch_max_retries})"
                                        )
                                        time.sleep(batch_retry_delay)
                                        batch_retry_delay *= 2
                                        continue
                                    else:
                                        raise ValueError("Jina AI rate limit exceeded during batch processing. Please try again later.")
                                
                                response.raise_for_status()
                                result = response.json()
                                
                                if "data" not in result or not result["data"]:
                                    raise ValueError("Invalid response from Jina AI API")
                                
                                for j, embedding_data in enumerate(result["data"]):
                                    original_idx = batch_indices[j]
                                    all_embeddings[original_idx] = embedding_data["embedding"]
                                
                                break  # Success, exit retry loop
                                
                            except requests.exceptions.RequestException as e:
                                if batch_attempt < batch_max_retries - 1:
                                    logger.warning(f"Jina AI batch request failed, retrying... (batch {batch_num}, attempt {batch_attempt + 1}/{batch_max_retries}): {e}")
                                    time.sleep(batch_retry_delay)
                                    batch_retry_delay *= 2
                                    continue
                                raise
                        
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
                            f"Error in batch {batch_num} for {provider}: {e}",
                            extra={
                                "model": model,
                                "provider": provider,
                                "batch_num": batch_num,
                                "error": str(e)
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

            elif provider == "voyage":
                api_key = self._get_client_for_model(model)
                voyage_model = model.replace(self.VOYAGE_MODEL_PREFIX, "")

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                all_embeddings = {}
                total_batches = (len(non_empty) + batch_size - 1) // batch_size
                max_retries = 8  # Increased from 5
                base_delay = 5  # Increased from 3 seconds

                for batch_start in range(0, len(non_empty), batch_size):
                    batch = non_empty[batch_start:batch_start + batch_size]
                    batch_texts = [t for _, t in batch]
                    batch_indices = [i for i, _ in batch]
                    batch_num = (batch_start // batch_size) + 1

                    # Add longer delay between batches to avoid rate limiting
                    if batch_start > 0:  # Don't delay first batch
                        batch_delay = random.uniform(2.0, 4.0)  # Increased from 1.0-2.0
                        logger.debug(
                            "VoyageAI: Waiting %.1f seconds before batch %d/%d",
                            batch_delay, batch_num, total_batches
                        )
                        time.sleep(batch_delay)

                    data = {
                        "input": batch_texts,
                        "model": voyage_model,
                        "input_type": input_type,  # Use parameter instead of hardcoded "document"
                    }

                    # Enhanced retry logic for rate limiting
                    for attempt in range(max_retries):
                        try:
                            response = requests.post(
                                f"{self.VOYAGE_BASE_URL}/embeddings",
                                headers=headers,
                                json=data,
                                timeout=30,
                            )
                            

                            if response.status_code == 429:
                                if attempt < max_retries - 1:
                                    # More aggressive exponential backoff with jitter
                                    delay = base_delay * (2 ** attempt) + random.uniform(1.0, 3.0)
                                    logger.warning(
                                        "VoyageAI batch rate limit hit, retrying in %.1f seconds (attempt %d/%d, batch %d/%d)",
                                        delay, attempt + 1, max_retries,
                                        batch_num, total_batches
                                    )
                                    time.sleep(delay)
                                    continue
                                else:
                                    logger.error(
                                        "VoyageAI rate limit exceeded after %d retries in batch %d/%d. "
                                        "Consider reducing batch size further or waiting longer.",
                                        max_retries, batch_num, total_batches
                                    )
                                    raise ValueError("VoyageAI rate limit exceeded. Please try again later.")
                            

                            response.raise_for_status()
                            result = response.json()
                            embeddings = result.get("data") or []
                            

                            if len(embeddings) != len(batch_texts):
                                raise ValueError("Invalid response from Voyage embeddings API")
                            

                            for j, embedding_data in enumerate(embeddings):
                                original_idx = batch_indices[j]
                                all_embeddings[original_idx] = embedding_data.get("embedding")
                            

                            logger.debug(
                                "VoyageAI: Successfully processed batch %d/%d with %d texts",
                                batch_num, total_batches, len(batch_texts)
                            )
                            break  # Success, exit retry loop
                            

                        except requests.exceptions.RequestException as e:
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt) + random.uniform(1.0, 3.0)
                                logger.warning(
                                    "VoyageAI batch request failed, retrying in %.1f seconds (attempt %d/%d, batch %d/%d): %s",
                                    delay, attempt + 1, max_retries,
                                    batch_num, total_batches, str(e)
                                )
                                time.sleep(delay)
                                continue
                            else:
                                logger.error(
                                    "VoyageAI batch request failed after %d retries in batch %d/%d: %s",
                                    max_retries, batch_num, total_batches, str(e)
                                )
                                raise

                result = []
                for i in range(len(texts)):
                    if i in all_embeddings:
                        result.append(all_embeddings[i])
                    else:
                        result.append([])

                return result
            
            # HuggingFace BGE models use HTTP requests
            elif provider == "huggingface":
                api_key = self._get_huggingface_client()
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
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
                        data = {
                            "model": "BAAI/bge-base-multilingual",
                            "input": batch_texts
                        }
                        
                        response = requests.post(
                            f"{self.HUGGINGFACE_BASE_URL}/v1/models/BAAI/bge-base-multilingual",
                            headers=headers,
                            json=data,
                            timeout=30
                        )
                        
                        if response.status_code == 429:
                            logger.warning(
                                f"HuggingFace rate limit hit in batch {batch_num}, waiting..."
                            )
                            time.sleep(2)
                            continue
                        
                        response.raise_for_status()
                        result = response.json()
                        
                        if not isinstance(result, list) or len(result) != len(batch_texts):
                            raise ValueError("Invalid response from HuggingFace API")
                        
                        for j, embedding in enumerate(result):
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
                            f"Error in batch {batch_num} for {provider}: {e}",
                            extra={
                                "model": model,
                                "provider": provider,
                                "batch_num": batch_num,
                                "error": str(e)
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
            
            # Ollama local models use HTTP requests
            elif provider == "ollama":
                logger.debug("ENTERING OLLAMA BATCH BRANCH")
                ollama_model = model.replace(self.OLLAMA_MODEL_PREFIX, "")
                logger.debug("Ollama batch model: %s", ollama_model)
                
                headers = {
                    "Content-Type": "application/json"
                }
                
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
                        # Ollama doesn't support batch embedding, so process one by one
                        batch_embeddings = []
                        for text in batch_texts:
                            data = {
                                "model": ollama_model,
                                "prompt": self._prepare_ollama_prompt(text)
                            }
                            
                            # Retry logic for each text in batch
                            max_retries = 3
                            retry_delay = 2
                            text_embedding = None
                            used_fallback = False
                            
                            for attempt in range(max_retries):
                                try:
                                    with self._ollama_lock:
                                        response = requests.post(
                                            f"{self.OLLAMA_BASE_URL}/api/embeddings",
                                            headers=headers,
                                            json=data,
                                            timeout=60  # Increased from 30 to 60 seconds
                                        )
                                    
                                    if response.status_code == 429:
                                        if attempt < max_retries - 1:
                                            logger.warning(
                                                f"Ollama rate limit hit in batch {batch_num}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                                            )
                                            time.sleep(retry_delay)
                                            retry_delay *= 2
                                            continue
                                        else:
                                            raise ValueError("Ollama rate limit exceeded in batch processing")
                                    
                                    response.raise_for_status()
                                    result = response.json()
                                    
                                    if "embedding" not in result:
                                        raise ValueError("Invalid response from Ollama API")
                                    
                                    text_embedding = result["embedding"]
                                    break
                                    
                                except requests.exceptions.RequestException as e:
                                    resp = getattr(e, "response", None)
                                    if resp is not None:
                                        if resp.status_code == 500 and self._is_ollama_nan_error(resp.text) and not used_fallback:
                                            fallback_model = self._get_ollama_fallback_model()
                                            requested_model = data.get("model")
                                            expected_dim = self._get_expected_ollama_dim(requested_model)

                                            if fallback_model and fallback_model != requested_model:
                                                logger.warning(
                                                    "Ollama returned NaN embedding for model '%s'; falling back to '%s' (batch)",
                                                    requested_model,
                                                    fallback_model,
                                                )
                                                used_fallback = True
                                                fallback_data = {
                                                    "model": fallback_model,
                                                    "prompt": data.get("prompt", ""),
                                                }
                                                with self._ollama_lock:
                                                    fallback_resp = requests.post(
                                                        f"{self.OLLAMA_BASE_URL}/api/embeddings",
                                                        headers=headers,
                                                        json=fallback_data,
                                                        timeout=60,
                                                    )
                                                fallback_resp.raise_for_status()
                                                fallback_json = fallback_resp.json()
                                                if "embedding" not in fallback_json:
                                                    raise ValueError("Invalid response from Ollama API (fallback model)")
                                                text_embedding = fallback_json["embedding"]
                                                if expected_dim is not None and len(text_embedding) != expected_dim:
                                                    logger.warning(
                                                        "Ollama fallback embedding dimension mismatch (batch): requested_dim=%s fallback_dim=%s. Using zero-vector.",
                                                        expected_dim,
                                                        len(text_embedding),
                                                    )
                                                    text_embedding = self._zero_vector(expected_dim)
                                                break

                                            if expected_dim is not None:
                                                logger.warning(
                                                    "Ollama returned NaN embedding and no compatible fallback available (batch). Using zero-vector dim=%s",
                                                    expected_dim,
                                                )
                                                text_embedding = self._zero_vector(expected_dim)
                                                break

                                        logger.error(
                                            "Ollama batch request failed: %s (status=%s, body=%s, prompt_len=%s)",
                                            e,
                                            resp.status_code,
                                            (resp.text or "")[:500],
                                            len(data.get("prompt") or ""),
                                        )
                                    else:
                                        logger.error(
                                            "Ollama batch request failed: %s (prompt_len=%s)",
                                            e,
                                            len(data.get("prompt") or ""),
                                        )
                                    if attempt < max_retries - 1:
                                        # Check if it's a timeout
                                        if "timeout" in str(e).lower():
                                            logger.warning(f"Ollama timeout in batch {batch_num}, retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                                        else:
                                            logger.warning(f"Ollama batch request failed, retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                                        time.sleep(retry_delay)
                                        retry_delay *= 2
                                        continue
                                    raise
                            
                            if text_embedding is None:
                                raise ValueError(f"Failed to generate embedding for text after {max_retries} attempts")
                            
                            batch_embeddings.append(text_embedding)
                        
                        # Map results back to original indices
                        for j, embedding in enumerate(batch_embeddings):
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
                            f"Error in batch {batch_num} for {provider}: {e}",
                            extra={
                                "model": model,
                                "provider": provider,
                                "batch_num": batch_num,
                                "error": str(e)
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
                    
                    # Debug: Check response structure
                    logger.debug(f"OpenRouter response structure: {type(response)}")
                    logger.debug(f"OpenRouter response dir: {[attr for attr in dir(response) if not attr.startswith('_')]}")
                    
                    if hasattr(response, 'data') and response.data:
                        logger.debug(f"OpenRouter response data length: {len(response.data)}")
                        logger.debug(f"OpenRouter first embedding shape: {len(response.data[0].embedding) if response.data[0].embedding else 'None'}")
                    else:
                        logger.error(f"OpenRouter response missing data attribute or empty data")
                        logger.error(f"Full response: {response}")
                        raise ValueError("No embedding data received from OpenRouter")
                    
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
                "jina": "JINA_AI_API_KEY",
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
            "qwen/qwen3-embedding-8b": 1024,
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
