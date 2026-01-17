"""Configuration management for the Semantic Chunker.

This module provides centralized configuration with environment variable
support and validation.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from app.services.chunker_exceptions import ConfigurationError, ErrorCode

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunkerConfig:
    """Centralized configuration for semantic chunking operations.
    
    All configuration can be loaded from environment variables with
    the CHUNKER_ prefix.
    
    Attributes:
        # Chunking parameters
        chunk_size: Target chunk size in characters (default: 500)
        min_chunk_size: Minimum chunk size (default: 150)
        max_chunk_size: Maximum chunk size (default: 2000)
        overlap: Overlap between chunks (default: 50)
        buffer_size: Sentences on each side for context (default: 1)
        
        # Threshold settings
        similarity_threshold: Base similarity threshold (default: 0.5)
        enable_adaptive_threshold: Use adaptive threshold (default: True)
        threshold_min: Minimum threshold value (default: 0.3)
        threshold_max: Maximum threshold value (default: 0.9)
        
        # Feature flags
        enable_qa_detection: Detect and merge Q&A pairs (default: True)
        enable_cache: Enable embedding cache (default: True)
        use_provider_manager: Use provider manager with fallback (default: True)
        
        # Embedding settings
        embedding_model: Default embedding model
        embedding_batch_size: Batch size for embeddings (default: 32)
        embedding_timeout: Timeout for embedding requests (default: 30.0)
        embedding_max_retries: Max retries for embedding (default: 3)
        
        # Preprocessing settings
        enable_preprocessing: Enable text preprocessing (default: True)
        merge_short_sentences: Merge short sentences (default: True)
        min_sentence_length: Minimum sentence length to keep (default: 10)
        max_sentence_length: Maximum sentence length before split (default: 1000)
        
        # Logging settings
        log_level: Logging level (default: INFO)
        log_preprocessing: Log preprocessing steps (default: False)
    """
    # Chunking parameters
    chunk_size: int = 500
    min_chunk_size: int = 150
    max_chunk_size: int = 2000
    overlap: int = 50
    buffer_size: int = 1
    
    # Threshold settings
    similarity_threshold: float = 0.5
    enable_adaptive_threshold: bool = True
    threshold_min: float = 0.3
    threshold_max: float = 0.9
    
    # Feature flags
    enable_qa_detection: bool = True
    enable_cache: bool = True
    use_provider_manager: bool = True
    
    # Embedding settings
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_batch_size: int = 32
    embedding_timeout: float = 30.0
    embedding_max_retries: int = 3
    
    # Preprocessing settings
    enable_preprocessing: bool = True
    merge_short_sentences: bool = True
    min_sentence_length: int = 10
    max_sentence_length: int = 1000
    
    # Logging settings
    log_level: str = "INFO"
    log_preprocessing: bool = False
    
    @classmethod
    def from_env(cls) -> "SemanticChunkerConfig":
        """Load configuration from environment variables.
        
        Environment variables use CHUNKER_ prefix:
        - CHUNKER_CHUNK_SIZE
        - CHUNKER_MIN_CHUNK_SIZE
        - CHUNKER_SIMILARITY_THRESHOLD
        - etc.
        
        Returns:
            SemanticChunkerConfig with values from environment
        """
        def get_env_int(key: str, default: int) -> int:
            value = os.environ.get(f"CHUNKER_{key}")
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                logger.warning(f"Invalid int for CHUNKER_{key}: {value}")
                return default
        
        def get_env_float(key: str, default: float) -> float:
            value = os.environ.get(f"CHUNKER_{key}")
            if value is None:
                return default
            try:
                return float(value)
            except ValueError:
                logger.warning(f"Invalid float for CHUNKER_{key}: {value}")
                return default
        
        def get_env_bool(key: str, default: bool) -> bool:
            value = os.environ.get(f"CHUNKER_{key}")
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes", "on")
        
        def get_env_str(key: str, default: str) -> str:
            return os.environ.get(f"CHUNKER_{key}", default)
        
        return cls(
            # Chunking parameters
            chunk_size=get_env_int("CHUNK_SIZE", 500),
            min_chunk_size=get_env_int("MIN_CHUNK_SIZE", 150),
            max_chunk_size=get_env_int("MAX_CHUNK_SIZE", 2000),
            overlap=get_env_int("OVERLAP", 50),
            buffer_size=get_env_int("BUFFER_SIZE", 1),
            
            # Threshold settings
            similarity_threshold=get_env_float("SIMILARITY_THRESHOLD", 0.5),
            enable_adaptive_threshold=get_env_bool("ENABLE_ADAPTIVE_THRESHOLD", True),
            threshold_min=get_env_float("THRESHOLD_MIN", 0.3),
            threshold_max=get_env_float("THRESHOLD_MAX", 0.9),
            
            # Feature flags
            enable_qa_detection=get_env_bool("ENABLE_QA_DETECTION", True),
            enable_cache=get_env_bool("ENABLE_CACHE", True),
            use_provider_manager=get_env_bool("USE_PROVIDER_MANAGER", True),
            
            # Embedding settings
            embedding_model=get_env_str("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            embedding_batch_size=get_env_int("EMBEDDING_BATCH_SIZE", 32),
            embedding_timeout=get_env_float("EMBEDDING_TIMEOUT", 30.0),
            embedding_max_retries=get_env_int("EMBEDDING_MAX_RETRIES", 3),
            
            # Preprocessing settings
            enable_preprocessing=get_env_bool("ENABLE_PREPROCESSING", True),
            merge_short_sentences=get_env_bool("MERGE_SHORT_SENTENCES", True),
            min_sentence_length=get_env_int("MIN_SENTENCE_LENGTH", 10),
            max_sentence_length=get_env_int("MAX_SENTENCE_LENGTH", 1000),
            
            # Logging settings
            log_level=get_env_str("LOG_LEVEL", "INFO"),
            log_preprocessing=get_env_bool("LOG_PREPROCESSING", False),
        )
    
    def validate(self) -> List[str]:
        """Validate configuration parameters.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Chunk size validation
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        if self.min_chunk_size <= 0:
            errors.append("min_chunk_size must be positive")
        if self.max_chunk_size <= self.min_chunk_size:
            errors.append("max_chunk_size must be greater than min_chunk_size")
        if self.chunk_size < self.min_chunk_size:
            errors.append("chunk_size should be >= min_chunk_size")
        if self.chunk_size > self.max_chunk_size:
            errors.append("chunk_size should be <= max_chunk_size")
        
        # Overlap validation
        if self.overlap < 0:
            errors.append("overlap cannot be negative")
        if self.overlap >= self.chunk_size:
            errors.append("overlap must be less than chunk_size")
        
        # Buffer validation
        if self.buffer_size < 0:
            errors.append("buffer_size cannot be negative")
        
        # Threshold validation
        if not 0 <= self.similarity_threshold <= 1:
            errors.append("similarity_threshold must be between 0 and 1")
        if not 0 <= self.threshold_min <= 1:
            errors.append("threshold_min must be between 0 and 1")
        if not 0 <= self.threshold_max <= 1:
            errors.append("threshold_max must be between 0 and 1")
        if self.threshold_min >= self.threshold_max:
            errors.append("threshold_min must be less than threshold_max")
        
        # Embedding validation
        if self.embedding_batch_size <= 0:
            errors.append("embedding_batch_size must be positive")
        if self.embedding_timeout <= 0:
            errors.append("embedding_timeout must be positive")
        if self.embedding_max_retries < 0:
            errors.append("embedding_max_retries cannot be negative")
        
        # Sentence length validation
        if self.min_sentence_length < 0:
            errors.append("min_sentence_length cannot be negative")
        if self.max_sentence_length <= self.min_sentence_length:
            errors.append("max_sentence_length must be greater than min_sentence_length")
        
        return errors
    
    def validate_or_raise(self) -> None:
        """Validate configuration and raise exception if invalid.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        errors = self.validate()
        if errors:
            raise ConfigurationError(
                f"Invalid configuration: {'; '.join(errors)}",
                details={"validation_errors": errors}
            )
    
    def check_api_keys(self) -> Dict[str, bool]:
        """Check if required API keys are configured.
        
        Returns:
            Dict mapping provider name to availability status
        """
        return {
            "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "chunk_size": self.chunk_size,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "overlap": self.overlap,
            "buffer_size": self.buffer_size,
            "similarity_threshold": self.similarity_threshold,
            "enable_adaptive_threshold": self.enable_adaptive_threshold,
            "threshold_min": self.threshold_min,
            "threshold_max": self.threshold_max,
            "enable_qa_detection": self.enable_qa_detection,
            "enable_cache": self.enable_cache,
            "use_provider_manager": self.use_provider_manager,
            "embedding_model": self.embedding_model,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_timeout": self.embedding_timeout,
            "embedding_max_retries": self.embedding_max_retries,
            "enable_preprocessing": self.enable_preprocessing,
            "merge_short_sentences": self.merge_short_sentences,
            "min_sentence_length": self.min_sentence_length,
            "max_sentence_length": self.max_sentence_length,
            "log_level": self.log_level,
            "log_preprocessing": self.log_preprocessing,
        }


# Global configuration instance (lazy loaded)
_global_config: Optional[SemanticChunkerConfig] = None


def get_chunker_config() -> SemanticChunkerConfig:
    """Get the global chunker configuration.
    
    Loads from environment variables on first call.
    
    Returns:
        SemanticChunkerConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = SemanticChunkerConfig.from_env()
    return _global_config


def reset_chunker_config() -> None:
    """Reset the global configuration (for testing)."""
    global _global_config
    _global_config = None
