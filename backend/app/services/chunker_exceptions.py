"""Custom exception classes for the Semantic Chunker.

This module defines a hierarchy of exceptions for error handling
in the semantic chunking pipeline.
"""

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    """Error codes for semantic chunker exceptions."""
    # General errors (1000-1099)
    UNKNOWN_ERROR = "CHUNK_1000"
    CONFIGURATION_ERROR = "CHUNK_1001"
    VALIDATION_ERROR = "CHUNK_1002"
    
    # Language detection errors (1100-1199)
    LANGUAGE_DETECTION_FAILED = "CHUNK_1100"
    UNSUPPORTED_LANGUAGE = "CHUNK_1101"
    
    # Sentence tokenization errors (1200-1299)
    TOKENIZATION_FAILED = "CHUNK_1200"
    EMPTY_TEXT = "CHUNK_1201"
    INVALID_TEXT_FORMAT = "CHUNK_1202"
    
    # Embedding provider errors (1300-1399)
    EMBEDDING_PROVIDER_UNAVAILABLE = "CHUNK_1300"
    EMBEDDING_API_ERROR = "CHUNK_1301"
    EMBEDDING_TIMEOUT = "CHUNK_1302"
    ALL_PROVIDERS_FAILED = "CHUNK_1303"
    
    # Cache errors (1400-1499)
    CACHE_READ_ERROR = "CHUNK_1400"
    CACHE_WRITE_ERROR = "CHUNK_1401"
    CACHE_CONNECTION_ERROR = "CHUNK_1402"


class SemanticChunkerError(Exception):
    """Base exception for all semantic chunker errors.
    
    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        details: Additional context about the error
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Error code from ErrorCode enum
            details: Additional context dictionary
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details
        }
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"[{self.error_code.value}] {self.message}"


class LanguageDetectionError(SemanticChunkerError):
    """Exception raised when language detection fails.
    
    This can occur when:
    - The langdetect library is not available
    - The text is too short for reliable detection
    - The text contains unsupported characters
    """
    
    def __init__(
        self,
        message: str,
        text_sample: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize language detection error.
        
        Args:
            message: Error description
            text_sample: Sample of text that failed detection (truncated)
            details: Additional context
        """
        error_details = details or {}
        if text_sample:
            # Truncate for safety
            error_details["text_sample"] = text_sample[:100]
            error_details["text_length"] = len(text_sample)
        
        super().__init__(
            message=message,
            error_code=ErrorCode.LANGUAGE_DETECTION_FAILED,
            details=error_details
        )


class SentenceTokenizationError(SemanticChunkerError):
    """Exception raised when sentence tokenization fails.
    
    This can occur when:
    - NLTK punkt data is not available
    - The text format is invalid
    - Custom tokenization patterns fail
    """
    
    def __init__(
        self,
        message: str,
        text_length: Optional[int] = None,
        language: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize tokenization error.
        
        Args:
            message: Error description
            text_length: Length of text that failed tokenization
            language: Detected language (if available)
            details: Additional context
        """
        error_details = details or {}
        if text_length is not None:
            error_details["text_length"] = text_length
        if language:
            error_details["language"] = language
        
        super().__init__(
            message=message,
            error_code=ErrorCode.TOKENIZATION_FAILED,
            details=error_details
        )


class CacheError(SemanticChunkerError):
    """Exception raised when cache operations fail.
    
    This can occur when:
    - Cache read/write operations fail
    - Cache connection is lost
    - Cache data is corrupted
    """
    
    def __init__(
        self,
        message: str,
        operation: str = "unknown",
        cache_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize cache error.
        
        Args:
            message: Error description
            operation: Cache operation that failed (get, set, delete)
            cache_key: Key involved in the failed operation
            details: Additional context
        """
        error_details = details or {}
        error_details["operation"] = operation
        if cache_key:
            # Truncate key for safety
            error_details["cache_key"] = cache_key[:50]
        
        # Determine specific error code
        if operation == "get":
            error_code = ErrorCode.CACHE_READ_ERROR
        elif operation == "set":
            error_code = ErrorCode.CACHE_WRITE_ERROR
        else:
            error_code = ErrorCode.CACHE_CONNECTION_ERROR
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=error_details
        )


class ConfigurationError(SemanticChunkerError):
    """Exception raised when configuration is invalid.
    
    This can occur when:
    - Required configuration values are missing
    - Configuration values are out of valid range
    - Conflicting configuration options are set
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize configuration error.
        
        Args:
            message: Error description
            config_key: Configuration key that is invalid
            config_value: Invalid value (if safe to include)
            details: Additional context
        """
        error_details = details or {}
        if config_key:
            error_details["config_key"] = config_key
        if config_value is not None:
            error_details["config_value"] = str(config_value)
        
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            details=error_details
        )
