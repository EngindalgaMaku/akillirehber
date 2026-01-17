"""Structured logging utilities for the Semantic Chunker.

This module provides enhanced logging with structured context
for better debugging and monitoring.
"""

import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class ChunkerLogContext:
    """Structured context for chunker logging.
    
    Attributes:
        text_length: Length of input text
        strategy: Chunking strategy used
        provider: Embedding provider name
        model: Embedding model name
        operation: Current operation being performed
        duration_ms: Operation duration in milliseconds
        error_type: Type of error if any
        error_message: Error message if any
        chunk_count: Number of chunks produced
        fallback_used: Whether fallback was triggered
    """
    text_length: int = 0
    strategy: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    operation: str = ""
    duration_ms: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    chunk_count: Optional[int] = None
    fallback_used: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def log_chunking_operation(
    level: int,
    message: str,
    context: ChunkerLogContext,
    exc_info: bool = False
) -> None:
    """Log a chunking operation with structured context.
    
    Args:
        level: Logging level (logging.INFO, logging.ERROR, etc.)
        message: Log message
        context: Structured context
        exc_info: Whether to include exception info
    """
    extra = {"chunker_context": context.to_dict()}
    logger.log(level, message, extra=extra, exc_info=exc_info)


def log_error(
    message: str,
    error: Exception,
    context: ChunkerLogContext
) -> None:
    """Log an error with full context.
    
    Args:
        message: Error description
        error: Exception that occurred
        context: Structured context
    """
    context.error_type = type(error).__name__
    context.error_message = str(error)
    log_chunking_operation(logging.ERROR, message, context, exc_info=True)


def log_warning(
    message: str,
    context: ChunkerLogContext
) -> None:
    """Log a warning with context.
    
    Args:
        message: Warning message
        context: Structured context
    """
    log_chunking_operation(logging.WARNING, message, context)


def log_info(
    message: str,
    context: ChunkerLogContext
) -> None:
    """Log info with context.
    
    Args:
        message: Info message
        context: Structured context
    """
    log_chunking_operation(logging.INFO, message, context)


def log_debug(
    message: str,
    context: ChunkerLogContext
) -> None:
    """Log debug info with context.
    
    Args:
        message: Debug message
        context: Structured context
    """
    log_chunking_operation(logging.DEBUG, message, context)


class ChunkerOperationLogger:
    """Context manager for logging chunking operations with timing.
    
    Usage:
        with ChunkerOperationLogger("embedding", text_length=1000) as op_logger:
            # perform operation
            op_logger.set_result(chunk_count=5)
    """
    
    def __init__(
        self,
        operation: str,
        text_length: int = 0,
        strategy: str = "",
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Initialize the operation logger.
        
        Args:
            operation: Name of the operation
            text_length: Length of input text
            strategy: Chunking strategy
            provider: Embedding provider
            model: Embedding model
        """
        self.context = ChunkerLogContext(
            text_length=text_length,
            strategy=strategy,
            provider=provider,
            model=model,
            operation=operation
        )
        self.start_time: Optional[float] = None
        self._error: Optional[Exception] = None
    
    def __enter__(self) -> "ChunkerOperationLogger":
        """Start timing the operation."""
        self.start_time = time.time()
        log_debug(f"Starting {self.context.operation}", self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Log operation completion or error."""
        if self.start_time:
            self.context.duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_val:
            self._error = exc_val
            log_error(
                f"Operation {self.context.operation} failed",
                exc_val,
                self.context
            )
        else:
            log_info(
                f"Operation {self.context.operation} completed",
                self.context
            )
        
        return False  # Don't suppress exceptions
    
    def set_result(
        self,
        chunk_count: Optional[int] = None,
        fallback_used: Optional[str] = None
    ) -> None:
        """Set operation result details.
        
        Args:
            chunk_count: Number of chunks produced
            fallback_used: Fallback strategy used, if any
        """
        if chunk_count is not None:
            self.context.chunk_count = chunk_count
        if fallback_used is not None:
            self.context.fallback_used = fallback_used
    
    def log_fallback(self, fallback_name: str, reason: str) -> None:
        """Log that a fallback was triggered.
        
        Args:
            fallback_name: Name of fallback strategy
            reason: Reason for fallback
        """
        self.context.fallback_used = fallback_name
        log_warning(
            f"Fallback to {fallback_name}: {reason}",
            self.context
        )


def with_logging(operation_name: str):
    """Decorator to add structured logging to a function.
    
    Args:
        operation_name: Name of the operation for logging
        
    Returns:
        Decorated function with logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract text_length if available
            text_length = 0
            if args and isinstance(args[0], str):
                text_length = len(args[0])
            elif 'text' in kwargs and isinstance(kwargs['text'], str):
                text_length = len(kwargs['text'])
            
            context = ChunkerLogContext(
                operation=operation_name,
                text_length=text_length
            )
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                context.duration_ms = (time.time() - start_time) * 1000
                log_info(f"{operation_name} completed", context)
                return result
            except Exception as e:
                context.duration_ms = (time.time() - start_time) * 1000
                log_error(f"{operation_name} failed", e, context)
                raise
        
        return wrapper
    return decorator



@dataclass
class PreprocessingDiagnostics:
    """Diagnostics for preprocessing transformations.
    
    Attributes:
        original_text_length: Length of original text
        processed_text_length: Length after preprocessing
        original_sentence_count: Number of sentences before preprocessing
        processed_sentence_count: Number of sentences after preprocessing
        transformations: List of transformations applied
        merged_sentences: Number of sentences merged
        split_sentences: Number of sentences split
        removed_sentences: Number of sentences removed
    """
    original_text_length: int = 0
    processed_text_length: int = 0
    original_sentence_count: int = 0
    processed_sentence_count: int = 0
    transformations: List[str] = field(default_factory=list)
    merged_sentences: int = 0
    split_sentences: int = 0
    removed_sentences: int = 0
    
    def add_transformation(self, description: str) -> None:
        """Add a transformation to the diagnostics.
        
        Args:
            description: Description of the transformation
        """
        self.transformations.append(description)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "original_text_length": self.original_text_length,
            "processed_text_length": self.processed_text_length,
            "original_sentence_count": self.original_sentence_count,
            "processed_sentence_count": self.processed_sentence_count,
            "transformations": self.transformations,
            "merged_sentences": self.merged_sentences,
            "split_sentences": self.split_sentences,
            "removed_sentences": self.removed_sentences,
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> str:
        """Generate human-readable summary of preprocessing."""
        parts = []
        
        if self.merged_sentences > 0:
            parts.append(f"merged {self.merged_sentences} short sentences")
        
        if self.split_sentences > 0:
            parts.append(f"split {self.split_sentences} long sentences")
        
        if self.removed_sentences > 0:
            parts.append(f"removed {self.removed_sentences} empty sentences")
        
        length_change = self.processed_text_length - self.original_text_length
        if length_change != 0:
            direction = "increased" if length_change > 0 else "decreased"
            parts.append(f"text length {direction} by {abs(length_change)} chars")
        
        if not parts:
            return "No significant transformations applied"
        
        return "; ".join(parts)


class PreprocessingLogger:
    """Logger for tracking preprocessing transformations.
    
    Usage:
        logger = PreprocessingLogger(original_text)
        logger.log_merge("Merged 2 short sentences")
        logger.set_result(processed_sentences)
        diagnostics = logger.get_diagnostics()
    """
    
    def __init__(self, original_text: str, original_sentences: List[str] = None):
        """Initialize preprocessing logger.
        
        Args:
            original_text: Original text before preprocessing
            original_sentences: Original sentences (optional)
        """
        self.diagnostics = PreprocessingDiagnostics(
            original_text_length=len(original_text),
            original_sentence_count=len(original_sentences) if original_sentences else 0
        )
        self._log_enabled = True
    
    def disable_logging(self) -> None:
        """Disable logging (for performance)."""
        self._log_enabled = False
    
    def log_merge(self, description: str) -> None:
        """Log a sentence merge operation.
        
        Args:
            description: Description of the merge
        """
        if self._log_enabled:
            self.diagnostics.add_transformation(f"MERGE: {description}")
            self.diagnostics.merged_sentences += 1
    
    def log_split(self, description: str) -> None:
        """Log a sentence split operation.
        
        Args:
            description: Description of the split
        """
        if self._log_enabled:
            self.diagnostics.add_transformation(f"SPLIT: {description}")
            self.diagnostics.split_sentences += 1
    
    def log_remove(self, description: str) -> None:
        """Log a sentence removal.
        
        Args:
            description: Description of the removal
        """
        if self._log_enabled:
            self.diagnostics.add_transformation(f"REMOVE: {description}")
            self.diagnostics.removed_sentences += 1
    
    def log_transform(self, description: str) -> None:
        """Log a general transformation.
        
        Args:
            description: Description of the transformation
        """
        if self._log_enabled:
            self.diagnostics.add_transformation(description)
    
    def set_result(
        self,
        processed_sentences: List[str],
        processed_text: str = None
    ) -> None:
        """Set the result of preprocessing.
        
        Args:
            processed_sentences: Sentences after preprocessing
            processed_text: Processed text (optional, computed if not provided)
        """
        self.diagnostics.processed_sentence_count = len(processed_sentences)
        
        if processed_text:
            self.diagnostics.processed_text_length = len(processed_text)
        else:
            self.diagnostics.processed_text_length = sum(
                len(s) for s in processed_sentences
            ) + len(processed_sentences) - 1  # Account for spaces
    
    def get_diagnostics(self) -> PreprocessingDiagnostics:
        """Get the preprocessing diagnostics.
        
        Returns:
            PreprocessingDiagnostics object
        """
        return self.diagnostics
