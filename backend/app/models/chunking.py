"""Pydantic models for text chunking functionality."""

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


class ChunkingStrategy(str, Enum):
    """Supported text chunking strategies."""

    FIXED_SIZE = "fixed_size"        # Character-based with overlap
    RECURSIVE = "recursive"           # Hierarchical separators
    SENTENCE = "sentence"             # NLP sentence boundaries
    SEMANTIC = "semantic"             # Embedding cosine similarity
    LATE_CHUNKING = "late_chunking"   # Long-context embedding first
    AGENTIC = "agentic"               # LLM-driven segmentation


class ChunkRequest(BaseModel):
    """Request model for text chunking API."""

    text: str = Field(..., min_length=1, description="Text to chunk")
    strategy: ChunkingStrategy = Field(default=ChunkingStrategy.FIXED_SIZE)
    chunk_size: int = Field(default=500, ge=10, le=10000)
    overlap: int = Field(default=50, ge=0)
    separators: Optional[List[str]] = Field(default=None)
    # Semantic chunking params
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    embedding_model: Optional[str] = Field(default="openai/text-embedding-3-small")
    # Agentic chunking params
    llm_model: Optional[str] = Field(default="gpt-4o-mini")
    # Enhanced semantic chunking params (Phase 6)
    enable_qa_detection: bool = Field(
        default=True,
        description="Enable Q&A pair detection and merging"
    )
    enable_adaptive_threshold: bool = Field(
        default=True,
        description="Enable adaptive threshold based on text characteristics"
    )
    enable_cache: bool = Field(
        default=True,
        description="Enable embedding caching for performance"
    )
    include_quality_metrics: bool = Field(
        default=False,
        description="Include quality metrics in response"
    )
    min_chunk_size: Optional[int] = Field(
        default=150,
        ge=10,
        description="Minimum characters per chunk"
    )
    max_chunk_size: Optional[int] = Field(
        default=2000,
        le=10000,
        description="Maximum characters per chunk"
    )
    buffer_size: Optional[int] = Field(
        default=1,
        ge=0,
        le=5,
        description="Sentences on each side for context in semantic chunking"
    )

    @model_validator(mode='after')
    def validate_request(self):
        """Validate request parameters."""
        # Validate that text is not whitespace-only
        if not self.text.strip():
            raise ValueError('Text cannot be empty or whitespace-only')
        # Validate that overlap is less than chunk_size
        if self.overlap >= self.chunk_size:
            raise ValueError('overlap must be less than chunk_size')
        # Validate min/max chunk size relationship
        if self.min_chunk_size and self.max_chunk_size:
            if self.min_chunk_size >= self.max_chunk_size:
                raise ValueError('min_chunk_size must be less than max_chunk_size')
        return self


class Chunk(BaseModel):
    """Model representing a single text chunk."""

    index: int
    content: str
    start_position: int
    end_position: int
    char_count: int
    has_overlap: bool = False


class ChunkStats(BaseModel):
    """Statistics about generated chunks."""

    total_chunks: int
    total_characters: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int


class ChunkResponse(BaseModel):
    """Response model for text chunking API."""

    chunks: List[Chunk]
    stats: ChunkStats
    strategy_used: ChunkingStrategy


# Quality metrics models (Phase 4 & 6)
class ChunkQualityMetricsResponse(BaseModel):
    """Quality metrics for a single chunk."""

    chunk_index: int
    semantic_coherence: float = Field(
        description="Intra-chunk sentence similarity (0-1)"
    )
    sentence_count: int
    topic_consistency: float = Field(
        description="How focused the chunk is on a single topic (0-1)"
    )
    has_questions: bool = False
    has_qa_pairs: bool = False


class QualityReportResponse(BaseModel):
    """Overall quality report for chunking results."""

    total_chunks: int
    avg_coherence: float
    min_coherence: float
    max_coherence: float
    chunks_below_threshold: List[int] = Field(
        default_factory=list,
        description="Indices of chunks with low coherence"
    )
    inter_chunk_similarities: List[float] = Field(
        default_factory=list,
        description="Similarity scores between consecutive chunks"
    )
    merge_recommendations: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="Pairs of chunk indices that should be merged"
    )
    split_recommendations: List[int] = Field(
        default_factory=list,
        description="Chunk indices that should be split"
    )
    overall_quality_score: float = Field(
        description="Overall quality score (0-1)"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Human-readable recommendations"
    )


class ChunkResponseWithQuality(BaseModel):
    """Response model with quality metrics included."""

    chunks: List[Chunk]
    stats: ChunkStats
    strategy_used: ChunkingStrategy
    quality_metrics: Optional[List[ChunkQualityMetricsResponse]] = None
    quality_report: Optional[QualityReportResponse] = None
    detected_language: Optional[str] = None
    adaptive_threshold_used: Optional[float] = None
    processing_time_ms: Optional[float] = None
    fallback_used: Optional[str] = None
    warning_message: Optional[str] = None


class ChunkingDiagnosticsResponse(BaseModel):
    """Diagnostics information for chunking operations."""

    strategy: str
    input_text_length: int
    processing_time: float
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    overlap_count: int
    error_message: Optional[str] = None
    performance_warnings: List[str] = Field(default_factory=list)
    quality_score: float = 0.0
    recommendations: List[str] = Field(default_factory=list)


class ChunkingProgressEvent(BaseModel):
    """Progress event for streaming chunking operations."""
    
    event_type: str = Field(
        description="Type of event: 'progress', 'complete', 'error'"
    )
    stage: str = Field(
        description="Current processing stage"
    )
    progress: float = Field(
        ge=0.0, le=100.0,
        description="Progress percentage (0-100)"
    )
    message: str = Field(
        description="Human-readable status message"
    )
    details: Optional[dict] = Field(
        default=None,
        description="Additional details about current stage"
    )
    result: Optional[dict] = Field(
        default=None,
        description="Final result when event_type is 'complete'"
    )
