"""Pydantic schemas for request/response validation."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import AliasChoices, BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models.db_models import UserRole


class Token(BaseModel):
    """JWT token response with access and refresh tokens."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    """Request schema for token refresh."""

    refresh_token: str


class TokenData(BaseModel):
    """JWT token payload data."""

    user_id: int
    email: Optional[str] = None
    role: Optional[UserRole] = None


class UserBase(BaseModel):
    """Base user schema."""

    email: EmailStr
    full_name: str
    role: UserRole


class UserCreate(UserBase):
    """Schema for creating a new user."""

    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    """Schema for updating user profile."""

    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


class PasswordChange(BaseModel):
    """Schema for changing password."""

    current_password: str
    new_password: str = Field(..., min_length=6)


class UserResponse(UserBase):
    """Schema for user response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


class CourseBase(BaseModel):
    """Base course schema."""

    name: str
    description: Optional[str] = None


class CourseCreate(CourseBase):
    """Schema for creating a new course."""

    pass


class CourseUpdate(BaseModel):
    """Schema for updating a course."""

    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class CourseResponse(CourseBase):
    """Schema for course response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    teacher_id: int
    is_active: bool = True
    created_at: datetime

    document_count: int = 0


class CourseListResponse(BaseModel):
    """Schema for course list response."""

    courses: List[CourseResponse]
    total: int


class DocumentBase(BaseModel):
    """Base document schema."""

    course_id: Optional[int] = None


class DocumentCreate(DocumentBase):
    """Schema for creating a new document."""

    pass


class DocumentResponse(DocumentBase):
    """Schema for document response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    char_count: Optional[int] = None
    is_processed: bool = False
    created_at: datetime
    chunk_count: int = 0
    embedding_status: Optional[str] = None
    embedding_model: Optional[str] = None
    embedded_at: Optional[datetime] = None
    vector_count: Optional[int] = 0


class DocumentListResponse(BaseModel):
    """Schema for document list response."""

    documents: List[DocumentResponse]
    total: int


class DocumentProcessRequest(BaseModel):
    """Schema for document processing request."""

    document_id: int


class ChunkingStrategy(str, Enum):
    """Chunking strategy enumeration."""

    FIXED = "fixed"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class ChunkingRequest(BaseModel):
    """Schema for chunking request."""

    text: str
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Validate that overlap is less than chunk size."""
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class ChunkResponse(BaseModel):
    """Schema for a single chunk response."""

    text: str
    start_index: int
    end_index: int
    metadata: dict = {}


class ChunkingResponse(BaseModel):
    """Schema for chunking response."""

    chunks: List[ChunkResponse]
    total_chunks: int
    strategy_used: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int


class ChunkQualityMetrics(BaseModel):
    """Schema for individual chunk quality metrics.
    
    Feature: semantic-chunker-enhancement, Task 7.12
    Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
    """

    chunk_index: int
    semantic_coherence: float  # 0-1, intra-chunk sentence similarity
    sentence_count: int
    avg_sentence_similarity: float  # 0-1
    topic_consistency: float  # 0-1
    has_questions: bool = False
    has_qa_pairs: bool = False


class QualityReportResponse(BaseModel):
    """Schema for quality report response.

    Feature: semantic-chunker-enhancement, Task 7.12
    Validates: Requirements 7.5
    """

    total_chunks: int
    avg_coherence: float
    min_coherence: float
    max_coherence: float
    chunks_below_threshold: List[int] = []  # chunk indices with low coherence
    inter_chunk_similarities: List[float] = []
    merge_recommendations: List[List[int]] = []  # pairs of chunk indices to merge
    split_recommendations: List[int] = []  # chunk indices to split
    overall_quality_score: float  # 0-1
    recommendations: List[str] = []


class ChunkingResponseWithQuality(BaseModel):
    """Schema for chunking response with quality metrics.

    Feature: semantic-chunker-enhancement, Task 7.12
    Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
    """

    chunks: List[ChunkResponse]
    total_chunks: int
    strategy_used: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    chunk_metrics: Optional[List[ChunkQualityMetrics]] = None
    quality_report: Optional[QualityReportResponse] = None


class ChatMessage(BaseModel):
    """Schema for chat message."""

    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    """Schema for chat request."""

    message: str
    history: List[ChatMessage] = []
    search_type: str = "hybrid"  # Keep for backward compatibility


class ChunkReference(BaseModel):
    """Schema for chunk reference in chat responses."""

    document_id: int
    document_name: str
    chunk_index: int
    content_preview: str
    full_content: str
    score: float


class ChunkDBResponse(BaseModel):
    """Schema for chunk database response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    content: str
    index: int
    start_position: int
    end_position: int
    char_count: int
    has_overlap: bool
    created_at: datetime


class ChunkListResponse(BaseModel):
    """Schema for chunk list response."""

    chunks: List[ChunkDBResponse]
    total: int
    document_id: int


class ChatResponse(BaseModel):
    """Schema for chat response."""

    message: str
    sources: List[ChunkReference] = []


class ChatHistoryMessage(BaseModel):
    """Schema for persisted chat history message."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime
    sources: Optional[List[ChunkReference]] = None
    response_time_ms: Optional[int] = None


class ChatHistoryResponse(BaseModel):
    """Schema for chat history response (paginated)."""

    messages: List[ChatHistoryMessage]
    has_more: bool


class ChatHistoryClearResponse(BaseModel):
    """Schema for clearing chat history."""

    success: bool
    deleted_count: int


class EmbeddingRequest(BaseModel):
    """Schema for embedding request."""

    text: str


class EmbeddingResponse(BaseModel):
    """Schema for embedding response."""

    embedding: List[float]
    model: str
    dimensions: int


class EmbedRequest(BaseModel):
    """Schema for document embedding request."""

    model: str = "text-embedding-3-small"


class EmbedResponse(BaseModel):
    """Schema for document embedding response."""

    document_id: int
    status: str
    vector_count: int
    model: str


class VectorCountResponse(BaseModel):
    """Schema for vector count response."""

    document_id: int
    vector_count: int


class CourseSettingsBase(BaseModel):
    """Base course settings schema."""

    default_chunk_strategy: str = "recursive"
    default_chunk_size: int = Field(default=500, ge=100, le=5000)
    default_overlap: int = Field(default=50, ge=0, le=500)
    default_embedding_model: str = "openai/text-embedding-3-small"
    search_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    search_top_k: int = Field(default=5, ge=1, le=20)
    min_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_provider: str = "openrouter"
    llm_model: str = "openai/gpt-4o-mini"
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1000, ge=100, le=4000)
    system_prompt: Optional[str] = None
    active_prompt_template_id: Optional[int] = None
    system_prompt_remembering: Optional[str] = None
    system_prompt_understanding_applying: Optional[str] = None
    system_prompt_analyzing_evaluating: Optional[str] = None
    enable_direct_llm: bool = False
    enable_pii_filter: bool = False
    enable_reranker: bool = False
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None
    reranker_top_k: int = Field(default=10, ge=5, le=20)
    vector_store: str = "weaviate"


class CourseSettingsCreate(CourseSettingsBase):
    """Schema for creating course settings."""

    course_id: int


class CourseSettingsUpdate(BaseModel):
    """Schema for updating course settings."""

    default_chunk_strategy: Optional[str] = None
    default_chunk_size: Optional[int] = Field(default=None, ge=100, le=5000)
    default_overlap: Optional[int] = Field(default=None, ge=0, le=500)
    default_embedding_model: Optional[str] = None
    search_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    search_top_k: Optional[int] = Field(default=None, ge=1, le=20)
    min_relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    llm_max_tokens: Optional[int] = Field(default=None, ge=100, le=4000)
    system_prompt: Optional[str] = Field(default=None, max_length=2000)
    active_prompt_template_id: Optional[int] = None
    system_prompt_remembering: Optional[str] = (
        Field(
            default=None,
            max_length=5000,
        )
    )
    system_prompt_understanding_applying: Optional[str] = (
        Field(
            default=None,
            max_length=5000,
        )
    )
    system_prompt_analyzing_evaluating: Optional[str] = (
        Field(
            default=None,
            max_length=5000,
        )
    )
    enable_direct_llm: Optional[bool] = None
    enable_pii_filter: Optional[bool] = None
    enable_reranker: Optional[bool] = None
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None
    reranker_top_k: Optional[int] = Field(default=None, ge=5, le=20)
    vector_store: Optional[str] = None

    @field_validator('reranker_provider')
    @classmethod
    def validate_reranker_provider(cls, v: Optional[str]) -> Optional[str]:
        """Validate reranker provider is one of the supported values."""
        if v is not None and v not in ['cohere', 'alibaba', 'jina', 'weaviate', 'bge', 'zeroentropy', 'voyage']:
            raise ValueError(f"Invalid reranker provider: {v}. Must be one of: cohere, alibaba, jina, weaviate, bge, zeroentropy, voyage")
        return v

    @field_validator('default_embedding_model')
    @classmethod
    def validate_embedding_model(cls, v: Optional[str]) -> Optional[str]:
        """Validate embedding model format (supports existing prefixes)."""
        if v is None:
            return v
        # Allow known prefixes and Voyage models
        allowed_prefixes = ['openai/', 'alibaba/', 'cohere/', 'jina/', 'bge/', 'ollama/', 'voyage/']
        if not any(v.startswith(p) for p in allowed_prefixes) and not v.startswith('voyage-'):
            raise ValueError(f"Invalid embedding model: {v}. Must start with one of: {', '.join(allowed_prefixes)} or voyage-")
        return v
    
    @field_validator('vector_store')
    @classmethod
    def validate_vector_store(cls, v: Optional[str]) -> Optional[str]:
        """Validate vector store is one of the supported values."""
        if v is not None and v not in ['weaviate']:
            raise ValueError(f"Invalid vector store: {v}. Must be: weaviate")
        return v


class CourseSettingsResponse(CourseSettingsBase):
    """Schema for course settings response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int
    created_at: datetime
    updated_at: datetime


class CoursePromptTemplateBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)


class CoursePromptTemplateCreate(CoursePromptTemplateBase):
    pass


class CoursePromptTemplateUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    content: Optional[str] = Field(default=None, min_length=1)


class CoursePromptTemplateResponse(CoursePromptTemplateBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int
    created_at: datetime
    updated_at: datetime


class CoursePromptTemplateListResponse(BaseModel):
    templates: List[CoursePromptTemplateResponse]


class LLMProvider(str, Enum):
    """LLM provider enumeration."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class LLMSettingsBase(BaseModel):
    """Base LLM settings schema."""

    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    system_prompt: Optional[str] = None


class LLMSettingsCreate(LLMSettingsBase):
    """Schema for creating LLM settings."""

    course_id: int


class LLMSettingsUpdate(BaseModel):
    """Schema for updating LLM settings."""

    provider: Optional[LLMProvider] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=100, le=4000)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    system_prompt: Optional[str] = None


class LLMSettingsResponse(LLMSettingsBase):
    """Schema for LLM settings response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int
    created_at: datetime
    updated_at: datetime


class DiagnosticMetrics(BaseModel):
    """Schema for diagnostic metrics."""

    total_documents: int
    total_chunks: int
    total_embeddings: int
    avg_chunk_size: float
    processing_errors: int


class DiagnosticResponse(BaseModel):
    """Schema for diagnostic response."""

    course_id: int
    course_name: str
    metrics: DiagnosticMetrics
    last_updated: datetime


class FileInfo(BaseModel):
    """Schema for file information."""

    filename: str
    file_size: int
    file_type: str
    char_count: Optional[int] = None


class ExtractionInfo(BaseModel):
    """Schema for extraction information."""

    success: bool
    char_count: Optional[int] = None
    method_used: Optional[str] = None


class ChunkingInfo(BaseModel):
    """Schema for chunking information."""

    success: bool
    total_chunks: int
    strategy_used: Optional[str] = None


class ErrorEntry(BaseModel):
    """Schema for error entry."""

    timestamp: str
    stage: str
    error_type: str
    error_message: str
    context: Optional[dict] = None


class PerformanceMetrics(BaseModel):
    """Schema for performance metrics."""

    total_processing_time: Optional[float] = None


class DiagnosticReportResponse(BaseModel):
    """Schema for diagnostic report response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    report_type: str
    file_info: Optional[FileInfo] = None
    extraction_info: Optional[ExtractionInfo] = None
    error_log: List[ErrorEntry] = []
    performance_metrics: Optional[PerformanceMetrics] = None
    recommendations: List[str] = []
    created_at: datetime


class ChunkQualityMetricsResponse(BaseModel):
    """Schema for chunk quality metrics response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    total_chunks: int
    avg_chunk_size: int
    min_chunk_size: int
    max_chunk_size: int
    size_distribution: dict
    overlap_analysis: dict
    content_quality_score: float
    recommendations: List[str]
    chunking_strategy: str
    chunk_size_config: int
    overlap_config: int
    created_at: datetime


class ProcessingStatusResponse(BaseModel):
    """Schema for processing status response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    status: str
    error_message: Optional[str] = None
    error_details: Optional[dict] = None
    processing_duration: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class SystemDiagnosticsResponse(BaseModel):
    """Schema for system diagnostics response."""

    system_health: str
    total_documents: int
    processing_stats: dict
    error_summary: dict
    performance_summary: dict
    recommendations: List[str]
    timestamp: datetime


# ==================== RAGAS Evaluation Schemas ====================

class TestQuestionBase(BaseModel):
    """Base test question schema."""

    question: str
    ground_truth: str
    alternative_ground_truths: Optional[List[str]] = None
    expected_contexts: Optional[List[str]] = None
    question_metadata: Optional[dict] = None


class TestQuestionCreate(TestQuestionBase):
    """Schema for creating a test question."""

    pass


class TestQuestionUpdate(BaseModel):
    """Schema for updating a test question."""

    question: Optional[str] = None
    ground_truth: Optional[str] = None
    alternative_ground_truths: Optional[List[str]] = None
    expected_contexts: Optional[List[str]] = None
    question_metadata: Optional[dict] = None


class TestQuestionResponse(TestQuestionBase):
    """Schema for test question response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    test_set_id: int
    created_at: datetime


class TestSetBase(BaseModel):
    """Base test set schema."""

    name: str
    description: Optional[str] = None


class TestSetCreate(TestSetBase):
    """Schema for creating a test set."""

    course_id: int


class TestSetUpdate(BaseModel):
    """Schema for updating a test set."""

    name: Optional[str] = None
    description: Optional[str] = None


class TestSetResponse(TestSetBase):
    """Schema for test set response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int
    created_by: int
    created_at: datetime
    updated_at: datetime
    question_count: int = 0


class TestSetDetailResponse(TestSetResponse):
    """Schema for test set detail response with questions."""

    questions: List[TestQuestionResponse] = []


class EvaluationConfig(BaseModel):
    """Schema for evaluation run configuration."""

    search_type: str = "hybrid"
    search_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)
    llm_model: Optional[str] = None
    llm_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class EvaluationRunCreate(BaseModel):
    """Schema for creating an evaluation run."""

    test_set_id: int
    course_id: int
    name: Optional[str] = None
    config: Optional[EvaluationConfig] = None
    evaluation_provider: Optional[str] = None
    evaluation_model: Optional[str] = None
    question_ids: Optional[List[int]] = None  # If provided, only evaluate these questions


class EvaluationRunResponse(BaseModel):
    """Schema for evaluation run response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    test_set_id: int
    test_set_name: Optional[str] = None
    course_id: int
    name: Optional[str] = None
    status: str
    config: Optional[dict] = None
    total_questions: int
    processed_questions: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime
    # Average metrics from summary
    avg_faithfulness: Optional[float] = None
    avg_answer_relevancy: Optional[float] = None
    avg_context_precision: Optional[float] = None
    avg_context_recall: Optional[float] = None
    avg_answer_correctness: Optional[float] = None


class EvaluationResultResponse(BaseModel):
    """Schema for evaluation result response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    question_id: int
    question_text: str
    ground_truth_text: str
    generated_answer: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    latency_ms: Optional[int] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    evaluation_model: Optional[str] = None
    search_alpha: Optional[float] = None
    search_top_k: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime


class RunSummaryResponse(BaseModel):
    """Schema for run summary response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    avg_faithfulness: Optional[float] = None
    avg_answer_relevancy: Optional[float] = None
    avg_context_precision: Optional[float] = None
    avg_context_recall: Optional[float] = None
    avg_answer_correctness: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    total_questions: int
    successful_questions: int
    failed_questions: int
    created_at: datetime


class EvaluationRunDetailResponse(EvaluationRunResponse):
    """Schema for detailed evaluation run with results."""

    results: List[EvaluationResultResponse] = []
    summary: Optional[RunSummaryResponse] = None


class RunComparisonRequest(BaseModel):
    """Schema for comparing multiple runs."""

    run_ids: List[int] = Field(..., min_length=2)


class RunComparisonResponse(BaseModel):
    """Schema for run comparison response."""

    runs: List[EvaluationRunResponse]
    summaries: List[RunSummaryResponse]


# ==================== Quick Test Result Schemas ====================

class QuickTestRequest(BaseModel):
    """Schema for quick test request."""

    course_id: int
    question: str
    ground_truth: str
    alternative_ground_truths: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    ragas_embedding_model: Optional[str] = None


class RetrievedContext(BaseModel):
    """Schema for a retrieved context with score."""

    text: str
    score: float


class QuickTestResponse(BaseModel):
    """Schema for quick test response."""

    question: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: List[RetrievedContext]
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    latency_ms: int
    system_prompt_used: str
    llm_provider_used: str
    llm_model_used: str
    evaluation_model_used: Optional[str] = None
    embedding_model_used: Optional[str] = None
    search_top_k_used: Optional[int] = None
    search_alpha_used: Optional[float] = None
    # Reranker metadata
    reranker_used: Optional[bool] = None
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None


# ==================== Quick Test Result Types ====================

class QuickTestResultCreate(BaseModel):
    """Schema for creating a quick test result."""

    course_id: int
    group_name: Optional[str] = None
    question: str
    ground_truth: str
    alternative_ground_truths: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    llm_provider: str
    llm_model: str
    evaluation_model: Optional[str] = None
    embedding_model: Optional[str] = None
    search_top_k: Optional[int] = None
    search_alpha: Optional[float] = None
    reranker_used: Optional[bool] = None
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None
    generated_answer: str
    retrieved_contexts: Optional[List[RetrievedContext]] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    latency_ms: int


class QuickTestResultResponse(BaseModel):
    """Schema for quick test result response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int
    group_name: Optional[str] = None
    question: str
    ground_truth: str
    alternative_ground_truths: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    llm_provider: str
    llm_model: str
    evaluation_model: Optional[str] = None
    embedding_model: Optional[str] = None
    search_top_k: Optional[int] = None
    search_alpha: Optional[float] = None
    generated_answer: str
    retrieved_contexts: Optional[List[RetrievedContext]] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    latency_ms: int
    created_by: int
    created_at: datetime
    # Reranker metadata
    reranker_used: Optional[bool] = None
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None


class RagasGroupInfo(BaseModel):
    """Schema for RAGAS group information with creation date."""

    name: str
    created_at: Optional[str] = None
    test_count: Optional[int] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    evaluation_model: Optional[str] = None
    embedding_model: Optional[str] = None
    search_top_k: Optional[int] = None
    search_alpha: Optional[float] = None
    reranker_used: Optional[bool] = None
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None
    avg_faithfulness: Optional[float] = None
    avg_answer_relevancy: Optional[float] = None
    avg_context_precision: Optional[float] = None
    avg_context_recall: Optional[float] = None
    avg_answer_correctness: Optional[float] = None


class QuickTestResultListResponse(BaseModel):
    """Schema for quick test result list response."""

    results: List[QuickTestResultResponse]
    total: int
    groups: List[RagasGroupInfo]
    # Contains avg metrics and test_parameters
    aggregate: Optional[dict] = None


# ==================== Custom LLM Model Schemas ====================

class CustomLLMModelBase(BaseModel):
    """Base custom LLM model schema."""

    provider: str
    model_id: str
    display_name: str


class CustomLLMModelCreate(CustomLLMModelBase):
    """Schema for creating a custom LLM model."""

    pass


class CustomLLMModelResponse(CustomLLMModelBase):
    """Schema for custom LLM model response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    is_active: bool
    created_by: int
    created_at: datetime


class CustomLLMModelListResponse(BaseModel):
    """Schema for custom LLM model list response."""

    models: List[CustomLLMModelResponse]
    total: int


class LLMModelsResponse(BaseModel):
    """Schema for combined LLM models response (default + custom)."""

    default_models: List[str]
    custom_models: List[CustomLLMModelResponse]


# ==================== Semantic Similarity Test Schemas ====================

class SemanticSimilarityQuickTestRequest(BaseModel):
    """Schema for semantic similarity quick test request."""

    course_id: int
    question: str
    ground_truth: str
    alternative_ground_truths: Optional[List[str]] = None
    generated_answer: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    use_direct_llm: Optional[bool] = None


class ScoreDetail(BaseModel):
    """Schema for individual similarity score detail."""

    model_config = ConfigDict(from_attributes=True)

    ground_truth: str
    score: float


class SemanticSimilarityQuickTestResponse(BaseModel):
    """Schema for semantic similarity quick test response."""

    question: str
    ground_truth: str
    generated_answer: str
    similarity_score: float  # 0.0 to 1.0 (cosine similarity)
    best_match_ground_truth: str  # Which ground truth had highest similarity
    all_scores: List[ScoreDetail]
    # ROUGE metrics
    rouge1: Optional[float] = None  # ROUGE-1 F1 score
    rouge2: Optional[float] = None  # ROUGE-2 F1 score
    rougel: Optional[float] = None  # ROUGE-L F1 score
    # BERTScore metrics
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None
    original_bertscore_precision: Optional[float] = None
    original_bertscore_recall: Optional[float] = None
    original_bertscore_f1: Optional[float] = None
    # Retrieval metrics
    hit_at_1: Optional[float] = None  # 1 if best match is rank 1, else 0
    mrr: Optional[float] = None  # Mean Reciprocal Rank (1/rank)
    # Metadata
    latency_ms: int
    embedding_model_used: str
    llm_model_used: Optional[str] = None  # Only if answer was generated
    retrieved_contexts: Optional[List[str]] = None  # Source contexts used
    system_prompt_used: Optional[str] = None  # System prompt used


class SemanticSimilarityTestCase(BaseModel):
    """Schema for a single test case in batch testing.
    
    Supports both English and Turkish field names:
    - question / Soru
    - ground_truth / İdeal Cevap (GT)
    - context / Bağlam (Context)
    - bloom_level / Bloom Seviyesi
    - topic / Konu
    - chunk_id / Chunk_ID
    """

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(..., validation_alias=AliasChoices("question", "Soru"))
    ground_truth: str = Field(..., validation_alias=AliasChoices("ground_truth", "İdeal Cevap (GT)"))
    alternative_ground_truths: Optional[List[str]] = None
    generated_answer: Optional[str] = None
    bloom_level: Optional[str] = Field(default=None, validation_alias=AliasChoices("bloom_level", "Bloom Seviyesi"))
    question_metadata: Optional[Dict[str, Any]] = None
    expected_contexts: Optional[List[str]] = Field(default=None, validation_alias=AliasChoices("expected_contexts", "Bağlam (Context)"))
    topic: Optional[str] = Field(default=None, validation_alias=AliasChoices("topic", "Konu"))
    chunk_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("chunk_id", "Chunk_ID"))


class SemanticSimilarityBatchTestRequest(BaseModel):
    """Schema for semantic similarity batch test request."""

    model_config = ConfigDict(populate_by_name=True)

    course_id: int
    test_cases: List[SemanticSimilarityTestCase]
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    llm_provider: Optional[str] = None  # Override course LLM provider
    llm_model: Optional[str] = None  # Override course LLM model
    search_top_k: Optional[int] = None  # Override search top_k
    search_alpha: Optional[float] = None  # Override search alpha
    reranker_used: Optional[bool] = None  # Override reranker usage
    reranker_provider: Optional[str] = None  # Override reranker provider
    reranker_model: Optional[str] = None  # Override reranker model
    use_direct_llm: Optional[bool] = None  # Direct LLM mode (bypass RAG)


class SemanticSimilarityBatchResult(BaseModel):
    """Schema for a single batch test result."""

    question: str
    ground_truth: str
    generated_answer: str
    similarity_score: Optional[float] = None
    best_match_ground_truth: Optional[str] = None
    # ROUGE metrics
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougel: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None
    original_bertscore_precision: Optional[float] = None
    original_bertscore_recall: Optional[float] = None
    original_bertscore_f1: Optional[float] = None
    # Retrieval metrics
    hit_at_1: Optional[float] = None
    mrr: Optional[float] = None
    # Metadata
    latency_ms: int
    error_message: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None  # Source contexts used


class AggregateStatistics(BaseModel):
    """Schema for aggregate statistics in batch testing."""

    avg_similarity: float
    min_similarity: float
    max_similarity: float
    total_latency_ms: int
    test_count: int
    successful_count: int
    failed_count: int


class SemanticSimilarityBatchTestResponse(BaseModel):
    """Schema for semantic similarity batch test response."""

    results: List[SemanticSimilarityBatchResult]
    aggregate: AggregateStatistics
    embedding_model_used: str
    llm_model_used: Optional[str] = None


class SemanticSimilarityBatchResultCreate(BaseModel):
    """Schema for batch saving multiple semantic similarity results at once."""

    course_id: int
    group_name: Optional[str] = None
    results: List['SemanticSimilarityResultItemCreate']


class SemanticSimilarityBatchSaveResponse(BaseModel):
    """Response for batch save operation."""

    saved_count: int
    failed_count: int
    group_name: Optional[str] = None


class SemanticSimilarityResultItemCreate(BaseModel):
    """Schema for a single result item within a batch save."""

    question: str
    ground_truth: str
    alternative_ground_truths: Optional[List[str]] = None
    generated_answer: str
    bloom_level: Optional[str] = None
    similarity_score: float
    best_match_ground_truth: str
    all_scores: Optional[List[ScoreDetail]] = None
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougel: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None
    original_bertscore_precision: Optional[float] = None
    original_bertscore_recall: Optional[float] = None
    original_bertscore_f1: Optional[float] = None
    hit_at_1: Optional[float] = None
    mrr: Optional[float] = None
    latency_ms: int = 0
    embedding_model_used: Optional[str] = None
    llm_model_used: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    system_prompt_used: Optional[str] = None
    search_top_k: Optional[int] = None
    search_alpha: Optional[float] = None
    reranker_used: Optional[bool] = None
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None


class SemanticSimilarityResultCreate(BaseModel):
    """Schema for creating a semantic similarity result."""

    course_id: int
    group_name: Optional[str] = None
    question: str
    ground_truth: str
    alternative_ground_truths: Optional[List[str]] = None
    generated_answer: str
    bloom_level: Optional[str] = None
    similarity_score: float
    best_match_ground_truth: str
    all_scores: Optional[List[ScoreDetail]] = None
    # ROUGE metrics
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougel: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None
    original_bertscore_precision: Optional[float] = None
    original_bertscore_recall: Optional[float] = None
    original_bertscore_f1: Optional[float] = None
    # Retrieval metrics
    hit_at_1: Optional[float] = None
    mrr: Optional[float] = None
    # Metadata
    latency_ms: int
    embedding_model_used: str
    llm_model_used: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    system_prompt_used: Optional[str] = None
    # Search configuration
    search_top_k: Optional[int] = None
    search_alpha: Optional[float] = None
    # Reranker configuration
    reranker_used: Optional[bool] = None
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None
    # These fields are set automatically by the backend
    # created_by: int  # Set from current_user
    # created_at: datetime  # Set automatically by database

    class SemanticSimilarityBatchResultCreate(BaseModel):
        """Schema for batch saving multiple semantic similarity results at once."""

        course_id: int
        group_name: Optional[str] = None
        results: List['SemanticSimilarityResultCreate']


    class SemanticSimilarityBatchSaveResponse(BaseModel):
        """Response for batch save operation."""

        saved_count: int
        failed_count: int
        group_name: Optional[str] = None


    class SemanticSimilarityResultCreate(BaseModel):
        """Schema for creating a semantic similarity result."""

        course_id: int
        group_name: Optional[str] = None
        question: str
        ground_truth: str
        alternative_ground_truths: Optional[List[str]] = None
        generated_answer: str
        bloom_level: Optional[str] = None
        similarity_score: float
        best_match_ground_truth: str
        all_scores: Optional[List[ScoreDetail]] = None
        # ROUGE metrics
        rouge1: Optional[float] = None
        rouge2: Optional[float] = None
        rougel: Optional[float] = None
        bertscore_precision: Optional[float] = None
        bertscore_recall: Optional[float] = None
        bertscore_f1: Optional[float] = None
        original_bertscore_precision: Optional[float] = None
        original_bertscore_recall: Optional[float] = None
        original_bertscore_f1: Optional[float] = None
        # Retrieval metrics
        hit_at_1: Optional[float] = None
        mrr: Optional[float] = None
        # Metadata
        latency_ms: int
        embedding_model_used: str
        llm_model_used: Optional[str] = None
        retrieved_contexts: Optional[List[str]] = None
        system_prompt_used: Optional[str] = None
        # Search configuration
        search_top_k: Optional[int] = None
        search_alpha: Optional[float] = None
        # Reranker configuration
        reranker_used: Optional[bool] = None
        reranker_provider: Optional[str] = None
        reranker_model: Optional[str] = None


class SemanticSimilarityResultResponse(BaseModel):
    """Schema for semantic similarity result response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int
    group_name: Optional[str] = None
    question: str
    ground_truth: str
    alternative_ground_truths: Optional[List[str]] = None
    generated_answer: str
    bloom_level: Optional[str] = None
    similarity_score: float
    best_match_ground_truth: str
    all_scores: Optional[List[ScoreDetail]] = None
    # ROUGE metrics
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougel: Optional[float] = None
    # BERTScore metrics
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None
    original_bertscore_precision: Optional[float] = None
    original_bertscore_recall: Optional[float] = None
    original_bertscore_f1: Optional[float] = None
    # Retrieval metrics
    hit_at_1: Optional[float] = None
    mrr: Optional[float] = None
    # Metadata
    latency_ms: int
    embedding_model_used: str
    llm_model_used: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    system_prompt_used: Optional[str] = None
    created_by: int
    created_at: datetime


class SemanticSimilarityGroupInfo(BaseModel):
    """Schema for group information with statistics."""

    name: str
    created_at: Optional[str] = None
    test_count: int = 0
    avg_rouge1: Optional[float] = None
    avg_rouge2: Optional[float] = None
    avg_rougel: Optional[float] = None
    avg_bertscore_precision: Optional[float] = None
    avg_bertscore_recall: Optional[float] = None
    avg_bertscore_f1: Optional[float] = None
    avg_original_bertscore_precision: Optional[float] = None
    avg_original_bertscore_recall: Optional[float] = None
    avg_original_bertscore_f1: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    search_top_k: Optional[int] = None
    search_alpha: Optional[float] = None
    reranker_used: Optional[bool] = None
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None


class SemanticSimilarityResultListResponse(BaseModel):
    """Schema for semantic similarity result list response."""

    results: List[SemanticSimilarityResultResponse]
    total: int
    groups: List[SemanticSimilarityGroupInfo]  # List of unique group names with creation dates
    # Aggregate statistics for all results (not just current page)
    aggregate: Optional[dict] = None  # Contains avg metrics for all results


# ==================== Admin User Management Schemas ====================

class AdminUserResponse(BaseModel):
    """Schema for admin user list response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class AdminUserListResponse(BaseModel):
    """Schema for admin user list with pagination."""

    users: List[AdminUserResponse]
    total: int
    page: int
    total_pages: int


class AdminUserUpdate(BaseModel):
    """Schema for updating a user by admin."""

    full_name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class AdminUserCreate(BaseModel):
    """Schema for creating a user by admin."""

    full_name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: UserRole = UserRole.STUDENT


class AdminUserUpdateResponse(BaseModel):
    """Schema for admin user update response."""

    success: bool
    user: AdminUserResponse
    message: str = "User updated successfully"


class AdminUserDeleteResponse(BaseModel):
    """Schema for admin user delete response."""

    success: bool
    message: str
    deleted_data: dict = Field(
        default_factory=dict,
        description="Summary of deleted related data"
    )


class AdminUserDeactivateResponse(BaseModel):
    """Schema for admin user deactivate/activate response."""

    success: bool
    message: str
    user: AdminUserResponse


class AdminStatisticsResponse(BaseModel):
    """Schema for admin statistics response."""

    total_users: int
    active_teachers: int
    active_students: int
    inactive_users: int
    new_users_this_month: int


class AdminPasswordResetResponse(BaseModel):
    """Schema for admin password reset response."""

    success: bool
    message: str
    temporary_password: str
    expires_at: datetime


# ==================== Batch Test Session Schemas ====================

class BatchTestSessionResponse(BaseModel):
    """Schema for batch test session response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int
    user_id: int
    group_name: str
    test_cases: str  # JSON string of test cases
    total_tests: int
    completed_tests: int
    failed_tests: int
    current_index: int
    status: str  # in_progress, completed, cancelled, failed
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model_used: Optional[str] = None
    # Search configuration
    search_top_k: Optional[int] = None
    search_alpha: Optional[float] = None
    # Reranker configuration
    reranker_used: Optional[bool] = None
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    updated_at: datetime


class BatchTestSessionListResponse(BaseModel):
    """Schema for batch test session list response."""

    sessions: List[BatchTestSessionResponse]
    total: int


class BatchTestSessionCreate(BaseModel):
    """Schema for creating a batch test session."""

    model_config = ConfigDict(populate_by_name=True)

    course_id: int
    test_cases: List[SemanticSimilarityTestCase]
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    # Search configuration
    search_top_k: Optional[int] = None
    search_alpha: Optional[float] = None
    # Reranker configuration
    reranker_used: Optional[bool] = None
    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None


class BatchTestSessionResumeRequest(BaseModel):
    """Schema for resuming a batch test session."""

    session_id: int


class TestDatasetCreate(BaseModel):
    """Schema for creating a test dataset."""
    
    course_id: int
    name: str
    description: Optional[str] = None
    test_cases: List[Dict[str, Any]]


# ==================== Backup Schemas ====================

class BackupInfo(BaseModel):
    """Schema for backup file information."""
    
    filename: str
    size: int  # Size in bytes
    created_at: datetime
    type: str  # postgres, weaviate, or full


class BackupListResponse(BaseModel):
    """Schema for backup list response."""
    
    backups: List[BackupInfo]
    total: int


class BackupCreateResponse(BaseModel):
    """Schema for backup creation response."""
    
    success: bool
    message: str
    filename: str
    size: int
    created_at: datetime


class BackupRestoreResponse(BaseModel):
    """Schema for backup restore response."""
    
    success: bool
    message: str
