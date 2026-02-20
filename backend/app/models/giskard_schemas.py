"""
Giskard Pydantic Schemas

This module contains Pydantic schemas for Giskard RAG testing,
including test sets, questions, evaluation runs, and results.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


# ==================== Test Set Schemas ====================


class GiskardTestSetCreate(BaseModel):
    """Schema for creating a Giskard test set."""
    course_id: int
    name: str
    description: Optional[str] = None


class GiskardTestSetUpdate(BaseModel):
    """Schema for updating a Giskard test set."""
    name: Optional[str] = None
    description: Optional[str] = None


class GiskardTestSetResponse(BaseModel):
    """Schema for Giskard test set response."""
    id: int
    course_id: int
    name: str
    description: Optional[str]
    created_by: int
    created_at: datetime
    updated_at: datetime
    question_count: int = 0


class GiskardTestSetDetailResponse(BaseModel):
    """Schema for Giskard test set detail response."""
    id: int
    course_id: int
    name: str
    description: Optional[str]
    created_by: int
    created_at: datetime
    updated_at: datetime
    question_count: int = 0
    questions: List["GiskardQuestionResponse"] = []


# ==================== Question Schemas ====================


class GiskardQuestionCreate(BaseModel):
    """Schema for creating a Giskard question."""
    question: str
    question_type: str = Field(
        ...,
        description="Type of question: 'relevant' or 'irrelevant'"
    )
    expected_answer: str
    question_metadata: Optional[dict] = None


class GiskardQuestionUpdate(BaseModel):
    """Schema for updating a Giskard question."""
    question: Optional[str] = None
    question_type: Optional[str] = None
    expected_answer: Optional[str] = None
    question_metadata: Optional[dict] = None


class GiskardQuestionResponse(BaseModel):
    """Schema for Giskard question response."""
    id: int
    test_set_id: int
    question: str
    question_type: str
    expected_answer: str
    question_metadata: Optional[dict]
    created_at: datetime


class GiskardRAGETGenerateTestsetRequest(BaseModel):
    course_id: int
    num_questions: int = Field(..., ge=1, le=300)
    language: Optional[str] = None
    agent_description: Optional[str] = None


class GiskardRAGETGenerateTestsetResponse(BaseModel):
    num_questions: int
    samples: List[dict]


# ==================== Evaluation Run Schemas ====================


class GiskardEvaluationRunCreate(BaseModel):
    """Schema for creating a Giskard evaluation run."""
    test_set_id: int
    course_id: int
    name: str
    config: Optional[dict] = None
    total_questions: int


class GiskardEvaluationRunResponse(BaseModel):
    """Schema for Giskard evaluation run response."""
    id: int
    test_set_id: int
    test_set_name: Optional[str] = None
    course_id: int
    name: str
    status: str
    config: Optional[dict]
    total_questions: int
    processed_questions: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    created_at: datetime
    overall_score: Optional[float] = None
    hallucination_rate: Optional[float] = None
    turkish_response_rate: Optional[float] = None


class GiskardEvaluationRunDetailResponse(BaseModel):
    """Schema for Giskard evaluation run detail response."""
    id: int
    test_set_id: int
    course_id: int
    name: str
    status: str
    config: Optional[dict]
    total_questions: int
    processed_questions: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    created_at: datetime
    results: List["GiskardResultResponse"] = []
    summary: Optional["GiskardSummaryResponse"] = None


# ==================== Result Schemas ====================


class GiskardResultResponse(BaseModel):
    """Schema for Giskard result response."""
    id: int
    run_id: int
    question_id: int
    question_text: str
    expected_answer: str
    generated_answer: str
    question_type: str

    # Metrics
    score: Optional[float]
    correct_refusal: Optional[bool]
    hallucinated: Optional[bool]
    provided_answer: Optional[bool]
    language: Optional[str]
    quality_score: Optional[float]

    # Model info
    llm_provider: Optional[str]
    llm_model: Optional[str]
    embedding_model: Optional[str]

    # Performance
    latency_ms: Optional[int]
    error_message: Optional[str]

    created_at: datetime


class GiskardSummaryResponse(BaseModel):
    """Schema for Giskard summary response."""
    id: int
    run_id: int

    # Relevant questions metrics
    relevant_count: int
    relevant_avg_score: Optional[float]
    relevant_success_rate: Optional[float]

    # Irrelevant questions metrics
    irrelevant_count: int
    irrelevant_avg_score: Optional[float]
    irrelevant_success_rate: Optional[float]
    hallucination_rate: Optional[float]
    correct_refusal_rate: Optional[float]

    # Language consistency
    language_consistency: Optional[float]
    turkish_response_rate: Optional[float]

    # Overall
    overall_score: Optional[float]
    total_questions: int
    successful_questions: int
    failed_questions: int
    avg_latency_ms: Optional[float]

    created_at: datetime


# ==================== Quick Test Schemas ====================


class GiskardQuickTestRequest(BaseModel):
    """Schema for Giskard quick test request."""
    course_id: int
    question: str
    question_type: str = Field(
        ...,
        description="Type of question: 'relevant' or 'irrelevant'"
    )
    expected_answer: str
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    system_prompt: Optional[str] = None


class GiskardQuickTestResponse(BaseModel):
    """Schema for Giskard quick test response."""
    question: str
    expected_answer: str
    generated_answer: str
    question_type: str

    # Metrics
    score: Optional[float]
    correct_refusal: Optional[bool]
    hallucinated: Optional[bool]
    provided_answer: Optional[bool]
    language: Optional[str]
    quality_score: Optional[float]

    # Model info
    system_prompt_used: Optional[str]
    llm_provider_used: Optional[str]
    llm_model_used: Optional[str]
    embedding_model_used: Optional[str]

    # Performance
    latency_ms: Optional[int]
    error_message: Optional[str]


class GiskardQuickTestResultCreate(BaseModel):
    """Schema for creating a Giskard quick test result."""
    course_id: int
    group_name: Optional[str] = None
    question: str
    question_type: str
    expected_answer: str
    generated_answer: str

    # Metrics
    score: Optional[float]
    correct_refusal: Optional[bool]
    hallucinated: Optional[bool]
    provided_answer: Optional[bool]
    language: Optional[str]
    quality_score: Optional[float]

    # Model info
    system_prompt: Optional[str]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    embedding_model: Optional[str]

    # Performance
    latency_ms: Optional[int]
    error_message: Optional[str]


class GiskardQuickTestResultResponse(BaseModel):
    """Schema for Giskard quick test result response."""
    id: int
    course_id: int
    group_name: Optional[str]
    question: str
    question_type: str
    expected_answer: str
    generated_answer: str

    # Metrics
    score: Optional[float]
    correct_refusal: Optional[bool]
    hallucinated: Optional[bool]
    provided_answer: Optional[bool]
    language: Optional[str]
    quality_score: Optional[float]

    # Model info
    system_prompt: Optional[str]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    embedding_model: Optional[str]

    # Performance
    latency_ms: Optional[int]
    error_message: Optional[str]

    created_by: int
    created_at: datetime


class GiskardQuickTestResultListResponse(BaseModel):
    """Schema for listing Giskard quick test results."""
    results: List[GiskardQuickTestResultResponse]
    total: int
    groups: List[str] = []
