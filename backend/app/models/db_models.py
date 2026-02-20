"""SQLAlchemy database models for RAG Educational Chatbot."""

import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Enum,
    Boolean,
    Float,
    JSON,
)
from sqlalchemy.orm import relationship

from app.database import Base


class UserRole(str, enum.Enum):
    """User role enumeration."""

    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"


class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.STUDENT)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    courses = relationship("Course", back_populates="teacher", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    batch_test_sessions = relationship("BatchTestSession", back_populates="user", cascade="all, delete-orphan")
    test_datasets = relationship("TestDataset", back_populates="user", cascade="all, delete-orphan")
    giskard_quick_test_results = relationship("GiskardQuickTestResult", back_populates="creator", cascade="all, delete-orphan")

    @property
    def is_admin(self) -> bool:
        """Check if user is admin (ID=1)."""
        return self.role == UserRole.ADMIN

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class Course(Base):
    """Course model for organizing educational content."""

    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    teacher = relationship("User", back_populates="courses")
    documents = relationship("Document", back_populates="course", cascade="all, delete-orphan")
    settings = relationship("CourseSettings", back_populates="course", uselist=False, cascade="all, delete-orphan")
    prompt_templates = relationship(
        "CoursePromptTemplate",
        back_populates="course",
        cascade="all, delete-orphan",
    )
    semantic_similarity_results = relationship("SemanticSimilarityResult", back_populates="course", cascade="all, delete-orphan")
    batch_test_sessions = relationship("BatchTestSession", back_populates="course", cascade="all, delete-orphan")
    test_datasets = relationship("TestDataset", back_populates="course", cascade="all, delete-orphan")
    giskard_test_sets = relationship("GiskardTestSet", back_populates="course", cascade="all, delete-orphan")
    giskard_runs = relationship("GiskardRun", back_populates="course", cascade="all, delete-orphan")
    giskard_quick_test_results = relationship("GiskardQuickTestResult", back_populates="course", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Course(id={self.id}, name={self.name})>"


class ChatMessageDB(Base):
    """Persistent chat message for a course conversation."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(
        Integer,
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return (
            f"<ChatMessageDB(id={self.id}, course_id={self.course_id}, "
            f"user_id={self.user_id}, role={self.role})>"
        )


class CourseSettings(Base):
    """Course-specific settings for RAG behavior."""

    __tablename__ = "course_settings"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), unique=True, nullable=False)
    vector_store = Column(String(50), default="weaviate", nullable=False)
    
    # Chunking defaults
    default_chunk_strategy = Column(String(50), default="recursive")
    default_chunk_size = Column(Integer, default=500)
    default_overlap = Column(Integer, default=50)
    
    # Embedding defaults
    default_embedding_model = Column(String(255), default="openai/text-embedding-3-small")
    
    # Search settings
    search_alpha = Column(Float, default=0.5)  # 0=keyword, 1=vector
    search_top_k = Column(Integer, default=5)
    min_relevance_score = Column(Float, default=0.0)  # 0-1, filter low-score results
    
    # LLM settings
    llm_provider = Column(String(50), default="openrouter")
    llm_model = Column(String(255), default="openai/gpt-4o-mini")
    llm_temperature = Column(Float, default=0.7)
    llm_max_tokens = Column(Integer, default=1000)
    
    # System prompt for course-specific AI behavior
    system_prompt = Column(Text, nullable=True)

    active_prompt_template_id = Column(
        Integer,
        ForeignKey("course_prompt_templates.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Bloom-specific system prompts for test generation
    system_prompt_remembering = Column(Text, nullable=True)
    system_prompt_understanding_applying = Column(Text, nullable=True)
    system_prompt_analyzing_evaluating = Column(Text, nullable=True)
    
    # Direct LLM mode (bypass RAG pipeline)
    enable_direct_llm = Column(Boolean, default=False, nullable=False)
    
    # PII filter (kişisel bilgi filtresi)
    enable_pii_filter = Column(Boolean, default=False, nullable=False)
    
    # Reranker settings
    enable_reranker = Column(Boolean, default=False, nullable=False)
    reranker_provider = Column(String(50), nullable=True)  # cohere/alibaba
    reranker_model = Column(String(100), nullable=True)
    reranker_top_k = Column(Integer, default=20, nullable=False)
    
    # Test Generation settings
    test_generation_persona = Column(String(500), nullable=True)  # Student persona for RAGAS test generation
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="settings")
    active_prompt_template = relationship(
        "CoursePromptTemplate",
        foreign_keys=[active_prompt_template_id],
    )

    def __repr__(self):
        return f"<CourseSettings(course_id={self.course_id})>"


class CoursePromptTemplate(Base):
    __tablename__ = "course_prompt_templates"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(
        Integer,
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    course = relationship("Course", back_populates="prompt_templates")

    def __repr__(self):
        return (
            f"<CoursePromptTemplate(id={self.id}, course_id={self.course_id}, "
            f"name={self.name})>"
        )


class EmbeddingStatus(str, enum.Enum):
    """Embedding status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class Document(Base):
    """Document model for uploaded files."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    char_count = Column(Integer, nullable=True)
    content = Column(Text, nullable=True)  # Extracted text content
    is_processed = Column(Boolean, default=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=True)  # Made nullable for user-specific docs
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # Required for user-specific docs
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Embedding fields
    embedding_status = Column(
        Enum(EmbeddingStatus),
        default=EmbeddingStatus.PENDING,
        nullable=False
    )
    embedding_model = Column(String(255), nullable=True)
    embedded_at = Column(DateTime, nullable=True)
    vector_count = Column(Integer, default=0)

    # Relationships
    course = relationship("Course", back_populates="documents")
    user = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename})>"


class Chunk(Base):
    """Chunk model for text segments."""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    index = Column(Integer, nullable=False)
    start_position = Column(Integer, nullable=False)
    end_position = Column(Integer, nullable=False)
    char_count = Column(Integer, nullable=False)
    has_overlap = Column(Boolean, default=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    # Weaviate vector ID (for future embedding storage)
    vector_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk(id={self.id}, index={self.index}, doc_id={self.document_id})>"


class ProcessingStatusEnum(str, enum.Enum):
    """Processing status enumeration for document processing pipeline."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    ERROR = "error"
    RETRYING = "retrying"


class ProcessingStatus(Base):
    """Processing status tracking for documents."""

    __tablename__ = "processing_status"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, unique=True)
    status = Column(String(50), nullable=False, default="pending")
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)  # Store detailed error information
    processing_duration = Column(Float, nullable=True)  # Duration in seconds
    retry_count = Column(Integer, default=0)
    last_retry_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    document = relationship("Document", backref="processing_status")

    def __repr__(self):
        return f"<ProcessingStatus(document_id={self.document_id}, status={self.status})>"


class DiagnosticReport(Base):
    """Diagnostic report for document processing analysis."""

    __tablename__ = "diagnostic_reports"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    report_type = Column(String(50), nullable=False)  # "processing", "chunking", "embedding"
    
    # File information
    file_info = Column(JSON, nullable=True)  # Original file metadata
    
    # Extraction information
    extraction_info = Column(JSON, nullable=True)  # Text extraction details
    
    # Chunking information
    chunking_info = Column(JSON, nullable=True)  # Chunking process details
    
    # Error log
    error_log = Column(JSON, nullable=True)  # List of errors encountered
    
    # Performance metrics
    performance_metrics = Column(JSON, nullable=True)  # Processing performance data
    
    # Recommendations
    recommendations = Column(JSON, nullable=True)  # List of improvement suggestions
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", backref="diagnostic_reports")

    def __repr__(self):
        return f"<DiagnosticReport(document_id={self.document_id}, type={self.report_type})>"


class ChunkQualityMetrics(Base):
    """Quality metrics for document chunks."""

    __tablename__ = "chunk_quality_metrics"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, unique=True)
    
    # Basic metrics
    total_chunks = Column(Integer, nullable=False)
    avg_chunk_size = Column(Integer, nullable=False)
    min_chunk_size = Column(Integer, nullable=False)
    max_chunk_size = Column(Integer, nullable=False)
    
    # Size distribution (JSON with size ranges and counts)
    size_distribution = Column(JSON, nullable=True)
    
    # Overlap analysis
    overlap_analysis = Column(JSON, nullable=True)
    
    # Content quality score (0.0 to 1.0)
    content_quality_score = Column(Float, nullable=False, default=0.0)
    
    # Quality recommendations
    recommendations = Column(JSON, nullable=True)
    
    # Processing metadata
    chunking_strategy = Column(String(50), nullable=True)
    chunk_size_config = Column(Integer, nullable=True)
    overlap_config = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    document = relationship("Document", backref="chunk_quality_metrics")

    def __repr__(self):
        return f"<ChunkQualityMetrics(document_id={self.document_id}, total_chunks={self.total_chunks})>"


# ==================== RAGAS Evaluation Models ====================


class EvaluationStatus(str, enum.Enum):
    """Evaluation run status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TestSet(Base):
    """Test set for RAGAS evaluation containing question-answer pairs."""

    __tablename__ = "test_sets"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    course = relationship("Course", backref="test_sets")
    creator = relationship("User", backref="test_sets")
    questions = relationship("TestQuestion", back_populates="test_set", cascade="all, delete-orphan")
    evaluation_runs = relationship("EvaluationRun", back_populates="test_set", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TestSet(id={self.id}, name={self.name})>"


class TestQuestion(Base):
    """Individual test question with ground truth answer."""

    __tablename__ = "test_questions"

    id = Column(Integer, primary_key=True, index=True)
    test_set_id = Column(Integer, ForeignKey("test_sets.id"), nullable=False)
    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=False)  # Primary expected answer
    alternative_ground_truths = Column(JSON, nullable=True)  # Alternative correct answers
    expected_contexts = Column(JSON, nullable=True)  # Array of expected source texts
    question_metadata = Column(JSON, nullable=True)  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    test_set = relationship("TestSet", back_populates="questions")
    results = relationship("EvaluationResult", back_populates="question", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TestQuestion(id={self.id}, question={self.question[:50]}...)>"


class EvaluationRun(Base):
    """Single evaluation run of a test set."""

    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    test_set_id = Column(Integer, ForeignKey("test_sets.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    name = Column(String(255), nullable=True)  # Optional run name
    status = Column(String(50), default="pending", nullable=False)
    config = Column(JSON, nullable=True)  # Run configuration (search_type, alpha, top_k, etc.)
    
    # Progress tracking
    total_questions = Column(Integer, default=0)
    processed_questions = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    test_set = relationship("TestSet", back_populates="evaluation_runs")
    course = relationship("Course", backref="evaluation_runs")
    results = relationship("EvaluationResult", back_populates="run", cascade="all, delete-orphan")
    summary = relationship("RunSummary", back_populates="run", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<EvaluationRun(id={self.id}, status={self.status})>"


class EvaluationResult(Base):
    """Individual question evaluation result with RAGAS metrics."""

    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("test_questions.id"), nullable=False)
    
    # Input/Output
    question_text = Column(Text, nullable=False)
    ground_truth_text = Column(Text, nullable=False)
    generated_answer = Column(Text, nullable=True)
    retrieved_contexts = Column(JSON, nullable=True)  # Array of retrieved context texts
    
    # RAGAS Metrics (0.0 to 1.0)
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_precision = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)
    answer_correctness = Column(Float, nullable=True)
    
    # Performance
    latency_ms = Column(Integer, nullable=True)
    
    # Model Information - what was used for THIS specific question
    llm_provider = Column(String, nullable=True)
    llm_model = Column(String, nullable=True)
    embedding_model = Column(String, nullable=True)
    evaluation_model = Column(String, nullable=True)  # Model used for RAGAS evaluation
    search_alpha = Column(Float, nullable=True)
    search_top_k = Column(Integer, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    run = relationship("EvaluationRun", back_populates="results")
    question = relationship("TestQuestion", back_populates="results")

    def __repr__(self):
        return f"<EvaluationResult(id={self.id}, run_id={self.run_id})>"


class SystemSettings(Base):
    """System-wide settings including registration keys."""

    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, index=True)
    teacher_registration_key = Column(String(255), nullable=True)
    student_registration_key = Column(String(255), nullable=True)
    hcaptcha_site_key = Column(String(255), nullable=True)
    hcaptcha_secret_key = Column(String(255), nullable=True)
    captcha_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<SystemSettings(id={self.id})>"


class RunSummary(Base):
    """Aggregated metrics summary for an evaluation run."""

    __tablename__ = "run_summaries"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False, unique=True)
    
    # Aggregated RAGAS Metrics
    avg_faithfulness = Column(Float, nullable=True)
    avg_answer_relevancy = Column(Float, nullable=True)
    avg_context_precision = Column(Float, nullable=True)
    avg_context_recall = Column(Float, nullable=True)
    avg_answer_correctness = Column(Float, nullable=True)
    
    # Performance
    avg_latency_ms = Column(Float, nullable=True)
    
    # Counts
    total_questions = Column(Integer, default=0)
    successful_questions = Column(Integer, default=0)
    failed_questions = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    run = relationship("EvaluationRun", back_populates="summary")

    def __repr__(self):
        return f"<RunSummary(run_id={self.run_id}, avg_faithfulness={self.avg_faithfulness})>"


class QuickTestResult(Base):
    """Saved quick test result for later viewing."""

    __tablename__ = "quick_test_results"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    group_name = Column(String(255), nullable=True)  # For grouping results
    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=False)
    alternative_ground_truths = Column(JSON, nullable=True)
    system_prompt = Column(Text, nullable=True)
    llm_provider = Column(String(100), nullable=False)
    llm_model = Column(String(255), nullable=False)
    evaluation_model = Column(String(255), nullable=True)  # Model used for RAGAS evaluation
    embedding_model = Column(String(255), nullable=True)  # Embedding model used at test time
    search_top_k = Column(Integer, nullable=True)  # Retrieval top_k used at test time
    search_alpha = Column(Float, nullable=True)  # Hybrid search alpha used at test time
    reranker_used = Column(Boolean, nullable=True)
    reranker_provider = Column(String(255), nullable=True)
    reranker_model = Column(String(255), nullable=True)
    generated_answer = Column(Text, nullable=False)
    retrieved_contexts = Column(JSON, nullable=True)
    
    # RAGAS Metrics (0.0 to 1.0)
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_precision = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)
    answer_correctness = Column(Float, nullable=True)
    
    # Performance
    latency_ms = Column(Integer, nullable=False)
    
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    course = relationship("Course", backref="quick_test_results")
    creator = relationship("User", backref="quick_test_results")

    def __repr__(self):
        return f"<QuickTestResult(id={self.id}, course_id={self.course_id}, question={self.question[:50]}...)>"


class CustomLLMModel(Base):
    """Custom LLM model added by users."""

    __tablename__ = "custom_llm_models"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), nullable=False)  # openrouter, groq, openai, etc.
    model_id = Column(String(255), nullable=False)  # The actual model identifier
    display_name = Column(String(255), nullable=False)  # User-friendly name
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    creator = relationship("User", backref="custom_llm_models")

    # Unique constraint on provider + model_id
    __table_args__ = (
        {"sqlite_autoincrement": True},
    )

    def __repr__(self):
        return f"<CustomLLMModel(id={self.id}, provider={self.provider}, model_id={self.model_id})>"


class RefreshToken(Base):
    """Refresh token model for session management."""

    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(500), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    revoked = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="refresh_tokens")

    def __repr__(self):
        return f"<RefreshToken(id={self.id}, user_id={self.user_id}, revoked={self.revoked})>"


class SemanticSimilarityResult(Base):
    """Semantic similarity test result for measuring answer similarity."""

    __tablename__ = "semantic_similarity_results"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    group_name = Column(String(255), nullable=True, index=True)
    batch_session_id = Column(Integer, ForeignKey("batch_test_sessions.id"), nullable=True)
    
    # Test inputs
    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=False)
    alternative_ground_truths = Column(JSON, nullable=True)
    generated_answer = Column(Text, nullable=False)
    bloom_level = Column(String(50), nullable=True)  # remembering, understanding_applying, analyzing_evaluating
    
    # Test results - Cosine Similarity
    similarity_score = Column(Float, nullable=False)
    best_match_ground_truth = Column(Text, nullable=False)
    all_scores = Column(JSON, nullable=True)  # List of {ground_truth, score}
    
    # Test results - ROUGE metrics
    rouge1 = Column(Float, nullable=True)  # ROUGE-1 F1 score
    rouge2 = Column(Float, nullable=True)  # ROUGE-2 F1 score
    rougel = Column(Float, nullable=True)  # ROUGE-L F1 score (lowercase)
    
    # Test results - BERTScore metrics
    bertscore_precision = Column(Float, nullable=True)
    bertscore_recall = Column(Float, nullable=True)
    bertscore_f1 = Column(Float, nullable=True)

    # Test results - Original BERTScore metrics
    original_bertscore_precision = Column(Float, nullable=True)
    original_bertscore_recall = Column(Float, nullable=True)
    original_bertscore_f1 = Column(Float, nullable=True)
    
    # Test results - Retrieval metrics
    hit_at_1 = Column(Float, nullable=True)  # 1 if best match is rank 1, else 0
    mrr = Column(Float, nullable=True)  # Mean Reciprocal Rank (1/rank)
    
    # RAG context
    retrieved_contexts = Column(JSON, nullable=True)  # List of context strings
    system_prompt_used = Column(Text, nullable=True)
    
    # Metadata
    latency_ms = Column(Integer, nullable=False)
    embedding_model_used = Column(String(255), nullable=False)
    llm_model_used = Column(String(255), nullable=True)
    
    # Search configuration used
    search_top_k = Column(Integer, nullable=True)
    search_alpha = Column(Float, nullable=True)
    
    # Reranker configuration
    reranker_used = Column(Boolean, nullable=True, default=False)
    reranker_provider = Column(String(50), nullable=True)
    reranker_model = Column(String(255), nullable=True)
    
    # Audit fields
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    course = relationship("Course", back_populates="semantic_similarity_results")
    creator = relationship("User", backref="semantic_similarity_results")
    batch_session = relationship("BatchTestSession", back_populates="results", foreign_keys=[batch_session_id])

    def __repr__(self):
        return f"<SemanticSimilarityResult(id={self.id}, course_id={self.course_id}, similarity_score={self.similarity_score})>"


class BatchTestSession(Base):
    """Batch test session for tracking progress and enabling resume functionality."""

    __tablename__ = "batch_test_sessions"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Session identification
    group_name = Column(String(255), nullable=False, index=True)  # Auto-generated group name

    # Test cases (stored as JSON)
    test_cases = Column(Text, nullable=False)  # JSON array of test cases

    # Progress tracking
    total_tests = Column(Integer, nullable=False, default=0)
    completed_tests = Column(Integer, nullable=False, default=0)
    failed_tests = Column(Integer, nullable=False, default=0)
    current_index = Column(Integer, nullable=False, default=0)  # Index of next test to process

    # Status
    status = Column(String(50), nullable=False, default="in_progress")  # in_progress, completed, cancelled, failed

    # Configuration used for this session
    llm_provider = Column(String(50), nullable=True)
    llm_model = Column(String(255), nullable=True)
    embedding_model_used = Column(String(255), nullable=True)
    # Search configuration
    search_top_k = Column(Integer, nullable=True)
    search_alpha = Column(Float, nullable=True)
    # Reranker configuration
    reranker_used = Column(Boolean, nullable=True, default=False)
    reranker_provider = Column(String(50), nullable=True)
    reranker_model = Column(String(255), nullable=True)

    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="batch_test_sessions")
    user = relationship("User", back_populates="batch_test_sessions")
    results = relationship("SemanticSimilarityResult", back_populates="batch_session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<BatchTestSession(id={self.id}, group_name={self.group_name}, status={self.status})>"


class TestDataset(Base):
    """Test dataset for storing reusable batch test data."""

    __tablename__ = "test_datasets"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Dataset identification
    name = Column(String(255), nullable=False)  # e.g., "Veri Seti 1", "Test Dataset 2"
    description = Column(Text, nullable=True)

    # Test cases (stored as JSON)
    test_cases = Column(Text, nullable=False)  # JSON array of test cases

    # Metadata
    total_test_cases = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="test_datasets")
    user = relationship("User", back_populates="test_datasets")

    def __repr__(self):
        return f"<TestDataset(id={self.id}, name={self.name}, course_id={self.course_id})>"



# ==================== Admin User Management Models ====================


class AuditLog(Base):
    """Audit log for tracking admin operations."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    admin_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String(50), nullable=False)  # 'edit', 'delete', 'deactivate', 'activate', 'reset_password'
    target_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # User being acted upon
    details = Column(JSON, nullable=True)  # Additional action details
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6

    # Relationships
    admin_user = relationship("User", foreign_keys=[admin_user_id])
    target_user = relationship("User", foreign_keys=[target_user_id])

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, admin_id={self.admin_user_id})>"


class TemporaryPassword(Base):
    """Temporary password for password reset functionality."""

    __tablename__ = "temporary_passwords"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False, nullable=False)

    # Relationships
    user = relationship("User")

    def __repr__(self):
        return f"<TemporaryPassword(id={self.id}, user_id={self.user_id}, used={self.used})>"
