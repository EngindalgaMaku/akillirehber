"""
Giskard Database Models

This module contains SQLAlchemy models for Giskard RAG testing,
including test sets, questions, evaluation runs, and results.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime,
    ForeignKey, JSON, Boolean
)
from sqlalchemy.orm import relationship

from app.database import Base


class GiskardTestSet(Base):
    """Giskard test set for hallucination testing."""
    __tablename__ = "giskard_test_sets"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(
        Integer,
        ForeignKey("courses.id"),
        nullable=False,
        index=True
    )
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    question_count = Column(Integer, nullable=False, default=0)
    created_by = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    # Relationships
    questions = relationship(
        "GiskardQuestion",
        back_populates="test_set",
        cascade="all, delete-orphan"
    )
    runs = relationship(
        "GiskardRun",
        back_populates="test_set",
        cascade="all, delete-orphan"
    )
    course = relationship("Course", back_populates="giskard_test_sets")


class GiskardQuestion(Base):
    """Giskard test question for hallucination testing."""
    __tablename__ = "giskard_questions"

    id = Column(Integer, primary_key=True, index=True)
    test_set_id = Column(
        Integer,
        ForeignKey("giskard_test_sets.id"),
        nullable=False,
        index=True
    )
    question = Column(Text, nullable=False)
    question_type = Column(
        String(50),
        nullable=False
    )  # "relevant"/"irrelevant"
    expected_answer = Column(Text, nullable=False)
    question_metadata = Column(JSON, nullable=True)  # topic, difficulty
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    test_set = relationship("GiskardTestSet", back_populates="questions")
    results = relationship(
        "GiskardResult",
        back_populates="question",
        cascade="all, delete-orphan"
    )


class GiskardRun(Base):
    """Giskard evaluation run."""
    __tablename__ = "giskard_runs"

    id = Column(Integer, primary_key=True, index=True)
    test_set_id = Column(
        Integer,
        ForeignKey("giskard_test_sets.id"),
        nullable=False,
        index=True
    )
    course_id = Column(
        Integer,
        ForeignKey("courses.id"),
        nullable=False,
        index=True
    )
    name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    config = Column(JSON, nullable=True)
    total_questions = Column(Integer, nullable=False, default=0)
    processed_questions = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    test_set = relationship("GiskardTestSet", back_populates="runs")
    results = relationship(
        "GiskardResult",
        back_populates="run",
        cascade="all, delete-orphan"
    )
    summary = relationship(
        "GiskardSummary",
        back_populates="run",
        uselist=False,
        cascade="all, delete-orphan"
    )
    course = relationship("Course", back_populates="giskard_runs")


class GiskardResult(Base):
    """Giskard evaluation result for a single question."""
    __tablename__ = "giskard_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(
        Integer,
        ForeignKey("giskard_runs.id"),
        nullable=False,
        index=True
    )
    question_id = Column(
        Integer,
        ForeignKey("giskard_questions.id"),
        nullable=False,
        index=True
    )
    question_text = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=False)
    generated_answer = Column(Text, nullable=False)
    question_type = Column(String(50), nullable=False)

    # Metrics
    score = Column(Float, nullable=True)
    correct_refusal = Column(Boolean, nullable=True)
    hallucinated = Column(Boolean, nullable=True)
    provided_answer = Column(Boolean, nullable=True)
    language = Column(String(50), nullable=True)
    quality_score = Column(Float, nullable=True)

    # Model info
    llm_provider = Column(String(100), nullable=True)
    llm_model = Column(String(100), nullable=True)
    embedding_model = Column(String(100), nullable=True)

    # Performance
    latency_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    run = relationship("GiskardRun", back_populates="results")
    question = relationship("GiskardQuestion", back_populates="results")


class GiskardSummary(Base):
    """Giskard evaluation run summary."""
    __tablename__ = "giskard_summaries"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(
        Integer,
        ForeignKey("giskard_runs.id"),
        nullable=False,
        unique=True,
        index=True
    )

    # Relevant questions metrics
    relevant_count = Column(Integer, nullable=False, default=0)
    relevant_avg_score = Column(Float, nullable=True)
    relevant_success_rate = Column(Float, nullable=True)

    # Irrelevant questions metrics
    irrelevant_count = Column(Integer, nullable=False, default=0)
    irrelevant_avg_score = Column(Float, nullable=True)
    irrelevant_success_rate = Column(Float, nullable=True)
    hallucination_rate = Column(Float, nullable=True)
    correct_refusal_rate = Column(Float, nullable=True)

    # Language consistency
    language_consistency = Column(Float, nullable=True)
    turkish_response_rate = Column(Float, nullable=True)

    # Overall
    overall_score = Column(Float, nullable=True)
    total_questions = Column(Integer, nullable=False, default=0)
    successful_questions = Column(Integer, nullable=False, default=0)
    failed_questions = Column(Integer, nullable=False, default=0)
    avg_latency_ms = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    run = relationship("GiskardRun", back_populates="summary")


class GiskardQuickTestResult(Base):
    """Giskard quick test result for single question testing."""
    __tablename__ = "giskard_quick_test_results"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(
        Integer,
        ForeignKey("courses.id"),
        nullable=False,
        index=True
    )
    group_name = Column(String(255), nullable=True, index=True)
    question = Column(Text, nullable=False)
    question_type = Column(String(50), nullable=False)
    expected_answer = Column(Text, nullable=False)
    generated_answer = Column(Text, nullable=False)

    # Metrics
    score = Column(Float, nullable=True)
    correct_refusal = Column(Boolean, nullable=True)
    hallucinated = Column(Boolean, nullable=True)
    provided_answer = Column(Boolean, nullable=True)
    language = Column(String(50), nullable=True)
    quality_score = Column(Float, nullable=True)

    # Model info
    system_prompt = Column(Text, nullable=True)
    llm_provider = Column(String(100), nullable=True)
    llm_model = Column(String(100), nullable=True)
    embedding_model = Column(String(100), nullable=True)

    # Performance
    latency_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    created_by = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    course = relationship(
        "Course",
        back_populates="giskard_quick_test_results"
    )
    creator = relationship(
        "User",
        back_populates="giskard_quick_test_results"
    )
