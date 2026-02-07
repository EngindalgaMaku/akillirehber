"""
Giskard Integration Service

This module provides integration with existing RAG system
for Giskard RAG testing, including hallucination detection,
relevance testing, and Turkish language support.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import func
from sqlalchemy.orm import Session
from fastapi import Depends

from app.database import get_db

from app.models.giskard_models import (
    GiskardTestSet, GiskardQuestion, GiskardRun,
    GiskardResult, GiskardSummary
)
from app.services.giskard_service import create_giskard_tester
from app.services.course_service import get_or_create_settings, DEFAULT_SYSTEM_PROMPT
from app.services.weaviate_service import WeaviateService
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class GiskardIntegrationService:
    """Service for integrating Giskard RAG testing with RAG system."""

    def __init__(self, db: Session):
        """
        Initialize Giskard integration service.

        Args:
            db: Database session
        """
        self.db = db
        self.weaviate_service = WeaviateService()
        self.embedding_service = EmbeddingService()

    def create_test_set(
        self,
        course_id: int,
        name: str,
        description: Optional[str],
        created_by: int
    ) -> GiskardTestSet:
        """
        Create a new Giskard test set.

        Args:
            course_id: Course ID
            name: Test set name
            description: Optional description
            created_by: User ID who created it

        Returns:
            Created GiskardTestSet
        """
        test_set = GiskardTestSet(
            course_id=course_id,
            name=name,
            description=description,
            created_by=created_by
        )
        self.db.add(test_set)
        self.db.commit()
        self.db.refresh(test_set)
        return test_set

    def add_question(
        self,
        test_set_id: int,
        question: str,
        question_type: str,
        expected_answer: str,
        metadata: Optional[dict] = None
    ) -> GiskardQuestion:
        """
        Add a question to a test set.

        Args:
            test_set_id: Test set ID
            question: Question text
            question_type: Type ("relevant" or "irrelevant")
            expected_answer: Expected answer
            metadata: Optional metadata

        Returns:
            Created GiskardQuestion
        """
        q = GiskardQuestion(
            test_set_id=test_set_id,
            question=question,
            question_type=question_type,
            expected_answer=expected_answer,
            question_metadata=metadata
        )
        self.db.add(q)
        self.db.commit()
        self.db.refresh(q)
        return q

    def start_evaluation_run(
        self,
        test_set_id: int,
        course_id: int,
        name: str,
        config: Optional[dict] = None
    ) -> GiskardRun:
        """
        Start a new Giskard evaluation run.

        Args:
            test_set_id: Test set ID
            course_id: Course ID
            name: Run name
            config: Optional configuration

        Returns:
            Created GiskardRun
        """
        # Get questions count
        question_count = self.db.query(
            func.count(GiskardQuestion.id)
        ).filter(
            GiskardQuestion.test_set_id == test_set_id
        ).scalar() or 0

        run = GiskardRun(
            test_set_id=test_set_id,
            course_id=course_id,
            name=name,
            status="pending",
            config=config,
            total_questions=question_count,
            processed_questions=0
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def generate_questions_for_test_set(
        self,
        test_set_id: int,
        num_relevant: int,
        num_irrelevant: int,
        language: str,
        replace_existing: bool = True,
    ) -> dict:
        test_set = self.db.query(GiskardTestSet).filter(
            GiskardTestSet.id == test_set_id
        ).first()
        if not test_set:
            raise ValueError("Test set not found")

        if language not in ("tr", "en"):
            raise ValueError("Invalid language")

        if replace_existing:
            self.db.query(GiskardQuestion).filter(
                GiskardQuestion.test_set_id == test_set_id
            ).delete(synchronize_session=False)
            self.db.commit()

        course_settings = get_or_create_settings(self.db, test_set.course_id)

        llm_service = LLMService(
            provider=course_settings.llm_provider,
            model=course_settings.llm_model,
            temperature=course_settings.llm_temperature,
            max_tokens=course_settings.llm_max_tokens,
        )

        tester = create_giskard_tester(
            model_name=course_settings.llm_model,
            provider=course_settings.llm_provider,
            llm_service=llm_service,
            language=language,
            num_test_questions=num_relevant,
            num_irrelevant_questions=num_irrelevant,
        )

        document_content = self._get_document_content(test_set.course_id)

        extra_contexts: List[str] = []
        try:
            from app.models.db_models import Course

            course = self.db.query(Course).filter(
                Course.id == test_set.course_id
            ).first()
            query_text = course.name if course and course.name else "course"
            weaviate_results = self.weaviate_service.keyword_search(
                course_id=test_set.course_id,
                query=query_text,
                limit=20,
            )
            extra_contexts = [r.content for r in weaviate_results if r.content]
        except Exception as e:
            logger.warning(f"Weaviate context fetch failed: {e}")

        combined_content = "\n\n".join(
            [c for c in [document_content] + extra_contexts if c]
        )

        if not combined_content.strip():
            raise ValueError(
                "No course content found to generate questions. "
                "Upload and process documents for this course first."
            )

        try:
            generated = tester.generate_test_questions(
                document_content=combined_content,
                num_relevant=num_relevant,
                num_irrelevant=num_irrelevant,
            )
        except Exception as e:
            raise ValueError(
                "Failed to generate questions with LLM. "
                "Check LLM credentials/config and backend logs for details."
            ) from e

        created_relevant = 0
        created_irrelevant = 0

        to_add: List[GiskardQuestion] = []

        for q in generated.get("relevant", []) or []:
            question_text = q.get("question")
            expected_answer = q.get("expected_answer")
            if not question_text or not expected_answer:
                continue

            metadata = {
                k: v
                for k, v in q.items()
                if k not in ("question", "expected_answer")
            }
            to_add.append(
                GiskardQuestion(
                    test_set_id=test_set_id,
                    question=question_text,
                    question_type="relevant",
                    expected_answer=expected_answer,
                    question_metadata=metadata or None,
                )
            )
            created_relevant += 1

        default_unknown = "I don't know" if language == "en" else "Bilmiyorum"
        for q in generated.get("irrelevant", []) or []:
            question_text = q.get("question")
            expected_answer = q.get("expected_answer") or default_unknown
            if not question_text:
                continue

            metadata = {
                k: v
                for k, v in q.items()
                if k not in ("question", "expected_answer")
            }
            to_add.append(
                GiskardQuestion(
                    test_set_id=test_set_id,
                    question=question_text,
                    question_type="irrelevant",
                    expected_answer=expected_answer,
                    question_metadata=metadata or None,
                )
            )
            created_irrelevant += 1

        if to_add:
            self.db.add_all(to_add)
            self.db.commit()

        test_set.question_count = self.db.query(GiskardQuestion).filter(
            GiskardQuestion.test_set_id == test_set_id
        ).count()
        self.db.commit()

        return {
            "test_set_id": test_set_id,
            "created_relevant": created_relevant,
            "created_irrelevant": created_irrelevant,
            "total_created": created_relevant + created_irrelevant,
        }

    def generate_raget_testset(
        self,
        course_id: int,
        num_questions: int,
        language: str | None = None,
        agent_description: str | None = None,
    ) -> dict:
        try:
            import pandas as pd
            from giskard.rag import KnowledgeBase, generate_testset
        except Exception as e:
            raise ValueError(
                "Failed to import Giskard RAGET dependencies in the backend "
                f"container: {e}. "
                "Rebuild Docker images or add missing packages."
            ) from e

        # Get course settings to use correct embedding model
        course_settings = get_or_create_settings(self.db, course_id)

        import os
        # For RAGET, use OpenAI text-embedding-3-small directly
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "RAGET requires OPENAI_API_KEY environment variable. "
                "Please set it in your .env file."
            )
        
        # Use OpenAI embedding model directly (no OpenRouter, no custom base URL)
        embedding_model = "text-embedding-3-small"
        
        # Clear any custom API base URLs for RAGET - use OpenAI directly
        if "OPENAI_API_BASE" in os.environ:
            del os.environ["OPENAI_API_BASE"]
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]

        document_content = self._get_document_content(course_id)

        extra_contexts: List[str] = []
        try:
            from app.models.db_models import Course

            course = (
                self.db.query(Course)
                .filter(Course.id == course_id)
                .first()
            )
            query_text = course.name if course and course.name else "course"
            weaviate_results = self.weaviate_service.keyword_search(
                course_id=course_id,
                query=query_text,
                limit=20,
            )
            extra_contexts = [r.content for r in weaviate_results if r.content]
        except Exception as e:
            logger.warning(f"Weaviate context fetch failed: {e}")

        combined_content = "\n\n".join(
            [c for c in [document_content] + extra_contexts if c]
        )

        if not combined_content.strip():
            raise ValueError(
                "No course content found to generate a testset. "
                "Upload and process documents for this course first."
            )

        df = pd.DataFrame({"samples": [combined_content]})
        knowledge_base = KnowledgeBase.from_pandas(df, columns=["samples"])

        kwargs = {
            "knowledge_base": knowledge_base,
            "num_questions": num_questions,
        }
        if language:
            kwargs["language"] = language
        if agent_description:
            kwargs["agent_description"] = agent_description

        testset = generate_testset(**kwargs)
        samples = testset.to_pandas().to_dict(orient="records")

        return {
            "num_questions": len(samples),
            "samples": samples,
        }

    def run_evaluation(
        self,
        run_id: int
    ):
        """
        Run Giskard evaluation for a run.

        Args:
            run_id: Evaluation run ID
        """
        # Get run
        run = self.db.query(GiskardRun).filter(
            GiskardRun.id == run_id
        ).first()
        if not run:
            logger.error(f"Run {run_id} not found")
            return

        # Update status to running
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        self.db.commit()

        try:
            # Get course settings
            course_settings = get_or_create_settings(self.db, run.course_id)

            # Get questions
            questions = self.db.query(GiskardQuestion).filter(
                GiskardQuestion.test_set_id == run.test_set_id
            ).all()

            # Create Giskard tester
            llm_service = LLMService(
                provider=course_settings.llm_provider,
                model=course_settings.llm_model,
                temperature=course_settings.llm_temperature,
                max_tokens=course_settings.llm_max_tokens
            )

            tester = create_giskard_tester(
                model_name=course_settings.llm_model,
                provider=course_settings.llm_provider,
                llm_service=llm_service
            )

            # Define RAG function
            def rag_function(question: str) -> str:
                """RAG function for testing."""
                # Get embedding for question
                query_vector = self.embedding_service.get_embedding(
                    question,
                    model=course_settings.default_embedding_model
                )

                # Search for relevant chunks
                search_results = self.weaviate_service.hybrid_search(
                    course_id=run.course_id,
                    query=question,
                    query_vector=query_vector,
                    alpha=course_settings.search_alpha,
                    limit=course_settings.search_top_k
                )

                # Filter by minimum relevance score
                min_score = (
                    getattr(course_settings, 'min_relevance_score', 0.0)
                    or 0.0
                )
                if min_score > 0 and search_results:
                    search_results = [
                        r for r in search_results
                        if r.score >= min_score
                    ]

                # Extract contexts
                retrieved_contexts = []
                for result in search_results:
                    if result.content:
                        retrieved_contexts.append(result.content)

                # Build context for LLM
                context_text = "\n\n---\n\n".join(
                    retrieved_contexts
                ) if retrieved_contexts else ""

                # Generate system prompt
                default_system_prompt = DEFAULT_SYSTEM_PROMPT

                system_prompt = (
                    course_settings.system_prompt
                    if course_settings.system_prompt
                    else default_system_prompt
                )

                # Generate answer
                user_prompt = f"""Aşağıda ders dokümanlarından alınan bağlam bilgileri verilmiştir.

Bağlam:
{context_text}

Soru: {question}

Yukarıdaki bağlam bilgilerini kullanarak soruyu yanıtla. Cevabında bağlamdaki teknik terimleri ve ifadeleri aynen kullan. Bağlamda olmayan bilgi ekleme."""

                try:
                    generated_answer = llm_service.generate_response([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ])
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    generated_answer = ""

                return generated_answer

            # Get document content for test question generation
            document_content = self._get_document_content(run.course_id)

            # Run test suite
            results = tester.run_test_suite(
                rag_function=rag_function,
                document_content=document_content
            )

            # Save results to database
            self._save_results(run, questions, results)

            # Calculate and save summary
            self._save_summary(run, results)

            # Update run status
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            run.processed_questions = run.total_questions
            self.db.commit()

            logger.info(
                f"Giskard evaluation run {run_id} completed: "
                f"{results['overall_score']:.2%}"
            )

        except Exception as e:
            logger.error(f"Giskard evaluation run {run_id} failed: {e}")
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc)
            self.db.commit()

    def _get_document_content(self, course_id: int) -> str:
        """
        Get document content for test question generation.

        Args:
            course_id: Course ID

        Returns:
            Combined document content
        """
        from app.models.db_models import Chunk, Document

        chunks = (
            self.db.query(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .filter(Document.course_id == course_id)
            .order_by(func.random())
            .limit(50)
            .all()
        )

        content_parts = []
        for chunk in chunks:
            if chunk.content:
                content_parts.append(chunk.content)

        return "\n\n".join(content_parts)

    def _save_results(
        self,
        run: GiskardRun,
        questions: List[GiskardQuestion],
        results: dict
    ):
        """
        Save evaluation results to database.

        Args:
            run: Evaluation run
            questions: List of questions
            results: Test results
        """
        evaluations = results.get("evaluations", [])

        for i, (question, eval_result) in enumerate(
            zip(questions, evaluations)
        ):
            start_time = time.time()

            try:
                # Generate answer
                generated_answer = eval_result.get("answer", "")

                # Evaluate response
                evaluation = self._evaluate_response(
                    question,
                    generated_answer,
                    question.question_type
                )

                latency_ms = int((time.time() - start_time) * 1000)

                # Save result
                result = GiskardResult(
                    run_id=run.id,
                    question_id=question.id,
                    question_text=question.question,
                    expected_answer=question.expected_answer,
                    generated_answer=generated_answer,
                    question_type=question.question_type,
                    score=evaluation["metrics"].get("score"),
                    correct_refusal=evaluation["metrics"].get(
                        "correct_refusal"
                    ),
                    hallucinated=evaluation["metrics"].get("hallucinated"),
                    provided_answer=evaluation["metrics"].get(
                        "provided_answer"
                    ),
                    language=evaluation["metrics"].get("language"),
                    quality_score=evaluation["metrics"].get("quality_score"),
                    llm_provider=(
                        run.config.get("llm_provider")
                        if run.config else None
                    ),
                    llm_model=(
                        run.config.get("llm_model")
                        if run.config else None
                    ),
                    embedding_model=(
                        run.config.get("embedding_model")
                        if run.config else None
                    ),
                    latency_ms=latency_ms,
                    error_message=eval_result.get("error")
                )
                self.db.add(result)

                # Update progress
                run.processed_questions = i + 1
                self.db.commit()

            except Exception as e:
                logger.error(f"Error evaluating question {question.id}: {e}")
                # Save error result
                result = GiskardResult(
                    run_id=run.id,
                    question_id=question.id,
                    question_text=question.question,
                    expected_answer=question.expected_answer,
                    generated_answer="",
                    question_type=question.question_type,
                    score=0.0,
                    correct_refusal=False,
                    hallucinated=False,
                    provided_answer=False,
                    language="",
                    quality_score=0.0,
                    llm_provider=(
                        run.config.get("llm_provider")
                        if run.config else None
                    ),
                    llm_model=(
                        run.config.get("llm_model")
                        if run.config else None
                    ),
                    embedding_model=(
                        run.config.get("embedding_model")
                        if run.config else None
                    ),
                    latency_ms=0,
                    error_message=str(e)
                )
                self.db.add(result)
                run.processed_questions = i + 1
                self.db.commit()

    def _evaluate_response(
        self,
        question: GiskardQuestion,
        answer: str,
        question_type: str
    ) -> dict:
        """
        Evaluate a single response.

        Args:
            question: The question
            answer: The model's response
            question_type: Type ("relevant" or "irrelevant")

        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}

        if question_type == "irrelevant":
            # Check for "Bilmiyorum" or similar responses
            negative_responses = [
                "bilmiyorum", "bilgim yok", "bilgim bulunmuyor",
                "bu konuda bilgi veremem", "notlarda bu konu yok"
            ]
            answer_lower = answer.lower()

            has_negative_response = any(
                neg in answer_lower for neg in negative_responses
            )

            metrics["correct_refusal"] = has_negative_response
            metrics["hallucinated"] = not has_negative_response
            metrics["score"] = 1.0 if has_negative_response else 0.0
            metrics["provided_answer"] = False
        else:
            # For relevant questions, check if answer is provided
            metrics["correct_refusal"] = False
            metrics["hallucinated"] = False
            metrics["provided_answer"] = bool(answer and len(answer) > 20)
            metrics["score"] = 1.0 if metrics["provided_answer"] else 0.0

        # Check language
        metrics["language"] = self._check_language(answer)

        # Calculate quality score
        metrics["quality_score"] = self._calculate_quality_score(metrics)

        return {"metrics": metrics}

    def _check_language(self, text: str) -> str:
        """Check if text is Turkish, English, or mixed."""
        turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        text_has_turkish = any(char in text for char in turkish_chars)

        english_words = ["the", "and", "is", "of", "to", "in", "that"]
        text_has_english = any(
            word.lower() in text.lower() for word in english_words
        )

        if text_has_turkish and not text_has_english:
            return "Türkçe"
        elif text_has_english and not text_has_turkish:
            return "İngilizce"
        else:
            return "Karışık"

    def _calculate_quality_score(self, metrics: dict) -> float:
        """Calculate overall quality score from metrics."""
        score = 0.0

        if "score" in metrics:
            score += metrics["score"] * 0.7

        if metrics.get("language") == "Türkçe":
            score += 0.3

        return min(score, 1.0)

    def _save_summary(self, run: GiskardRun, results: dict):
        """
        Save evaluation summary to database.

        Args:
            run: Evaluation run
            results: Test results
        """
        metrics = results.get("metrics", {})
        relevant_metrics = metrics.get("relevant_questions", {})
        irrelevant_metrics = metrics.get("irrelevant_questions", {})

        summary = GiskardSummary(
            run_id=run.id,
            relevant_count=relevant_metrics.get("count", 0),
            relevant_avg_score=relevant_metrics.get("avg_score"),
            relevant_success_rate=relevant_metrics.get("success_rate"),
            irrelevant_count=irrelevant_metrics.get("count", 0),
            irrelevant_avg_score=irrelevant_metrics.get("avg_score"),
            irrelevant_success_rate=irrelevant_metrics.get("success_rate"),
            hallucination_rate=irrelevant_metrics.get(
                "hallucination_rate"
            ),
            correct_refusal_rate=irrelevant_metrics.get(
                "correct_refusal_rate"
            ),
            language_consistency=metrics.get("language_consistency"),
            turkish_response_rate=metrics.get("turkish_response_rate"),
            overall_score=results.get("overall_score"),
            total_questions=run.total_questions,
            successful_questions=metrics.get("total_evaluations", 0),
            failed_questions=0,
            avg_latency_ms=metrics.get("avg_latency_ms")
        )
        self.db.add(summary)
        self.db.commit()

    def run_quick_test(
        self,
        course_id: int,
        question: str,
        question_type: str,
        expected_answer: str,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Run a quick Giskard test on a single question.

        Args:
            course_id: Course ID
            question: Question text
            question_type: Type ("relevant" or "irrelevant")
            expected_answer: Expected answer
            llm_provider: Optional LLM provider override
            llm_model: Optional LLM model override
            system_prompt: Optional system prompt override

        Returns:
            Dictionary with test results
        """
        # Get course settings
        course_settings = get_or_create_settings(self.db, course_id)

        # Use override or course settings for LLM
        actual_provider = llm_provider or course_settings.llm_provider
        actual_model = llm_model or course_settings.llm_model

        start_time = time.time()

        try:
            # Create LLM service
            llm_service = LLMService(
                provider=actual_provider,
                model=actual_model,
                temperature=course_settings.llm_temperature,
                max_tokens=course_settings.llm_max_tokens
            )

            # Get embedding for question
            query_vector = self.embedding_service.get_embedding(
                question,
                model=course_settings.default_embedding_model
            )

            # Search for relevant chunks
            search_results = self.weaviate_service.hybrid_search(
                course_id=course_id,
                query=question,
                query_vector=query_vector,
                alpha=course_settings.search_alpha,
                limit=course_settings.search_top_k
            )

            # Filter by minimum relevance score
            min_score = (
                getattr(course_settings, 'min_relevance_score', 0.0)
                or 0.0
            )
            if min_score > 0 and search_results:
                search_results = [
                    r for r in search_results if r.score >= min_score
                ]

            # Extract contexts
            retrieved_contexts = []
            for result in search_results:
                if result.content:
                    retrieved_contexts.append(result.content)

            # Build context for LLM
            context_text = "\n\n---\n\n".join(
                    retrieved_contexts
                ) if retrieved_contexts else ""

            # Generate system prompt
            default_system_prompt = DEFAULT_SYSTEM_PROMPT

            actual_system_prompt = (
                system_prompt
                if system_prompt
                else default_system_prompt
            )

            # Generate answer
            user_prompt = f"""Aşağıda ders dokümanlarından alınan bağlam bilgileri verilmiştir.

Bağlam:
{context_text}

Soru: {question}

Yukarıdaki bağlam bilgilerini kullanarak soruyu yanıtla. Cevabında bağlamdaki teknik terimleri ve ifadeleri aynen kullan. Bağlamda olmayan bilgi ekleme."""

            generated_answer = llm_service.generate_response([
                {"role": "system", "content": actual_system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            latency_ms = int((time.time() - start_time) * 1000)

            # Evaluate response
            evaluation = self._evaluate_response(
                question,
                generated_answer,
                question_type
            )
            eval_metrics = evaluation["metrics"]

            return {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "question_type": question_type,
                "score": eval_metrics.get("score"),
                "correct_refusal": eval_metrics.get("correct_refusal"),
                "hallucinated": eval_metrics.get("hallucinated"),
                "provided_answer": eval_metrics.get("provided_answer"),
                "language": eval_metrics.get("language"),
                "quality_score": eval_metrics.get("quality_score"),
                "system_prompt_used": actual_system_prompt,
                "llm_provider_used": actual_provider,
                "llm_model_used": actual_model,
                "embedding_model_used": (
                    course_settings.default_embedding_model
                ),
                "latency_ms": latency_ms,
                "retrieved_contexts": (
                    retrieved_contexts
                    if retrieved_contexts else None
                ),
                "error_message": None
            }

        except Exception as e:
            logger.error(f"Quick test error: {e}")
            return {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": "",
                "question_type": question_type,
                "score": 0.0,
                "correct_refusal": False,
                "hallucinated": False,
                "provided_answer": False,
                "language": "",
                "quality_score": 0.0,
                "system_prompt_used": system_prompt or "",
                "llm_provider_used": actual_provider,
                "llm_model_used": actual_model,
                "embedding_model_used": (
                    course_settings.default_embedding_model
                ),
                "latency_ms": 0,
                "retrieved_contexts": None,
                "error_message": str(e)
            }


def get_giskard_integration_service(
    db: Session = Depends(get_db)
) -> GiskardIntegrationService:
    """
    Factory function to create a Giskard integration service.

    Args:
        db: Database session

    Returns:
        GiskardIntegrationService instance
    """
    return GiskardIntegrationService(db)
