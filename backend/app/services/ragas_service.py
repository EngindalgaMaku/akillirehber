"""RAGAS Evaluation Service for running RAG evaluations."""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.db_models import (
    EvaluationRun, EvaluationResult, RunSummary, 
    TestQuestion, Course
)
from app.services.course_service import get_or_create_settings, DEFAULT_SYSTEM_PROMPT
from app.services.weaviate_service import WeaviateService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


try:
    import wandb  # type: ignore

    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


def get_embedding_provider_from_model(model_name: str) -> str:
    """Extract embedding provider from model name.
    
    Examples:
        "voyage-3" -> "voyage"
        "openai/text-embedding-3-small" -> "openrouter"
        "text-embedding-3-small" -> "openai"
    """
    if not model_name:
        return "openrouter"
    
    model_lower = model_name.lower()
    
    if model_lower.startswith("voyage"):
        return "voyage"
    elif "/" in model_name:
        # Format: provider/model (e.g., openai/text-embedding-3-small via OpenRouter)
        return "openrouter"
    elif "text-embedding" in model_lower:
        return "openai"
    else:
        return "openrouter"


class RagasEvaluationService:
    """Service for running RAGAS evaluations."""
    
    def __init__(self, db: Session):
        self.db = db
        settings = get_settings()
        self.ragas_url = getattr(settings, 'ragas_url', 'http://rag-ragas:8001')
        self.weaviate_service = WeaviateService()
    
    def run_evaluation(self, run_id: int):
        """Run evaluation for all questions in a test set."""

        # Get run
        run = self.db.query(EvaluationRun).filter(
            EvaluationRun.id == run_id
        ).first()
        if not run:
            logger.error("Run %s not found", run_id)
            return

        # DEBUG: Log run config
        logger.info(f"[SELECTIVE EVAL DEBUG] Run {run_id} config: {run.config}")
        logger.info(f"[SELECTIVE EVAL DEBUG] Run {run_id} test_set_id: {run.test_set_id}")

        # Update status to running
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        self.db.commit()

        # Get course settings for RAG config BEFORE W&B init
        course_settings = get_or_create_settings(self.db, run.course_id)

        # Get questions (filtered by question_ids if provided in config)
        question_query = self.db.query(TestQuestion).filter(
            TestQuestion.test_set_id == run.test_set_id
        )
        
        # Filter by question_ids if provided
        if run.config and run.config.get("question_ids"):
            question_ids = run.config["question_ids"]
            logger.info("[SELECTIVE EVAL DEBUG] Filtering by question_ids: %s", question_ids)
            question_query = question_query.filter(TestQuestion.id.in_(question_ids))
        else:
            logger.info("[SELECTIVE EVAL DEBUG] No question_ids filter - evaluating all questions")
        
        questions = question_query.all()
        logger.info("[SELECTIVE EVAL DEBUG] Found %d questions to evaluate", len(questions))
        logger.info("[SELECTIVE EVAL DEBUG] Question IDs: %s", [q.id for q in questions])

        # Get course for collection name
        course = self.db.query(Course).filter(Course.id == run.course_id).first()

        wb_enabled = False
        wb_run = None
        wb_table = None

        try:
            wb_project = os.getenv("WANDB_PROJECT", "akilli-rehber")
            wb_mode = os.getenv("WANDB_MODE")
            wb_has_key = bool(os.getenv("WANDB_API_KEY"))
            if WANDB_AVAILABLE and (wb_has_key or wb_mode == "offline"):
                wb_entity = os.getenv("WANDB_ENTITY")
                run_name = f"ragas-run-{run.course_id}-{run.test_set_id}-{run.id}"
                wb_run = wandb.init(
                    project=wb_project,
                    entity=wb_entity,
                    name=run_name,
                    config={
                        "course_id": run.course_id,
                        "run_id": run.id,
                        "run_name": run.name,
                        "evaluation_provider": (
                            run.config.get("evaluation_provider") if run.config else None
                        ),
                        "evaluation_model": (
                            run.config.get("evaluation_model") if run.config else None
                        ),
                        "llm_provider": course_settings.llm_provider,
                        "llm_model": course_settings.llm_model,
                        "embedding_model": course_settings.default_embedding_model,
                        "system_prompt": course_settings.system_prompt,
                        "search_alpha": course_settings.search_alpha,
                        "search_top_k": course_settings.search_top_k,
                        "min_relevance_score": getattr(
                            course_settings, "min_relevance_score", None
                        ),
                        "reranker_enabled": getattr(
                            course_settings, "enable_reranker", False
                        ),
                        "reranker_provider": getattr(
                            course_settings, "reranker_provider", None
                        ),
                        "reranker_model": getattr(
                            course_settings, "reranker_model", None
                        ),
                        "total_questions": len(questions),
                    },
                    tags=["ragas", "evaluation"],
                )
                wb_enabled = True

                try:
                    # Semantic similarity gibi basit metric definitions - step parametresini kullan!
                    wandb.define_metric("question_index")
                    wandb.define_metric("faithfulness", step_metric="question_index")
                    wandb.define_metric("answer_relevancy", step_metric="question_index")
                    wandb.define_metric("context_precision", step_metric="question_index")
                    wandb.define_metric("context_recall", step_metric="question_index")
                    wandb.define_metric("answer_correctness", step_metric="question_index")
                    wandb.define_metric("latency_ms", step_metric="question_index")
                    wandb.define_metric("contexts_count", step_metric="question_index")
                except Exception as e:
                    logger.warning(f"W&B metric definition failed: {e}")

                wb_url = getattr(wb_run, "url", None)
                if wb_url:
                    cfg = dict(run.config or {})
                    cfg["wandb_run_url"] = wb_url
                    cfg["wandb_run_id"] = getattr(wb_run, "id", None)
                    run.config = cfg
                    self.db.commit()

                wb_table = wandb.Table(
                    columns=[
                        "result_id",
                        "question_id",
                        "question",
                        "ground_truth",
                        "generated_answer",
                        "retrieved_contexts",
                        "system_prompt_used",
                        "llm_provider",
                        "llm_model",
                        "embedding_model",
                        "evaluation_model",
                        "faithfulness",
                        "answer_relevancy",
                        "context_precision",
                        "context_recall",
                        "answer_correctness",
                        "latency_ms",
                        "error_message",
                    ]
                )
        except Exception as e:
            logger.warning("W&B init failed: %s", e)
            wb_enabled = False
            wb_run = None
            wb_table = None
        
        try:
            
            # Process each question
            successful = 0
            failed = 0
            total_latency = 0
            
            metrics_sum = {
                "faithfulness": 0,
                "answer_relevancy": 0,
                "context_precision": 0,
                "context_recall": 0,
                "answer_correctness": 0,
            }
            
            for i, question in enumerate(questions):
                try:
                    result = self._evaluate_question(
                        run, question, course, course_settings, run.config
                    )

                    # DEBUG: Log W&B status
                    logger.info(f"[W&B DEBUG] Question {i}: wb_enabled={wb_enabled}, wb_run={wb_run is not None}, result.faithfulness={result.faithfulness}")
                    
                    if wb_enabled and wb_run is not None:
                        try:
                            if wb_table is not None:
                                wb_table.add_data(
                                    result.id,
                                    question.id,
                                    question.question,
                                    question.ground_truth,
                                    result.generated_answer,
                                    result.retrieved_contexts,
                                    course_settings.system_prompt,
                                    result.llm_provider,
                                    result.llm_model,
                                    result.embedding_model,
                                    result.evaluation_model,
                                    result.faithfulness,
                                    result.answer_relevancy,
                                    result.context_precision,
                                    result.context_recall,
                                    result.answer_correctness,
                                    result.latency_ms,
                                    result.error_message,
                                )
                            # Semantic similarity gibi - question_index'i hem key hem step olarak gönder!
                            log_data = {
                                "question_index": i,
                                "faithfulness": result.faithfulness,
                                "answer_relevancy": result.answer_relevancy,
                                "context_precision": result.context_precision,
                                "context_recall": result.context_recall,
                                "answer_correctness": result.answer_correctness,
                                "latency_ms": result.latency_ms,
                                "contexts_count": len(result.retrieved_contexts) if result.retrieved_contexts else 0,
                            }
                            wb_run.log(log_data, step=i)
                            
                            # W&B'ye hemen gönder (buffer'ı flush et) - Semantic similarity gibi!
                            try:
                                if hasattr(wb_run, '_flush'):
                                    wb_run._flush()
                            except Exception:
                                pass
                        except Exception as e:
                            logger.warning("W&B log failed: %s", e)
                    
                    if result.error_message:
                        failed += 1
                    else:
                        successful += 1
                        total_latency += result.latency_ms or 0
                        
                        # Sum metrics
                        if result.faithfulness:
                            metrics_sum["faithfulness"] += result.faithfulness
                        if result.answer_relevancy:
                            metrics_sum["answer_relevancy"] += result.answer_relevancy
                        if result.context_precision:
                            metrics_sum["context_precision"] += result.context_precision
                        if result.context_recall:
                            metrics_sum["context_recall"] += result.context_recall
                        if result.answer_correctness:
                            metrics_sum["answer_correctness"] += result.answer_correctness
                    
                    # Update progress
                    run.processed_questions = i + 1
                    self.db.commit()
                    
                except Exception as e:
                    logger.error(f"Error evaluating question {question.id}: {e}")
                    failed += 1
                    
                    # Best-effort: still persist latency and config fields
                    alpha = (
                        course_settings.search_alpha
                        if course_settings.search_alpha is not None
                        else (run.config.get("search_alpha", 0.5) if run.config else 0.5)
                    )
                    top_k = (
                        course_settings.search_top_k
                        if course_settings.search_top_k is not None
                        else (run.config.get("top_k", 5) if run.config else 5)
                    )
                    evaluation_model = run.config.get("evaluation_model") if run.config else None

                    # Create error result
                    error_result = EvaluationResult(
                        run_id=run.id,
                        question_id=question.id,
                        question_text=question.question,
                        ground_truth_text=question.ground_truth,
                        error_message=str(e),
                        latency_ms=0,
                        llm_provider=course_settings.llm_provider,
                        llm_model=course_settings.llm_model,
                        embedding_model=course_settings.default_embedding_model,
                        evaluation_model=evaluation_model,
                        search_alpha=alpha,
                        search_top_k=top_k,
                    )
                    self.db.add(error_result)
                    
                    run.processed_questions = i + 1
                    self.db.commit()

                    if wb_enabled and wb_run is not None:
                        try:
                            if wb_table is not None:
                                wb_table.add_data(
                                    None,
                                    question.id,
                                    question.question,
                                    question.ground_truth,
                                    None,
                                    None,
                                    course_settings.system_prompt,
                                    course_settings.llm_provider,
                                    course_settings.llm_model,
                                    course_settings.default_embedding_model,
                                    evaluation_model,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    0,
                                    str(e),
                                )
                            wb_run.log(
                                {
                                    "progress/processed_questions": i + 1,
                                    "progress/total_questions": len(questions),
                                    "error": 1,
                                    "error_message": str(e),
                                },
                                step=i + 1,
                            )
                        except Exception as e:
                            logger.warning("W&B log failed: %s", e)
            
            # Create summary
            total = successful + failed
            summary = RunSummary(
                run_id=run.id,
                avg_faithfulness=metrics_sum["faithfulness"] / successful if successful > 0 else None,
                avg_answer_relevancy=metrics_sum["answer_relevancy"] / successful if successful > 0 else None,
                avg_context_precision=metrics_sum["context_precision"] / successful if successful > 0 else None,
                avg_context_recall=metrics_sum["context_recall"] / successful if successful > 0 else None,
                avg_answer_correctness=metrics_sum["answer_correctness"] / successful if successful > 0 else None,
                avg_latency_ms=total_latency / successful if successful > 0 else None,
                total_questions=total,
                successful_questions=successful,
                failed_questions=failed,
            )
            self.db.add(summary)
            
            # Update run status
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            self.db.commit()

            if wb_enabled and wb_run is not None:
                try:
                    wb_run.log(
                        {
                            "aggregate/successful_questions": successful,
                            "aggregate/failed_questions": failed,
                            "aggregate/total_questions": total,
                        }
                    )
                    if wb_table is not None:
                        wb_run.log({"results": wb_table})
                    wb_run.finish()
                except Exception as e:
                    logger.warning("W&B finish failed: %s", e)
            
            logger.info(f"Evaluation run {run_id} completed: {successful}/{total} successful")
            
        except Exception as e:
            logger.error(f"Evaluation run {run_id} failed: {e}")
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc)
            self.db.commit()

            if wb_enabled and wb_run is not None:
                try:
                    wb_run.log({"fatal_error": str(e)})
                    wb_run.finish(exit_code=1)
                except Exception:
                    pass
    
    def _evaluate_question(
        self,
        run: EvaluationRun,
        question: TestQuestion,
        course: Course,
        course_settings,
        config: Optional[dict]
    ) -> EvaluationResult:
        """Evaluate a single question."""

        start_time = time.time()

        # Get config values - prefer course settings, fallback to run config
        alpha = course_settings.search_alpha if course_settings.search_alpha is not None else (config.get("search_alpha", 0.5) if config else 0.5)
        top_k = course_settings.search_top_k if course_settings.search_top_k is not None else (config.get("top_k", 5) if config else 5)
        
        # Get evaluation model from config if provided
        evaluation_model = config.get("evaluation_model") if config else None
        
        # DEBUG: Log evaluation model source
        print(
            f"[RAGAS DEBUG] evaluate_with_ragas - input_data.evaluation_model: {evaluation_model}",
            flush=True
        )
        
        # If no evaluation model specified, use course LLM model as fallback
        if not evaluation_model:
            evaluation_model = f"{course_settings.llm_provider}/{course_settings.llm_model}"
            print(
                f"[RAGAS DEBUG] No evaluation_model in config, using course LLM: {evaluation_model}",
                flush=True
            )
        
        # Get embedding for the question
        from app.services.embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        query_vector = embedding_service.get_embedding(
            question.question,
            model=course_settings.default_embedding_model
        )
        
        # Search for relevant chunks
        search_results = self.weaviate_service.hybrid_search(
            course_id=course.id,
            query=question.question,
            query_vector=query_vector,
            alpha=alpha,
            limit=top_k
        )

        # Filter results by minimum relevance score
        min_score = getattr(course_settings, 'min_relevance_score', 0.0) or 0.0
        if min_score > 0 and search_results:
            search_results = [r for r in search_results if r.score >= min_score]

        # Extract contexts
        retrieved_contexts = []
        for result in search_results:
            content = result.content
            if content:
                retrieved_contexts.append(content)
        
        # Build context for LLM
        context_text = "\n\n---\n\n".join(retrieved_contexts) if retrieved_contexts else ""
        
        # Generate answer using LLM
        system_prompt = course_settings.system_prompt or DEFAULT_SYSTEM_PROMPT
        
        user_prompt = f"""Aşağıda ders dokümanlarından alınan bağlam bilgileri verilmiştir.

Bağlam:
{context_text}

Soru: {question.question}

Yukarıdaki bağlam bilgilerini kullanarak soruyu yanıtla. Cevabında bağlamdaki teknik terimleri ve ifadeleri aynen kullan. Bağlamda olmayan bilgi ekleme."""

        try:
            llm_service = LLMService(
                provider=course_settings.llm_provider,
                model=course_settings.llm_model,
                temperature=course_settings.llm_temperature,
                max_tokens=course_settings.llm_max_tokens
            )
            generated_answer = llm_service.generate_response([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
        except Exception as e:
            logger.error(f"LLM error: {e}")
            generated_answer = ""
        
        # Prepare ground truths (combine primary + alternatives)
        ground_truths = [question.ground_truth]
        if question.alternative_ground_truths:
            ground_truths.extend(question.alternative_ground_truths)
        
        # Call RAGAS service for metrics (sync version for BackgroundTask)
        metrics = self._get_ragas_metrics_sync(
            question.question,
            ground_truths,
            generated_answer,
            retrieved_contexts,
            evaluation_model,
            reranker_provider=course_settings.reranker_provider if course_settings.enable_reranker else None,
            reranker_model=course_settings.reranker_model if course_settings.enable_reranker else None,
            embedding_provider=get_embedding_provider_from_model(course_settings.default_embedding_model),
            embedding_model=course_settings.default_embedding_model
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Create result
        result = EvaluationResult(
            run_id=run.id,
            question_id=question.id,
            question_text=question.question,
            ground_truth_text=question.ground_truth,
            generated_answer=generated_answer,
            retrieved_contexts=retrieved_contexts,
            faithfulness=metrics.get("faithfulness"),
            answer_relevancy=metrics.get("answer_relevancy"),
            context_precision=metrics.get("context_precision"),
            context_recall=metrics.get("context_recall"),
            answer_correctness=metrics.get("answer_correctness"),
            latency_ms=latency_ms,
            error_message=metrics.get("error"),
            # Model information used for THIS question
            llm_provider=course_settings.llm_provider,
            llm_model=course_settings.llm_model,
            embedding_model=course_settings.default_embedding_model,
            evaluation_model=evaluation_model,
            search_alpha=alpha,
            search_top_k=top_k,
        )
        
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        
        return result
    
    def _get_ragas_metrics_sync(
        self,
        question: str,
        ground_truths: list,
        generated_answer: str,
        retrieved_contexts: list,
        evaluation_model: str = None,
        reranker_provider: str = None,
        reranker_model: str = None,
        embedding_provider: str = None,
        embedding_model: str = None
    ) -> dict:
        """Get RAGAS metrics from evaluation service (sync version with retry).
        
        Args:
            question: The question text
            ground_truths: List of acceptable ground truth answers
            generated_answer: The generated answer to evaluate
            retrieved_contexts: List of retrieved context chunks
            evaluation_model: Optional OpenRouter model to use for evaluation
            reranker_provider: Optional reranker provider used (cohere/alibaba)
            reranker_model: Optional reranker model used
            embedding_provider: Embedding provider to use for RAGAS metrics
            embedding_model: Embedding model to use for RAGAS metrics
        """
        # Prepare payload with ground_truths list if available
        payload = {
            "question": question,
            "ground_truth": ground_truths[0] if ground_truths else "",  # Primary (backward compat)
            "generated_answer": generated_answer,
            "retrieved_contexts": retrieved_contexts,
        }
        
        # Add embedding info for RAGAS answer_relevancy metric
        if embedding_provider:
            payload["embedding_provider"] = embedding_provider
        if embedding_model:
            payload["embedding_model"] = embedding_model
        
        # Send all ground truths if we have alternatives
        if ground_truths and len(ground_truths) > 1:
            payload["ground_truths"] = ground_truths  # RAGAS will use all for answer_correctness
            logger.info(
                f"[RAGAS ALT GT] Sending {len(ground_truths)} ground truths to RAGAS service"
            )
        
        # Add evaluation_model if provided - RAGAS service zaten modeli kullanıyor, mapping gereksiz!
        if evaluation_model:
            payload["evaluation_model"] = evaluation_model
            
            # DEBUG: Log the model being sent
            print(
                f"[RAGAS DEBUG] Sending evaluation_model to RAGAS service: {evaluation_model}",
                flush=True
            )
        
        # Add reranker metadata if provided
        if reranker_provider:
            payload["reranker_provider"] = reranker_provider
            payload["reranker_model"] = reranker_model
        
        # DEBUG: Log the payload being sent to RAGAS service
        print(
            f"[RAGAS DEBUG] Sending to RAGAS service - "
            f"evaluation_model param: {evaluation_model}, "
            f"reranker: {reranker_provider}/{reranker_model if reranker_model else 'none'}",
            flush=True
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=300.0) as client:  # Increased to 5 minutes for RAGAS
                    response = client.post(
                        f"{self.ragas_url}/evaluate",
                        json=payload,
                    )

                    if response.status_code == 200:
                        return response.json()
                    logger.error("RAGAS API error: %s", response.status_code)
                    return {"error": f"RAGAS API error: {response.status_code}"}

            except Exception as e:
                logger.error("Error calling RAGAS API (attempt %d/%d): %s",
                           attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                else:
                    return {"error": str(e)}

    def run_evaluation_for_questions(self, run_id: int, question_ids: list):
        """Run evaluation for specific questions only (for incremental evaluation).
        
        This is used when re-evaluating specific questions in an existing run.
        It preserves the original total_questions count and only updates the
        specific questions being re-evaluated.
        """
        
        # Get run
        run = self.db.query(EvaluationRun).filter(
            EvaluationRun.id == run_id
        ).first()
        if not run:
            logger.error("Run %s not found", run_id)
            return

        logger.info(f"[SELECTIVE EVAL DEBUG] Running evaluation for run {run_id}, questions: {question_ids}")

        # Store original total_questions - don't change it during re-evaluation
        original_total = run.total_questions

        # Update status to running if not already
        if run.status != "running":
            run.status = "running"
            if not run.started_at:
                run.started_at = datetime.now(timezone.utc)
            self.db.commit()
        
        try:
            # Get specific questions
            questions = self.db.query(TestQuestion).filter(
                TestQuestion.id.in_(question_ids)
            ).all()
            
            logger.info(f"[SELECTIVE EVAL DEBUG] Found {len(questions)} questions to evaluate")
            
            # Get course settings for RAG config
            course_settings = get_or_create_settings(self.db, run.course_id)
            
            # Get course for collection name
            course = self.db.query(Course).filter(Course.id == run.course_id).first()
            
            # Process each question
            new_successful = 0
            new_failed = 0
            
            for question in questions:
                try:
                    result = self._evaluate_question(
                        run, question, course, course_settings, run.config
                    )
                    
                    if result.error_message:
                        new_failed += 1
                    else:
                        new_successful += 1
                    
                except Exception as e:
                    logger.error(f"Error evaluating question {question.id}: {e}")
                    new_failed += 1
                    
                    # Create error result
                    error_result = EvaluationResult(
                        run_id=run.id,
                        question_id=question.id,
                        question_text=question.question,
                        ground_truth_text=question.ground_truth,
                        error_message=str(e),
                    )
                    self.db.add(error_result)
            
            self.db.commit()
            
            # Recalculate summary with all results
            # Get successful results (no error)
            successful_results = self.db.query(EvaluationResult).filter(
                EvaluationResult.run_id == run.id,
                EvaluationResult.error_message.is_(None)
            ).all()
            
            # Get failed results (with error)
            failed_count = self.db.query(func.count(EvaluationResult.id)).filter(
                EvaluationResult.run_id == run.id,
                EvaluationResult.error_message.isnot(None)
            ).scalar() or 0
            
            successful_count = len(successful_results)
            actual_total = successful_count + failed_count
            
            # Update processed_questions to actual results count
            run.processed_questions = actual_total
            
            # Keep total_questions as the original value or actual results, whichever is larger
            # This ensures we don't lose track of the original test set size
            run.total_questions = max(original_total, actual_total)
            
            if successful_results:
                metrics_sum = {
                    "faithfulness": sum(r.faithfulness for r in successful_results if r.faithfulness),
                    "answer_relevancy": sum(r.answer_relevancy for r in successful_results if r.answer_relevancy),
                    "context_precision": sum(r.context_precision for r in successful_results if r.context_precision),
                    "context_recall": sum(r.context_recall for r in successful_results if r.context_recall),
                    "answer_correctness": sum(r.answer_correctness for r in successful_results if r.answer_correctness),
                }
                total_latency = sum(r.latency_ms for r in successful_results if r.latency_ms)
                
                # Update or create summary
                summary = self.db.query(RunSummary).filter(RunSummary.run_id == run.id).first()
                if summary:
                    summary.avg_faithfulness = metrics_sum["faithfulness"] / successful_count if successful_count > 0 else None
                    summary.avg_answer_relevancy = metrics_sum["answer_relevancy"] / successful_count if successful_count > 0 else None
                    summary.avg_context_precision = metrics_sum["context_precision"] / successful_count if successful_count > 0 else None
                    summary.avg_context_recall = metrics_sum["context_recall"] / successful_count if successful_count > 0 else None
                    summary.avg_answer_correctness = metrics_sum["answer_correctness"] / successful_count if successful_count > 0 else None
                    summary.avg_latency_ms = total_latency / successful_count if successful_count > 0 else None
                    summary.total_questions = actual_total
                    summary.successful_questions = successful_count
                    summary.failed_questions = failed_count
                else:
                    summary = RunSummary(
                        run_id=run.id,
                        avg_faithfulness=metrics_sum["faithfulness"] / successful_count if successful_count > 0 else None,
                        avg_answer_relevancy=metrics_sum["answer_relevancy"] / successful_count if successful_count > 0 else None,
                        avg_context_precision=metrics_sum["context_precision"] / successful_count if successful_count > 0 else None,
                        avg_context_recall=metrics_sum["context_recall"] / successful_count if successful_count > 0 else None,
                        avg_answer_correctness=metrics_sum["answer_correctness"] / successful_count if successful_count > 0 else None,
                        avg_latency_ms=total_latency / successful_count if successful_count > 0 else None,
                        total_questions=actual_total,
                        successful_questions=successful_count,
                        failed_questions=failed_count,
                    )
                    self.db.add(summary)
            
            # Always mark as completed after re-evaluation finishes
            # The status should only be "failed" if there was an exception during processing
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            run.error_message = None  # Clear any previous error message
            
            self.db.commit()
            
            logger.info(f"Evaluation for run {run_id} completed: {new_successful}/{len(questions)} new questions successful, total: {successful_count}/{actual_total}")
            
        except Exception as e:
            logger.error(f"Evaluation run {run_id} failed: {e}")
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc)
            self.db.commit()

