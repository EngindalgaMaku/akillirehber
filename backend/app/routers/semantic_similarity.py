"""Semantic Similarity Test API endpoints."""

import logging
import os
import time
import json
import asyncio
from typing import Optional, List
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models.db_models import (
    User, SemanticSimilarityResult, BatchTestSession, TestDataset
)
from app.models.schemas import (
    SemanticSimilarityQuickTestRequest,
    SemanticSimilarityQuickTestResponse,
    SemanticSimilarityBatchTestRequest,
    SemanticSimilarityBatchTestResponse,
    SemanticSimilarityResultCreate,
    SemanticSimilarityResultResponse,
    SemanticSimilarityResultListResponse,
    AggregateStatistics,
    BatchTestSessionResponse,
    BatchTestSessionListResponse,
    BatchTestSessionCreate,
    SemanticSimilarityTestCase,
    TestDatasetCreate,
)
from app.services.auth_service import get_current_user, get_current_teacher
from app.services.course_service import (
    verify_course_access,
    get_or_create_settings
)
from app.services.semantic_similarity_service import (
    SemanticSimilarityService
)

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/semantic-similarity",
    tags=["semantic-similarity"]
)


class WandbExportRequest(BaseModel):
    course_id: int
    group_name: str


class WandbRunUpdateRequest(BaseModel):
    run_id: str
    group_name: str
    course_id: int
    tags: Optional[List[str]] = None
    llm_model_used: Optional[str] = None
    embedding_model_used: Optional[str] = None
    llm_provider: Optional[str] = None
    total_tests: Optional[int] = None


# ==================== Quick Test Endpoint ====================

@router.post(
    "/quick-test",
    response_model=SemanticSimilarityQuickTestResponse
)
async def quick_test(
    data: SemanticSimilarityQuickTestRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run a quick semantic similarity test on a single question-answer pair.
    """
    # Verify course access
    verify_course_access(db, data.course_id, current_user)

    # Get course settings for embedding/LLM models
    course_settings = get_or_create_settings(db, data.course_id)

    start_time = time.time()

    try:
        # Initialize service
        service = SemanticSimilarityService(db)

        # Log incoming LLM parameters for debugging
        logger.info(
            "Quick test request - llm_provider: %s, llm_model: %s",
            data.llm_provider,
            data.llm_model
        )

        # Generate answer if not provided - use RAG pipeline
        generated_answer = data.generated_answer
        llm_model_used = None
        retrieved_contexts = []
        
        # Get system prompt from course settings
        default_system_prompt = (
            "Sen bir eğitim asistanısın. Verilen bağlam bilgilerini "
            "kullanarak öğrencilerin sorularını yanıtla. Yanıtlarını "
            "Türkçe ver."
        )
        system_prompt_used = (
            course_settings.system_prompt
            if course_settings.system_prompt
            else default_system_prompt
        )
        
        if not generated_answer:
            # Use the service's generate_answer method which includes RAG
            # (retrieves context from Weaviate and generates answer with LLM)
            generated_answer, retrieved_contexts, llm_model_used = (
                service.generate_answer(
                    course_id=data.course_id,
                    question=data.question,
                    llm_provider=data.llm_provider,
                    llm_model=data.llm_model
                )
            )
            logger.info(
                "Generated answer using RAG pipeline with %d contexts, "
                "LLM: %s",
                len(retrieved_contexts),
                llm_model_used
            )

        # Prepare reference answers
        reference_answers = [data.ground_truth]
        if data.alternative_ground_truths:
            reference_answers.extend(data.alternative_ground_truths)

        # Compute all metrics (cosine similarity, ROUGE, BERTScore, Hit@1, MRR)
        embedding_model = course_settings.default_embedding_model
        metrics = service.compute_all_metrics(
            generated_answer,
            reference_answers,
            embedding_model,
            retrieved_contexts=retrieved_contexts,
            lang="tr"  # Turkish language for BERTScore
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return SemanticSimilarityQuickTestResponse(
            question=data.question,
            ground_truth=data.ground_truth,
            generated_answer=generated_answer,
            similarity_score=metrics['similarity_score'],
            best_match_ground_truth=metrics['best_match_ground_truth'],
            all_scores=metrics['all_scores'],
            rouge1=metrics.get('rouge1'),
            rouge2=metrics.get('rouge2'),
            rougel=metrics.get('rougel'),
            bertscore_precision=metrics.get('bertscore_precision'),
            bertscore_recall=metrics.get('bertscore_recall'),
            bertscore_f1=metrics.get('bertscore_f1'),
            original_bertscore_precision=metrics.get(
                'original_bertscore_precision'
            ),
            original_bertscore_recall=metrics.get('original_bertscore_recall'),
            original_bertscore_f1=metrics.get('original_bertscore_f1'),
            hit_at_1=metrics.get('hit_at_1'),
            mrr=metrics.get('mrr'),
            embedding_model_used=embedding_model,
            latency_ms=latency_ms,
            llm_model_used=llm_model_used,
            retrieved_contexts=(
                retrieved_contexts if retrieved_contexts else None
            ),
            system_prompt_used=system_prompt_used,
        )

    except Exception as e:
        logger.error("Quick test error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== Batch Test Endpoint ====================

@router.post(
    "/batch-test",
    response_model=SemanticSimilarityBatchTestResponse
)
async def batch_test(
    data: SemanticSimilarityBatchTestRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run batch semantic similarity tests on multiple question-answer pairs.
    """
    # Verify course access
    verify_course_access(db, data.course_id, current_user)

    # Get course settings
    course_settings = get_or_create_settings(db, data.course_id)

    default_system_prompt = (
        "Sen bir eğitim asistanısın. Verilen bağlam bilgilerini "
        "kullanarak öğrencilerin sorularını yanıtla. Yanıtlarını "
        "Türkçe ver."
    )
    system_prompt_used = (
        course_settings.system_prompt
        if course_settings.system_prompt
        else default_system_prompt
    )

    start_time = time.time()

    try:
        # Initialize service
        service = SemanticSimilarityService(db)
        embedding_model = course_settings.default_embedding_model
        llm_model_used = None

        # Process each test case
        results = []
        total_similarity = 0.0
        min_similarity = 1.0
        max_similarity = 0.0
        successful_count = 0
        failed_count = 0

        for test_case in data.test_cases:
            try:
                # Generate answer if not provided - use RAG pipeline
                generated_answer = test_case.generated_answer
                retrieved_contexts = []
                
                if not generated_answer:
                    generated_answer, retrieved_contexts, llm_model_used = (
                        service.generate_answer(
                            course_id=data.course_id,
                            question=test_case.question,
                            llm_provider=data.llm_provider,
                            llm_model=data.llm_model
                        )
                    )
                
                # Prepare reference answers
                reference_answers = [test_case.ground_truth]
                if test_case.alternative_ground_truths:
                    reference_answers.extend(
                        test_case.alternative_ground_truths
                    )

                # Compute all metrics
                metrics = service.compute_all_metrics(
                    generated_answer,
                    reference_answers,
                    embedding_model,
                    retrieved_contexts=retrieved_contexts,
                    lang="tr"
                )

                # Update statistics
                total_similarity += metrics['similarity_score']
                min_similarity = min(
                    min_similarity, metrics['similarity_score']
                )
                max_similarity = max(
                    max_similarity, metrics['similarity_score']
                )
                successful_count += 1

                # Add result
                results.append({
                    "question": test_case.question,
                    "ground_truth": test_case.ground_truth,
                    "generated_answer": generated_answer,
                    "similarity_score": metrics['similarity_score'],
                    "best_match_ground_truth": metrics[
                        'best_match_ground_truth'
                    ],
                    "rouge1": metrics.get('rouge1'),
                    "rouge2": metrics.get('rouge2'),
                    "rougel": metrics.get('rougel'),
                    "bertscore_precision": metrics.get('bertscore_precision'),
                    "bertscore_recall": metrics.get('bertscore_recall'),
                    "bertscore_f1": metrics.get('bertscore_f1'),
                    "original_bertscore_precision": metrics.get(
                        'original_bertscore_precision'
                    ),
                    "original_bertscore_recall": metrics.get(
                        'original_bertscore_recall'
                    ),
                    "original_bertscore_f1": metrics.get(
                        'original_bertscore_f1'
                    ),
                    "hit_at_1": metrics.get('hit_at_1'),
                    "mrr": metrics.get('mrr'),
                    "latency_ms": 0,  # Individual latency not tracked in batch
                    "error_message": None,
                    "retrieved_contexts": (
                        retrieved_contexts if retrieved_contexts else None
                    ),
                    "system_prompt_used": system_prompt_used,
                })

            except Exception as e:
                # Handle individual test case failure gracefully
                logger.error("Batch test case error: %s", str(e))
                failed_count += 1
                results.append({
                    "question": test_case.question,
                    "ground_truth": test_case.ground_truth,
                    "generated_answer": test_case.generated_answer or "",
                    "similarity_score": None,
                    "best_match_ground_truth": None,
                    "latency_ms": 0,
                    "error_message": str(e),
                    "retrieved_contexts": None,
                    "system_prompt_used": system_prompt_used,
                })

        total_latency_ms = int((time.time() - start_time) * 1000)

        # Compute aggregate statistics
        avg_similarity = (
            total_similarity / successful_count
            if successful_count > 0 else None
        )

        return SemanticSimilarityBatchTestResponse(
            results=results,
            aggregate=AggregateStatistics(
                avg_similarity=avg_similarity,
                min_similarity=(
                    min_similarity if successful_count > 0 else None
                ),
                max_similarity=(
                    max_similarity if successful_count > 0 else None
                ),
                total_latency_ms=total_latency_ms,
                test_count=len(data.test_cases),
                successful_count=successful_count,
                failed_count=failed_count,
            ),
            embedding_model_used=embedding_model,
            llm_model_used=llm_model_used,
        )

    except Exception as e:
        logger.error("Batch test error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== Batch Test Streaming Endpoint ====================

@router.post("/batch-test-stream")
async def batch_test_stream(
    data: SemanticSimilarityBatchTestRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream batch semantic similarity test results as they complete."""
    verify_course_access(db, data.course_id, current_user)
    course_settings = get_or_create_settings(db, data.course_id)

    async def generate():
        try:
            service = SemanticSimilarityService(db)
            embedding_model = data.embedding_model or course_settings.default_embedding_model
            llm_model_used = None
            
            # Get system prompt to include in results
            default_system_prompt = (
                "Sen bir eğitim asistanısın. Verilen bağlam bilgilerini "
                "kullanarak öğrencilerin sorularını yanıtla. Yanıtlarını "
                "Türkçe ver."
            )
            system_prompt_used = (
                course_settings.system_prompt
                if course_settings.system_prompt
                else default_system_prompt
            )
            
            total_count = len(data.test_cases)
            completed_count = 0
            failed_count = 0
            MAX_RETRIES = 2  # Retry failed questions up to 2 times
            MAX_WORKERS = 1 if (embedding_model or "").startswith("ollama/") else 5  # Ollama is sensitive to concurrency

            # Process test cases in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all test cases to executor
                future_to_index = {
                    executor.submit(
                        _process_test_case,
                        idx,
                        test_case,
                        service,
                        data.course_id,
                        data.llm_provider,
                        data.llm_model,
                        embedding_model,
                        system_prompt_used,
                        MAX_RETRIES
                    ): idx
                    for idx, test_case in enumerate(data.test_cases)
                }

                # Collect results as they complete
                for future in as_completed(future_to_index.keys()):
                    result = future.result()
                    
                    if result["success"]:
                        completed_count += 1
                    else:
                        failed_count += 1
                        completed_count += 1  # Count as completed

                    # Send progress event
                    yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)

            # Send completion event
            completion = {
                "event": "complete",
                "total": total_count,
                "completed": completed_count,
                "embedding_model_used": embedding_model,
                "llm_model_used": llm_model_used
            }
            yield f"data: {json.dumps(completion, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)

        except Exception as e:
            logger.error("Stream error: %s", str(e))
            error = {"event": "error", "error": str(e)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


def _process_test_case(
    idx: int,
    test_case,
    service,
    course_id: int,
    llm_provider: str,
    llm_model: str,
    embedding_model: str,
    system_prompt_used: str,
    MAX_RETRIES: int
) -> dict:
    """Process a single test case with retry mechanism."""
    retry_count = 0
    last_error = None
    
    while retry_count <= MAX_RETRIES:
        try:
            # Start timing for this test case
            case_start_time = time.time()
            
            # Generate answer if not provided
            generated_answer = test_case.generated_answer
            retrieved_contexts = []
            
            if not generated_answer:
                generated_answer, retrieved_contexts, llm_model_used = (
                    service.generate_answer(
                        course_id=course_id,
                        question=test_case.question,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        embedding_model=embedding_model,
                    )
                )
            
            # Prepare reference answers
            reference_answers = [test_case.ground_truth]
            if test_case.alternative_ground_truths:
                reference_answers.extend(
                    test_case.alternative_ground_truths
                )

            # Compute all metrics
            metrics = service.compute_all_metrics(
                generated_answer,
                reference_answers,
                embedding_model,
                retrieved_contexts=retrieved_contexts,
                lang="tr"
            )

            # Calculate latency for this test case
            case_latency_ms = int(
                (time.time() - case_start_time) * 1000
            )

            # Send progress event
            result = {
                "event": "progress",
                "index": idx,
                "total": None,  # Will be set by caller
                "completed": None,  # Will be set by caller
                "result": {
                    "question": test_case.question,
                    "ground_truth": test_case.ground_truth,
                    "generated_answer": generated_answer,
                    "bloom_level": test_case.bloom_level if hasattr(test_case, 'bloom_level') else None,
                    "similarity_score": metrics['similarity_score'],
                    "best_match_ground_truth":
                        metrics['best_match_ground_truth'],
                    "rouge1": metrics.get('rouge1'),
                    "rouge2": metrics.get('rouge2'),
                    "rougel": metrics.get('rougel'),
                    "bertscore_precision":
                        metrics.get('bertscore_precision'),
                    "bertscore_recall":
                        metrics.get('bertscore_recall'),
                    "bertscore_f1": metrics.get('bertscore_f1'),
                    "original_bertscore_precision": metrics.get(
                        'original_bertscore_precision'
                    ),
                    "original_bertscore_recall": metrics.get(
                        'original_bertscore_recall'
                    ),
                    "original_bertscore_f1": metrics.get(
                        'original_bertscore_f1'
                    ),
                    "hit_at_1": metrics.get('hit_at_1'),
                    "mrr": metrics.get('mrr'),
                    "latency_ms": case_latency_ms,
                    "retrieved_contexts":
                        retrieved_contexts if retrieved_contexts else None,
                    "system_prompt_used": system_prompt_used,
                },
                "success": True
            }
            return result

        except Exception as e:
            last_error = str(e)
            retry_count += 1
            
            if retry_count <= MAX_RETRIES:
                # Log retry attempt
                logger.warning(
                    "Test case %d attempt %d failed: %s. Retrying...",
                    idx, retry_count, last_error
                )
                time.sleep(1)  # Wait 1 second before retry
            else:
                # All retries exhausted, send error result
                logger.error(
                    "Test case %d failed after %d attempts: %s.",
                    idx, MAX_RETRIES, last_error
                )
                
                # Send error result so frontend can display it
                error_result = {
                    "event": "progress",
                    "index": idx,
                    "total": None,
                    "completed": None,
                    "result": {
                        "question": test_case.question,
                        "ground_truth": test_case.ground_truth,
                        "generated_answer":
                            test_case.generated_answer or "N/A",
                        "similarity_score": None,
                        "best_match_ground_truth": None,
                        "rouge1": None,
                        "rouge2": None,
                        "rougel": None,
                        "bertscore_precision": None,
                        "bertscore_recall": None,
                        "bertscore_f1": None,
                        "original_bertscore_precision": None,
                        "original_bertscore_recall": None,
                        "original_bertscore_f1": None,
                        "hit_at_1": None,
                        "mrr": None,
                        "latency_ms": 0,
                        "retrieved_contexts": None,
                        "system_prompt_used": system_prompt_used,
                        "error_message": last_error,
                    },
                    "success": False
                }
                return error_result


# ==================== Results Endpoints ====================

@router.post(
    "/results",
    response_model=SemanticSimilarityResultResponse,
    status_code=status.HTTP_201_CREATED
)
async def save_result(
    data: SemanticSimilarityResultCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a semantic similarity test result for later viewing."""
    verify_course_access(db, data.course_id, current_user)

    # Convert ScoreDetail objects to dicts for JSON serialization
    all_scores_dicts = None
    if data.all_scores:
        all_scores_dicts = [
            score.model_dump() if hasattr(score, 'model_dump') else score
            for score in data.all_scores
        ]
    
    result = SemanticSimilarityResult(
        course_id=data.course_id,
        group_name=data.group_name,
        question=data.question,
        ground_truth=data.ground_truth,
        alternative_ground_truths=data.alternative_ground_truths,
        generated_answer=data.generated_answer,
        bloom_level=data.bloom_level,
        similarity_score=data.similarity_score,
        best_match_ground_truth=data.best_match_ground_truth,
        all_scores=all_scores_dicts,
        rouge1=data.rouge1,
        rouge2=data.rouge2,
        rougel=data.rougel,
        bertscore_precision=data.bertscore_precision,
        bertscore_recall=data.bertscore_recall,
        bertscore_f1=data.bertscore_f1,
        original_bertscore_precision=data.original_bertscore_precision,
        original_bertscore_recall=data.original_bertscore_recall,
        original_bertscore_f1=data.original_bertscore_f1,
        hit_at_1=data.hit_at_1,
        mrr=data.mrr,
        retrieved_contexts=data.retrieved_contexts,
        system_prompt_used=data.system_prompt_used,
        embedding_model_used=data.embedding_model_used,
        llm_model_used=data.llm_model_used,
        latency_ms=data.latency_ms,
        created_by=current_user.id,
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    return SemanticSimilarityResultResponse(
        id=result.id,
        course_id=result.course_id,
        group_name=result.group_name,
        question=result.question,
        ground_truth=result.ground_truth,
        alternative_ground_truths=result.alternative_ground_truths,
        generated_answer=result.generated_answer,
        bloom_level=result.bloom_level,
        similarity_score=result.similarity_score,
        best_match_ground_truth=result.best_match_ground_truth,
        all_scores=result.all_scores,
        rouge1=result.rouge1,
        rouge2=result.rouge2,
        rougel=result.rougel,
        bertscore_precision=result.bertscore_precision,
        bertscore_recall=result.bertscore_recall,
        bertscore_f1=result.bertscore_f1,
        original_bertscore_precision=result.original_bertscore_precision,
        original_bertscore_recall=result.original_bertscore_recall,
        original_bertscore_f1=result.original_bertscore_f1,
        hit_at_1=result.hit_at_1,
        mrr=result.mrr,
        retrieved_contexts=result.retrieved_contexts,
        system_prompt_used=result.system_prompt_used,
        embedding_model_used=result.embedding_model_used,
        llm_model_used=result.llm_model_used,
        latency_ms=result.latency_ms,
        created_by=result.created_by,
        created_at=result.created_at,
    )


@router.get(
    "/results",
    response_model=SemanticSimilarityResultListResponse
)
async def list_results(
    course_id: int,
    group_name: Optional[str] = None,
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List saved semantic similarity test results for a course with
    pagination.
    """
    verify_course_access(db, course_id, current_user)

    query = db.query(SemanticSimilarityResult).filter(
        SemanticSimilarityResult.course_id == course_id
    )

    if group_name is not None:
        query = query.filter(
            SemanticSimilarityResult.group_name == group_name
        )

    # Get total count before pagination
    total_count = query.count()

    # Apply pagination
    results = (
        query.order_by(SemanticSimilarityResult.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Get unique group names with their latest creation date,
    # sorted by newest first
    from sqlalchemy import func

    groups_query = db.query(
        SemanticSimilarityResult.group_name,
        func.max(SemanticSimilarityResult.created_at).label(
            'latest_created_at'
        )
    ).filter(
        SemanticSimilarityResult.course_id == course_id,
        SemanticSimilarityResult.group_name.isnot(None)
    ).group_by(
        SemanticSimilarityResult.group_name
    ).order_by(
        func.max(SemanticSimilarityResult.created_at).desc()
    ).all()

    # Return as list of dicts with group name and creation date
    groups = [
        {
            "name": g[0],
            "created_at": g[1].isoformat() if g[1] else None
        }
        for g in groups_query if g[0]
    ]

    # Calculate aggregate statistics for ALL results (not just current page)
    all_results_query = db.query(SemanticSimilarityResult).filter(
        SemanticSimilarityResult.course_id == course_id
    )
    if group_name is not None:
        all_results_query = all_results_query.filter(
            SemanticSimilarityResult.group_name == group_name
        )
    
    all_results = all_results_query.all()
    
    aggregate = None
    if all_results:
        # Calculate averages
        avg_similarity = (
            sum(r.similarity_score for r in all_results) /
            len(all_results)
        )
        
        # ROUGE metrics
        rouge1_results = [
            r.rouge1 for r in all_results if r.rouge1 is not None
        ]
        rouge2_results = [
            r.rouge2 for r in all_results if r.rouge2 is not None
        ]
        rougel_results = [
            r.rougel for r in all_results if r.rougel is not None
        ]

        # BERTScore metrics
        bert_p_results = [
            r.bertscore_precision for r in all_results
            if r.bertscore_precision is not None
        ]
        bert_r_results = [
            r.bertscore_recall for r in all_results
            if r.bertscore_recall is not None
        ]
        bert_f1_results = [
            r.bertscore_f1 for r in all_results
            if r.bertscore_f1 is not None
        ]

        orig_bert_p_results = [
            r.original_bertscore_precision for r in all_results
            if r.original_bertscore_precision is not None
        ]
        orig_bert_r_results = [
            r.original_bertscore_recall for r in all_results
            if r.original_bertscore_recall is not None
        ]
        orig_bert_f1_results = [
            r.original_bertscore_f1 for r in all_results
            if r.original_bertscore_f1 is not None
        ]

        # Retrieval metrics
        hit_at_1_results = [
            r.hit_at_1 for r in all_results if r.hit_at_1 is not None
        ]
        mrr_results = [r.mrr for r in all_results if r.mrr is not None]
        
        aggregate = {
            "avg_similarity": avg_similarity,
            "avg_rouge1": (
                sum(rouge1_results) / len(rouge1_results)
                if rouge1_results else None
            ),
            "avg_rouge2": (
                sum(rouge2_results) / len(rouge2_results)
                if rouge2_results else None
            ),
            "avg_rougel": (
                sum(rougel_results) / len(rougel_results)
                if rougel_results else None
            ),
            "avg_bertscore_precision": (
                sum(bert_p_results) / len(bert_p_results)
                if bert_p_results else None
            ),
            "avg_bertscore_recall": (
                sum(bert_r_results) / len(bert_r_results)
                if bert_r_results else None
            ),
            "avg_bertscore_f1": (
                sum(bert_f1_results) / len(bert_f1_results)
                if bert_f1_results else None
            ),
            "avg_original_bertscore_precision": (
                sum(orig_bert_p_results) / len(orig_bert_p_results)
                if orig_bert_p_results else None
            ),
            "avg_original_bertscore_recall": (
                sum(orig_bert_r_results) / len(orig_bert_r_results)
                if orig_bert_r_results else None
            ),
            "avg_original_bertscore_f1": (
                sum(orig_bert_f1_results) / len(orig_bert_f1_results)
                if orig_bert_f1_results else None
            ),
            "avg_hit_at_1": (
                sum(hit_at_1_results) / len(hit_at_1_results)
                if hit_at_1_results else None
            ),
            "avg_mrr": (
                sum(mrr_results) / len(mrr_results)
                if mrr_results else None
            ),
            "test_count": len(all_results),
        }

    return SemanticSimilarityResultListResponse(
        results=[SemanticSimilarityResultResponse(
            id=r.id,
            course_id=r.course_id,
            group_name=r.group_name,
            question=r.question,
            ground_truth=r.ground_truth,
            alternative_ground_truths=r.alternative_ground_truths,
            generated_answer=r.generated_answer,
            similarity_score=r.similarity_score,
            best_match_ground_truth=r.best_match_ground_truth,
            all_scores=r.all_scores,
            rouge1=r.rouge1,
            rouge2=r.rouge2,
            rougel=r.rougel,
            bertscore_precision=r.bertscore_precision,
            bertscore_recall=r.bertscore_recall,
            bertscore_f1=r.bertscore_f1,
            original_bertscore_precision=r.original_bertscore_precision,
            original_bertscore_recall=r.original_bertscore_recall,
            original_bertscore_f1=r.original_bertscore_f1,
            hit_at_1=r.hit_at_1,
            mrr=r.mrr,
            retrieved_contexts=r.retrieved_contexts,
            system_prompt_used=r.system_prompt_used,
            embedding_model_used=r.embedding_model_used,
            llm_model_used=r.llm_model_used,
            latency_ms=r.latency_ms,
            created_by=r.created_by,
            created_at=r.created_at,
        ) for r in results],
        total=total_count,
        groups=groups,
        aggregate=aggregate,
    )


@router.post("/wandb-export")
async def wandb_export_group(
    data: WandbExportRequest,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Export an existing saved semantic similarity group to Weights & Biases."""
    verify_course_access(db, data.course_id, current_user)

    wb_project = os.getenv("WANDB_PROJECT")
    wb_mode = os.getenv("WANDB_MODE")
    wb_has_key = bool(os.getenv("WANDB_API_KEY"))
    if (
        wandb is None
        or not wb_project
        or (not wb_has_key and wb_mode != "offline")
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "W&B is not configured. Set WANDB_PROJECT and WANDB_API_KEY "
                "(or WANDB_MODE=offline)."
            ),
        )

    # Load all results for this group
    results = (
        db.query(SemanticSimilarityResult)
        .filter(
            SemanticSimilarityResult.course_id == data.course_id,
            SemanticSimilarityResult.group_name == data.group_name,
        )
        .order_by(SemanticSimilarityResult.created_at.asc())
        .all()
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No results found for group",
        )

    # Compute aggregates
    valid_similarity = [
        r.similarity_score
        for r in results
        if r.similarity_score is not None
    ]
    rouge1_vals = [r.rouge1 for r in results if r.rouge1 is not None]
    rouge2_vals = [r.rouge2 for r in results if r.rouge2 is not None]
    rougel_vals = [r.rougel for r in results if r.rougel is not None]
    bert_f1_vals = [
        r.original_bertscore_f1
        for r in results
        if r.original_bertscore_f1 is not None
    ]
    hit_at_1_vals = [r.hit_at_1 for r in results if r.hit_at_1 is not None]
    mrr_vals = [r.mrr for r in results if r.mrr is not None]
    latency_vals = [r.latency_ms for r in results if r.latency_ms is not None]

    def _avg(vals):
        return (sum(vals) / len(vals)) if vals else None

    course_settings = get_or_create_settings(db, data.course_id)

    wb_entity = os.getenv("WANDB_ENTITY")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"semantic-similarity-{data.course_id}-{data.group_name}-{timestamp}"
    )
    wb_run = wandb.init(
        project=wb_project,
        entity=wb_entity,
        name=run_name,
        config={
            "course_id": data.course_id,
            "group_name": data.group_name,
            "embedding_model_used": results[0].embedding_model_used,
            "llm_model_used": results[0].llm_model_used,
            "reranker_used": results[0].reranker_used,
            "reranker_provider": results[0].reranker_provider,
            "reranker_model": results[0].reranker_model,
            "search_alpha": course_settings.search_alpha,
            "search_top_k": course_settings.search_top_k,
            "min_relevance_score": course_settings.min_relevance_score,
            "test_count": len(results),
        },
        tags=["semantic-similarity", "backfill"],
    )

    table = wandb.Table(
        columns=[
            "id",
            "question",
            "ground_truth",
            "generated_answer",
            "similarity_score",
            "rouge1",
            "rouge2",
            "rougel",
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
            "hit_at_1",
            "mrr",
            "latency_ms",
            "created_at",
        ]
    )

    for r in results:
        table.add_data(
            r.id,
            r.question,
            r.ground_truth,
            r.generated_answer,
            r.similarity_score,
            r.rouge1,
            r.rouge2,
            r.rougel,
            r.original_bertscore_precision,
            r.original_bertscore_recall,
            r.original_bertscore_f1,
            r.hit_at_1,
            r.mrr,
            r.latency_ms,
            r.created_at.isoformat() if r.created_at else None,
        )

    wandb.log(
        {
            "aggregate/avg_similarity": _avg(valid_similarity),
            "aggregate/avg_rouge1": _avg(rouge1_vals),
            "aggregate/avg_rouge2": _avg(rouge2_vals),
            "aggregate/avg_rougel": _avg(rougel_vals),
            "aggregate/avg_bertscore_f1": _avg(bert_f1_vals),
            "aggregate/avg_hit_at_1": _avg(hit_at_1_vals),
            "aggregate/avg_mrr": _avg(mrr_vals),
            "aggregate/avg_latency_ms": _avg(latency_vals),
            "test_count": len(results),
        }
    )
    wandb.log({"results": table})

    run_url = getattr(wb_run, "url", None)
    wb_run.finish()
    return {
        "success": True,
        "run_name": run_name,
        "run_url": run_url,
        "exported_count": len(results),
    }


@router.get("/wandb-runs")
async def list_wandb_runs(
    course_id: int,
    page: int = 1,
    limit: int = 25,  # Increased default limit
    search: Optional[str] = None,
    state: Optional[str] = None,
    tag: Optional[str] = None,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """List W&B runs for the project with server-side pagination and filtering."""
    verify_course_access(db, course_id, current_user)

    wb_project = os.getenv("WANDB_PROJECT")
    wb_entity = os.getenv("WANDB_ENTITY")
    wb_has_key = bool(os.getenv("WANDB_API_KEY"))
    if wandb is None or not wb_project or not wb_has_key:
        raise HTTPException(
            status_code=400,
            detail="W&B is not configured. Set WANDB_PROJECT and WANDB_API_KEY.",
        )

    try:
        api = wandb.Api()
        runs_path = f"{wb_entity}/{wb_project}" if wb_entity else wb_project
        
        # Apply filters at the API level for better performance
        filters = {}
        if state and state != "all":
            filters["state"] = state
            
        # Get runs with filters - this is much more efficient
        runs = api.runs(
            path=runs_path,
            filters=filters,
            per_page=limit * 5  # Get more items for filtering, but still limited
        )
        
        # Convert to list and apply remaining filters
        all_filtered_runs = []
        
        for run in runs:
            cfg = run.config or {}
            
            # Filter by course_id if it exists in the config
            if cfg.get("course_id") and cfg.get("course_id") != course_id:
                continue
                
            # Apply search filter
            if search:
                search_lower = search.lower()
                if (search_lower not in run.name.lower() and 
                    search_lower not in run.id.lower() and
                    (cfg.get("group_name") and search_lower not in cfg.get("group_name").lower())):
                    continue
            
            # Apply tag filter
            if tag:
                tags = getattr(run, 'tags', [])
                tag_lower = tag.lower()
                if not any(tag_lower in t.lower() for t in tags):
                    continue
            
            missing = []
            if not cfg.get("llm_model_used"):
                missing.append("llm_model_used")
            if not cfg.get("embedding_model_used"):
                # Only add to updated if we're actually setting it
                cfg["embedding_model_used"] = "openai/text-embedding-3-small"
                missing.append("embedding_model_used")
            if not cfg.get("llm_provider"):
                cfg["llm_provider"] = "zai"
                missing.append("llm_provider")
            if not cfg.get("total_tests"):
                cfg["total_tests"] = 50
                missing.append("total_tests")

            # Get tags from W&B run object (not from config)
            tags = getattr(run, 'tags', [])
            
            all_filtered_runs.append({
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "config": {**cfg, "tags": tags},
                "missing_fields": missing,
            })

        # Sort by created_at descending (newest first)
        all_filtered_runs.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        
        # Apply pagination
        total_items = len(all_filtered_runs)
        total_pages = (total_items + limit - 1) // limit
        
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_result = all_filtered_runs[start_idx:end_idx]
        
        # Return paginated result
        return {
            "runs": paginated_result,
            "pagination": {
                "currentPage": page,
                "totalPages": total_pages,
                "totalItems": total_items,
                "itemsPerPage": limit,
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching W&B runs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch W&B runs: {str(e)}"
        )


@router.post("/wandb-runs/update")
async def update_wandb_run(
    data: WandbRunUpdateRequest,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Update a W&B run's missing config fields using DB values for the group."""
    verify_course_access(db, data.course_id, current_user)

    wb_project = os.getenv("WANDB_PROJECT")
    wb_entity = os.getenv("WANDB_ENTITY")
    wb_has_key = bool(os.getenv("WANDB_API_KEY"))
    if wandb is None or not wb_project or not wb_has_key:
        raise HTTPException(
            status_code=400,
            detail="W&B is not configured. Set WANDB_PROJECT and WANDB_API_KEY.",
        )

    # Find a result in DB for this group to get model fields
    sample_result = (
        db.query(SemanticSimilarityResult)
        .filter(
            SemanticSimilarityResult.course_id == data.course_id,
            SemanticSimilarityResult.group_name == data.group_name,
        )
        .order_by(SemanticSimilarityResult.created_at.asc())
        .first()
    )
    
    # If no DB result found, use defaults for missing fields
    if not sample_result:
        # Use default values for runs that were logged directly to W&B
        api = wandb.Api()
        runs_path = f"{wb_entity}/{wb_project}" if wb_entity else wb_project
        try:
            run = api.run(f"{runs_path}/{data.run_id}")
        except Exception as e:
            raise HTTPException(status_code=404, detail="W&B run not found.") from e

        cfg = run.config or {}
        updated = []
        
        # Update tags if provided
        if data.tags is not None:
            cfg["tags"] = data.tags
            updated.append("tags")
        
        if not cfg.get("llm_model_used"):
            cfg["llm_model_used"] = "zai/glm-4.7"
            updated.append("llm_model_used")
        if not cfg.get("embedding_model_used"):
            # Only add to updated if we're actually setting it
            cfg["embedding_model_used"] = "openai/text-embedding-3-small"
            updated.append("embedding_model_used")
        if not cfg.get("llm_provider"):
            cfg["llm_provider"] = "zai"
            updated.append("llm_provider")
        if not cfg.get("total_tests"):
            cfg["total_tests"] = 50
            updated.append("total_tests")

        if not updated:
            return {"success": False, "message": "No fields needed update."}

        # Update run config using wandb.Api
        run.config.update(cfg)
        run.save()

        return {
            "success": True,
            "updated_fields": updated,
            "run_name": run.name,
        }

    api = wandb.Api()
    runs_path = f"{wb_entity}/{wb_project}" if wb_entity else wb_project
    try:
        run = api.run(f"{runs_path}/{data.run_id}")
    except Exception as e:
        raise HTTPException(status_code=404, detail="W&B run not found.") from e

    cfg = run.config or {}
    updated = []
    
    # Update tags if provided
    if data.tags is not None:
        cfg["tags"] = data.tags
        updated.append("tags")
    
    # Update model fields if provided in the request
    if data.llm_model_used is not None:
        cfg["llm_model_used"] = data.llm_model_used
        updated.append("llm_model_used")
    elif not cfg.get("llm_model_used") and sample_result.llm_model_used:
        # Only use DB sample if field is missing and not provided in request
        cfg["llm_model_used"] = sample_result.llm_model_used
        updated.append("llm_model_used")
        
    if data.embedding_model_used is not None:
        cfg["embedding_model_used"] = data.embedding_model_used
        updated.append("embedding_model_used")
    elif not cfg.get("embedding_model_used") and sample_result.embedding_model_used:
        cfg["embedding_model_used"] = sample_result.embedding_model_used
        updated.append("embedding_model_used")
    
    # Update other fields if provided
    if data.llm_provider is not None:
        cfg["llm_provider"] = data.llm_provider
        updated.append("llm_provider")
        
    if data.total_tests is not None:
        cfg["total_tests"] = data.total_tests
        updated.append("total_tests")

    if not updated:
        return {"success": False, "message": "No fields needed update."}

    # Update run config using wandb.Api proper methods
    run.config.update(cfg)
    run.save()

    return {
        "success": True,
        "updated_fields": updated,
        "run_name": run.name,
    }


@router.get(
    "/results/{result_id}",
    response_model=SemanticSimilarityResultResponse
)
async def get_result(
    result_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a single semantic similarity test result."""
    result = db.query(SemanticSimilarityResult).filter(
        SemanticSimilarityResult.id == result_id
    ).first()

    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    verify_course_access(db, result.course_id, current_user)

    return SemanticSimilarityResultResponse(
        id=result.id,
        course_id=result.course_id,
        group_name=result.group_name,
        question=result.question,
        ground_truth=result.ground_truth,
        alternative_ground_truths=result.alternative_ground_truths,
        generated_answer=result.generated_answer,
        similarity_score=result.similarity_score,
        best_match_ground_truth=result.best_match_ground_truth,
        all_scores=result.all_scores,
        rouge1=result.rouge1,
        rouge2=result.rouge2,
        rougel=result.rougel,
        bertscore_precision=result.bertscore_precision,
        bertscore_recall=result.bertscore_recall,
        bertscore_f1=result.bertscore_f1,
        original_bertscore_precision=result.original_bertscore_precision,
        original_bertscore_recall=result.original_bertscore_recall,
        original_bertscore_f1=result.original_bertscore_f1,
        retrieved_contexts=result.retrieved_contexts,
        system_prompt_used=result.system_prompt_used,
        embedding_model_used=result.embedding_model_used,
        llm_model_used=result.llm_model_used,
        latency_ms=result.latency_ms,
        created_by=result.created_by,
        created_at=result.created_at,
    )


@router.delete(
    "/results/{result_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_result(
    result_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Delete a semantic similarity test result."""
    result = db.query(SemanticSimilarityResult).filter(
        SemanticSimilarityResult.id == result_id
    ).first()

    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    verify_course_access(db, result.course_id, current_user)

    db.delete(result)
    db.commit()
    return None


@router.put("/groups/rename")
async def rename_group(
    course_id: int,
    old_group_name: str,
    new_group_name: str,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Rename all results in a group."""
    # Verify course access
    verify_course_access(db, course_id, current_user)

    # Check if old group exists
    results = db.query(SemanticSimilarityResult).filter(
        SemanticSimilarityResult.course_id == course_id,
        SemanticSimilarityResult.group_name == old_group_name
    ).all()

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Group not found"
        )

    # Update all results in the group
    for result in results:
        result.group_name = new_group_name

    db.commit()

    return {
        "success": True,
        "message": (
            f"Group renamed from '{old_group_name}' to '{new_group_name}'"
        ),
        "updated_count": len(results)
    }


@router.delete("/groups/{group_name}")
async def delete_group(
    course_id: int,
    group_name: str,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Delete all results in a group."""
    # Verify course access
    verify_course_access(db, course_id, current_user)

    # Check if group exists
    results = db.query(SemanticSimilarityResult).filter(
        SemanticSimilarityResult.course_id == course_id,
        SemanticSimilarityResult.group_name == group_name
    ).all()

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Group not found"
        )

    # Delete all results in the group
    for result in results:
        db.delete(result)

    db.commit()

    return {
        "success": True,
        "message": f"Group '{group_name}' deleted with {len(results)} results",
        "deleted_count": len(results)
    }


# ==================== Batch Test Session Endpoints ====================

@router.post(
    "/batch-test-sessions",
    response_model=BatchTestSessionResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_batch_test_session(
    data: BatchTestSessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new batch test session with auto-generated group name."""
    # Debug log to see what data is received
    logger.info(f"Received batch test session data: {data.model_dump()}")
    
    # Verify course access
    verify_course_access(db, data.course_id, current_user)

    # Get course settings for embedding model
    course_settings = get_or_create_settings(db, data.course_id)

    # Auto-generate unique group name with timestamp and random ID
    import uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID
    group_name = f"Batch Test {timestamp}_{unique_id}"

    logger.info(
        "Creating batch test session - course_id: %s, user_id: %s, "
        "test_cases: %d, group_name: %s",
        data.course_id, current_user.id, len(data.test_cases), group_name
    )

    # Check if a session with the same test cases already exists
    # (to prevent duplicate sessions)
    test_cases_json = json.dumps(
        [tc.model_dump() for tc in data.test_cases],
        ensure_ascii=False
    )

    existing_session = db.query(BatchTestSession).filter(
        BatchTestSession.course_id == data.course_id,
        BatchTestSession.user_id == current_user.id,
        BatchTestSession.test_cases == test_cases_json,
        BatchTestSession.status.in_(["in_progress", "failed"])
    ).first()

    if existing_session:
        logger.warning(
            "Session already exists - id: %s, group_name: %s, "
            "status: %s, current_index: %d",
            existing_session.id, existing_session.group_name,
            existing_session.status, existing_session.current_index
        )
        raise HTTPException(
            status_code=400,
            detail=f"An active session with these test cases already exists. "
            f"Please resume the existing session (ID: {existing_session.id}) "
            f"instead of creating a new one."
        )

    # Create batch test session
    session = BatchTestSession(
        course_id=data.course_id,
        user_id=current_user.id,
        group_name=group_name,
        test_cases=test_cases_json,
        total_tests=len(data.test_cases),
        completed_tests=0,
        failed_tests=0,
        current_index=0,
        status="in_progress",
        llm_provider=data.llm_provider,
        llm_model=data.llm_model,
        embedding_model_used=(data.embedding_model or course_settings.default_embedding_model),
        reranker_used=data.reranker_used,
        reranker_provider=data.reranker_provider,
        reranker_model=data.reranker_model,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    logger.info(
        "Batch test session created - id: %s, group_name: %s, "
        "total_tests: %d, current_index: %d, status: %s",
        session.id, session.group_name, session.total_tests,
        session.current_index, session.status
    )

    return BatchTestSessionResponse(
        id=session.id,
        course_id=session.course_id,
        user_id=session.user_id,
        group_name=session.group_name,
        test_cases=session.test_cases,
        total_tests=session.total_tests,
        completed_tests=session.completed_tests,
        failed_tests=session.failed_tests,
        current_index=session.current_index,
        status=session.status,
        llm_provider=session.llm_provider,
        llm_model=session.llm_model,
        embedding_model_used=session.embedding_model_used,
        reranker_used=session.reranker_used,
        reranker_provider=session.reranker_provider,
        reranker_model=session.reranker_model,
        started_at=session.started_at,
        completed_at=session.completed_at,
        updated_at=session.updated_at,
    )


@router.get(
    "/batch-test-sessions",
    response_model=BatchTestSessionListResponse
)
async def list_batch_test_sessions(
    course_id: int,
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List batch test sessions for a course."""
    # Verify course access
    verify_course_access(db, course_id, current_user)

    logger.info(
        "Listing batch test sessions - course_id: %s, user_id: %s",
        course_id, current_user.id
    )

    query = db.query(BatchTestSession).filter(
        BatchTestSession.course_id == course_id
    )

    # Get total count before pagination
    total_count = query.count()

    # Apply pagination
    sessions = (
        query.order_by(BatchTestSession.started_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    logger.info(
        "Found %d batch test sessions for course_id: %s",
        len(sessions), course_id
    )

    return BatchTestSessionListResponse(
        sessions=[
            BatchTestSessionResponse(
                id=s.id,
                course_id=s.course_id,
                user_id=s.user_id,
                group_name=s.group_name,
                test_cases=s.test_cases,
                total_tests=s.total_tests,
                completed_tests=s.completed_tests,
                failed_tests=s.failed_tests,
                current_index=s.current_index,
                status=s.status,
                llm_provider=s.llm_provider,
                llm_model=s.llm_model,
                embedding_model_used=s.embedding_model_used,
                started_at=s.started_at,
                completed_at=s.completed_at,
                updated_at=s.updated_at,
            )
            for s in sessions
        ],
        total=total_count,
    )


@router.get(
    "/batch-test-sessions/{session_id}",
    response_model=BatchTestSessionResponse
)
async def get_batch_test_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a specific batch test session."""
    session = db.query(BatchTestSession).filter(
        BatchTestSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify course access
    verify_course_access(db, session.course_id, current_user)

    return BatchTestSessionResponse(
        id=session.id,
        course_id=session.course_id,
        user_id=session.user_id,
        group_name=session.group_name,
        test_cases=session.test_cases,
        total_tests=session.total_tests,
        completed_tests=session.completed_tests,
        failed_tests=session.failed_tests,
        current_index=session.current_index,
        status=session.status,
        llm_provider=session.llm_provider,
        llm_model=session.llm_model,
        embedding_model_used=session.embedding_model_used,
        started_at=session.started_at,
        completed_at=session.completed_at,
        updated_at=session.updated_at,
    )


@router.post("/batch-test-sessions/{session_id}/resume")
async def resume_batch_test_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Resume a batch test session from where it left off."""
    # Get the session
    session = db.query(BatchTestSession).filter(
        BatchTestSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify course access
    verify_course_access(db, session.course_id, current_user)

    # Check if session can be resumed
    if session.status not in ["in_progress", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume session with status '{session.status}'"
        )

    logger.info(
        "Resuming batch test session - id: %s, group_name: %s, "
        "status: %s, current_index: %d, completed_tests: %d, "
        "total_tests: %d",
        session.id, session.group_name, session.status,
        session.current_index, session.completed_tests, session.total_tests
    )

    # Get course settings
    course_settings = get_or_create_settings(db, session.course_id)

    # Parse test cases from JSON
    test_cases = json.loads(session.test_cases)

    logger.info(
        "Test cases loaded - total: %d, starting from index: %d",
        len(test_cases), session.current_index
    )

    async def generate():
        try:
            service = SemanticSimilarityService(db)
            embedding_model = session.embedding_model_used
            llm_model_used = session.llm_model

            wb_run = None
            wb_table = None
            wb_enabled = False
            wb_project = os.getenv("WANDB_PROJECT") or "akilli-rehber"
            wb_mode = os.getenv("WANDB_MODE")
            wb_has_key = bool(os.getenv("WANDB_API_KEY"))
            if (
                wandb is not None
                and (wb_has_key or wb_mode == "offline")
            ):
                wb_enabled = True
                wb_entity = os.getenv("WANDB_ENTITY")
                wb_run = wandb.init(
                    project=wb_project,
                    entity=wb_entity,
                    name=f"semantic-similarity-session-{session.id}",
                    config={
                        "course_id": session.course_id,
                        "session_id": session.id,
                        "group_name": session.group_name,
                        "llm_provider": session.llm_provider,
                        "llm_model": session.llm_model,
                        "llm_model_used": session.llm_model,
                        "embedding_model_used": session.embedding_model_used,
                        "reranker_used": session.reranker_used,
                        "reranker_provider": session.reranker_provider,
                        "reranker_model": session.reranker_model,
                        "search_alpha": course_settings.search_alpha,
                        "search_top_k": course_settings.search_top_k,
                        "min_relevance_score": (
                            course_settings.min_relevance_score
                        ),
                        "total_tests": session.total_tests,
                    },
                    tags=["semantic-similarity", "batch"],
                )
                wb_table = wandb.Table(
                    columns=[
                        "index",
                        "question",
                        "ground_truth",
                        "generated_answer",
                        "similarity_score",
                        "rouge1",
                        "rouge2",
                        "rougel",
                        "bertscore_precision",
                        "bertscore_recall",
                        "bertscore_f1",
                        "original_bertscore_precision",
                        "original_bertscore_recall",
                        "original_bertscore_f1",
                        "hit_at_1",
                        "mrr",
                        "latency_ms",
                        "error_message",
                    ]
                )

            sum_similarity = 0.0
            cnt_similarity = 0
            sum_latency_ms = 0
            cnt_latency = 0
            sum_rouge1 = 0.0
            cnt_rouge1 = 0
            sum_rouge2 = 0.0
            cnt_rouge2 = 0
            sum_rougel = 0.0
            cnt_rougel = 0
            sum_bertscore_f1 = 0.0
            cnt_bertscore_f1 = 0
            sum_hit_at_1 = 0.0
            cnt_hit_at_1 = 0
            sum_mrr = 0.0
            cnt_mrr = 0

            # Get system prompt to include in results
            default_system_prompt = (
                "Sen bir eğitim asistanısın. Verilen bağlam bilgilerini "
                "kullanarak öğrencilerin sorularını yanıtla. Yanıtlarını "
                "Türkçe ver."
            )
            system_prompt_used = (
                course_settings.system_prompt
                if course_settings.system_prompt
                else default_system_prompt
            )

            # Start from current_index
            MAX_RETRIES = 2  # Retry failed questions up to 2 times
            
            for idx in range(session.current_index, len(test_cases)):
                # Refresh session from database before each iteration
                db.expire_all()
                fresh_session = db.query(BatchTestSession).filter(
                    BatchTestSession.id == session.id
                ).first()
                
                if not fresh_session:
                    logger.error(
                        "Session %d not found in database",
                        session.id
                    )
                    raise HTTPException(
                        status_code=404,
                        detail=f"Session {session.id} not found"
                    )
                
                test_case_data = test_cases[idx]
                test_case = SemanticSimilarityTestCase(**test_case_data)

                # Retry logic for failed test cases
                retry_count = 0
                last_error = None
                success = False
                
                while retry_count <= MAX_RETRIES and not success:
                    try:
                        # Start timing for this test case
                        case_start_time = time.time()

                        # Generate answer if not provided
                        generated_answer = test_case.generated_answer
                        retrieved_contexts = []
                        
                        if not generated_answer:
                            (
                                generated_answer,
                                retrieved_contexts,
                                llm_model_used,
                            ) = service.generate_answer(
                                course_id=fresh_session.course_id,
                                question=test_case.question,
                                llm_provider=fresh_session.llm_provider,
                                llm_model=fresh_session.llm_model,
                            )

                        # Prepare reference answers
                        reference_answers = [test_case.ground_truth]
                        if test_case.alternative_ground_truths:
                            reference_answers.extend(
                                test_case.alternative_ground_truths
                            )

                        # Compute all metrics
                        metrics = service.compute_all_metrics(
                            generated_answer,
                            reference_answers,
                            embedding_model,
                            retrieved_contexts=retrieved_contexts,
                            lang="tr"
                        )

                        # Calculate latency for this test case
                        case_latency_ms = int(
                            (time.time() - case_start_time) * 1000
                        )

                        if metrics.get("similarity_score") is not None:
                            sum_similarity += float(
                                metrics["similarity_score"]
                            )
                            cnt_similarity += 1
                        sum_latency_ms += int(case_latency_ms)
                        cnt_latency += 1
                        if metrics.get("rouge1") is not None:
                            sum_rouge1 += float(metrics["rouge1"])
                            cnt_rouge1 += 1
                        if metrics.get("rouge2") is not None:
                            sum_rouge2 += float(metrics["rouge2"])
                            cnt_rouge2 += 1
                        if metrics.get("rougel") is not None:
                            sum_rougel += float(metrics["rougel"])
                            cnt_rougel += 1
                        if metrics.get("bertscore_f1") is not None:
                            sum_bertscore_f1 += float(metrics["bertscore_f1"])
                            cnt_bertscore_f1 += 1
                        if metrics.get("hit_at_1") is not None:
                            sum_hit_at_1 += float(metrics["hit_at_1"])
                            cnt_hit_at_1 += 1
                        if metrics.get("mrr") is not None:
                            sum_mrr += float(metrics["mrr"])
                            cnt_mrr += 1

                        # Save result to database with group name
                        result = SemanticSimilarityResult(
                            course_id=fresh_session.course_id,
                            group_name=fresh_session.group_name,
                            batch_session_id=fresh_session.id,
                            question=test_case.question,
                            ground_truth=test_case.ground_truth,
                            alternative_ground_truths=(
                                test_case.alternative_ground_truths
                            ),
                            generated_answer=generated_answer,
                            bloom_level=test_case.bloom_level if hasattr(test_case, 'bloom_level') else None,
                            similarity_score=metrics['similarity_score'],
                            best_match_ground_truth=metrics[
                                'best_match_ground_truth'
                            ],
                            rouge1=metrics.get('rouge1'),
                            rouge2=metrics.get('rouge2'),
                            rougel=metrics.get('rougel'),
                            bertscore_precision=metrics.get(
                                'bertscore_precision'
                            ),
                            bertscore_recall=metrics.get('bertscore_recall'),
                            bertscore_f1=metrics.get('bertscore_f1'),
                            original_bertscore_precision=metrics.get(
                                'original_bertscore_precision'
                            ),
                            original_bertscore_recall=metrics.get(
                                'original_bertscore_recall'
                            ),
                            original_bertscore_f1=metrics.get(
                                'original_bertscore_f1'
                            ),
                            hit_at_1=metrics.get('hit_at_1'),
                            mrr=metrics.get('mrr'),
                            retrieved_contexts=retrieved_contexts,
                            system_prompt_used=system_prompt_used,
                            embedding_model_used=embedding_model,
                            llm_model_used=llm_model_used,
                            latency_ms=case_latency_ms,
                            created_by=current_user.id,
                        )
                        db.add(result)

                        # Update session progress using fresh_session
                        fresh_session.completed_tests += 1
                        fresh_session.current_index = idx + 1
                        fresh_session.updated_at = datetime.now(timezone.utc)
                        db.commit()
                        db.flush()  # Ensure DB changes are written

                        logger.info(
                            "Test case %d completed - session_id: %s, "
                            "current_index: %d, completed_tests: %d, "
                            "total_tests: %d",
                            idx, fresh_session.id, fresh_session.current_index,
                            fresh_session.completed_tests,
                            fresh_session.total_tests
                        )

                        # Send progress event
                        progress = {
                            "event": "progress",
                            "index": idx,
                            "total": fresh_session.total_tests,
                            "completed": fresh_session.completed_tests,
                            "result": {
                                "question": test_case.question,
                                "ground_truth": test_case.ground_truth,
                                "generated_answer": generated_answer,
                                "similarity_score": metrics[
                                    'similarity_score'
                                ],
                                "best_match_ground_truth": metrics[
                                    'best_match_ground_truth'
                                ],
                                "rouge1": metrics.get('rouge1'),
                                "rouge2": metrics.get('rouge2'),
                                "rougel": metrics.get('rougel'),
                                "bertscore_precision": metrics.get(
                                    'bertscore_precision'
                                ),
                                "bertscore_recall": metrics.get(
                                    'bertscore_recall'
                                ),
                                "bertscore_f1": metrics.get('bertscore_f1'),
                                "original_bertscore_precision": metrics.get(
                                    'original_bertscore_precision'
                                ),
                                "original_bertscore_recall": metrics.get(
                                    'original_bertscore_recall'
                                ),
                                "original_bertscore_f1": metrics.get(
                                    'original_bertscore_f1'
                                ),
                                "hit_at_1": metrics.get('hit_at_1'),
                                "mrr": metrics.get('mrr'),
                                "latency_ms": case_latency_ms,
                                "retrieved_contexts": (
                                    retrieved_contexts if retrieved_contexts
                                    else None
                                ),
                                "system_prompt_used": system_prompt_used,
                            }
                        }

                        if wb_enabled and wb_run is not None:
                            wandb.log(
                                {
                                    "similarity_score": metrics.get(
                                        "similarity_score"
                                    ),
                                    "rouge1": metrics.get("rouge1"),
                                    "rouge2": metrics.get("rouge2"),
                                    "rougel": metrics.get("rougel"),
                                    "bertscore_precision": metrics.get(
                                        "bertscore_precision"
                                    ),
                                    "bertscore_recall": metrics.get(
                                        "bertscore_recall"
                                    ),
                                    "bertscore_f1": metrics.get(
                                        "bertscore_f1"
                                    ),
                                    "original_bertscore_precision": (
                                        metrics.get(
                                            "original_bertscore_precision"
                                        )
                                    ),
                                    "original_bertscore_recall": metrics.get(
                                        "original_bertscore_recall"
                                    ),
                                    "original_bertscore_f1": metrics.get(
                                        "original_bertscore_f1"
                                    ),
                                    "hit_at_1": metrics.get("hit_at_1"),
                                    "mrr": metrics.get("mrr"),
                                    "latency_ms": case_latency_ms,
                                    "completed_tests": (
                                        fresh_session.completed_tests
                                    ),
                                    "failed_tests": fresh_session.failed_tests,
                                },
                                step=idx,
                            )
                            if wb_table is not None:
                                wb_table.add_data(
                                    idx,
                                    test_case.question,
                                    test_case.ground_truth,
                                    generated_answer,
                                    metrics.get("similarity_score"),
                                    metrics.get("rouge1"),
                                    metrics.get("rouge2"),
                                    metrics.get("rougel"),
                                    metrics.get("bertscore_precision"),
                                    metrics.get("bertscore_recall"),
                                    metrics.get("bertscore_f1"),
                                    metrics.get(
                                        "original_bertscore_precision"
                                    ),
                                    metrics.get("original_bertscore_recall"),
                                    metrics.get("original_bertscore_f1"),
                                    metrics.get("hit_at_1"),
                                    metrics.get("mrr"),
                                    case_latency_ms,
                                    None,
                                )
                        progress_json = json.dumps(
                            progress, ensure_ascii=False
                        )
                        yield f"data: {progress_json}\n\n"
                        await asyncio.sleep(0)

                        success = True  # Mark as successful

                    except Exception as e:
                        last_error = str(e)
                        retry_count += 1

                        if retry_count <= MAX_RETRIES:
                            # Log retry attempt
                            logger.warning(
                                "Test case %d attempt %d failed: %s. "
                                "Retrying...",
                                idx, retry_count, last_error
                            )
                            time.sleep(1)  # Wait 1 second before retry
                        else:
                            # All retries exhausted
                            logger.error(
                                "Test case %d failed after %d attempts: %s.",
                                idx, MAX_RETRIES, last_error
                            )

                            # Update session progress
                            fresh_session.failed_tests += 1
                            fresh_session.current_index = idx + 1
                            fresh_session.updated_at = datetime.now(
                                timezone.utc
                            )
                            db.commit()
                            db.flush()

                            logger.info(
                                "Test case %d failed - session_id: %s, "
                                "current_index: %d, failed_tests: %d, "
                                "error: %s",
                                idx, fresh_session.id,
                                fresh_session.current_index,
                                fresh_session.failed_tests, last_error
                            )

                            # Send error result
                            error_result = {
                                "event": "progress",
                                "index": idx,
                                "total": fresh_session.total_tests,
                                "completed": fresh_session.completed_tests,
                                "result": {
                                    "question": test_case.question,
                                    "ground_truth": test_case.ground_truth,
                                    "generated_answer": (
                                        test_case.generated_answer
                                        or "N/A"
                                    ),
                                    "similarity_score": None,
                                    "best_match_ground_truth": None,
                                    "rouge1": None,
                                    "rouge2": None,
                                    "rougel": None,
                                    "bertscore_precision": None,
                                    "bertscore_recall": None,
                                    "bertscore_f1": None,
                                    "hit_at_1": None,
                                    "mrr": None,
                                    "latency_ms": 0,
                                    "retrieved_contexts": None,
                                    "system_prompt_used": system_prompt_used,
                                    "error_message": last_error,
                                }
                            }

                            if wb_enabled and wb_run is not None:
                                wandb.log(
                                    {
                                        "error": 1,
                                        "completed_tests": (
                                            fresh_session.completed_tests
                                        ),
                                        "failed_tests": (
                                            fresh_session.failed_tests
                                        ),
                                    },
                                    step=idx,
                                )
                                if wb_table is not None:
                                    wb_table.add_data(
                                        idx,
                                        test_case.question,
                                        test_case.ground_truth,
                                        test_case.generated_answer or "N/A",
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        0,
                                        last_error,
                                    )
                            error_json = json.dumps(
                                error_result, ensure_ascii=False
                            )
                            yield f"data: {error_json}\n\n"
                            await asyncio.sleep(0)

            # Mark session as completed using fresh_session
            db.expire_all()
            fresh_session = db.query(BatchTestSession).filter(
                BatchTestSession.id == session.id
            ).first()

            if fresh_session:
                fresh_session.status = "completed"
                fresh_session.completed_at = datetime.now(
                    timezone.utc
                )
                fresh_session.updated_at = datetime.now(
                    timezone.utc
                )
                db.commit()
                db.flush()

                logger.info(
                    "Session completed - id: %s, group_name: %s, "
                    "total_tests: %d, completed_tests: %d, failed_tests: %d",
                    fresh_session.id, fresh_session.group_name,
                    fresh_session.total_tests,
                    fresh_session.completed_tests, fresh_session.failed_tests
                )

                # Send completion event
                completion = {
                    "event": "complete",
                    "total": fresh_session.total_tests,
                    "completed": fresh_session.completed_tests,
                    "failed": fresh_session.failed_tests,
                    "embedding_model_used": embedding_model,
                    "llm_model_used": llm_model_used
                }

                if wb_enabled and wb_run is not None:
                    wandb.log(
                        {
                            "aggregate/avg_similarity": (
                                (sum_similarity / cnt_similarity)
                                if cnt_similarity
                                else None
                            ),
                            "aggregate/avg_latency_ms": (
                                (sum_latency_ms / cnt_latency)
                                if cnt_latency
                                else None
                            ),
                            "aggregate/avg_rouge1": (
                                (sum_rouge1 / cnt_rouge1)
                                if cnt_rouge1
                                else None
                            ),
                            "aggregate/avg_rouge2": (
                                (sum_rouge2 / cnt_rouge2)
                                if cnt_rouge2
                                else None
                            ),
                            "aggregate/avg_rougel": (
                                (sum_rougel / cnt_rougel)
                                if cnt_rougel
                                else None
                            ),
                            "aggregate/avg_bertscore_f1": (
                                (sum_bertscore_f1 / cnt_bertscore_f1)
                                if cnt_bertscore_f1
                                else None
                            ),
                            "aggregate/avg_hit_at_1": (
                                (sum_hit_at_1 / cnt_hit_at_1)
                                if cnt_hit_at_1
                                else None
                            ),
                            "aggregate/avg_mrr": (
                                (sum_mrr / cnt_mrr) if cnt_mrr else None
                            ),
                            "total_tests": fresh_session.total_tests,
                            "completed_tests": fresh_session.completed_tests,
                            "failed_tests": fresh_session.failed_tests,
                        }
                    )
                    if wb_table is not None:
                        wandb.log({"results": wb_table})
                    wb_run.finish()

                yield f"data: {json.dumps(completion, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

        except Exception as e:
            logger.error("Resume stream error: %s", str(e))
            # Mark session as failed using fresh_session
            db.expire_all()
            fresh_session = db.query(BatchTestSession).filter(
                BatchTestSession.id == session.id
            ).first()

            if fresh_session:
                fresh_session.status = "failed"
                fresh_session.updated_at = datetime.now(
                    timezone.utc
                )
                db.commit()
                db.flush()
            
            error = {"event": "error", "error": str(e)}
            if "wb_enabled" in locals() and wb_enabled and wb_run is not None:
                try:
                    wandb.log({"fatal_error": str(e)})
                    wb_run.finish(exit_code=1)
                except Exception:
                    pass
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@router.delete(
    "/batch-test-sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def cancel_batch_test_session(
    session_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Cancel a batch test session."""
    session = db.query(BatchTestSession).filter(
        BatchTestSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify course access
    verify_course_access(db, session.course_id, current_user)

    # Mark session as cancelled
    session.status = "cancelled"
    session.completed_at = datetime.now(timezone.utc)
    session.updated_at = datetime.now(timezone.utc)
    db.commit()

    return None


@router.delete(
    "/batch-test-sessions/{session_id}/delete",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_batch_test_session(
    session_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Permanently delete a batch test session."""
    session = db.query(BatchTestSession).filter(
        BatchTestSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify course access
    verify_course_access(db, session.course_id, current_user)

    logger.info(
        "Deleting batch test session - id: %s, group_name: %s, "
        "course_id: %s",
        session.id, session.group_name, session.course_id
    )

    # Delete associated results first
    db.query(SemanticSimilarityResult).filter(
        SemanticSimilarityResult.batch_session_id == session_id
    ).delete(synchronize_session=False)

    # Delete the session
    db.delete(session)
    db.commit()

    logger.info(
        "Batch test session deleted - id: %s, group_name: %s",
        session.id, session.group_name
    )

    return None


# ==================== Test Dataset Management ====================

@router.post("/test-datasets")
async def save_test_dataset(
    data: TestDatasetCreate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Save a test dataset for batch testing."""
    # Verify course access
    verify_course_access(db, data.course_id, current_user)
    
    # Create dataset
    dataset = TestDataset(
        course_id=data.course_id,
        user_id=current_user.id,
        name=data.name,
        description=data.description,
        test_cases=json.dumps(data.test_cases),
        total_test_cases=len(data.test_cases)
    )
    
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    
    logger.info(
        "Test dataset saved - id: %s, name: %s, course_id: %s, test_cases: %d",
        dataset.id, dataset.name, data.course_id, len(data.test_cases)
    )
    
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "total_test_cases": dataset.total_test_cases,
        "created_at": dataset.created_at.isoformat()
    }


@router.get("/test-datasets")
async def get_test_datasets(
    course_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Get all test datasets for a course."""
    from app.models.db_models import TestDataset
    
    # Verify course access
    verify_course_access(db, course_id, current_user)
    
    datasets = db.query(TestDataset).filter(
        TestDataset.course_id == course_id
    ).order_by(TestDataset.created_at.desc()).all()
    
    return {
        "datasets": [
            {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "total_test_cases": dataset.total_test_cases,
                "created_at": dataset.created_at.isoformat(),
                "updated_at": dataset.updated_at.isoformat()
            }
            for dataset in datasets
        ]
    }


@router.get("/test-datasets/{dataset_id}")
async def get_test_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Get a specific test dataset."""
    from app.models.db_models import TestDataset
    
    dataset = db.query(TestDataset).filter(
        TestDataset.id == dataset_id
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Verify course access
    verify_course_access(db, dataset.course_id, current_user)
    
    try:
        test_cases = json.loads(dataset.test_cases)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid test cases data")
    
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "test_cases": test_cases,
        "total_test_cases": dataset.total_test_cases,
        "created_at": dataset.created_at.isoformat(),
        "updated_at": dataset.updated_at.isoformat()
    }


@router.delete("/test-datasets/{dataset_id}")
async def delete_test_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Delete a test dataset."""
    from app.models.db_models import TestDataset
    
    dataset = db.query(TestDataset).filter(
        TestDataset.id == dataset_id
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Verify course access
    verify_course_access(db, dataset.course_id, current_user)
    
    db.delete(dataset)
    db.commit()
    
    logger.info(
        "Test dataset deleted - id: %s, name: %s",
        dataset.id, dataset.name
    )
    
    return {"message": "Dataset deleted successfully"}
