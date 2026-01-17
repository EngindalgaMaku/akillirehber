"""Semantic Similarity Test API endpoints."""

import logging
import time
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import User, SemanticSimilarityResult
from app.models.schemas import (
    SemanticSimilarityQuickTestRequest,
    SemanticSimilarityQuickTestResponse,
    SemanticSimilarityBatchTestRequest,
    SemanticSimilarityBatchTestResponse,
    SemanticSimilarityResultCreate,
    SemanticSimilarityResultResponse,
    SemanticSimilarityResultListResponse,
    AggregateStatistics,
)
from app.services.auth_service import get_current_user, get_current_teacher
from app.services.course_service import (
    verify_course_access,
    get_or_create_settings
)
from app.services.semantic_similarity_service import (
    SemanticSimilarityService
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/semantic-similarity",
    tags=["semantic-similarity"]
)


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
            retrieved_contexts=retrieved_contexts,  # Pass retrieved contexts for Hit@1/MRR
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
            hit_at_1=metrics.get('hit_at_1'),
            mrr=metrics.get('mrr'),
            embedding_model_used=embedding_model,
            latency_ms=latency_ms,
            llm_model_used=llm_model_used,
            retrieved_contexts=retrieved_contexts if retrieved_contexts else None,
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
                    reference_answers.extend(test_case.alternative_ground_truths)

                # Compute all metrics
                metrics = service.compute_all_metrics(
                    generated_answer,
                    reference_answers,
                    embedding_model,
                    retrieved_contexts=retrieved_contexts,  # Pass retrieved contexts
                    lang="tr"
                )

                # Update statistics
                total_similarity += metrics['similarity_score']
                min_similarity = min(min_similarity, metrics['similarity_score'])
                max_similarity = max(max_similarity, metrics['similarity_score'])
                successful_count += 1

                # Add result
                results.append({
                    "question": test_case.question,
                    "ground_truth": test_case.ground_truth,
                    "generated_answer": generated_answer,
                    "similarity_score": metrics['similarity_score'],
                    "best_match_ground_truth": metrics['best_match_ground_truth'],
                    "rouge1": metrics.get('rouge1'),
                    "rouge2": metrics.get('rouge2'),
                    "rougel": metrics.get('rougel'),
                    "bertscore_precision": metrics.get('bertscore_precision'),
                    "bertscore_recall": metrics.get('bertscore_recall'),
                    "bertscore_f1": metrics.get('bertscore_f1'),
                    "latency_ms": 0,  # Individual latency not tracked in batch
                    "error_message": None,
                    "retrieved_contexts": retrieved_contexts if retrieved_contexts else None,
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
                })

        total_latency_ms = int((time.time() - start_time) * 1000)

        # Compute aggregate statistics
        avg_similarity = (
            total_similarity / successful_count
            if successful_count > 0
            else None
        )

        return SemanticSimilarityBatchTestResponse(
            results=results,
            aggregate=AggregateStatistics(
                avg_similarity=avg_similarity,
                min_similarity=min_similarity if successful_count > 0 else None,
                max_similarity=max_similarity if successful_count > 0 else None,
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
            embedding_model = course_settings.default_embedding_model
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

            for idx, test_case in enumerate(data.test_cases):
                try:
                    # Start timing for this test case
                    case_start_time = time.time()
                    
                    # Generate answer if not provided
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
                        reference_answers.extend(test_case.alternative_ground_truths)

                    # Compute all metrics
                    metrics = service.compute_all_metrics(
                        generated_answer,
                        reference_answers,
                        embedding_model,
                        retrieved_contexts=retrieved_contexts,  # Pass retrieved contexts
                        lang="tr"
                    )

                    # Calculate latency for this test case
                    case_latency_ms = int((time.time() - case_start_time) * 1000)

                    completed_count += 1

                    # Send progress event
                    result = {
                        "event": "progress",
                        "index": idx,
                        "total": total_count,
                        "completed": completed_count,
                        "result": {
                            "question": test_case.question,
                            "ground_truth": test_case.ground_truth,
                            "generated_answer": generated_answer,
                            "similarity_score": metrics['similarity_score'],
                            "best_match_ground_truth": metrics['best_match_ground_truth'],
                            "rouge1": metrics.get('rouge1'),
                            "rouge2": metrics.get('rouge2'),
                            "rougel": metrics.get('rougel'),
                            "bertscore_precision": metrics.get('bertscore_precision'),
                            "bertscore_recall": metrics.get('bertscore_recall'),
                            "bertscore_f1": metrics.get('bertscore_f1'),
                            "hit_at_1": metrics.get('hit_at_1'),
                            "mrr": metrics.get('mrr'),
                            "latency_ms": case_latency_ms,
                            "retrieved_contexts": retrieved_contexts if retrieved_contexts else None,
                            "system_prompt_used": system_prompt_used,
                        }
                    }
                    yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

                except Exception as e:
                    logger.error("Test case %d error: %s", idx, str(e))
                    error_result = {
                        "event": "error",
                        "index": idx,
                        "error": str(e)
                    }
                    yield f"data: {json.dumps(error_result, ensure_ascii=False)}\n\n"

            # Send completion event
            completion = {
                "event": "complete",
                "total": total_count,
                "completed": completed_count,
                "embedding_model_used": embedding_model,
                "llm_model_used": llm_model_used
            }
            yield f"data: {json.dumps(completion, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error("Stream error: %s", str(e))
            error = {"event": "error", "error": str(e)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


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
        similarity_score=data.similarity_score,
        best_match_ground_truth=data.best_match_ground_truth,
        all_scores=all_scores_dicts,
        rouge1=data.rouge1,
        rouge2=data.rouge2,
        rougel=data.rougel,
        bertscore_precision=data.bertscore_precision,
        bertscore_recall=data.bertscore_recall,
        bertscore_f1=data.bertscore_f1,
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
        similarity_score=result.similarity_score,
        best_match_ground_truth=result.best_match_ground_truth,
        all_scores=result.all_scores,
        rouge1=result.rouge1,
        rouge2=result.rouge2,
        rougel=result.rougel,
        bertscore_precision=result.bertscore_precision,
        bertscore_recall=result.bertscore_recall,
        bertscore_f1=result.bertscore_f1,
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

    if group_name:
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

    # Get unique group names
    groups_query = db.query(SemanticSimilarityResult.group_name).filter(
        SemanticSimilarityResult.course_id == course_id,
        SemanticSimilarityResult.group_name.isnot(None)
    ).distinct().all()
    groups = [g[0] for g in groups_query if g[0]]

    # Calculate aggregate statistics for ALL results (not just current page)
    all_results_query = db.query(SemanticSimilarityResult).filter(
        SemanticSimilarityResult.course_id == course_id
    )
    if group_name:
        all_results_query = all_results_query.filter(
            SemanticSimilarityResult.group_name == group_name
        )
    
    all_results = all_results_query.all()
    
    aggregate = None
    if all_results:
        # Calculate averages
        avg_similarity = sum(r.similarity_score for r in all_results) / len(all_results)
        
        # ROUGE metrics
        rouge1_results = [r.rouge1 for r in all_results if r.rouge1 is not None]
        rouge2_results = [r.rouge2 for r in all_results if r.rouge2 is not None]
        rougel_results = [r.rougel for r in all_results if r.rougel is not None]
        
        # BERTScore metrics
        bert_p_results = [r.bertscore_precision for r in all_results if r.bertscore_precision is not None]
        bert_r_results = [r.bertscore_recall for r in all_results if r.bertscore_recall is not None]
        bert_f1_results = [r.bertscore_f1 for r in all_results if r.bertscore_f1 is not None]
        
        # Retrieval metrics
        hit_at_1_results = [r.hit_at_1 for r in all_results if r.hit_at_1 is not None]
        mrr_results = [r.mrr for r in all_results if r.mrr is not None]
        
        aggregate = {
            "avg_similarity": avg_similarity,
            "avg_rouge1": sum(rouge1_results) / len(rouge1_results) if rouge1_results else None,
            "avg_rouge2": sum(rouge2_results) / len(rouge2_results) if rouge2_results else None,
            "avg_rougel": sum(rougel_results) / len(rougel_results) if rougel_results else None,
            "avg_bertscore_precision": sum(bert_p_results) / len(bert_p_results) if bert_p_results else None,
            "avg_bertscore_recall": sum(bert_r_results) / len(bert_r_results) if bert_r_results else None,
            "avg_bertscore_f1": sum(bert_f1_results) / len(bert_f1_results) if bert_f1_results else None,
            "avg_hit_at_1": sum(hit_at_1_results) / len(hit_at_1_results) if hit_at_1_results else None,
            "avg_mrr": sum(mrr_results) / len(mrr_results) if mrr_results else None,
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
