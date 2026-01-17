"""RAGAS Evaluation API endpoints."""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)

from app.database import get_db
from app.models.db_models import (
    User, TestSet, TestQuestion, EvaluationRun,
    EvaluationResult, RunSummary, QuickTestResult
)
from app.models.schemas import (
    TestSetCreate, TestSetUpdate, TestSetResponse, TestSetDetailResponse,
    TestQuestionCreate, TestQuestionUpdate, TestQuestionResponse,
    TestSetImport, TestSetExport,
    EvaluationRunCreate, EvaluationRunResponse, EvaluationRunDetailResponse,
    EvaluationResultResponse, RunSummaryResponse,
    RunComparisonRequest, RunComparisonResponse,
    QuickTestRequest, QuickTestResponse,
    QuickTestResultCreate, QuickTestResultResponse, QuickTestResultListResponse,
)
from app.services.auth_service import get_current_user, get_current_teacher
from app.services.course_service import verify_course_access
from app.services.ragas_service import RagasEvaluationService

router = APIRouter(prefix="/api/ragas", tags=["ragas"])


# ==================== Test Set Endpoints ====================

@router.post("/test-sets", response_model=TestSetResponse, status_code=status.HTTP_201_CREATED)
async def create_test_set(
    data: TestSetCreate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Create a new test set for a course."""
    verify_course_access(db, data.course_id, current_user)
    
    test_set = TestSet(
        course_id=data.course_id,
        name=data.name,
        description=data.description,
        created_by=current_user.id,
    )
    db.add(test_set)
    db.commit()
    db.refresh(test_set)
    
    return TestSetResponse(
        id=test_set.id,
        course_id=test_set.course_id,
        name=test_set.name,
        description=test_set.description,
        created_by=test_set.created_by,
        created_at=test_set.created_at,
        updated_at=test_set.updated_at,
        question_count=0,
    )


@router.get("/test-sets", response_model=List[TestSetResponse])
async def list_test_sets(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all test sets for a course."""
    verify_course_access(db, course_id, current_user)
    
    test_sets = db.query(TestSet).filter(TestSet.course_id == course_id).all()
    
    result = []
    for ts in test_sets:
        question_count = db.query(func.count(TestQuestion.id)).filter(
            TestQuestion.test_set_id == ts.id
        ).scalar()
        
        result.append(TestSetResponse(
            id=ts.id,
            course_id=ts.course_id,
            name=ts.name,
            description=ts.description,
            created_by=ts.created_by,
            created_at=ts.created_at,
            updated_at=ts.updated_at,
            question_count=question_count or 0,
        ))
    
    return result


@router.get("/test-sets/{test_set_id}", response_model=TestSetDetailResponse)
async def get_test_set(
    test_set_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get test set details with questions."""
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    questions = db.query(TestQuestion).filter(
        TestQuestion.test_set_id == test_set_id
    ).all()
    
    return TestSetDetailResponse(
        id=test_set.id,
        course_id=test_set.course_id,
        name=test_set.name,
        description=test_set.description,
        created_by=test_set.created_by,
        created_at=test_set.created_at,
        updated_at=test_set.updated_at,
        question_count=len(questions),
        questions=[TestQuestionResponse(
            id=q.id,
            test_set_id=q.test_set_id,
            question=q.question,
            ground_truth=q.ground_truth,
            alternative_ground_truths=q.alternative_ground_truths,
            expected_contexts=q.expected_contexts,
            question_metadata=q.question_metadata,
            created_at=q.created_at,
        ) for q in questions],
    )


@router.put("/test-sets/{test_set_id}", response_model=TestSetResponse)
async def update_test_set(
    test_set_id: int,
    data: TestSetUpdate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Update a test set."""
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    if data.name is not None:
        test_set.name = data.name
    if data.description is not None:
        test_set.description = data.description
    
    test_set.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(test_set)
    
    question_count = db.query(func.count(TestQuestion.id)).filter(
        TestQuestion.test_set_id == test_set.id
    ).scalar()
    
    return TestSetResponse(
        id=test_set.id,
        course_id=test_set.course_id,
        name=test_set.name,
        description=test_set.description,
        created_by=test_set.created_by,
        created_at=test_set.created_at,
        updated_at=test_set.updated_at,
        question_count=question_count or 0,
    )


@router.delete("/test-sets/{test_set_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_test_set(
    test_set_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Delete a test set."""
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    db.delete(test_set)
    db.commit()
    return None


@router.post("/test-sets/{test_set_id}/duplicate", response_model=TestSetResponse)
async def duplicate_test_set(
    test_set_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Duplicate a test set with all its questions (but not evaluation runs)."""
    # Get original test set
    original = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not original:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, original.course_id, current_user)
    
    # Create duplicate test set
    duplicate = TestSet(
        course_id=original.course_id,
        name=f"{original.name} (Kopya)",
        description=original.description,
        created_by=current_user.id,
    )
    db.add(duplicate)
    db.commit()
    db.refresh(duplicate)
    
    # Copy all questions
    original_questions = db.query(TestQuestion).filter(
        TestQuestion.test_set_id == test_set_id
    ).all()
    
    for q in original_questions:
        new_question = TestQuestion(
            test_set_id=duplicate.id,
            question=q.question,
            ground_truth=q.ground_truth,
            alternative_ground_truths=q.alternative_ground_truths,
            expected_contexts=q.expected_contexts,
            question_metadata=q.question_metadata,
        )
        db.add(new_question)
    
    db.commit()
    
    question_count = len(original_questions)
    
    return TestSetResponse(
        id=duplicate.id,
        course_id=duplicate.course_id,
        name=duplicate.name,
        description=duplicate.description,
        created_by=duplicate.created_by,
        created_at=duplicate.created_at,
        updated_at=duplicate.updated_at,
        question_count=question_count,
    )


@router.post("/test-sets/{test_set_id}/import", response_model=TestSetDetailResponse)
async def import_questions(
    test_set_id: int,
    data: TestSetImport,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Import questions into a test set."""
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    for q_data in data.questions:
        question = TestQuestion(
            test_set_id=test_set_id,
            question=q_data.question,
            ground_truth=q_data.ground_truth,
            alternative_ground_truths=q_data.alternative_ground_truths,
            expected_contexts=q_data.expected_contexts,
            question_metadata=q_data.question_metadata,
        )
        db.add(question)
    
    db.commit()
    
    # Return updated test set
    return await get_test_set(test_set_id, current_user, db)


@router.get("/test-sets/{test_set_id}/export", response_model=TestSetExport)
async def export_test_set(
    test_set_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export test set to JSON format."""
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    questions = db.query(TestQuestion).filter(
        TestQuestion.test_set_id == test_set_id
    ).all()
    
    return TestSetExport(
        name=test_set.name,
        description=test_set.description,
        questions=[{
            "question": q.question,
            "ground_truth": q.ground_truth,
            "alternative_ground_truths": q.alternative_ground_truths,
            "expected_contexts": q.expected_contexts,
            "question_metadata": q.question_metadata,
        } for q in questions],
    )


# ==================== Question Endpoints ====================

@router.post("/test-sets/{test_set_id}/questions", response_model=TestQuestionResponse)
async def add_question(
    test_set_id: int,
    data: TestQuestionCreate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Add a question to a test set."""
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    question = TestQuestion(
        test_set_id=test_set_id,
        question=data.question,
        ground_truth=data.ground_truth,
        alternative_ground_truths=data.alternative_ground_truths,
        expected_contexts=data.expected_contexts,
        question_metadata=data.question_metadata,
    )
    db.add(question)
    db.commit()
    db.refresh(question)
    
    return TestQuestionResponse(
        id=question.id,
        test_set_id=question.test_set_id,
        question=question.question,
        ground_truth=question.ground_truth,
        alternative_ground_truths=question.alternative_ground_truths,
        expected_contexts=question.expected_contexts,
        question_metadata=question.question_metadata,
        created_at=question.created_at,
    )


@router.put("/questions/{question_id}", response_model=TestQuestionResponse)
async def update_question(
    question_id: int,
    data: TestQuestionUpdate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Update a question."""
    question = db.query(TestQuestion).filter(TestQuestion.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    test_set = db.query(TestSet).filter(TestSet.id == question.test_set_id).first()
    verify_course_access(db, test_set.course_id, current_user)
    
    if data.question is not None:
        question.question = data.question
    if data.ground_truth is not None:
        question.ground_truth = data.ground_truth
    if data.alternative_ground_truths is not None:
        question.alternative_ground_truths = data.alternative_ground_truths
    if data.expected_contexts is not None:
        question.expected_contexts = data.expected_contexts
    if data.question_metadata is not None:
        question.question_metadata = data.question_metadata
    
    db.commit()
    db.refresh(question)
    
    return TestQuestionResponse(
        id=question.id,
        test_set_id=question.test_set_id,
        question=question.question,
        ground_truth=question.ground_truth,
        alternative_ground_truths=question.alternative_ground_truths,
        expected_contexts=question.expected_contexts,
        question_metadata=question.question_metadata,
        created_at=question.created_at,
    )


@router.delete("/questions/{question_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_question(
    question_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Delete a question."""
    question = db.query(TestQuestion).filter(TestQuestion.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    test_set = db.query(TestSet).filter(TestSet.id == question.test_set_id).first()
    verify_course_access(db, test_set.course_id, current_user)
    
    db.delete(question)
    db.commit()
    return None



# ==================== Evaluation Endpoints ====================

@router.post("/evaluate", response_model=EvaluationRunResponse)
async def start_evaluation(
    data: EvaluationRunCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Start a new evaluation run or add to existing run."""
    # DEBUG: Log incoming request
    logger.info(f"[SELECTIVE EVAL DEBUG] Received evaluation request:")
    logger.info(f"[SELECTIVE EVAL DEBUG] - test_set_id: {data.test_set_id}")
    logger.info(f"[SELECTIVE EVAL DEBUG] - course_id: {data.course_id}")
    logger.info(f"[SELECTIVE EVAL DEBUG] - question_ids: {data.question_ids}")
    logger.info(f"[SELECTIVE EVAL DEBUG] - evaluation_model: {data.evaluation_model}")
    
    # Verify access
    verify_course_access(db, data.course_id, current_user)
    
    # Verify test set exists and belongs to course
    test_set = db.query(TestSet).filter(TestSet.id == data.test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    if test_set.course_id != data.course_id:
        raise HTTPException(status_code=400, detail="Test set does not belong to this course")
    
    # If question_ids provided, check for existing run for this test set
    existing_run = None
    if data.question_ids:
        # Find the most recent run for this test set (any status)
        existing_run = db.query(EvaluationRun).filter(
            EvaluationRun.test_set_id == data.test_set_id
        ).order_by(EvaluationRun.created_at.desc()).first()
        
        if existing_run:
            logger.info(f"[SELECTIVE EVAL DEBUG] Found existing run {existing_run.id} (status: {existing_run.status}) for test set")
            
            # Get already evaluated question IDs from this run
            evaluated_question_ids = db.query(EvaluationResult.question_id).filter(
                EvaluationResult.run_id == existing_run.id
            ).all()
            evaluated_question_ids = {q[0] for q in evaluated_question_ids}
            
            logger.info(f"[SELECTIVE EVAL DEBUG] Already evaluated questions: {evaluated_question_ids}")
            
            # Check which questions are already evaluated
            already_evaluated = [qid for qid in data.question_ids if qid in evaluated_question_ids]
            new_question_ids = [qid for qid in data.question_ids if qid not in evaluated_question_ids]
            
            # If some questions are already evaluated, delete their old results to re-evaluate
            if already_evaluated:
                logger.info(f"[SELECTIVE EVAL DEBUG] Re-evaluating questions: {already_evaluated}")
                db.query(EvaluationResult).filter(
                    EvaluationResult.run_id == existing_run.id,
                    EvaluationResult.question_id.in_(already_evaluated)
                ).delete(synchronize_session=False)
                
                # Update processed_questions count
                existing_run.processed_questions = max(0, existing_run.processed_questions - len(already_evaluated))
                
                db.commit()
                # Add them to new_question_ids for re-evaluation
                new_question_ids.extend(already_evaluated)
            
            logger.info(f"[SELECTIVE EVAL DEBUG] Questions to evaluate: {new_question_ids}")
            
            # Don't update config or total_questions - keep original test set intact
            # Just re-evaluate the specific questions
            
            # Set status to running if it was completed or failed
            if existing_run.status in ["completed", "failed"]:
                existing_run.status = "running"
                existing_run.error_message = None
            
            db.commit()
            db.refresh(existing_run)
            
            logger.info(f"[SELECTIVE EVAL DEBUG] Updated run {existing_run.id} status to: {existing_run.status}")
            
            # Start background evaluation for questions only
            ragas_service = RagasEvaluationService(db)
            background_tasks.add_task(
                ragas_service.run_evaluation_for_questions, 
                existing_run.id, 
                new_question_ids
            )
            
            return EvaluationRunResponse(
                id=existing_run.id,
                test_set_id=existing_run.test_set_id,
                course_id=existing_run.course_id,
                name=existing_run.name,
                status=existing_run.status,
                config=existing_run.config,
                total_questions=existing_run.total_questions,
                processed_questions=existing_run.processed_questions,
                started_at=existing_run.started_at,
                completed_at=existing_run.completed_at,
                error_message=existing_run.error_message,
                created_at=existing_run.created_at,
            )
    
    # No existing run or no question_ids - create new run
    question_query = db.query(func.count(TestQuestion.id)).filter(
        TestQuestion.test_set_id == data.test_set_id
    )
    if data.question_ids:
        logger.info(f"[SELECTIVE EVAL DEBUG] Filtering count by question_ids: {data.question_ids}")
        question_query = question_query.filter(TestQuestion.id.in_(data.question_ids))
    
    question_count = question_query.scalar()
    logger.info(f"[SELECTIVE EVAL DEBUG] Question count: {question_count}")
    
    if question_count == 0:
        raise HTTPException(status_code=400, detail="No questions to evaluate")
    
    # Build config with evaluation_model and question_ids if provided
    run_config = data.config.model_dump() if data.config else {}
    if data.evaluation_model:
        run_config["evaluation_model"] = data.evaluation_model
    if data.question_ids:
        run_config["question_ids"] = data.question_ids
    
    logger.info(f"[SELECTIVE EVAL DEBUG] Final run_config: {run_config}")
    
    # Create evaluation run
    run = EvaluationRun(
        test_set_id=data.test_set_id,
        course_id=data.course_id,
        name=data.name,
        status="pending",
        config=run_config if run_config else None,
        total_questions=question_count,
        processed_questions=0,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    
    logger.info(f"[SELECTIVE EVAL DEBUG] Created run {run.id} with config: {run.config}")
    
    # Start background evaluation
    ragas_service = RagasEvaluationService(db)
    background_tasks.add_task(ragas_service.run_evaluation, run.id)
    
    return EvaluationRunResponse(
        id=run.id,
        test_set_id=run.test_set_id,
        course_id=run.course_id,
        name=run.name,
        status=run.status,
        config=run.config,
        total_questions=run.total_questions,
        processed_questions=run.processed_questions,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error_message=run.error_message,
        created_at=run.created_at,
    )


@router.get("/runs", response_model=List[EvaluationRunResponse])
async def list_runs(
    course_id: int,
    test_set_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List evaluation runs for a course, optionally filtered by test set."""
    verify_course_access(db, course_id, current_user)
    
    query = db.query(EvaluationRun).filter(
        EvaluationRun.course_id == course_id
    )
    
    # Filter by test_set_id if provided
    if test_set_id is not None:
        query = query.filter(EvaluationRun.test_set_id == test_set_id)
    
    runs = query.order_by(EvaluationRun.created_at.desc()).all()
    
    # Get test set names for all runs
    test_set_ids = list(set(r.test_set_id for r in runs))
    test_sets = db.query(TestSet).filter(TestSet.id.in_(test_set_ids)).all()
    test_set_names = {ts.id: ts.name for ts in test_sets}
    
    # Get summaries for all runs
    run_ids = [r.id for r in runs]
    summaries = db.query(RunSummary).filter(RunSummary.run_id.in_(run_ids)).all()
    summary_map = {s.run_id: s for s in summaries}
    
    return [EvaluationRunResponse(
        id=r.id,
        test_set_id=r.test_set_id,
        test_set_name=test_set_names.get(r.test_set_id),
        course_id=r.course_id,
        name=r.name,
        status=r.status,
        config=r.config,
        total_questions=r.total_questions,
        processed_questions=r.processed_questions,
        started_at=r.started_at,
        completed_at=r.completed_at,
        error_message=r.error_message,
        created_at=r.created_at,
        avg_faithfulness=summary_map[r.id].avg_faithfulness if r.id in summary_map else None,
        avg_answer_relevancy=summary_map[r.id].avg_answer_relevancy if r.id in summary_map else None,
        avg_context_precision=summary_map[r.id].avg_context_precision if r.id in summary_map else None,
        avg_context_recall=summary_map[r.id].avg_context_recall if r.id in summary_map else None,
        avg_answer_correctness=summary_map[r.id].avg_answer_correctness if r.id in summary_map else None,
    ) for r in runs]


@router.get("/runs/{run_id}", response_model=EvaluationRunDetailResponse)
async def get_run(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get evaluation run details with results."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    verify_course_access(db, run.course_id, current_user)
    
    results = db.query(EvaluationResult).filter(
        EvaluationResult.run_id == run_id
    ).all()
    
    summary = db.query(RunSummary).filter(RunSummary.run_id == run_id).first()
    
    return EvaluationRunDetailResponse(
        id=run.id,
        test_set_id=run.test_set_id,
        course_id=run.course_id,
        name=run.name,
        status=run.status,
        config=run.config,
        total_questions=run.total_questions,
        processed_questions=run.processed_questions,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error_message=run.error_message,
        created_at=run.created_at,
        results=[EvaluationResultResponse(
            id=r.id,
            run_id=r.run_id,
            question_id=r.question_id,
            question_text=r.question_text,
            ground_truth_text=r.ground_truth_text,
            generated_answer=r.generated_answer,
            retrieved_contexts=r.retrieved_contexts,
            faithfulness=r.faithfulness,
            answer_relevancy=r.answer_relevancy,
            context_precision=r.context_precision,
            context_recall=r.context_recall,
            answer_correctness=r.answer_correctness,
            latency_ms=r.latency_ms,
            llm_provider=r.llm_provider,
            llm_model=r.llm_model,
            embedding_model=r.embedding_model,
            evaluation_model=r.evaluation_model,
            search_alpha=r.search_alpha,
            search_top_k=r.search_top_k,
            error_message=r.error_message,
            created_at=r.created_at,
        ) for r in results],
        summary=RunSummaryResponse(
            id=summary.id,
            run_id=summary.run_id,
            avg_faithfulness=summary.avg_faithfulness,
            avg_answer_relevancy=summary.avg_answer_relevancy,
            avg_context_precision=summary.avg_context_precision,
            avg_context_recall=summary.avg_context_recall,
            avg_answer_correctness=summary.avg_answer_correctness,
            avg_latency_ms=summary.avg_latency_ms,
            total_questions=summary.total_questions,
            successful_questions=summary.successful_questions,
            failed_questions=summary.failed_questions,
            created_at=summary.created_at,
        ) if summary else None,
    )


@router.get("/runs/{run_id}/status")
async def get_run_status(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get evaluation run status (for polling)."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    verify_course_access(db, run.course_id, current_user)
    
    return {
        "id": run.id,
        "status": run.status,
        "total_questions": run.total_questions,
        "processed_questions": run.processed_questions,
        "error_message": run.error_message,
    }


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_run(
    run_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Delete an evaluation run."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    verify_course_access(db, run.course_id, current_user)
    
    db.delete(run)
    db.commit()
    return None


@router.post("/compare", response_model=RunComparisonResponse)
async def compare_runs(
    data: RunComparisonRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Compare multiple evaluation runs."""
    runs = []
    summaries = []
    
    for run_id in data.run_ids:
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        
        verify_course_access(db, run.course_id, current_user)
        
        runs.append(EvaluationRunResponse(
            id=run.id,
            test_set_id=run.test_set_id,
            course_id=run.course_id,
            name=run.name,
            status=run.status,
            config=run.config,
            total_questions=run.total_questions,
            processed_questions=run.processed_questions,
            started_at=run.started_at,
            completed_at=run.completed_at,
            error_message=run.error_message,
            created_at=run.created_at,
        ))
        
        summary = db.query(RunSummary).filter(RunSummary.run_id == run_id).first()
        if summary:
            summaries.append(RunSummaryResponse(
                id=summary.id,
                run_id=summary.run_id,
                avg_faithfulness=summary.avg_faithfulness,
                avg_answer_relevancy=summary.avg_answer_relevancy,
                avg_context_precision=summary.avg_context_precision,
                avg_context_recall=summary.avg_context_recall,
                avg_answer_correctness=summary.avg_answer_correctness,
                avg_latency_ms=summary.avg_latency_ms,
                total_questions=summary.total_questions,
                successful_questions=summary.successful_questions,
                failed_questions=summary.failed_questions,
                created_at=summary.created_at,
            ))
    
    return RunComparisonResponse(runs=runs, summaries=summaries)


# ==================== RAGAS Settings Proxy ====================

@router.get("/settings")
async def get_ragas_settings(
    current_user: User = Depends(get_current_teacher),
):
    """Get RAGAS evaluation settings."""
    import httpx
    from app.config import get_settings
    settings = get_settings()
    ragas_url = getattr(settings, 'ragas_url', 'http://rag-ragas:8001')
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ragas_url}/settings")
            if response.status_code == 200:
                return response.json()
            raise HTTPException(status_code=response.status_code, detail="RAGAS service error")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"RAGAS service unavailable: {e}")


@router.post("/settings")
async def update_ragas_settings(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    current_user: User = Depends(get_current_teacher),
):
    """Update RAGAS evaluation settings."""
    import httpx
    from app.config import get_settings
    settings = get_settings()
    ragas_url = getattr(settings, 'ragas_url', 'http://rag-ragas:8001')
    
    payload = {}
    if provider is not None:
        payload["provider"] = provider
    if model is not None:
        payload["model"] = model
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{ragas_url}/settings", json=payload)
            if response.status_code == 200:
                return response.json()
            raise HTTPException(status_code=response.status_code, detail="RAGAS service error")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"RAGAS service unavailable: {e}")


@router.get("/providers")
async def get_ragas_providers(
    current_user: User = Depends(get_current_teacher),
):
    """Get available RAGAS LLM providers."""
    import httpx
    from app.config import get_settings
    settings = get_settings()
    ragas_url = getattr(settings, 'ragas_url', 'http://rag-ragas:8001')
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ragas_url}/providers")
            if response.status_code == 200:
                return response.json()
            raise HTTPException(status_code=response.status_code, detail="RAGAS service error")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"RAGAS service unavailable: {e}")


@router.post("/quick-test", response_model=QuickTestResponse)
async def quick_test(
    data: QuickTestRequest,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Run a quick RAGAS evaluation on a single question."""
    import time
    from app.services.course_service import get_or_create_settings
    from app.services.weaviate_service import WeaviateService
    from app.services.embedding_service import EmbeddingService
    from app.services.llm_service import LLMService
    
    # Verify course access
    verify_course_access(db, data.course_id, current_user)
    
    # Get course settings
    course_settings = get_or_create_settings(db, data.course_id)
    
    # Use provided system_prompt or get from course settings
    system_prompt = data.system_prompt
    if system_prompt is None:
        system_prompt = course_settings.system_prompt or \
            "Sen yardımcı bir asistansın. Verilen bağlama göre soruları yanıtla."
    
    # Use provided LLM settings or get from course settings
    llm_provider = data.llm_provider or course_settings.llm_provider
    llm_model = data.llm_model or course_settings.llm_model
    
    start_time = time.time()
    
    try:
        # Get course for collection name
        from app.models.db_models import Course
        course = db.query(Course).filter(Course.id == data.course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        # Get embedding for the question
        embedding_service = EmbeddingService()
        query_vector = embedding_service.get_embedding(
            data.question,
            model=course_settings.default_embedding_model
        )
        
        # Search for relevant chunks
        weaviate_service = WeaviateService()
        search_results = weaviate_service.hybrid_search(
            course_id=course.id,
            query=data.question,
            query_vector=query_vector,
            alpha=course_settings.search_alpha,
            limit=course_settings.search_top_k
        )
        
        # Filter results by minimum relevance score
        min_score = getattr(course_settings, 'min_relevance_score', 0.0) or 0.0
        if min_score > 0 and search_results:
            search_results = [r for r in search_results if r.score >= min_score]
        
        # Extract contexts with scores
        retrieved_contexts = []
        context_texts = []
        for result in search_results:
            content = result.content
            if content:
                retrieved_contexts.append({
                    "text": content,
                    "score": result.score
                })
                context_texts.append(content)
        
        # Build context for LLM
        context_text = "\n\n---\n\n".join(context_texts) if context_texts else ""
        
        # Generate answer using LLM
        user_prompt = f"""Bağlam:
{context_text}

Soru: {data.question}

Lütfen yukarıdaki bağlama dayanarak soruyu yanıtla."""
        
        llm_service = LLMService(
            provider=llm_provider,
            model=llm_model,
            temperature=course_settings.llm_temperature,
            max_tokens=course_settings.llm_max_tokens
        )
        generated_answer = llm_service.generate_response([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # Prepare ground truths (combine primary + alternatives)
        ground_truths = [data.ground_truth]
        if data.alternative_ground_truths:
            ground_truths.extend(data.alternative_ground_truths)
        
        # Call RAGAS service for metrics (RAGAS expects plain text contexts)
        ragas_service = RagasEvaluationService(db)
        
        # DEBUG: Log the llm_model being passed to RAGAS evaluation
        print(f"[RAGAS DEBUG] Quick test - llm_model for RAGAS evaluation: {llm_model}", flush=True)
        print(f"[RAGAS DEBUG] Quick test - data.llm_model: {data.llm_model}, course_settings.llm_model: {course_settings.llm_model}", flush=True)
        
        metrics = ragas_service._get_ragas_metrics_sync(
            data.question,
            ground_truths,
            generated_answer,
            context_texts,
            llm_model  # RAGAS değerlendirmesi için de aynı modeli kullan
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return QuickTestResponse(
            question=data.question,
            ground_truth=data.ground_truth,
            generated_answer=generated_answer,
            retrieved_contexts=retrieved_contexts,
            faithfulness=metrics.get("faithfulness"),
            answer_relevancy=metrics.get("answer_relevancy"),
            context_precision=metrics.get("context_precision"),
            context_recall=metrics.get("context_recall"),
            answer_correctness=metrics.get("answer_correctness"),
            latency_ms=latency_ms,
            system_prompt_used=system_prompt,
            llm_provider_used=llm_provider,
            llm_model_used=llm_model,
        )
        
    except Exception as e:
        logger.error(f"Quick test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Quick Test Results Endpoints ====================


@router.post(
    "/quick-test-results",
    response_model=QuickTestResultResponse,
    status_code=status.HTTP_201_CREATED
)
async def save_quick_test_result(
    data: QuickTestResultCreate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Save a quick test result for later viewing."""
    verify_course_access(db, data.course_id, current_user)
    
    # Convert Pydantic models to dicts for JSON serialization
    contexts_for_db = None
    if data.retrieved_contexts:
        contexts_for_db = [
            ctx.model_dump() if hasattr(ctx, 'model_dump') else ctx
            for ctx in data.retrieved_contexts
        ]
    
    result = QuickTestResult(
        course_id=data.course_id,
        group_name=data.group_name,
        question=data.question,
        ground_truth=data.ground_truth,
        alternative_ground_truths=data.alternative_ground_truths,
        system_prompt=data.system_prompt,
        llm_provider=data.llm_provider,
        llm_model=data.llm_model,
        generated_answer=data.generated_answer,
        retrieved_contexts=contexts_for_db,
        faithfulness=data.faithfulness,
        answer_relevancy=data.answer_relevancy,
        context_precision=data.context_precision,
        context_recall=data.context_recall,
        answer_correctness=data.answer_correctness,
        latency_ms=data.latency_ms,
        created_by=current_user.id,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    
    return QuickTestResultResponse(
        id=result.id,
        course_id=result.course_id,
        group_name=result.group_name,
        question=result.question,
        ground_truth=result.ground_truth,
        alternative_ground_truths=result.alternative_ground_truths,
        system_prompt=result.system_prompt,
        llm_provider=result.llm_provider,
        llm_model=result.llm_model,
        generated_answer=result.generated_answer,
        retrieved_contexts=result.retrieved_contexts,
        faithfulness=result.faithfulness,
        answer_relevancy=result.answer_relevancy,
        context_precision=result.context_precision,
        context_recall=result.context_recall,
        answer_correctness=result.answer_correctness,
        latency_ms=result.latency_ms,
        created_by=result.created_by,
        created_at=result.created_at,
    )


@router.get("/quick-test-results", response_model=QuickTestResultListResponse)
async def list_quick_test_results(
    course_id: int,
    group_name: Optional[str] = None,
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List saved quick test results for a course with pagination."""
    verify_course_access(db, course_id, current_user)
    
    query = db.query(QuickTestResult).filter(
        QuickTestResult.course_id == course_id
    )
    
    if group_name:
        query = query.filter(QuickTestResult.group_name == group_name)
    
    # Get total count before pagination
    total_count = query.count()
    
    # Apply pagination
    results = query.order_by(QuickTestResult.created_at.desc()).offset(skip).limit(limit).all()
    
    # Get unique group names
    groups_query = db.query(QuickTestResult.group_name).filter(
        QuickTestResult.course_id == course_id,
        QuickTestResult.group_name.isnot(None)
    ).distinct().all()
    groups = [g[0] for g in groups_query if g[0]]
    
    return QuickTestResultListResponse(
        results=[QuickTestResultResponse(
            id=r.id,
            course_id=r.course_id,
            group_name=r.group_name,
            question=r.question,
            ground_truth=r.ground_truth,
            alternative_ground_truths=r.alternative_ground_truths,
            system_prompt=r.system_prompt,
            llm_provider=r.llm_provider,
            llm_model=r.llm_model,
            generated_answer=r.generated_answer,
            retrieved_contexts=r.retrieved_contexts,
            faithfulness=r.faithfulness,
            answer_relevancy=r.answer_relevancy,
            context_precision=r.context_precision,
            context_recall=r.context_recall,
            answer_correctness=r.answer_correctness,
            latency_ms=r.latency_ms,
            created_by=r.created_by,
            created_at=r.created_at,
        ) for r in results],
        total=total_count,
        groups=groups,
    )


@router.get(
    "/quick-test-results/{result_id}",
    response_model=QuickTestResultResponse
)
async def get_quick_test_result(
    result_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a single quick test result."""
    result = db.query(QuickTestResult).filter(
        QuickTestResult.id == result_id
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    verify_course_access(db, result.course_id, current_user)
    
    return QuickTestResultResponse(
        id=result.id,
        course_id=result.course_id,
        group_name=result.group_name,
        question=result.question,
        ground_truth=result.ground_truth,
        alternative_ground_truths=result.alternative_ground_truths,
        system_prompt=result.system_prompt,
        llm_provider=result.llm_provider,
        llm_model=result.llm_model,
        generated_answer=result.generated_answer,
        retrieved_contexts=result.retrieved_contexts,
        faithfulness=result.faithfulness,
        answer_relevancy=result.answer_relevancy,
        context_precision=result.context_precision,
        context_recall=result.context_recall,
        answer_correctness=result.answer_correctness,
        latency_ms=result.latency_ms,
        created_by=result.created_by,
        created_at=result.created_at,
    )


@router.delete(
    "/quick-test-results/{result_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_quick_test_result(
    result_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Delete a quick test result."""
    result = db.query(QuickTestResult).filter(
        QuickTestResult.id == result_id
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    verify_course_access(db, result.course_id, current_user)
    
    db.delete(result)
    db.commit()
    return None


@router.post("/runs/{run_id}/fix-summary")
async def fix_run_summary(
    run_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Fix the summary for a run by recalculating based on actual results.
    
    This fixes issues where deleted questions are incorrectly counted as failed.
    Also removes results for questions that no longer exist in the test set.
    """
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    verify_course_access(db, run.course_id, current_user)
    
    # Get current question IDs in the test set
    current_question_ids = db.query(TestQuestion.id).filter(
        TestQuestion.test_set_id == run.test_set_id
    ).all()
    current_question_ids = {q[0] for q in current_question_ids}
    
    # Find and delete results for questions that no longer exist
    orphan_results = db.query(EvaluationResult).filter(
        EvaluationResult.run_id == run.id,
        ~EvaluationResult.question_id.in_(current_question_ids)
    ).all()
    
    deleted_count = len(orphan_results)
    if deleted_count > 0:
        for orphan in orphan_results:
            db.delete(orphan)
        db.commit()
        logger.info(
            f"Deleted {deleted_count} orphan results for run {run_id}"
        )
    
    # Count actual results (after cleanup)
    total_results = db.query(func.count(EvaluationResult.id)).filter(
        EvaluationResult.run_id == run.id
    ).scalar() or 0
    
    # Count successful results (no error)
    successful_count = db.query(func.count(EvaluationResult.id)).filter(
        EvaluationResult.run_id == run.id,
        EvaluationResult.error_message.is_(None)
    ).scalar() or 0
    
    # Count failed results (with error)
    failed_count = db.query(func.count(EvaluationResult.id)).filter(
        EvaluationResult.run_id == run.id,
        EvaluationResult.error_message.isnot(None)
    ).scalar() or 0
    
    # Update run
    run.total_questions = total_results
    run.processed_questions = total_results
    
    # If there are no actual failures, mark as completed
    if failed_count == 0 and run.status == "failed":
        run.status = "completed"
        run.error_message = None
    
    # Get successful results for metrics calculation
    successful_results = db.query(EvaluationResult).filter(
        EvaluationResult.run_id == run.id,
        EvaluationResult.error_message.is_(None)
    ).all()
    
    # Calculate metrics
    if successful_results:
        metrics_sum = {
            "faithfulness": sum(
                r.faithfulness for r in successful_results if r.faithfulness
            ),
            "answer_relevancy": sum(
                r.answer_relevancy for r in successful_results if r.answer_relevancy
            ),
            "context_precision": sum(
                r.context_precision for r in successful_results
                if r.context_precision
            ),
            "context_recall": sum(
                r.context_recall for r in successful_results if r.context_recall
            ),
            "answer_correctness": sum(
                r.answer_correctness for r in successful_results
                if r.answer_correctness
            ),
        }
        total_latency = sum(
            r.latency_ms for r in successful_results if r.latency_ms
        )
        
        # Update or create summary
        summary = db.query(RunSummary).filter(
            RunSummary.run_id == run.id
        ).first()
        
        if summary:
            if successful_count > 0:
                summary.avg_faithfulness = (
                    metrics_sum["faithfulness"] / successful_count
                )
                summary.avg_answer_relevancy = (
                    metrics_sum["answer_relevancy"] / successful_count
                )
                summary.avg_context_precision = (
                    metrics_sum["context_precision"] / successful_count
                )
                summary.avg_context_recall = (
                    metrics_sum["context_recall"] / successful_count
                )
                summary.avg_answer_correctness = (
                    metrics_sum["answer_correctness"] / successful_count
                )
                summary.avg_latency_ms = total_latency / successful_count
            summary.total_questions = total_results
            summary.successful_questions = successful_count
            summary.failed_questions = failed_count
        else:
            summary = RunSummary(
                run_id=run.id,
                avg_faithfulness=(
                    metrics_sum["faithfulness"] / successful_count
                    if successful_count > 0 else None
                ),
                avg_answer_relevancy=(
                    metrics_sum["answer_relevancy"] / successful_count
                    if successful_count > 0 else None
                ),
                avg_context_precision=(
                    metrics_sum["context_precision"] / successful_count
                    if successful_count > 0 else None
                ),
                avg_context_recall=(
                    metrics_sum["context_recall"] / successful_count
                    if successful_count > 0 else None
                ),
                avg_answer_correctness=(
                    metrics_sum["answer_correctness"] / successful_count
                    if successful_count > 0 else None
                ),
                avg_latency_ms=(
                    total_latency / successful_count
                    if successful_count > 0 else None
                ),
                total_questions=total_results,
                successful_questions=successful_count,
                failed_questions=failed_count,
            )
            db.add(summary)
    
    db.commit()
    
    return {
        "message": "Summary fixed successfully",
        "run_id": run.id,
        "status": run.status,
        "total_questions": total_results,
        "successful_questions": successful_count,
        "failed_questions": failed_count,
    }
