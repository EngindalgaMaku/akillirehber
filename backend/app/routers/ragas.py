"""RAGAS Evaluation API endpoints."""

import asyncio
import json
import logging
import os
import uuid
from typing import List, Optional, Dict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from pydantic import BaseModel

from app.database import get_db, SessionLocal
from app.models.db_models import (
    User, TestSet, TestQuestion, EvaluationRun,
    EvaluationResult, RunSummary, QuickTestResult
)
from app.models.schemas import (
    TestSetCreate, TestSetUpdate, TestSetResponse,
    TestSetDetailResponse,
    TestQuestionCreate, TestQuestionUpdate, TestQuestionResponse,
    EvaluationRunCreate, EvaluationRunResponse, EvaluationRunDetailResponse,
    EvaluationResultResponse, RunSummaryResponse,
    RunComparisonRequest, RunComparisonResponse,
    QuickTestRequest, QuickTestResponse,
    QuickTestResultCreate, QuickTestResultResponse, QuickTestResultListResponse,
)
from app.services.auth_service import get_current_user, get_current_teacher
from app.services.course_service import (
    verify_course_access,
    get_or_create_settings,
    DEFAULT_SYSTEM_PROMPT,
)
from app.services.ragas_service import RagasEvaluationService, get_embedding_provider_from_model

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ragas", tags=["ragas"])


# ==================== Cancellation Manager ====================
# Global dictionary to track active batch tests and their cancellation/pause flags
_active_batch_tests: Dict[str, Dict] = {}

def create_batch_test_id() -> str:
    """Generate a unique batch test ID."""
    return str(uuid.uuid4())

def register_batch_test(test_id: str) -> None:
    """Register a new batch test with cancellation and pause flags."""
    _active_batch_tests[test_id] = {
        "cancelled": False,
        "paused": False,
        "start_time": datetime.now(timezone.utc)
    }
    logger.info(f"Batch test registered: {test_id}")

def is_batch_test_cancelled(test_id: str) -> bool:
    """Check if a batch test has been cancelled."""
    return _active_batch_tests.get(test_id, {}).get("cancelled", False)

def is_batch_test_paused(test_id: str) -> bool:
    """Check if a batch test is paused."""
    return _active_batch_tests.get(test_id, {}).get("paused", False)

def cancel_batch_test(test_id: str) -> bool:
    """Cancel a running batch test."""
    if test_id in _active_batch_tests:
        _active_batch_tests[test_id]["cancelled"] = True
        logger.info(f"Batch test cancelled: {test_id}")
        return True
    # Even if not found locally, register it as cancelled
    # (handles multi-worker scenarios where test runs on different worker)
    _active_batch_tests[test_id] = {
        "cancelled": True,
        "paused": False,
        "start_time": datetime.now(timezone.utc)
    }
    logger.info(f"Batch test force-cancelled (not found locally): {test_id}")
    return True

def pause_batch_test(test_id: str) -> bool:
    """Pause a running batch test."""
    if test_id in _active_batch_tests:
        _active_batch_tests[test_id]["paused"] = True
        logger.info(f"Batch test paused: {test_id}")
        return True
    _active_batch_tests[test_id] = {
        "cancelled": False,
        "paused": True,
        "start_time": datetime.now(timezone.utc)
    }
    logger.info(f"Batch test force-paused (not found locally): {test_id}")
    return True

def resume_batch_test(test_id: str) -> bool:
    """Resume a paused batch test."""
    if test_id in _active_batch_tests:
        _active_batch_tests[test_id]["paused"] = False
        logger.info(f"Batch test resumed: {test_id}")
        return True
    _active_batch_tests[test_id] = {
        "cancelled": False,
        "paused": False,
        "start_time": datetime.now(timezone.utc)
    }
    logger.info(f"Batch test force-resumed (not found locally): {test_id}")
    return True

def unregister_batch_test(test_id: str) -> None:
    """Remove batch test from active tests."""
    if test_id in _active_batch_tests:
        del _active_batch_tests[test_id]
        logger.info(f"Batch test unregistered: {test_id}")

def get_active_batch_tests() -> Dict[str, Dict]:
    """Get all active batch tests."""
    return {
        test_id: {
            "paused": info["paused"],
            "cancelled": info["cancelled"],
            "start_time": info["start_time"].isoformat()
        }
        for test_id, info in _active_batch_tests.items()
    }


def clean_context_text(text: str) -> str:
    """Temizlenmiş context metni döndürür.
    
    Chunk'lardaki gereksiz whitespace karakterlerini temizler:
    - Birden fazla \n -> tek boşluk
    - \t -> tek boşluk  
    - Birden fazla boşluk -> tek boşluk
    - Başta/sonda boşluk -> kaldır
    """
    import re
    
    if not text:
        return text
    
    # \n ve \t karakterlerini boşluğa çevir
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # Birden fazla boşluğu tek boşluğa indir
    text = re.sub(r'\s+', ' ', text)
    
    # Başta ve sonda boşluk varsa kaldır
    text = text.strip()
    
    return text


def _run_evaluation_background(run_id: int) -> None:
    db = SessionLocal()
    try:
        service = RagasEvaluationService(db)
        service.run_evaluation(run_id)
    except Exception:
        logger.exception("RAGAS background evaluation failed (run_id=%s)", run_id)
    finally:
        db.close()


def _run_evaluation_for_questions_background(run_id: int, question_ids: List[int]) -> None:
    db = SessionLocal()
    try:
        service = RagasEvaluationService(db)
        service.run_evaluation_for_questions(run_id, question_ids)
    except Exception:
        logger.exception(
            "RAGAS background evaluation failed (run_id=%s, question_ids=%s)",
            run_id,
            question_ids,
        )
    finally:
        db.close()


class RagasWandbExportRequest(BaseModel):
    course_id: int
    run_id: int


class RagasWandbRunUpdateRequest(BaseModel):
    run_id: str
    course_id: int
    evaluation_run_id: int
    tags: Optional[List[str]] = None


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
    
    from datetime import timezone
    test_set.updated_at = datetime.now(timezone.utc)
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


@router.post("/test-sets/{test_set_id}/merge/{source_test_set_id}", response_model=TestSetDetailResponse)
async def merge_test_sets(
    test_set_id: int,
    source_test_set_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Merge questions from source test set into target test set.
    
    Copies all questions from source_test_set_id to test_set_id (target).
    The source test set remains unchanged.
    """
    # Get and verify target test set
    target_test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not target_test_set:
        raise HTTPException(status_code=404, detail="Target test set not found")
    
    verify_course_access(db, target_test_set.course_id, current_user)
    
    # Get and verify source test set
    source_test_set = db.query(TestSet).filter(TestSet.id == source_test_set_id).first()
    if not source_test_set:
        raise HTTPException(status_code=404, detail="Source test set not found")
    
    verify_course_access(db, source_test_set.course_id, current_user)
    
    # Both test sets must belong to the same course
    if target_test_set.course_id != source_test_set.course_id:
        raise HTTPException(
            status_code=400,
            detail="Test sets must belong to the same course"
        )
    
    # Get all questions from source test set
    source_questions = db.query(TestQuestion).filter(
        TestQuestion.test_set_id == source_test_set_id
    ).all()
    
    if not source_questions:
        raise HTTPException(
            status_code=400,
            detail="Source test set has no questions to merge"
        )
    
    # Copy questions to target test set
    merged_count = 0
    for q in source_questions:
        new_question = TestQuestion(
            test_set_id=test_set_id,
            question=q.question,
            ground_truth=q.ground_truth,
            alternative_ground_truths=q.alternative_ground_truths,
            expected_contexts=q.expected_contexts,
            question_metadata=q.question_metadata,
        )
        db.add(new_question)
        merged_count += 1
    
    db.commit()
    
    logger.info(
        f"Merged {merged_count} questions from test set {source_test_set_id} "
        f"to test set {test_set_id}"
    )
    
    # Return updated target test set
    return await get_test_set(test_set_id, current_user, db)


@router.post("/test-sets/{test_set_id}/import", response_model=TestSetDetailResponse)
async def import_questions(
    test_set_id: int,
    data: dict,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Import questions into a test set."""
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)

    questions = None
    if isinstance(data, dict):
        questions = data.get("test_cases")
        if questions is None:
            questions = data.get("questions")
    if not isinstance(questions, list):
        raise HTTPException(
            status_code=400,
            detail="Invalid payload: 'test_cases' must be a list",
        )

    for q_data in questions:
        if not isinstance(q_data, dict):
            continue
        question_text = q_data.get("question")
        ground_truth = q_data.get("ground_truth")
        if not question_text or not ground_truth:
            continue

        question = TestQuestion(
            test_set_id=test_set_id,
            question=question_text,
            ground_truth=ground_truth,
            alternative_ground_truths=q_data.get("alternative_ground_truths"),
            expected_contexts=q_data.get("expected_contexts"),
            question_metadata=q_data.get("question_metadata"),
        )
        db.add(question)
    
    db.commit()
    
    # Return updated test set
    return await get_test_set(test_set_id, current_user, db)


@router.get("/test-sets/{test_set_id}/export")
async def export_test_set(
    test_set_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export test set to JSON format."""
    print(f"[EXPORT] Received request for test_set_id: {test_set_id}")
    
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    print(f"[EXPORT] Found test set: ID={test_set.id}, Name={test_set.name}")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    questions = db.query(TestQuestion).filter(
        TestQuestion.test_set_id == test_set_id
    ).all()
    
    print(f"[EXPORT] Found {len(questions)} questions for test_set_id={test_set_id}")
    
    test_cases = [
        {
            "question": q.question,
            "ground_truth": q.ground_truth,
            "alternative_ground_truths": q.alternative_ground_truths,
            "expected_contexts": q.expected_contexts,
            "question_metadata": q.question_metadata,
        }
        for q in questions
    ]

    result = {
        "id": test_set.id,
        "name": test_set.name,
        "description": test_set.description,
        "test_cases": test_cases,
        # Backward-compatible alias
        "questions": test_cases,
    }
    
    print(f"[EXPORT] Returning: ID={result['id']}, Name={result['name']}, Questions={len(result['questions'])}")
    
    return result


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
    logger.info("[SELECTIVE EVAL DEBUG] Received evaluation request:")
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
            background_tasks.add_task(
                _run_evaluation_for_questions_background,
                existing_run.id,
                new_question_ids,
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
    if data.evaluation_provider:
        run_config["evaluation_provider"] = data.evaluation_provider
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
    background_tasks.add_task(_run_evaluation_background, run.id)
    
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
        "wandb_run_url": (run.config or {}).get("wandb_run_url"),
        "wandb_run_id": (run.config or {}).get("wandb_run_id"),
    }


@router.get("/runs/{run_id}/stream")
async def stream_run_progress(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream evaluation progress and results (Server-Sent Events).

    This endpoint is designed for UI live updates during an active run.
    It polls DB for new EvaluationResult rows and pushes them as SSE events.
    """

    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    verify_course_access(db, run.course_id, current_user)

    async def generate():
        last_result_id: int = 0
        last_status: Optional[str] = None
        last_processed: Optional[int] = None

        while True:
            db.expire_all()
            current_run = (
                db.query(EvaluationRun)
                .filter(EvaluationRun.id == run_id)
                .first()
            )
            if not current_run:
                payload = {"event": "error", "error": "Run not found"}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return

            # Send status/progress changes
            if (
                current_run.status != last_status
                or current_run.processed_questions != last_processed
            ):
                last_status = current_run.status
                last_processed = current_run.processed_questions
                payload = {
                    "event": "status",
                    "run_id": current_run.id,
                    "status": current_run.status,
                    "total_questions": current_run.total_questions,
                    "processed_questions": current_run.processed_questions,
                    "error_message": current_run.error_message,
                    "wandb_run_url": (current_run.config or {}).get(
                        "wandb_run_url"
                    ),
                    "wandb_run_id": (current_run.config or {}).get(
                        "wandb_run_id"
                    ),
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # Send new results
            new_results = (
                db.query(EvaluationResult)
                .filter(
                    EvaluationResult.run_id == run_id,
                    EvaluationResult.id > last_result_id,
                )
                .order_by(EvaluationResult.id.asc())
                .all()
            )

            for r in new_results:
                last_result_id = max(last_result_id, r.id)
                payload = {
                    "event": "result",
                    "result": {
                        "id": r.id,
                        "run_id": r.run_id,
                        "question_id": r.question_id,
                        "question_text": r.question_text,
                        "ground_truth_text": r.ground_truth_text,
                        "generated_answer": r.generated_answer,
                        "retrieved_contexts": r.retrieved_contexts,
                        "faithfulness": r.faithfulness,
                        "answer_relevancy": r.answer_relevancy,
                        "context_precision": r.context_precision,
                        "context_recall": r.context_recall,
                        "answer_correctness": r.answer_correctness,
                        "latency_ms": r.latency_ms,
                        "llm_provider": r.llm_provider,
                        "llm_model": r.llm_model,
                        "embedding_model": r.embedding_model,
                        "evaluation_model": r.evaluation_model,
                        "search_alpha": r.search_alpha,
                        "search_top_k": r.search_top_k,
                        "error_message": r.error_message,
                        "created_at": (
                            r.created_at.isoformat()
                            if r.created_at
                            else None
                        ),
                    },
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # Stop once run finished and there are no new results to send.
            if current_run.status in ["completed", "failed", "cancelled"]:
                if not new_results:
                    payload = {
                        "event": "complete",
                        "run_id": current_run.id,
                        "status": current_run.status,
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    return

            await asyncio.sleep(1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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


@router.post("/wandb-export")
async def wandb_export_run(
    data: RagasWandbExportRequest,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Export an evaluation run to Weights & Biases."""
    verify_course_access(db, data.course_id, current_user)

    wb_project = os.getenv("WANDB_PROJECT")
    wb_mode = os.getenv("WANDB_MODE")
    wb_has_key = bool(os.getenv("WANDB_API_KEY"))
    if wandb is None or not wb_project or (not wb_has_key and wb_mode != "offline"):
        raise HTTPException(
            status_code=400,
            detail=(
                "W&B is not configured. Set WANDB_PROJECT and WANDB_API_KEY "
                "(or WANDB_MODE=offline)."
            ),
        )

    run = db.query(EvaluationRun).filter(EvaluationRun.id == data.run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.course_id != data.course_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    results = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.run_id == run.id)
        .order_by(EvaluationResult.id.asc())
        .all()
    )
    if not results:
        raise HTTPException(status_code=404, detail="No results found for run")

    summary = db.query(RunSummary).filter(RunSummary.run_id == run.id).first()
    test_set = db.query(TestSet).filter(TestSet.id == run.test_set_id).first()
    course_settings = get_or_create_settings(db, data.course_id)

    sample = next(
        (
            r
            for r in results
            if r.llm_model or r.embedding_model or r.evaluation_model
        ),
        results[0],
    )

    wb_entity = os.getenv("WANDB_ENTITY")
    from datetime import timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"ragas-{data.course_id}-{run.test_set_id}-{run.id}-{timestamp}"
    )

    wb_run = wandb.init(
        project=wb_project,
        entity=wb_entity,
        name=run_name,
        config={
            "course_id": data.course_id,
            "evaluation_run_id": run.id,
            "test_set_id": run.test_set_id,
            "test_set_name": test_set.name if test_set else None,
            "run_name": run.name,
            # Generation (RAG) models
            "generation_llm_provider": sample.llm_provider,
            "generation_llm_model": sample.llm_model,
            "embedding_model": sample.embedding_model,
            # Evaluation (RAGAS) models
            "evaluation_llm_provider": (run.config or {}).get("evaluation_provider"),
            "evaluation_llm_model": sample.evaluation_model,
            "search_alpha": sample.search_alpha or course_settings.search_alpha,
            "search_top_k": sample.search_top_k or course_settings.search_top_k,
            "min_relevance_score": course_settings.min_relevance_score,
            "total_questions": run.total_questions,
        },
        tags=["RAGAS", "ragas", "evaluation"],
    )

    table = wandb.Table(
        columns=[
            "id",
            "question_id",
            "question",
            "ground_truth",
            "generated_answer",
            "contexts_count",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
            "latency_ms",
            "error_message",
            "created_at",
        ]
    )

    for r in results:
        table.add_data(
            r.id,
            r.question_id,
            r.question_text,
            r.ground_truth_text,
            r.generated_answer,
            len(r.retrieved_contexts) if r.retrieved_contexts else 0,
            r.faithfulness,
            r.answer_relevancy,
            r.context_precision,
            r.context_recall,
            r.answer_correctness,
            r.latency_ms,
            r.error_message,
            r.created_at.isoformat() if r.created_at else None,
        )

    payload = {
        "results": table,
        "test_count": len(results),
    }
    if summary:
        payload.update(
            {
                "aggregate/avg_faithfulness": summary.avg_faithfulness,
                "aggregate/avg_answer_relevancy": summary.avg_answer_relevancy,
                "aggregate/avg_context_precision": summary.avg_context_precision,
                "aggregate/avg_context_recall": summary.avg_context_recall,
                "aggregate/avg_answer_correctness": summary.avg_answer_correctness,
                "aggregate/avg_latency_ms": summary.avg_latency_ms,
                "aggregate/successful_questions": summary.successful_questions,
                "aggregate/failed_questions": summary.failed_questions,
            }
        )
    wandb.log(payload)

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
    limit: int = 25,
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

        filters = {}
        if state and state != "all":
            filters["state"] = state

        runs = api.runs(
            path=runs_path,
            filters=filters,
            per_page=limit * 5,
        )

        all_filtered_runs = []
        for r in runs:
            cfg = r.config or {}

            if cfg.get("course_id") and cfg.get("course_id") != course_id:
                continue

            if search:
                search_lower = search.lower()
                if (
                    search_lower not in r.name.lower()
                    and search_lower not in r.id.lower()
                    and (
                        cfg.get("evaluation_run_id")
                        and search_lower
                        not in str(cfg.get("evaluation_run_id")).lower()
                    )
                ):
                    continue

            if tag:
                tags = getattr(r, "tags", [])
                tag_lower = tag.lower()
                if not any(tag_lower in t.lower() for t in tags):
                    continue

            missing = []
            if not cfg.get("evaluation_run_id"):
                missing.append("evaluation_run_id")
            if not cfg.get("test_set_id"):
                missing.append("test_set_id")
            if not cfg.get("evaluation_model"):
                missing.append("evaluation_model")

            tags = getattr(r, "tags", [])
            all_filtered_runs.append(
                {
                    "id": r.id,
                    "name": r.name,
                    "state": r.state,
                    "created_at": r.created_at,
                    "config": {**cfg, "tags": tags},
                    "missing_fields": missing,
                }
            )

        all_filtered_runs.sort(
            key=lambda x: x.get("created_at") or "",
            reverse=True,
        )

        total_items = len(all_filtered_runs)
        total_pages = (total_items + limit - 1) // limit

        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_result = all_filtered_runs[start_idx:end_idx]

        return {
            "runs": paginated_result,
            "pagination": {
                "currentPage": page,
                "totalPages": total_pages,
                "totalItems": total_items,
                "itemsPerPage": limit,
            },
        }

    except Exception as e:
        logger.error("Error fetching W&B runs: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch W&B runs: {str(e)}",
        )


@router.post("/wandb-runs/update")
async def update_wandb_run(
    data: RagasWandbRunUpdateRequest,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Update a W&B run's missing config fields using DB values for an evaluation run."""
    verify_course_access(db, data.course_id, current_user)

    wb_project = os.getenv("WANDB_PROJECT")
    wb_entity = os.getenv("WANDB_ENTITY")
    wb_has_key = bool(os.getenv("WANDB_API_KEY"))
    if wandb is None or not wb_project or not wb_has_key:
        raise HTTPException(
            status_code=400,
            detail="W&B is not configured. Set WANDB_PROJECT and WANDB_API_KEY.",
        )

    eval_run = (
        db.query(EvaluationRun)
        .filter(EvaluationRun.id == data.evaluation_run_id)
        .first()
    )
    if not eval_run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    if eval_run.course_id != data.course_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    sample = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.run_id == eval_run.id)
        .order_by(EvaluationResult.id.asc())
        .first()
    )

    api = wandb.Api()
    runs_path = f"{wb_entity}/{wb_project}" if wb_entity else wb_project
    try:
        wb_run = api.run(f"{runs_path}/{data.run_id}")
    except Exception as e:
        raise HTTPException(status_code=404, detail="W&B run not found.") from e

    cfg = wb_run.config or {}
    updated = []

    if data.tags is not None:
        cfg["tags"] = data.tags
        updated.append("tags")

    if not cfg.get("course_id"):
        cfg["course_id"] = data.course_id
        updated.append("course_id")

    if not cfg.get("evaluation_run_id"):
        cfg["evaluation_run_id"] = eval_run.id
        updated.append("evaluation_run_id")

    if not cfg.get("test_set_id"):
        cfg["test_set_id"] = eval_run.test_set_id
        updated.append("test_set_id")

    if sample:
        if not cfg.get("evaluation_model") and sample.evaluation_model:
            cfg["evaluation_model"] = sample.evaluation_model
            updated.append("evaluation_model")
        if not cfg.get("llm_provider") and sample.llm_provider:
            cfg["llm_provider"] = sample.llm_provider
            updated.append("llm_provider")
        if not cfg.get("llm_model") and sample.llm_model:
            cfg["llm_model"] = sample.llm_model
            updated.append("llm_model")
        if not cfg.get("embedding_model") and sample.embedding_model:
            cfg["embedding_model"] = sample.embedding_model
            updated.append("embedding_model")

    if not updated:
        return {"success": False, "message": "No fields needed update."}

    wb_run.config.update(cfg)
    wb_run.save()

    return {
        "success": True,
        "updated_fields": updated,
        "run_name": wb_run.name,
    }


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
        system_prompt = course_settings.system_prompt or DEFAULT_SYSTEM_PROMPT
    
    # Use provided LLM settings or get from course settings
    llm_provider = data.llm_provider or course_settings.llm_provider
    llm_model = data.llm_model or course_settings.llm_model
    
    # Get evaluation model from RAGAS service
    evaluation_model_used = None
    try:
        ragas_service = RagasEvaluationService(db)
        import httpx
        response = httpx.get(f"{ragas_service.ragas_url}/settings", timeout=5.0)
        if response.status_code == 200:
            ragas_settings = response.json()
            evaluation_model_used = ragas_settings.get("current_model")
    except Exception:
        pass  # If RAGAS service is not available, continue without evaluation model info
    
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
            model=course_settings.default_embedding_model,
            input_type="query"
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
                # 🔥 Temiz context kullan - whitespace karakterlerini kaldır
                clean_content = clean_context_text(content)
                
                retrieved_contexts.append({
                    "text": clean_content,
                    "score": result.score
                })
                context_texts.append(clean_content)

        logger.info(
            "[RAGAS QUICK TEST] Retrieved contexts: %s (search_results=%s, min_score=%s, top_k=%s)",
            len(context_texts),
            len(search_results) if search_results else 0,
            getattr(course_settings, 'min_relevance_score', 0.0) or 0.0,
            course_settings.search_top_k,
        )
        
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

        logger.info(
            "[RAGAS QUICK TEST] Generated answer length: %s (provider=%s, model=%s)",
            len(generated_answer) if generated_answer else 0,
            llm_provider,
            llm_model,
        )
        
        # Prepare ground truths (combine primary + alternatives)
        ground_truths = [data.ground_truth]
        if data.alternative_ground_truths:
            ground_truths.extend(data.alternative_ground_truths)
        
        # Call RAGAS service for metrics (RAGAS expects plain text contexts)
        ragas_service = RagasEvaluationService(db)
        
        # Determine embedding model for RAGAS metrics
        ragas_emb_model = data.ragas_embedding_model or course_settings.default_embedding_model
        ragas_emb_provider = get_embedding_provider_from_model(ragas_emb_model)
        # Clean provider prefix for embedding model name (e.g., "openai/text-embedding-3-small" -> "text-embedding-3-small" for openai provider)
        ragas_emb_model_clean = ragas_emb_model
        if "/" in ragas_emb_model and ragas_emb_provider != "openrouter":
            ragas_emb_model_clean = ragas_emb_model.split("/", 1)[1] if "/" in ragas_emb_model else ragas_emb_model
        
        metrics = ragas_service._get_ragas_metrics_sync(
            data.question,
            ground_truths,
            generated_answer,
            context_texts,
            evaluation_model_used,
            reranker_provider=course_settings.reranker_provider if course_settings.enable_reranker else None,
            reranker_model=course_settings.reranker_model if course_settings.enable_reranker else None,
            embedding_provider=ragas_emb_provider,
            embedding_model=ragas_emb_model_clean
        )

        logger.info(
            "[RAGAS QUICK TEST] Metrics keys: %s | error=%s",
            list(metrics.keys()) if isinstance(metrics, dict) else type(metrics),
            metrics.get("error") if isinstance(metrics, dict) else None,
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
            evaluation_model_used=evaluation_model_used,
            embedding_model_used=course_settings.default_embedding_model,
            search_top_k_used=course_settings.search_top_k,
            search_alpha_used=course_settings.search_alpha,
            reranker_used=course_settings.enable_reranker,
            reranker_provider=course_settings.reranker_provider if course_settings.enable_reranker else None,
            reranker_model=course_settings.reranker_model if course_settings.enable_reranker else None,
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
        evaluation_model=data.evaluation_model,
        embedding_model=data.embedding_model,
        search_top_k=data.search_top_k,
        search_alpha=data.search_alpha,
        reranker_used=data.reranker_used,
        reranker_provider=data.reranker_provider,
        reranker_model=data.reranker_model,
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
        evaluation_model=result.evaluation_model,
        embedding_model=result.embedding_model,
        search_top_k=result.search_top_k,
        search_alpha=result.search_alpha,
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
        reranker_used=result.reranker_used,
        reranker_provider=result.reranker_provider,
        reranker_model=result.reranker_model,
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
    try:
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
        
        # Get unique group names with their creation dates and basic metadata
        from sqlalchemy import func as sql_func

        # Group stats: earliest created_at per group, count, and representative llm fields
        group_stats_subq = (
            db.query(
                QuickTestResult.group_name.label("group_name"),
                sql_func.min(QuickTestResult.created_at).label("created_at"),
                sql_func.count(QuickTestResult.id).label("test_count"),
            )
            .filter(
                QuickTestResult.course_id == course_id,
                QuickTestResult.group_name.isnot(None),
                QuickTestResult.group_name != "",
            )
            .group_by(QuickTestResult.group_name)
            .subquery()
        )

        # Representative row per group: earliest entry (by created_at) to capture test-time settings
        rep_subq = (
            db.query(
                QuickTestResult.group_name.label("group_name"),
                sql_func.min(QuickTestResult.created_at).label("created_at"),
            )
            .filter(
                QuickTestResult.course_id == course_id,
                QuickTestResult.group_name.isnot(None),
                QuickTestResult.group_name != "",
            )
            .group_by(QuickTestResult.group_name)
            .subquery()
        )

        rep_rows = (
            db.query(
                QuickTestResult.group_name,
                QuickTestResult.llm_provider,
                QuickTestResult.llm_model,
                QuickTestResult.evaluation_model,
                QuickTestResult.embedding_model,
                QuickTestResult.search_top_k,
                QuickTestResult.search_alpha,
                QuickTestResult.reranker_used,
                QuickTestResult.reranker_provider,
                QuickTestResult.reranker_model,
            )
            .join(
                rep_subq,
                (QuickTestResult.group_name == rep_subq.c.group_name)
                & (QuickTestResult.created_at == rep_subq.c.created_at),
            )
            .all()
        )
        rep_map = {
            r[0]: {
                "llm_provider": r[1],
                "llm_model": r[2],
                "evaluation_model": r[3],
                "embedding_model": r[4],
                "search_top_k": r[5],
                "search_alpha": r[6],
                "reranker_used": r[7],
                "reranker_provider": r[8],
                "reranker_model": r[9],
            }
            for r in rep_rows
            if r and r[0]
        }

        groups_query = (
            db.query(
                group_stats_subq.c.group_name,
                group_stats_subq.c.created_at,
                group_stats_subq.c.test_count,
            )
            .order_by(group_stats_subq.c.created_at.desc())
            .all()
        )

        # Calculate average metrics per group (SQL AVG ignores NULL; avoids NaN and keeps 0 values)
        metrics_query = (
            db.query(
                QuickTestResult.group_name.label("group_name"),
                sql_func.avg(QuickTestResult.faithfulness).label("avg_faithfulness"),
                sql_func.avg(QuickTestResult.answer_relevancy).label("avg_answer_relevancy"),
                sql_func.avg(QuickTestResult.context_precision).label("avg_context_precision"),
                sql_func.avg(QuickTestResult.context_recall).label("avg_context_recall"),
                sql_func.avg(QuickTestResult.answer_correctness).label("avg_answer_correctness"),
            )
            .filter(
                QuickTestResult.course_id == course_id,
                QuickTestResult.group_name.isnot(None),
                QuickTestResult.group_name != "",
            )
        )
        if group_name:
            metrics_query = metrics_query.filter(QuickTestResult.group_name == group_name)

        metrics_rows = metrics_query.group_by(QuickTestResult.group_name).all()
        group_metrics = {
            r.group_name: {
                "avg_faithfulness": r.avg_faithfulness,
                "avg_answer_relevancy": r.avg_answer_relevancy,
                "avg_context_precision": r.avg_context_precision,
                "avg_context_recall": r.avg_context_recall,
                "avg_answer_correctness": r.avg_answer_correctness,
            }
            for r in metrics_rows
            if r and r.group_name
        }

        groups = []
        for gname, created_at_dt, test_count in groups_query:
            if not gname:
                continue
            rep = rep_map.get(gname, {})
            metrics = group_metrics.get(gname, {})
            groups.append(
                {
                    "name": gname,
                    "created_at": created_at_dt.isoformat() if created_at_dt else None,
                    "test_count": int(test_count) if test_count is not None else None,
                    "llm_provider": rep.get("llm_provider"),
                    "llm_model": rep.get("llm_model"),
                    "evaluation_model": rep.get("evaluation_model"),
                    "embedding_model": rep.get("embedding_model"),
                    "search_top_k": rep.get("search_top_k"),
                    "search_alpha": rep.get("search_alpha"),
                    "reranker_used": rep.get("reranker_used"),
                    "reranker_provider": rep.get("reranker_provider"),
                    "reranker_model": rep.get("reranker_model"),
                    "avg_faithfulness": metrics.get("avg_faithfulness"),
                    "avg_answer_relevancy": metrics.get("avg_answer_relevancy"),
                    "avg_context_precision": metrics.get("avg_context_precision"),
                    "avg_context_recall": metrics.get("avg_context_recall"),
                    "avg_answer_correctness": metrics.get("avg_answer_correctness"),
                }
            )
        
        # Calculate aggregate statistics for the filtered results (not just current page)
        all_results_query = db.query(QuickTestResult).filter(
            QuickTestResult.course_id == course_id
        )
        if group_name:
            all_results_query = all_results_query.filter(QuickTestResult.group_name == group_name)
        
        all_results = all_results_query.all()
        
        # Collect test parameters from most recent result
        test_parameters = None
        if all_results:
            latest = all_results[0]
            test_parameters = {
                "llm_model": latest.llm_model,
                "llm_provider": latest.llm_provider,
                "embedding_model": latest.embedding_model,
                "evaluation_model": latest.evaluation_model,
                "search_alpha": latest.search_alpha,
                "search_top_k": latest.search_top_k,
                "reranker_used": latest.reranker_used,
                "reranker_provider": latest.reranker_provider,
                "reranker_model": latest.reranker_model,
            }
        
        # Calculate average metrics
        successful_results = [r for r in all_results if r.faithfulness is not None]
        aggregate = None
        if successful_results:
            aggregate = {
                "avg_faithfulness": sum(r.faithfulness for r in successful_results if r.faithfulness) / len([r for r in successful_results if r.faithfulness]),
                "avg_answer_relevancy": sum(r.answer_relevancy for r in successful_results if r.answer_relevancy) / len([r for r in successful_results if r.answer_relevancy]) if any(r.answer_relevancy for r in successful_results) else None,
                "avg_context_precision": sum(r.context_precision for r in successful_results if r.context_precision) / len([r for r in successful_results if r.context_precision]) if any(r.context_precision for r in successful_results) else None,
                "avg_context_recall": sum(r.context_recall for r in successful_results if r.context_recall) / len([r for r in successful_results if r.context_recall]) if any(r.context_recall for r in successful_results) else None,
                "avg_answer_correctness": sum(r.answer_correctness for r in successful_results if r.answer_correctness) / len([r for r in successful_results if r.answer_correctness]) if any(r.answer_correctness for r in successful_results) else None,
                "test_count": len(all_results),
                "test_parameters": test_parameters
            }
        
        response_data = {
            "results": [QuickTestResultResponse(
                id=r.id,
                course_id=r.course_id,
                group_name=r.group_name,
                question=r.question,
                ground_truth=r.ground_truth,
                alternative_ground_truths=r.alternative_ground_truths,
                system_prompt=r.system_prompt,
                llm_provider=r.llm_provider,
                llm_model=r.llm_model,
                evaluation_model=r.evaluation_model,
                embedding_model=r.embedding_model,
                search_top_k=r.search_top_k,
                search_alpha=r.search_alpha,
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
                reranker_used=r.reranker_used,
                reranker_provider=r.reranker_provider,
                reranker_model=r.reranker_model,
            ) for r in results],
            "total": total_count,
            "groups": groups,
        }
        
        # Add aggregate if available
        if aggregate:
            response_data["aggregate"] = aggregate
        
        return QuickTestResultListResponse(**response_data)
    except Exception as e:
        logger.error(f"Error listing quick test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick-test-results/existing-questions")
async def get_existing_quick_test_questions(
    course_id: int,
    group_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    verify_course_access(db, course_id, current_user)
    if not group_name:
        raise HTTPException(status_code=400, detail="group_name is required")

    rows = (
        db.query(QuickTestResult.question)
        .filter(
            QuickTestResult.course_id == course_id,
            QuickTestResult.group_name == group_name,
        )
        .distinct()
        .all()
    )
    return {
        "questions": [r[0] for r in rows if r and r[0]],
    }


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
        evaluation_model=result.evaluation_model,
        embedding_model=result.embedding_model,
        search_top_k=result.search_top_k,
        search_alpha=result.search_alpha,
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
        reranker_used=result.reranker_used,
        reranker_provider=result.reranker_provider,
        reranker_model=result.reranker_model,
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


@router.post("/quick-test-results/wandb-export")
async def export_quick_test_results_to_wandb(
    data: dict,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Export quick test results group to Weights & Biases."""
    course_id = data.get("course_id")
    group_name = data.get("group_name")
    
    if not course_id or not group_name:
        raise HTTPException(
            status_code=400,
            detail="course_id and group_name are required"
        )
    
    verify_course_access(db, course_id, current_user)

    wb_project = os.getenv("WANDB_PROJECT")
    wb_mode = os.getenv("WANDB_MODE")
    wb_has_key = bool(os.getenv("WANDB_API_KEY"))
    if wandb is None or not wb_project or (not wb_has_key and wb_mode != "offline"):
        raise HTTPException(
            status_code=400,
            detail=(
                "W&B is not configured. Set WANDB_PROJECT and WANDB_API_KEY "
                "(or WANDB_MODE=offline)."
            ),
        )

    # Get all results for this group
    results = (
        db.query(QuickTestResult)
        .filter(
            QuickTestResult.course_id == course_id,
            QuickTestResult.group_name == group_name
        )
        .order_by(QuickTestResult.created_at.asc())
        .all()
    )
    
    if not results:
        raise HTTPException(status_code=404, detail="No results found for this group")

    # Get course info
    from app.models.db_models import Course
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Calculate aggregate metrics
    metrics_with_values = {
        'faithfulness': [r.faithfulness for r in results if r.faithfulness is not None],
        'answer_relevancy': [r.answer_relevancy for r in results if r.answer_relevancy is not None],
        'context_precision': [r.context_precision for r in results if r.context_precision is not None],
        'context_recall': [r.context_recall for r in results if r.context_recall is not None],
        'answer_correctness': [r.answer_correctness for r in results if r.answer_correctness is not None],
        'latency_ms': [r.latency_ms for r in results if r.latency_ms is not None],
    }
    
    def safe_avg(values):
        return sum(values) / len(values) if values else None

    avg_metrics = {
        'avg_faithfulness': safe_avg(metrics_with_values['faithfulness']),
        'avg_answer_relevancy': safe_avg(metrics_with_values['answer_relevancy']),
        'avg_context_precision': safe_avg(metrics_with_values['context_precision']),
        'avg_context_recall': safe_avg(metrics_with_values['context_recall']),
        'avg_answer_correctness': safe_avg(metrics_with_values['answer_correctness']),
        'avg_latency_ms': safe_avg(metrics_with_values['latency_ms']),
    }

    # Get representative test parameters from first result
    sample = results[0]
    
    wb_entity = os.getenv("WANDB_ENTITY")
    from datetime import timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"ragas-quick-{course_id}-{group_name}-{timestamp}"

    # Initialize W&B run
    wb_run = wandb.init(
        project=wb_project,
        entity=wb_entity,
        name=run_name,
        config={
            "course_id": course_id,
            "course_name": course.name,
            "group_name": group_name,
            "test_type": "quick_test",
            # Generation (RAG) models
            "generation_llm_provider": sample.llm_provider,
            "generation_llm_model": sample.llm_model,
            "embedding_model": sample.embedding_model,
            # Evaluation (RAGAS) models
            "evaluation_model": sample.evaluation_model,
            # Search parameters
            "search_alpha": sample.search_alpha,
            "search_top_k": sample.search_top_k,
            # Reranker settings
            "reranker_used": sample.reranker_used,
            "reranker_provider": sample.reranker_provider,
            "reranker_model": sample.reranker_model,
            # System prompt
            "system_prompt": sample.system_prompt,
            # Counts
            "total_questions": len(results),
        },
        tags=["RAGAS", "ragas", "quick_test", "evaluation", group_name],
    )

    # Create detailed results table
    table = wandb.Table(
        columns=[
            "id",
            "question",
            "ground_truth",
            "alternative_ground_truths",
            "generated_answer",
            "contexts_count",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
            "latency_ms",
            "llm_provider",
            "llm_model",
            "evaluation_model",
            "embedding_model",
            "search_top_k",
            "search_alpha",
            "reranker_used",
            "reranker_provider",
            "reranker_model",
            "created_at",
        ]
    )

    for r in results:
        contexts_count = 0
        if r.retrieved_contexts:
            if isinstance(r.retrieved_contexts, list):
                contexts_count = len(r.retrieved_contexts)
        
        table.add_data(
            r.id,
            r.question,
            r.ground_truth,
            r.alternative_ground_truths,
            r.generated_answer,
            contexts_count,
            r.faithfulness,
            r.answer_relevancy,
            r.context_precision,
            r.context_recall,
            r.answer_correctness,
            r.latency_ms,
            r.llm_provider,
            r.llm_model,
            r.evaluation_model,
            r.embedding_model,
            r.search_top_k,
            r.search_alpha,
            r.reranker_used,
            r.reranker_provider,
            r.reranker_model,
            r.created_at.isoformat() if r.created_at else None,
        )

    # Log everything to W&B
    payload = {
        "results": table,
        "test_count": len(results),
    }
    
    # Add aggregate metrics
    for key, value in avg_metrics.items():
        if value is not None:
            payload[f"aggregate/{key}"] = value
    
    wandb.log(payload)

    # Get run URL
    run_url = getattr(wb_run, "url", None)
    run_id = getattr(wb_run, "id", None)
    
    wb_run.finish()
    
    return {
        "success": True,
        "run_name": run_name,
        "run_id": run_id,
        "run_url": run_url,
        "exported_count": len(results),
        "aggregate_metrics": avg_metrics,
    }


# ==================== Batch Test Streaming Endpoint ====================

class BatchTestStreamRequest(BaseModel):
    course_id: int
    test_cases: List[dict]
    group_name: str
    enable_wandb: bool = False
    only_indices: Optional[List[int]] = None
    ragas_embedding_model: Optional[str] = None


@router.post("/quick-test-results/batch-stream")
async def batch_test_stream(
    data: BatchTestStreamRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream batch RAGAS test results with W&B logging and cancellation/pause support."""
    verify_course_access(db, data.course_id, current_user)
    course_settings = get_or_create_settings(db, data.course_id)
    
    # Generate unique test ID for cancellation/pause
    test_id = create_batch_test_id()
    register_batch_test(test_id)

    async def generate():
        try:
            import time
            from app.services.weaviate_service import WeaviateService
            from app.services.embedding_service import EmbeddingService
            from app.services.llm_service import LLMService
            
            # Services
            ragas_service = RagasEvaluationService(db)
            
            # RAGAS Ayarlarından evaluation model'i al (W&B init'ten ÖNCE!)
            evaluation_model_for_batch = None
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10.0) as http_client:
                    ragas_settings_response = await http_client.get(f"{ragas_service.ragas_url}/settings")
                    if ragas_settings_response.status_code == 200:
                        ragas_settings_data = ragas_settings_response.json()
                        evaluation_model_for_batch = ragas_settings_data.get("current_model")
                        logger.info(f"[BATCH TEST] Using evaluation model from RAGAS settings: {evaluation_model_for_batch}")
                    else:
                        logger.warning("[BATCH TEST] Could not fetch RAGAS settings, will use default")
            except Exception as e:
                logger.warning(f"[BATCH TEST] Error fetching RAGAS settings: {e}")
            
            # W&B Setup
            wb_run = None
            wb_table = None
            wb_enabled = False
            
            if data.enable_wandb:
                wb_project = os.getenv("WANDB_PROJECT")
                wb_mode = os.getenv("WANDB_MODE")
                wb_has_key = bool(os.getenv("WANDB_API_KEY"))
                
                if wandb is not None and (wb_has_key or wb_mode == "offline"):
                    wb_enabled = True
                    wb_entity = os.getenv("WANDB_ENTITY")
                    from datetime import timezone
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    
                    # Finish any existing wandb run to avoid step conflicts
                    try:
                        if wandb.run is not None:
                            wandb.finish()
                            logger.info("[BATCH W&B] Finished previous wandb run")
                    except Exception as e:
                        logger.warning(f"[BATCH W&B] Error finishing previous run: {e}")
                    
                    system_prompt = course_settings.system_prompt or DEFAULT_SYSTEM_PROMPT
                    
                    wb_run = wandb.init(
                        project=wb_project,
                        entity=wb_entity,
                        name=f"ragas-batch-{data.course_id}-{data.group_name}-{timestamp}",
                        reinit=True,  # Allow creating new run even if one exists
                        config={
                            "course_id": data.course_id,
                            "group_name": data.group_name,
                            # Generation LLM (cevap üretmek için - ders ayarlarından)
                            "generation_llm_provider": course_settings.llm_provider,
                            "generation_llm_model": course_settings.llm_model,
                            "llm_model_used": course_settings.llm_model,  # Backward compatibility
                            # Evaluation LLM (RAGAS metrikleri için - RAGAS ayarlarından)
                            "evaluation_llm_model": evaluation_model_for_batch,
                            "evaluation_model": evaluation_model_for_batch,  # Backward compatibility
                            # Embedding & RAG settings
                            "embedding_model": course_settings.default_embedding_model,
                            "embedding_model_used": course_settings.default_embedding_model,
                            "search_alpha": course_settings.search_alpha,
                            "search_top_k": course_settings.search_top_k,
                            "min_relevance_score": course_settings.min_relevance_score,
                            # Reranker settings
                            "reranker_used": course_settings.enable_reranker,
                            "reranker_provider": course_settings.reranker_provider if course_settings.enable_reranker else None,
                            "reranker_model": course_settings.reranker_model if course_settings.enable_reranker else None,
                            # Test info
                            "total_tests": len(data.test_cases),
                            "system_prompt": system_prompt,
                        },
                        tags=["RAGAS", "batch", "ragas"],
                    )
                    
                    # Metric'leri tanımla ki Charts'ta görünsün
                    try:
                        wandb.define_metric("test_index")
                        wandb.define_metric("faithfulness", step_metric="test_index")
                        wandb.define_metric("answer_relevancy", step_metric="test_index")
                        wandb.define_metric("context_precision", step_metric="test_index")
                        wandb.define_metric("context_recall", step_metric="test_index")
                        wandb.define_metric("answer_correctness", step_metric="test_index")
                        wandb.define_metric("latency_ms", step_metric="test_index")
                        wandb.define_metric("contexts_count", step_metric="test_index")
                    except Exception as e:
                        logger.warning(f"W&B metric definition failed: {e}")
                    
                    wb_table = wandb.Table(
                        columns=[
                            "index",
                            "question",
                            "ground_truth",
                            "generated_answer",
                            "faithfulness",
                            "answer_relevancy",
                            "context_precision",
                            "context_recall",
                            "answer_correctness",
                            "latency_ms",
                            "contexts_count",
                            "error_message",
                        ],
                        log_mode="MUTABLE"  # Allow re-logging after mutations
                    )
            
            # Metrics aggregation
            sum_metrics = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
            }
            cnt_metrics = {
                "faithfulness": 0,
                "answer_relevancy": 0,
                "context_precision": 0,
                "context_recall": 0,
                "answer_correctness": 0,
            }
            sum_latency = 0
            cnt_latency = 0
            
            system_prompt = course_settings.system_prompt or DEFAULT_SYSTEM_PROMPT
            
            llm_provider = course_settings.llm_provider
            llm_model = course_settings.llm_model
            
            # Get course
            from app.models.db_models import Course
            course = db.query(Course).filter(Course.id == data.course_id).first()
            if not course:
                raise HTTPException(status_code=404, detail="Course not found")
            
            # Services
            embedding_service = EmbeddingService()
            weaviate_service = WeaviateService()
            llm_service = LLMService(
                provider=llm_provider,
                model=llm_model,
                temperature=course_settings.llm_temperature,
                max_tokens=course_settings.llm_max_tokens
            )
            ragas_service = RagasEvaluationService(db)
            
            # RAGAS Ayarlarından evaluation model'i al
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10.0) as http_client:
                    ragas_settings_response = await http_client.get(f"{ragas_service.ragas_url}/settings")
                    if ragas_settings_response.status_code == 200:
                        ragas_settings_data = ragas_settings_response.json()
                        evaluation_model_for_batch = ragas_settings_data.get("current_model")
                        logger.info(f"[BATCH TEST] Using evaluation model from RAGAS settings: {evaluation_model_for_batch}")
                    else:
                        evaluation_model_for_batch = None
                        logger.warning("[BATCH TEST] Could not fetch RAGAS settings, will use default")
            except Exception as e:
                evaluation_model_for_batch = None
                logger.warning(f"[BATCH TEST] Error fetching RAGAS settings: {e}")
            
            # OPTIMIZED PROCESSING with small batch parallelism (2-3 tests at a time)
            # Benefits:
            # 1. Embedding cache reduces API calls for repeated contexts
            # 2. Small parallelism (2-3 workers) provides speedup without instability
            # 3. More stable than 10 workers, faster than sequential
            logger.info(f"[BATCH TEST] Processing {len(data.test_cases)} tests with SMALL BATCH PARALLELISM (2-3 workers)")
            
            # Embedding cache for repeated contexts (saves API calls)
            embedding_cache = {}  # {text: embedding_vector}
            cache_lock = asyncio.Lock()
            
            completed_count = 0
            
            def process_single_test(idx, test_case):
                """Process a single test case with retry for missing metrics and low scores - runs in thread pool"""
                # Check cancellation before starting
                if is_batch_test_cancelled(test_id):
                    return {
                        "success": False,
                        "idx": idx,
                        "question": test_case.get("question", ""),
                        "ground_truth": test_case.get("ground_truth", ""),
                        "error": "cancelled",
                        "cancelled": True,
                    }
                
                # Wait while paused
                while is_batch_test_paused(test_id):
                    time.sleep(0.5)
                    if is_batch_test_cancelled(test_id):
                        return {
                            "success": False,
                            "idx": idx,
                            "question": test_case.get("question", ""),
                            "ground_truth": test_case.get("ground_truth", ""),
                            "error": "cancelled",
                            "cancelled": True,
                        }
                
                MAX_RETRIES = 2  # Reduced from 3 to 2 for speed
                
                # Metric-specific thresholds
                FAITHFULNESS_THRESHOLD = 0.5  # 50% - critical for factual accuracy
                RELEVANCY_THRESHOLD = 0.4  # 40% - critical for answer quality
                
                retry_count = 0
                last_error = None
                previous_answer = None  # Track previous answer to avoid infinite loops
                
                while retry_count <= MAX_RETRIES:
                    # Check cancellation at each retry
                    if is_batch_test_cancelled(test_id):
                        return {
                            "success": False,
                            "idx": idx,
                            "question": test_case.get("question", ""),
                            "ground_truth": test_case.get("ground_truth", ""),
                            "error": "cancelled",
                            "cancelled": True,
                        }
                    
                    # Wait while paused
                    while is_batch_test_paused(test_id):
                        time.sleep(0.5)
                        if is_batch_test_cancelled(test_id):
                            return {
                                "success": False,
                                "idx": idx,
                                "question": test_case.get("question", ""),
                                "ground_truth": test_case.get("ground_truth", ""),
                                "error": "cancelled",
                                "cancelled": True,
                            }
                    
                    try:
                        start_time = time.time()
                        
                        question = test_case.get("question")
                        ground_truth = test_case.get("ground_truth")
                        alternative_ground_truths = test_case.get("alternative_ground_truths", [])
                        
                        # Get embedding with cache
                        cache_key = f"{course_settings.default_embedding_model}:{question}"
                        if cache_key in embedding_cache:
                            query_vector = embedding_cache[cache_key]
                            logger.debug(f"[CACHE HIT] Test {idx}: Using cached embedding for question")
                        else:
                            query_vector = embedding_service.get_embedding(
                                question,
                                model=course_settings.default_embedding_model,
                                input_type="query"
                            )
                            embedding_cache[cache_key] = query_vector
                            logger.debug(f"[CACHE MISS] Test {idx}: Generated new embedding for question")
                        
                        # Search
                        search_results = weaviate_service.hybrid_search(
                            course_id=course.id,
                            query=question,
                            query_vector=query_vector,
                            alpha=course_settings.search_alpha,
                            limit=course_settings.search_top_k
                        )
                        
                        # Filter by relevance
                        min_score = getattr(course_settings, 'min_relevance_score', 0.0) or 0.0
                        if min_score > 0 and search_results:
                            search_results = [r for r in search_results if r.score >= min_score]
                        
                        # Extract contexts with caching
                        retrieved_contexts = []
                        context_texts = []
                        cache_hits = 0
                        cache_misses = 0
                        
                        for result in search_results:
                            content = result.content
                            if content:
                                # Clean context
                                clean_content = clean_context_text(content)
                                
                                retrieved_contexts.append({
                                    "text": clean_content,
                                    "score": result.score
                                })
                                context_texts.append(clean_content)
                        
                        logger.debug(
                            f"[CACHE STATS] Test {idx}: Question cache hit, "
                            f"Context cache hits: {cache_hits}, misses: {cache_misses}"
                        )
                        
                        # Generate answer
                        context_text = "\n\n---\n\n".join(context_texts) if context_texts else ""
                        user_prompt = f"""Bağlam:
{context_text}

Soru: {question}

Lütfen yukarıdaki bağlama dayanarak soruyu yanıtla."""
                        
                        generated_answer = llm_service.generate_response([
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ])
                        
                        # Prepare ground truths
                        ground_truths = [ground_truth]
                        if alternative_ground_truths:
                            ground_truths.extend(alternative_ground_truths)
                        
                        # Get RAGAS metrics
                        # Use ragas_embedding_model override if provided
                        batch_ragas_emb_model = data.ragas_embedding_model or course_settings.default_embedding_model
                        batch_ragas_emb_provider = get_embedding_provider_from_model(batch_ragas_emb_model)
                        batch_ragas_emb_model_clean = batch_ragas_emb_model
                        if "/" in batch_ragas_emb_model and batch_ragas_emb_provider != "openrouter":
                            batch_ragas_emb_model_clean = batch_ragas_emb_model.split("/", 1)[1]
                        
                        metrics = ragas_service._get_ragas_metrics_sync(
                            question,
                            ground_truths,
                            generated_answer,
                            context_texts,
                            evaluation_model_for_batch,
                            embedding_provider=batch_ragas_emb_provider,
                            embedding_model=batch_ragas_emb_model_clean
                        )
                        
                        # VALIDATE METRICS - Retry if critical metrics are missing or None
                        critical_metrics = ['context_precision', 'faithfulness', 'answer_relevancy']
                        missing_metrics = [m for m in critical_metrics if metrics.get(m) is None]
                        
                        # CHECK FOR LOW SCORES - Retry if scores are critically low
                        low_score_metrics = []
                        
                        # Check faithfulness (50% threshold)
                        faithfulness = metrics.get('faithfulness')
                        if faithfulness is not None and faithfulness < FAITHFULNESS_THRESHOLD:
                            low_score_metrics.append(f"faithfulness={faithfulness:.1%}")
                        
                        # Check answer_relevancy (40% threshold)
                        answer_relevancy = metrics.get('answer_relevancy')
                        if answer_relevancy is not None and answer_relevancy < RELEVANCY_THRESHOLD:
                            low_score_metrics.append(f"answer_relevancy={answer_relevancy:.1%}")
                        
                        # Decide if we should retry
                        should_retry = False
                        retry_reason = None
                        
                        if missing_metrics and retry_count < MAX_RETRIES:
                            should_retry = True
                            retry_reason = f"Missing metrics: {missing_metrics}"
                        elif low_score_metrics and retry_count < MAX_RETRIES:
                            # Only retry for low scores if the answer is different from previous attempt
                            if previous_answer is None or generated_answer != previous_answer:
                                should_retry = True
                                retry_reason = f"Low scores: {low_score_metrics}"
                            else:
                                logger.warning(
                                    f"Test {idx}: Same answer repeated with low scores {low_score_metrics}. "
                                    f"Accepting result to avoid infinite loop."
                                )
                        
                        if should_retry:
                            retry_count += 1
                            previous_answer = generated_answer
                            logger.warning(
                                f"Test {idx} attempt {retry_count}: {retry_reason}. Retrying..."
                            )
                            time.sleep(0.5)
                            continue
                        
                        # REJECT RESULT if still missing critical metrics after all retries
                        if missing_metrics:
                            error_msg = f"CRITICAL: Missing metrics after {retry_count} retries: {missing_metrics}"
                            logger.error(f"Test {idx}: {error_msg}")
                            return {
                                "success": False,
                                "idx": idx,
                                "question": question,
                                "ground_truth": ground_truth,
                                "error": error_msg,
                            }
                        
                        # WARN but ACCEPT if low scores after retries
                        if low_score_metrics:
                            logger.warning(
                                f"Test {idx} completed after {retry_count} retries: Still has low scores {low_score_metrics}"
                            )
                        
                        latency_ms = int((time.time() - start_time) * 1000)
                        
                        # Save to DB (thread-safe with new session)
                        from app.database import SessionLocal
                        thread_db = SessionLocal()
                        try:
                            contexts_for_db = [
                                {"text": ctx["text"], "score": ctx["score"]}
                                for ctx in retrieved_contexts
                            ]
                            
                            result = QuickTestResult(
                                course_id=data.course_id,
                                group_name=data.group_name,
                                question=question,
                                ground_truth=ground_truth,
                                alternative_ground_truths=alternative_ground_truths,
                                system_prompt=system_prompt,
                                llm_provider=llm_provider,
                                llm_model=llm_model,
                                evaluation_model=evaluation_model_for_batch,
                                embedding_model=course_settings.default_embedding_model,
                                search_top_k=course_settings.search_top_k,
                                search_alpha=course_settings.search_alpha,
                                reranker_used=course_settings.enable_reranker,
                                reranker_provider=course_settings.reranker_provider if course_settings.enable_reranker else None,
                                reranker_model=course_settings.reranker_model if course_settings.enable_reranker else None,
                                generated_answer=generated_answer,
                                retrieved_contexts=contexts_for_db,
                                faithfulness=metrics.get("faithfulness"),
                                answer_relevancy=metrics.get("answer_relevancy"),
                                context_precision=metrics.get("context_precision"),
                                context_recall=metrics.get("context_recall"),
                                answer_correctness=metrics.get("answer_correctness"),
                                latency_ms=latency_ms,
                                created_by=current_user.id,
                            )
                            thread_db.add(result)
                            thread_db.commit()
                        finally:
                            thread_db.close()
                        
                        return {
                            "success": True,
                            "idx": idx,
                            "question": question,
                            "ground_truth": ground_truth,
                            "generated_answer": generated_answer,
                            "metrics": metrics,
                            "latency_ms": latency_ms,
                            "retrieved_contexts": retrieved_contexts,
                            "context_texts": context_texts,
                            "retry_count": retry_count,
                            "missing_metrics": missing_metrics if missing_metrics else None,
                            "low_score_metrics": low_score_metrics if low_score_metrics else None,
                        }
                    
                    except Exception as e:
                        last_error = str(e)
                        retry_count += 1
                        
                        if retry_count <= MAX_RETRIES:
                            logger.warning(
                                f"Test {idx} attempt {retry_count} failed: {last_error}. Retrying..."
                            )
                            time.sleep(0.5)
                        else:
                            logger.error(
                                f"Test {idx} failed after {MAX_RETRIES} retries: {last_error}"
                            )
                            return {
                                "success": False,
                                "idx": idx,
                                "question": test_case.get("question", ""),
                                "ground_truth": test_case.get("ground_truth", ""),
                                "error": last_error,
                            }
                
                # Should never reach here
                return {
                    "success": False,
                    "idx": idx,
                    "question": test_case.get("question", ""),
                    "ground_truth": test_case.get("ground_truth", ""),
                    "error": last_error or "Unknown error after retries",
                }
            
            # Process tests with SMALL BATCH PARALLELISM (2-3 workers)
            # This provides speedup without the instability of 10 workers
            import concurrent.futures
            
            PARALLEL_WORKERS = int(os.getenv("BATCH_PARALLEL_WORKERS", 3))  # Conservative parallelism to avoid LLM rate limits
            logger.info(f"[BATCH TEST] Using {PARALLEL_WORKERS} parallel workers")
            
            indices_to_run: List[int]
            if data.only_indices is None:
                indices_to_run = list(range(len(data.test_cases)))
            else:
                indices_to_run = [
                    int(i)
                    for i in data.only_indices
                    if isinstance(i, int)
                    or (isinstance(i, str) and str(i).isdigit())
                ]
                indices_to_run = [
                    i
                    for i in indices_to_run
                    if 0 <= i < len(data.test_cases)
                ]

            total_to_run = len(indices_to_run)
            
            # Send init event with test_id for cancellation/pause support
            init_event = {
                "event": "init",
                "test_id": test_id,
                "total": total_to_run
            }
            yield f"data: {json.dumps(init_event, ensure_ascii=True)}\n\n"
            await asyncio.sleep(0)

            with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                # Submit selected tasks
                future_to_idx = {
                    executor.submit(process_single_test, idx, data.test_cases[idx]): idx
                    for idx in indices_to_run
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    result_data = future.result()
                    
                    # Check if task was cancelled internally
                    if result_data.get("cancelled"):
                        logger.info(f"Batch test {test_id} cancelled, stopping...")
                        # Cancel remaining futures
                        for f in future_to_idx.keys():
                            f.cancel()
                        
                        cancel_event = {
                            "event": "cancelled",
                            "test_id": test_id,
                            "completed": completed_count,
                            "total": total_to_run
                        }
                        yield f"data: {json.dumps(cancel_event, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0)
                        break
                    
                    # Check for cancellation at loop level too
                    if is_batch_test_cancelled(test_id):
                        logger.info(f"Batch test {test_id} cancelled at loop level, stopping...")
                        for f in future_to_idx.keys():
                            f.cancel()
                        
                        cancel_event = {
                            "event": "cancelled",
                            "test_id": test_id,
                            "completed": completed_count,
                            "total": total_to_run
                        }
                        yield f"data: {json.dumps(cancel_event, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0)
                        break
                    
                    idx = result_data["idx"]
                    completed_count += 1
                    
                    if result_data["success"]:
                        metrics = result_data["metrics"]
                        latency_ms = result_data["latency_ms"]
                        context_texts = result_data["context_texts"]
                        
                        # Aggregate
                        for key in sum_metrics.keys():
                            if metrics.get(key) is not None:
                                sum_metrics[key] += metrics[key]
                                cnt_metrics[key] += 1
                        sum_latency += latency_ms
                        cnt_latency += 1
                        
                        # W&B logging
                        if wb_enabled and wb_run is not None:
                            try:
                                wandb.log(
                                    {
                                        "faithfulness": metrics.get("faithfulness"),
                                        "answer_relevancy": metrics.get(
                                            "answer_relevancy"
                                        ),
                                        "context_precision": metrics.get(
                                            "context_precision"
                                        ),
                                        "context_recall": metrics.get(
                                            "context_recall"
                                        ),
                                        "answer_correctness": metrics.get(
                                            "answer_correctness"
                                        ),
                                        "latency_ms": latency_ms,
                                        "contexts_count": len(context_texts),
                                    },
                                    step=idx,
                                )
                            except Exception as e:
                                logger.error(
                                    "W&B log failed for test %s: %s",
                                    idx,
                                    e,
                                )

                            # W&B'ye hemen gönder (buffer'ı flush et)
                            try:
                                if hasattr(wb_run, "_flush"):
                                    wb_run._flush()
                            except Exception:
                                pass

                            if wb_table is not None:
                                wb_table.add_data(
                                    idx,
                                    result_data["question"],
                                    result_data["ground_truth"],
                                    result_data["generated_answer"],
                                    metrics.get("faithfulness"),
                                    metrics.get("answer_relevancy"),
                                    metrics.get("context_precision"),
                                    metrics.get("context_recall"),
                                    metrics.get("answer_correctness"),
                                    latency_ms,
                                    len(context_texts),
                                    None,
                                )
                                # HER ADIMDA TABLE'I LOG ET
                                try:
                                    wandb.log({"results": wb_table}, step=idx)
                                except Exception as e:
                                    logger.error(
                                        "W&B table log failed for test %s: %s",
                                        idx,
                                        e,
                                    )
                        
                        # Send progress
                        progress = {
                            "event": "progress",
                            "index": idx,
                            "total": total_to_run,
                            "completed": completed_count,
                            "result": {
                                "question": result_data["question"],
                                "ground_truth": result_data["ground_truth"],
                                "generated_answer": result_data["generated_answer"],
                                "faithfulness": metrics.get("faithfulness"),
                                "answer_relevancy": metrics.get("answer_relevancy"),
                                "context_precision": metrics.get("context_precision"),
                                "context_recall": metrics.get("context_recall"),
                                "answer_correctness": metrics.get("answer_correctness"),
                                "latency_ms": latency_ms,
                                "retrieved_contexts": result_data["retrieved_contexts"],
                                "retry_count": result_data.get("retry_count", 0),
                                "missing_metrics": result_data.get("missing_metrics"),
                                "low_score_metrics": result_data.get("low_score_metrics"),
                            }
                        }
                        yield f"data: {json.dumps(progress, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0)
                        
                    else:
                        # Send error
                        error_result = {
                            "event": "progress",
                            "index": idx,
                            "total": total_to_run,
                            "completed": completed_count,
                            "result": {
                                "question": result_data["question"],
                                "ground_truth": result_data["ground_truth"],
                                "generated_answer": "",
                                "error_message": result_data["error"],
                            }
                        }
                        
                        if wb_enabled and wb_run is not None and wb_table is not None:
                            wb_table.add_data(
                                idx,
                                result_data["question"],
                                result_data["ground_truth"],
                                "",
                                None, None, None, None, None,
                                0,
                                0,
                                result_data["error"],
                            )
                        
                        yield f"data: {json.dumps(error_result, ensure_ascii=True)}\n\n"
                        await asyncio.sleep(0)
            
            # Log cache statistics
            logger.info(
                f"[CACHE STATS] Batch test completed. "
                f"Embedding cache size: {len(embedding_cache)} entries"
            )
            
            # Only finalize if not cancelled
            if not is_batch_test_cancelled(test_id):
                # W&B finalize
                if wb_enabled and wb_run is not None:
                    avg_metrics = {
                        f"aggregate/avg_{key}": (
                            sum_metrics[key] / cnt_metrics[key]
                            if cnt_metrics[key] > 0 else None
                        )
                        for key in sum_metrics.keys()
                    }
                    avg_metrics["aggregate/avg_latency_ms"] = (
                        sum_latency / cnt_latency if cnt_latency > 0 else None
                    )
                    avg_metrics["total_tests"] = len(data.test_cases)
                    
                    wandb.log(avg_metrics)
                    if wb_table is not None:
                        wandb.log({"results": wb_table})
                    
                    wb_run_url = getattr(wb_run, "url", None)
                    wb_run.finish()
                else:
                    wb_run_url = None
                
                # Send completion
                completion = {
                    "event": "complete",
                    "test_id": test_id,
                    "total": total_to_run,
                    "completed": completed_count,
                    "wandb_url": wb_run_url,
                }
                # ✅ ensure_ascii=True to prevent JSON parse errors
                yield f"data: {json.dumps(completion, ensure_ascii=True)}\n\n"
                await asyncio.sleep(0)
            
        except Exception as e:
            logger.error(f"Batch stream error: {e}")
            error = {"event": "error", "test_id": test_id, "error": str(e)}
            if "wb_run" in locals() and wb_run is not None:
                try:
                    wb_run.finish(exit_code=1)
                except Exception:
                    pass
            # ✅ ensure_ascii=True to prevent JSON parse errors
            yield f"data: {json.dumps(error, ensure_ascii=True)}\n\n"
            await asyncio.sleep(0)
        finally:
            # Clean up
            unregister_batch_test(test_id)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ==================== Batch Test Control Endpoints ====================

@router.post("/batch-test/{test_id}/cancel")
async def cancel_batch_test_endpoint(
    test_id: str,
    current_user: User = Depends(get_current_user),
):
    """Cancel a running batch test."""
    if cancel_batch_test(test_id):
        return {"success": True, "message": f"Batch test {test_id} cancelled"}
    raise HTTPException(status_code=404, detail="Batch test not found or already completed")


@router.post("/batch-test/{test_id}/pause")
async def pause_batch_test_endpoint(
    test_id: str,
    current_user: User = Depends(get_current_user),
):
    """Pause a running batch test."""
    if pause_batch_test(test_id):
        return {"success": True, "message": f"Batch test {test_id} paused"}
    raise HTTPException(status_code=404, detail="Batch test not found or already completed")


@router.post("/batch-test/{test_id}/resume")
async def resume_batch_test_endpoint(
    test_id: str,
    current_user: User = Depends(get_current_user),
):
    """Resume a paused batch test."""
    if resume_batch_test(test_id):
        return {"success": True, "message": f"Batch test {test_id} resumed"}
    raise HTTPException(status_code=404, detail="Batch test not found or already completed")


@router.get("/batch-test/active")
async def get_active_batch_tests_endpoint(
    current_user: User = Depends(get_current_user),
):
    """Get all active batch tests."""
    return {"tests": get_active_batch_tests()}


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


# ==================== Group Management Endpoints ====================

@router.put("/groups/rename")
async def rename_ragas_group(
    course_id: int,
    old_group_name: str,
    new_group_name: str,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Rename all quick test results in a group."""
    verify_course_access(db, course_id, current_user)

    # Check if old group exists
    results = db.query(QuickTestResult).filter(
        QuickTestResult.course_id == course_id,
        QuickTestResult.group_name == old_group_name
    ).all()

    if not results:
        raise HTTPException(status_code=404, detail="Group not found")

    # Update all results in the group
    for result in results:
        result.group_name = new_group_name

    db.commit()

    return {
        "success": True,
        "message": f"Group renamed from '{old_group_name}' to '{new_group_name}'",
        "updated_count": len(results)
    }


@router.delete("/groups/{group_name}")
async def delete_ragas_group(
    course_id: int,
    group_name: str,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Delete all quick test results in a group."""
    verify_course_access(db, course_id, current_user)

    # Check if group exists
    results = db.query(QuickTestResult).filter(
        QuickTestResult.course_id == course_id,
        QuickTestResult.group_name == group_name
    ).all()

    if not results:
        raise HTTPException(status_code=404, detail="Group not found")

    # Delete all results in the group
    for result in results:
        db.delete(result)

    db.commit()

    return {
        "success": True,
        "message": f"Group '{group_name}' deleted with {len(results)} results",
        "deleted_count": len(results)
    }


# ==================== Test Generation Endpoint ====================

@router.post("/test-sets/{test_set_id}/generate-questions")
async def generate_questions_from_documents(
    test_set_id: int,
    num_questions: int = 50,
    persona: Optional[str] = None,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Generate test questions from course documents using RAGAS.
    
    Uses RAGAS TestsetGenerator with Bloom taxonomy to create questions from
    course documents. Supports custom persona for student-level adaptation.
    """
    import httpx
    from app.config import get_settings
    from app.models.db_models import Document
    
    # Get test set and verify access
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    # Get persona from course settings if not provided
    if not persona:
        settings = get_or_create_settings(db, test_set.course_id)
        persona = settings.test_generation_persona or \
            "Öğrenci, temel seviye bilgisayar kullanımı"
    
    # Get processed documents for the course
    documents = db.query(Document).filter(
        Document.course_id == test_set.course_id,
        Document.is_processed == True
    ).all()
    
    if not documents:
        raise HTTPException(
            status_code=400,
            detail="No processed documents found in this course"
        )
    
    # Extract document contents
    doc_contents = []
    for doc in documents:
        if doc.content:
            doc_contents.append(doc.content)
    
    if not doc_contents:
        raise HTTPException(
            status_code=400,
            detail="No document content available"
        )
    
    logger.info(
        f"[TEST GEN] Generating {num_questions} questions from {len(doc_contents)} documents"
    )
    
    # Call RAGAS service
    settings_obj = get_settings()
    ragas_url = getattr(settings_obj, 'ragas_url', 'http://rag-ragas:8001')
    
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:  # 10 min timeout
            response = await client.post(
                f"{ragas_url}/generate-testset",
                json={
                    "documents": doc_contents,
                    "persona": persona,
                    "test_size": num_questions,
                    "distributions": {
                        "simple": 0.4,
                        "reasoning": 0.4,
                        "multi_context": 0.2
                    }
                }
            )
            
            if response.status_code != 200:
                error_detail = response.text
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"RAGAS service error: {error_detail}"
                )
            
            result = response.json()
            generated_questions = result.get("questions", [])
            
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Test generation timed out. Try with fewer questions."
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAGAS service unavailable: {e}"
        )
    
    # Save generated questions to test set
    saved_count = 0
    for q_data in generated_questions:
        # Extract fields from RAGAS response
        question_text = q_data.get("question")
        ground_truth = q_data.get("ground_truth")
        
        if not question_text or not ground_truth:
            logger.warning(f"[TEST GEN] Skipping invalid question: {q_data}")
            continue
        
        # Create TestQuestion
        question = TestQuestion(
            test_set_id=test_set_id,
            question=question_text,
            ground_truth=ground_truth,
            expected_contexts=q_data.get("contexts"),
            question_metadata={
                "generated_by": "ragas",
                "persona": persona,
                "evolution_type": q_data.get("evolution_type"),
                "episode_done": q_data.get("episode_done")
            }
        )
        db.add(question)
        saved_count += 1
    
    db.commit()
    
    logger.info(f"[TEST GEN] Saved {saved_count} questions to test set {test_set_id}")
    
    return {
        "success": True,
        "test_set_id": test_set_id,
        "generated_count": len(generated_questions),
        "saved_count": saved_count,
        "persona_used": persona,
        "llm_used": result.get("llm_used")
    }
