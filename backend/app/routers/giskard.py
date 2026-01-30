"""
Giskard router for RAG testing and hallucination detection.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.models.db_models import User
from app.models import (
    GiskardTestSetCreate,
    GiskardTestSetResponse,
    GiskardQuestionCreate,
    GiskardQuestionResponse,
    GiskardEvaluationRunCreate,
    GiskardEvaluationRunResponse,
    GiskardResultResponse,
    GiskardSummaryResponse,
    GiskardQuickTestRequest,
    GiskardQuickTestResponse,
    GiskardQuickTestResultCreate,
    GiskardQuickTestResultResponse,
    GiskardQuickTestResultListResponse,
    GiskardTestSet,
    GiskardQuestion,
    GiskardRun,
    GiskardResult,
    GiskardSummary,
    GiskardQuickTestResult,
)
from app.models.giskard_schemas import (
    GiskardRAGETGenerateTestsetRequest,
    GiskardRAGETGenerateTestsetResponse,
)
from app.services.auth_service import get_current_teacher
from app.services.course_service import verify_course_access
from app.services.giskard_integration_service import (
    GiskardIntegrationService,
    get_giskard_integration_service,
)

router = APIRouter(prefix="/api/giskard", tags=["giskard"])


# Test Sets
@router.post("/test-sets", response_model=GiskardTestSetResponse)
async def create_test_set(
    test_set: GiskardTestSetCreate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
) -> GiskardTestSetResponse:
    """Create a new test set."""
    verify_course_access(db, test_set.course_id, current_user)
    db_test_set = GiskardTestSet(
        course_id=test_set.course_id,
        name=test_set.name,
        description=test_set.description,
        created_by=current_user.id,
    )
    db.add(db_test_set)
    db.commit()
    db.refresh(db_test_set)
    return GiskardTestSetResponse(
        id=db_test_set.id,
        course_id=db_test_set.course_id,
        name=db_test_set.name,
        description=db_test_set.description,
        created_by=db_test_set.created_by,
        created_at=db_test_set.created_at,
        updated_at=db_test_set.updated_at,
        question_count=db_test_set.question_count,
    )


@router.get("/test-sets/{test_set_id}", response_model=GiskardTestSetResponse)
async def get_test_set(
    test_set_id: int,
    db: Session = Depends(get_db),
) -> GiskardTestSetResponse:
    """Get a test set by ID."""
    test_set = db.query(GiskardTestSet).filter(
        GiskardTestSet.id == test_set_id
    ).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    return GiskardTestSetResponse(
        id=test_set.id,
        course_id=test_set.course_id,
        name=test_set.name,
        description=test_set.description,
        created_by=test_set.created_by,
        created_at=test_set.created_at,
        updated_at=test_set.updated_at,
        question_count=test_set.question_count,
    )


@router.get(
    "/courses/{course_id}/test-sets",
    response_model=List[GiskardTestSetResponse]
)
async def get_course_test_sets(
    course_id: int,
    db: Session = Depends(get_db),
) -> List[GiskardTestSetResponse]:
    """Get all test sets for a course."""
    test_sets = (
        db.query(GiskardTestSet)
        .filter(GiskardTestSet.course_id == course_id)
        .order_by(GiskardTestSet.created_at.desc())
        .all()
    )
    return [
        GiskardTestSetResponse(
            id=ts.id,
            course_id=ts.course_id,
            name=ts.name,
            description=ts.description,
            created_by=ts.created_by,
            created_at=ts.created_at,
            updated_at=ts.updated_at,
            question_count=ts.question_count,
        )
        for ts in test_sets
    ]


@router.delete("/test-sets/{test_set_id}")
async def delete_test_set(
    test_set_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Delete a test set."""
    test_set = db.query(GiskardTestSet).filter(
        GiskardTestSet.id == test_set_id
    ).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    db.delete(test_set)
    db.commit()
    return {"message": "Test set deleted"}


@router.post(
    "/raget/generate-testset",
    response_model=GiskardRAGETGenerateTestsetResponse,
)
async def raget_generate_testset(
    request: GiskardRAGETGenerateTestsetRequest,
    current_user: User = Depends(get_current_teacher),
    giskard_service: GiskardIntegrationService = Depends(
        get_giskard_integration_service
    ),
    db: Session = Depends(get_db),
) -> GiskardRAGETGenerateTestsetResponse:
    verify_course_access(db, request.course_id, current_user)
    try:
        res = giskard_service.generate_raget_testset(
            course_id=request.course_id,
            num_questions=request.num_questions,
            language=request.language,
            agent_description=request.agent_description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return GiskardRAGETGenerateTestsetResponse(
        num_questions=res.get("num_questions", 0),
        samples=res.get("samples", []),
    )


# Questions
@router.post(
    "/test-sets/{test_set_id}/questions",
    response_model=GiskardQuestionResponse
)
async def create_question(
    test_set_id: int,
    question: GiskardQuestionCreate,
    db: Session = Depends(get_db),
) -> GiskardQuestionResponse:
    """Add a question to a test set."""
    test_set = db.query(GiskardTestSet).filter(
        GiskardTestSet.id == test_set_id
    ).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")

    db_question = GiskardQuestion(
        test_set_id=test_set_id,
        question=question.question,
        expected_answer=question.expected_answer,
        question_type=question.question_type,
        question_metadata=question.question_metadata,
    )
    db.add(db_question)
    db.commit()
    db.refresh(db_question)

    # Update test set question count
    test_set.question_count = (
        db.query(GiskardQuestion)
        .filter(GiskardQuestion.test_set_id == test_set_id)
        .count()
    )
    db.commit()

    return GiskardQuestionResponse(
        id=db_question.id,
        test_set_id=db_question.test_set_id,
        question=db_question.question,
        question_type=db_question.question_type,
        expected_answer=db_question.expected_answer,
        question_metadata=db_question.question_metadata,
        created_at=db_question.created_at,
    )


@router.get(
    "/test-sets/{test_set_id}/questions",
    response_model=List[GiskardQuestionResponse]
)
async def get_test_set_questions(
    test_set_id: int,
    db: Session = Depends(get_db),
) -> List[GiskardQuestionResponse]:
    """Get all questions in a test set."""
    questions = (
        db.query(GiskardQuestion)
        .filter(GiskardQuestion.test_set_id == test_set_id)
        .order_by(GiskardQuestion.created_at.asc())
        .all()
    )
    return [
        GiskardQuestionResponse(
            id=q.id,
            test_set_id=q.test_set_id,
            question=q.question,
            question_type=q.question_type,
            expected_answer=q.expected_answer,
            question_metadata=q.question_metadata,
            created_at=q.created_at,
        )
        for q in questions
    ]


@router.delete("/questions/{question_id}")
async def delete_question(
    question_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Delete a question."""
    question = db.query(GiskardQuestion).filter(
        GiskardQuestion.id == question_id
    ).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    test_set_id = question.test_set_id
    db.delete(question)
    db.commit()

    # Update test set question count
    test_set = db.query(GiskardTestSet).filter(
        GiskardTestSet.id == test_set_id
    ).first()
    if test_set:
        test_set.question_count = (
            db.query(GiskardQuestion)
            .filter(GiskardQuestion.test_set_id == test_set_id)
            .count()
        )
        db.commit()

    return {"message": "Question deleted"}


# Evaluation Runs
@router.post(
    "/test-sets/{test_set_id}/runs",
    response_model=GiskardEvaluationRunResponse
)
async def create_evaluation_run(
    test_set_id: int,
    run: GiskardEvaluationRunCreate,
    giskard_service: GiskardIntegrationService = Depends(
        get_giskard_integration_service
    ),
) -> GiskardEvaluationRunResponse:
    """Create and start an evaluation run."""
    test_set = giskard_service.db.query(GiskardTestSet).filter(
        GiskardTestSet.id == test_set_id
    ).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")

    db_run = GiskardRun(
        test_set_id=test_set_id,
        course_id=run.course_id,
        name=run.name,
        config=run.config,
        status="running",
        total_questions=run.total_questions,
        processed_questions=0,
    )
    giskard_service.db.add(db_run)
    giskard_service.db.commit()
    giskard_service.db.refresh(db_run)

    # Start evaluation in background
    giskard_service.run_evaluation(db_run.id)

    return GiskardEvaluationRunResponse(
        id=db_run.id,
        test_set_id=db_run.test_set_id,
        test_set_name=test_set.name,
        course_id=db_run.course_id,
        name=db_run.name,
        status=db_run.status,
        config=db_run.config,
        total_questions=db_run.total_questions,
        processed_questions=db_run.processed_questions,
        started_at=db_run.started_at,
        completed_at=db_run.completed_at,
        error_message=db_run.error_message,
        created_at=db_run.created_at,
        overall_score=None,
        hallucination_rate=None,
        turkish_response_rate=None,
    )


@router.get("/runs/{run_id}", response_model=GiskardEvaluationRunResponse)
async def get_evaluation_run(
    run_id: int,
    db: Session = Depends(get_db),
) -> GiskardEvaluationRunResponse:
    """Get an evaluation run by ID."""
    run = db.query(GiskardRun).filter(GiskardRun.id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=404,
            detail="Evaluation run not found"
        )
    
    # Get summary for metrics if available
    summary = db.query(GiskardSummary).filter(
        GiskardSummary.run_id == run_id
    ).first()
    
    return GiskardEvaluationRunResponse(
        id=run.id,
        test_set_id=run.test_set_id,
        test_set_name=None,
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
        overall_score=summary.overall_score if summary else None,
        hallucination_rate=summary.hallucination_rate if summary else None,
        turkish_response_rate=(
            summary.turkish_response_rate if summary else None
        ),
    )


@router.get(
    "/courses/{course_id}/runs",
    response_model=List[GiskardEvaluationRunResponse]
)
async def get_course_evaluation_runs(
    course_id: int,
    db: Session = Depends(get_db),
) -> List[GiskardEvaluationRunResponse]:
    """Get all evaluation runs for a course."""
    runs = (
        db.query(GiskardRun)
        .join(GiskardTestSet)
        .filter(GiskardTestSet.course_id == course_id)
        .order_by(GiskardRun.created_at.desc())
        .all()
    )
    
    results = []
    for r in runs:
        # Get summary for metrics if available
        summary = db.query(GiskardSummary).filter(
            GiskardSummary.run_id == r.id
        ).first()
        
        results.append(GiskardEvaluationRunResponse(
            id=r.id,
            test_set_id=r.test_set_id,
            test_set_name=None,
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
            overall_score=summary.overall_score if summary else None,
            hallucination_rate=(
                summary.hallucination_rate if summary else None
            ),
            turkish_response_rate=(
                summary.turkish_response_rate if summary else None
            ),
        ))
    
    return results


@router.delete("/runs/{run_id}")
async def delete_evaluation_run(
    run_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Delete an evaluation run."""
    run = db.query(GiskardRun).filter(GiskardRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    db.delete(run)
    db.commit()
    return {"message": "Evaluation run deleted"}


# Results
@router.get(
    "/runs/{run_id}/results",
    response_model=List[GiskardResultResponse]
)
async def get_run_results(
    run_id: int,
    db: Session = Depends(get_db),
) -> List[GiskardResultResponse]:
    """Get all results for an evaluation run."""
    results = (
        db.query(GiskardResult)
        .filter(GiskardResult.run_id == run_id)
        .order_by(GiskardResult.created_at.asc())
        .all()
    )
    return [
        GiskardResultResponse(
            id=r.id,
            run_id=r.run_id,
            question_id=r.question_id,
            question_text=r.question_text,
            expected_answer=r.expected_answer,
            generated_answer=r.generated_answer,
            question_type=r.question_type,
            score=r.score,
            correct_refusal=r.correct_refusal,
            hallucinated=r.hallucinated,
            provided_answer=r.provided_answer,
            language=r.language,
            quality_score=r.quality_score,
            llm_provider=r.llm_provider,
            llm_model=r.llm_model,
            embedding_model=r.embedding_model,
            latency_ms=r.latency_ms,
            error_message=r.error_message,
            created_at=r.created_at,
        )
        for r in results
    ]


@router.get("/runs/{run_id}/summary", response_model=GiskardSummaryResponse)
async def get_run_summary(
    run_id: int,
    db: Session = Depends(get_db),
) -> GiskardSummaryResponse:
    """Get summary statistics for an evaluation run."""
    summary = db.query(GiskardSummary).filter(
        GiskardSummary.run_id == run_id
    ).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    return GiskardSummaryResponse(
        id=summary.id,
        run_id=summary.run_id,
        relevant_count=summary.relevant_count,
        relevant_avg_score=summary.relevant_avg_score,
        relevant_success_rate=summary.relevant_success_rate,
        irrelevant_count=summary.irrelevant_count,
        irrelevant_avg_score=summary.irrelevant_avg_score,
        irrelevant_success_rate=summary.irrelevant_success_rate,
        hallucination_rate=summary.hallucination_rate,
        correct_refusal_rate=summary.correct_refusal_rate,
        language_consistency=summary.language_consistency,
        turkish_response_rate=summary.turkish_response_rate,
        overall_score=summary.overall_score,
        total_questions=summary.total_questions,
        successful_questions=summary.successful_questions,
        failed_questions=summary.failed_questions,
        avg_latency_ms=summary.avg_latency_ms,
        created_at=summary.created_at,
    )


# Quick Test
@router.post("/quick-test", response_model=GiskardQuickTestResponse)
async def quick_test(
    request: GiskardQuickTestRequest,
    giskard_service: GiskardIntegrationService = Depends(
        get_giskard_integration_service
    ),
) -> GiskardQuickTestResponse:
    """Run a quick single-question test."""
    result = giskard_service.run_quick_test(
        course_id=request.course_id,
        question=request.question,
        question_type=request.question_type,
        expected_answer=request.expected_answer,
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
        system_prompt=request.system_prompt,
    )
    return result


# Saved Quick Test Results
@router.post(
    "/quick-test/save",
    response_model=GiskardQuickTestResultResponse
)
async def save_quick_test_result(
    result: GiskardQuickTestResultCreate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
) -> GiskardQuickTestResultResponse:
    """Save a quick test result."""
    # Verify course access
    verify_course_access(db, result.course_id, current_user)
    
    db_result = GiskardQuickTestResult(
        course_id=result.course_id,
        group_name=result.group_name,
        question=result.question,
        question_type=result.question_type,
        expected_answer=result.expected_answer,
        generated_answer=result.generated_answer,
        score=result.score,
        correct_refusal=result.correct_refusal,
        hallucinated=result.hallucinated,
        provided_answer=result.provided_answer,
        language=result.language,
        quality_score=result.quality_score,
        system_prompt=result.system_prompt,
        llm_provider=result.llm_provider,
        llm_model=result.llm_model,
        embedding_model=result.embedding_model,
        latency_ms=result.latency_ms,
        error_message=result.error_message,
        created_by=current_user.id,
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    return GiskardQuickTestResultResponse(
        id=db_result.id,
        course_id=db_result.course_id,
        group_name=db_result.group_name,
        question=db_result.question,
        question_type=db_result.question_type,
        expected_answer=db_result.expected_answer,
        generated_answer=db_result.generated_answer,
        score=db_result.score,
        correct_refusal=db_result.correct_refusal,
        hallucinated=db_result.hallucinated,
        provided_answer=db_result.provided_answer,
        language=db_result.language,
        quality_score=db_result.quality_score,
        system_prompt=db_result.system_prompt,
        llm_provider=db_result.llm_provider,
        llm_model=db_result.llm_model,
        embedding_model=db_result.embedding_model,
        latency_ms=db_result.latency_ms,
        error_message=db_result.error_message,
        created_by=db_result.created_by,
        created_at=db_result.created_at,
    )


@router.get(
    "/courses/{course_id}/quick-test-results",
    response_model=GiskardQuickTestResultListResponse,
)
async def get_quick_test_results(
    course_id: int,
    group_name: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
) -> GiskardQuickTestResultListResponse:
    """Get saved quick test results for a course."""
    # Verify course access
    verify_course_access(db, course_id, current_user)
    
    query = db.query(GiskardQuickTestResult).filter(
        GiskardQuickTestResult.course_id == course_id
    )

    if group_name is not None:
        if group_name == "":
            query = query.filter(GiskardQuickTestResult.group_name == "")
        else:
            query = query.filter(
                GiskardQuickTestResult.group_name == group_name
            )

    total = query.count()

    # Get unique groups
    groups_query = (
        db.query(GiskardQuickTestResult.group_name)
        .filter(GiskardQuickTestResult.course_id == course_id)
        .distinct()
        .all()
    )
    groups = [g[0] for g in groups_query if g[0] is not None]

    results = (
        query.order_by(GiskardQuickTestResult.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return GiskardQuickTestResultListResponse(
        results=[
            GiskardQuickTestResultResponse(
                id=r.id,
                course_id=r.course_id,
                group_name=r.group_name,
                question=r.question,
                question_type=r.question_type,
                expected_answer=r.expected_answer,
                generated_answer=r.generated_answer,
                score=r.score,
                correct_refusal=r.correct_refusal,
                hallucinated=r.hallucinated,
                provided_answer=r.provided_answer,
                language=r.language,
                quality_score=r.quality_score,
                system_prompt=r.system_prompt,
                llm_provider=r.llm_provider,
                llm_model=r.llm_model,
                embedding_model=r.embedding_model,
                latency_ms=r.latency_ms,
                error_message=r.error_message,
                created_by=r.created_by,
                created_at=r.created_at,
            )
            for r in results
        ],
        total=total,
        groups=groups,
    )


@router.delete("/quick-test-results/{result_id}")
async def delete_quick_test_result(
    result_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Delete a quick test result."""
    result = db.query(GiskardQuickTestResult).filter(
        GiskardQuickTestResult.id == result_id
    ).first()
    if not result:
        raise HTTPException(
            status_code=404, detail="Quick test result not found"
        )
    db.delete(result)
    db.commit()
    return {"message": "Quick test result deleted"}
