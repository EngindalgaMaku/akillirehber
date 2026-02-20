"""Duplicate question detection endpoint for RAGAS test sets."""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models.db_models import User, TestSet, TestQuestion
from app.services.auth_service import get_current_teacher
from app.services.course_service import verify_course_access
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ragas", tags=["ragas"])


class DuplicateGroup(BaseModel):
    """Group of similar questions."""
    similarity_score: float
    questions: List[dict]  # List of {id, question, ground_truth}


class FindDuplicatesRequest(BaseModel):
    """Request to find duplicate questions."""
    test_set_id: int
    similarity_threshold: float = 0.85  # Cosine similarity threshold (0-1)


class FindDuplicatesResponse(BaseModel):
    """Response with duplicate question groups."""
    test_set_id: int
    test_set_name: str
    total_questions: int
    duplicate_groups: List[DuplicateGroup]
    total_duplicates: int


class DeleteDuplicatesRequest(BaseModel):
    """Request to delete specific questions."""
    question_ids: List[int]


@router.post("/test-sets/find-duplicates", response_model=FindDuplicatesResponse)
async def find_duplicate_questions(
    data: FindDuplicatesRequest,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Find duplicate/similar questions in a test set using cosine similarity.
    
    Returns groups of similar questions where similarity >= threshold.
    """
    # Get test set
    test_set = db.query(TestSet).filter(TestSet.id == data.test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    # Get all questions
    questions = db.query(TestQuestion).filter(
        TestQuestion.test_set_id == data.test_set_id
    ).all()
    
    if len(questions) < 2:
        return FindDuplicatesResponse(
            test_set_id=test_set.id,
            test_set_name=test_set.name,
            total_questions=len(questions),
            duplicate_groups=[],
            total_duplicates=0,
        )
    
    # Get embeddings for all questions
    embedding_service = EmbeddingService()
    
    # Use course's default embedding model
    from app.services.course_service import get_or_create_settings
    course_settings = get_or_create_settings(db, test_set.course_id)
    embedding_model = course_settings.default_embedding_model
    
    logger.info(f"Computing embeddings for {len(questions)} questions using {embedding_model}")
    
    question_embeddings = []
    for q in questions:
        try:
            embedding = embedding_service.get_embedding(q.question, model=embedding_model)
            question_embeddings.append({
                "id": q.id,
                "question": q.question,
                "ground_truth": q.ground_truth,
                "embedding": embedding,
            })
        except Exception as e:
            logger.error(f"Failed to get embedding for question {q.id}: {e}")
            continue
    
    # Find similar pairs using cosine similarity
    import numpy as np
    
    def cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Track which questions are already in a group
    processed = set()
    duplicate_groups = []
    
    for i in range(len(question_embeddings)):
        if question_embeddings[i]["id"] in processed:
            continue
        
        # Find all questions similar to this one
        similar_questions = [question_embeddings[i]]
        
        for j in range(i + 1, len(question_embeddings)):
            if question_embeddings[j]["id"] in processed:
                continue
            
            similarity = cosine_similarity(
                question_embeddings[i]["embedding"],
                question_embeddings[j]["embedding"]
            )
            
            if similarity >= data.similarity_threshold:
                similar_questions.append(question_embeddings[j])
                processed.add(question_embeddings[j]["id"])
        
        # If we found duplicates, add to groups
        if len(similar_questions) > 1:
            processed.add(question_embeddings[i]["id"])
            
            # Calculate average similarity for the group
            similarities = []
            for k in range(len(similar_questions)):
                for l in range(k + 1, len(similar_questions)):
                    sim = cosine_similarity(
                        similar_questions[k]["embedding"],
                        similar_questions[l]["embedding"]
                    )
                    similarities.append(sim)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
            
            duplicate_groups.append(DuplicateGroup(
                similarity_score=float(avg_similarity),
                questions=[
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "ground_truth": q["ground_truth"],
                    }
                    for q in similar_questions
                ]
            ))
    
    # Sort groups by similarity score (highest first)
    duplicate_groups.sort(key=lambda g: g.similarity_score, reverse=True)
    
    # Count total duplicates (excluding one from each group)
    total_duplicates = sum(len(g.questions) - 1 for g in duplicate_groups)
    
    logger.info(
        f"Found {len(duplicate_groups)} duplicate groups with {total_duplicates} duplicates "
        f"(threshold: {data.similarity_threshold})"
    )
    
    return FindDuplicatesResponse(
        test_set_id=test_set.id,
        test_set_name=test_set.name,
        total_questions=len(questions),
        duplicate_groups=duplicate_groups,
        total_duplicates=total_duplicates,
    )


@router.post("/test-sets/{test_set_id}/delete-questions")
async def delete_multiple_questions(
    test_set_id: int,
    data: DeleteDuplicatesRequest,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """Delete multiple questions from a test set."""
    # Get test set
    test_set = db.query(TestSet).filter(TestSet.id == test_set_id).first()
    if not test_set:
        raise HTTPException(status_code=404, detail="Test set not found")
    
    verify_course_access(db, test_set.course_id, current_user)
    
    # Delete questions
    deleted_count = db.query(TestQuestion).filter(
        TestQuestion.test_set_id == test_set_id,
        TestQuestion.id.in_(data.question_ids)
    ).delete(synchronize_session=False)
    
    db.commit()
    
    logger.info(f"Deleted {deleted_count} questions from test set {test_set_id}")
    
    return {
        "deleted_count": deleted_count,
        "question_ids": data.question_ids,
    }
