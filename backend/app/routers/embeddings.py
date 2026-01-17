"""Embedding management API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import User, Document, EmbeddingStatus
from app.models.schemas import EmbedRequest, EmbedResponse, VectorCountResponse
from app.services.auth_service import get_current_teacher
from app.services.course_service import verify_course_ownership
from app.services.document_service import get_document_by_id, get_document_chunks
from app.services.embedding_service import get_embedding_service
from app.services.weaviate_service import (
    get_weaviate_service,
    ChunkWithEmbedding,
)

router = APIRouter(prefix="/api", tags=["embeddings"])

DOC_NOT_FOUND = "Document not found"


@router.post("/documents/{document_id}/embed", response_model=EmbedResponse)
async def embed_document(
    document_id: int,
    request: EmbedRequest,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Generate embeddings for document chunks and store in Weaviate.

    Only teachers can embed documents from their own courses.
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    # Verify teacher owns the course
    verify_course_ownership(db, document.course_id, current_user)

    # Get chunks
    chunks = get_document_chunks(db, document_id)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document has no chunks. Process the document first.",
        )

    # Update status to processing
    document.embedding_status = EmbeddingStatus.PROCESSING
    db.commit()

    try:
        # Get embeddings
        embedding_service = get_embedding_service()
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_service.get_embeddings(texts, model=request.model)

        # Prepare chunks with embeddings
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:  # Skip empty embeddings
                chunks_with_embeddings.append(ChunkWithEmbedding(
                    chunk_id=chunk.id,
                    document_id=document_id,
                    content=chunk.content,
                    chunk_index=chunk.index,
                    vector=embedding
                ))

        # Store in Weaviate
        weaviate_service = get_weaviate_service()
        weaviate_service.store_chunks(
            course_id=document.course_id,
            document_id=document_id,
            chunks=chunks_with_embeddings
        )

        # Update document status
        document.embedding_status = EmbeddingStatus.COMPLETED
        document.embedding_model = request.model
        document.embedded_at = datetime.utcnow()
        document.vector_count = len(chunks_with_embeddings)
        db.commit()

        return EmbedResponse(
            document_id=document_id,
            status=EmbeddingStatus.COMPLETED,
            vector_count=len(chunks_with_embeddings),
            model=request.model
        )

    except Exception as e:
        document.embedding_status = EmbeddingStatus.ERROR
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {str(e)}"
        )


@router.delete(
    "/documents/{document_id}/vectors",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_document_vectors(
    document_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Delete all vectors for a document from Weaviate.

    Only teachers can delete vectors from their own courses.
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    # Verify teacher owns the course
    verify_course_ownership(db, document.course_id, current_user)

    # Delete from Weaviate
    weaviate_service = get_weaviate_service()
    weaviate_service.delete_by_document(document.course_id, document_id)

    # Update document status
    document.embedding_status = EmbeddingStatus.PENDING
    document.embedding_model = None
    document.embedded_at = None
    document.vector_count = 0
    db.commit()

    return None


@router.get(
    "/documents/{document_id}/vectors/count",
    response_model=VectorCountResponse
)
async def get_document_vector_count(
    document_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Get vector count for a document.
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    # Verify teacher owns the course
    verify_course_ownership(db, document.course_id, current_user)

    # Get count from Weaviate
    weaviate_service = get_weaviate_service()
    count = weaviate_service.get_document_vector_count(
        document.course_id, document_id
    )

    return VectorCountResponse(document_id=document_id, vector_count=count)
