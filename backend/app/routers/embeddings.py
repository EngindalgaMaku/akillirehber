"""Embedding management API endpoints."""

import asyncio
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
from app.services.vector_store_factory import get_vector_store_for_course
from app.services.vector_store_interface import ChunkWithEmbedding

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
        
        print(f"=== Embedding Debug for Document {document_id} ===")
        print(f"Model: {request.model}")
        print(f"Text count: {len(texts)}")
        print(f"First text preview: {texts[0][:100] if texts else 'No texts'}...")
        
        try:
            embeddings = embedding_service.get_embeddings(
                texts, 
                model=request.model,
                input_type="document"
            )
            print(f"Embeddings generated: {len(embeddings)}")
            print(f"First embedding length: {len(embeddings[0]) if embeddings else 0}")
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            raise
        
        print("=== End Embedding Debug ===")

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

        # Get vector store for this course
        vector_store = get_vector_store_for_course(document.course_id, db)
        vector_store_name = vector_store.__class__.__name__
        
        print(f"Using vector store: {vector_store_name} for course {document.course_id}")
        
        # Store the embeddings
        vector_store.store_chunks(
            course_id=document.course_id,
            document_id=document_id,
            chunks=chunks_with_embeddings
        )
        print(f"Successfully stored {len(chunks_with_embeddings)} embeddings in {vector_store_name}")

        # Verify vectors were actually stored - retry to handle eventual consistency
        expected_count = len(chunks_with_embeddings)
        actual_count = 0
        max_retries = 5
        for attempt in range(max_retries):
            actual_count = vector_store.get_document_vector_count(
                document.course_id, document_id
            )
            print(f"Verification attempt {attempt + 1}/{max_retries}: {actual_count}/{expected_count} vectors found")
            if actual_count >= expected_count:
                break
            if actual_count > 0 and attempt >= 2:
                # Partial persistence is enough to proceed
                break
            await asyncio.sleep(1.5 * (attempt + 1))
        
        print(f"Final verification: {actual_count} vectors found in {vector_store_name} for document {document_id}")
        
        if actual_count == 0:
            document.embedding_status = EmbeddingStatus.ERROR
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    f"Vectors were not persisted in {vector_store_name}. "
                    f"Expected {expected_count} but found 0. "
                    "Please check vector store connection and try again."
                )
            )
        
        if actual_count < expected_count:
            print(
                f"⚠️ WARNING: Only {actual_count}/{expected_count} "
                f"vectors persisted for document {document_id}"
            )

        # Update document status
        document.embedding_status = EmbeddingStatus.COMPLETED
        document.embedding_model = request.model
        document.embedded_at = datetime.utcnow()
        document.vector_count = actual_count
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
    Delete all vectors for a document from the configured vector store.

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

    # Delete from vector store (Weaviate or ChromaDB based on settings)
    vector_store = get_vector_store_for_course(document.course_id, db)
    vector_store.delete_by_document(document.course_id, document_id)

    # Update document status
    document.embedding_status = EmbeddingStatus.PENDING
    document.embedding_model = None
    document.embedded_at = None
    document.vector_count = 0
    db.commit()

    # If no other documents in this course have vectors, drop the
    # entire collection so the vector index is reset.  This avoids
    # dimension-mismatch errors when re-embedding with a different model.
    from app.services.document_service import get_documents_by_course
    course_docs = get_documents_by_course(db, document.course_id)
    has_remaining_vectors = any(
        d.vector_count and d.vector_count > 0
        for d in course_docs
        if d.id != document_id
    )
    if not has_remaining_vectors:
        vector_store.delete_by_course(document.course_id)
        print(
            f"No vectors remaining for course {document.course_id}, "
            f"collection dropped."
        )

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
    Get vector count for a document from the configured vector store.
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    # Verify teacher owns the course
    verify_course_ownership(db, document.course_id, current_user)

    # Get count from vector store (Weaviate or ChromaDB based on settings)
    vector_store = get_vector_store_for_course(document.course_id, db)
    count = vector_store.get_document_vector_count(
        document.course_id, document_id
    )

    return VectorCountResponse(document_id=document_id, vector_count=count)


@router.delete(
    "/courses/{course_id}/collection",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_course_collection(
    course_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Delete the entire vector collection for a course.

    This removes the Weaviate collection and resets all document
    embedding statuses.  Useful when switching embedding models.
    """
    verify_course_ownership(db, course_id, current_user)

    vector_store = get_vector_store_for_course(course_id, db)
    vector_store.delete_by_course(course_id)

    # Reset all documents in this course
    from app.services.document_service import get_documents_by_course
    docs = get_documents_by_course(db, course_id)
    for doc in docs:
        doc.embedding_status = EmbeddingStatus.PENDING
        doc.embedding_model = None
        doc.embedded_at = None
        doc.vector_count = 0
    db.commit()

    return None
