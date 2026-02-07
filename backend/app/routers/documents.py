"""Document management API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import User, UserRole
from app.models.schemas import (
    DocumentResponse,
    DocumentListResponse,
    ChunkDBResponse,
    ChunkListResponse,
    ProcessingStatusResponse,
    DiagnosticReportResponse,
    ChunkQualityMetricsResponse,
    SystemDiagnosticsResponse,
)
from app.models.chunking import ChunkingStrategy
from app.services.auth_service import get_current_user, get_current_teacher
from app.services.course_service import verify_course_access, verify_course_ownership
from app.services.document_service import (
    delete_document,
    get_document_by_id,
    get_document_chunks,
    get_document_with_chunk_count,
    get_documents_by_course,
    get_documents_by_user,
    save_chunks_to_db,
    upload_document,
    upload_user_document,
    process_document_chunking,
    retry_document_processing,
    get_document_processing_status,
    delete_document_chunks as del_chunks,
    delete_single_chunk,
)
from app.services.weaviate_service import get_weaviate_service
from app.services.chunker import ChunkerService
from app.services.diagnostic_service import DiagnosticService
from app.services.embedding_provider import EmbeddingProviderError

router = APIRouter(prefix="/api", tags=["documents"])

chunker_service = ChunkerService()

DOC_NOT_FOUND = "Document not found"
OVERLAP_ERROR = "Overlap must be less than chunk_size"


def _verify_document_access(db: Session, document, current_user: User) -> None:
    if document.course_id is not None:
        verify_course_access(db, document.course_id, current_user)
        return

    if current_user.role == UserRole.ADMIN or document.user_id == current_user.id:
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You don't have access to this document",
    )


def _verify_document_ownership(db: Session, document, current_user: User) -> None:
    if document.course_id is not None:
        verify_course_ownership(db, document.course_id, current_user)
        return

    if current_user.role == UserRole.ADMIN or document.user_id == current_user.id:
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You can only access your own documents",
    )


class ChunkingConfigUpdate(BaseModel):
    """Request body for updating chunking configuration."""

    strategy: Optional[ChunkingStrategy] = None
    chunk_size: Optional[int] = Field(None, ge=100, le=5000)
    overlap: Optional[int] = Field(None, ge=0, le=500)
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    embedding_model: Optional[str] = None
    min_chunk_size: Optional[int] = Field(None, ge=50, le=1000)
    max_chunk_size: Optional[int] = Field(None, ge=500, le=10000)


class ProcessDocumentRequest(BaseModel):
    """Request body for document processing."""

    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 500
    overlap: int = 50
    similarity_threshold: Optional[float] = 0.5
    embedding_model: Optional[str] = "openai/text-embedding-3-small"
    min_chunk_size: Optional[int] = 150
    max_chunk_size: Optional[int] = 2000


@router.get("/courses/{course_id}/documents", response_model=DocumentListResponse)
async def list_course_documents(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all documents in a course.

    All authenticated users can view documents.
    """
    verify_course_access(db, course_id, current_user)
    documents = get_documents_by_course(db, course_id)

    doc_responses = [
        DocumentResponse(**get_document_with_chunk_count(db, doc))
        for doc in documents
    ]

    return DocumentListResponse(documents=doc_responses, total=len(doc_responses))


@router.post(
    "/courses/{course_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_course_document(
    course_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Upload a document to a course.

    Only teachers can upload documents to their own courses.

    Supported formats: PDF, Markdown (.md), Word (.docx), Text (.txt)
    """
    course = verify_course_ownership(db, course_id, current_user)
    document = await upload_document(db, course, file, current_user)
    return DocumentResponse(**get_document_with_chunk_count(db, document))


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get document details by ID.

    All authenticated users can view document details.
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    _verify_document_access(db, document, current_user)

    return DocumentResponse(**get_document_with_chunk_count(db, document))


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course_document(
    document_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Delete a document.

    Only teachers can delete documents from their own courses.
    """
    delete_document(db, document_id, current_user)
    return None


@router.get("/documents/{document_id}/chunks", response_model=ChunkListResponse)
async def list_document_chunks(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all chunks for a document.

    All authenticated users can view chunks.
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    _verify_document_access(db, document, current_user)

    chunks = get_document_chunks(db, document_id)
    chunk_responses = [ChunkDBResponse.model_validate(chunk) for chunk in chunks]

    return ChunkListResponse(
        chunks=chunk_responses, total=len(chunk_responses), document_id=document_id
    )


@router.delete("/documents/{document_id}/chunks", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_chunks(
    document_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Delete all chunks for a document.

    Only teachers can delete chunks from their own courses.
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    _verify_document_ownership(db, document, current_user)

    # Delete all chunks
    del_chunks(db, document_id)
    return None


@router.delete("/documents/{document_id}/chunks/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_single_chunk_endpoint(
    document_id: int,
    chunk_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Delete a single chunk by ID.

    Also removes the corresponding vector from Weaviate if it exists.
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    _verify_document_ownership(db, document, current_user)

    deleted = delete_single_chunk(db, chunk_id, document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found",
        )

    # Also delete from Weaviate if document has a course
    if document.course_id:
        try:
            weaviate_svc = get_weaviate_service()
            weaviate_svc.delete_by_chunk_id(document.course_id, chunk_id)
        except Exception:
            pass  # Vector may not exist yet

    return None


@router.post("/documents/{document_id}/process", response_model=ChunkListResponse)
async def process_document(
    document_id: int,
    request: ProcessDocumentRequest,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Process a document with chunking.

    Only teachers can process documents from their own courses.

    - **strategy**: Chunking strategy (recursive, semantic)
    - **chunk_size**: Size of each chunk in characters
    - **overlap**: Overlap between chunks in characters
    - **similarity_threshold**: Threshold for semantic chunking (0-1)
    - **embedding_model**: Model for semantic chunking embeddings
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    _verify_document_ownership(db, document, current_user)

    if not document.content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document has no content to process",
        )

    # Validate parameters
    if request.overlap >= request.chunk_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=OVERLAP_ERROR,
        )

    try:
        chunks = chunker_service.chunk_text(
            document.content,
            request.strategy,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            similarity_threshold=request.similarity_threshold,
            embedding_model=request.embedding_model,
            min_chunk_size=request.min_chunk_size,
            max_chunk_size=request.max_chunk_size,
        )
    except EmbeddingProviderError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except ValueError as e:
        if "No embedding data received" in str(e):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e),
            )
        raise

    # Convert to dict format for saving
    chunks_data = [
        {
            "content": chunk.content,
            "index": chunk.index,
            "start_position": chunk.start_position,
            "end_position": chunk.end_position,
            "char_count": chunk.char_count,
            "has_overlap": chunk.has_overlap,
        }
        for chunk in chunks
    ]

    # Save chunks to database
    db_chunks = save_chunks_to_db(db, document, chunks_data)
    chunk_responses = [ChunkDBResponse.model_validate(chunk) for chunk in db_chunks]

    return ChunkListResponse(
        chunks=chunk_responses, total=len(chunk_responses), document_id=document_id
    )


class RetryProcessingRequest(BaseModel):
    """Request body for retrying document processing."""

    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 500
    overlap: int = 50


@router.get("/documents/{document_id}/status", response_model=ProcessingStatusResponse)
async def get_document_status(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get processing status for a document.

    All authenticated users can view processing status.
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    _verify_document_access(db, document, current_user)

    status_response = get_document_processing_status(db, document_id)
    if not status_response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processing status not found",
        )

    return status_response


@router.post("/documents/{document_id}/retry", response_model=DocumentResponse)
async def retry_document_process(
    document_id: int,
    request: RetryProcessingRequest,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Retry processing for a failed document.

    Only teachers can retry processing for documents in their own courses.
    Document must be in ERROR status to be retried.

    - **strategy**: Chunking strategy (recursive, semantic)
    - **chunk_size**: Size of each chunk in characters
    - **overlap**: Overlap between chunks in characters
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    _verify_document_ownership(db, document, current_user)

    # Validate parameters
    if request.overlap >= request.chunk_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=OVERLAP_ERROR,
        )

    # Retry processing
    updated_document = retry_document_processing(
        db, document_id, request.strategy.value, request.chunk_size, request.overlap
    )

    return DocumentResponse(**get_document_with_chunk_count(db, updated_document))


@router.get("/documents/{document_id}/diagnostics", response_model=DiagnosticReportResponse)
async def get_document_diagnostics(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get diagnostic information for a document.

    All authenticated users can view diagnostic information.
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    _verify_document_access(db, document, current_user)

    # Get diagnostic information from diagnostic service
    diagnostic_service = DiagnosticService(db)
    
    diagnostics = diagnostic_service.get_document_diagnostics(document_id)
    if not diagnostics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Diagnostic information not found",
        )

    return diagnostics


@router.post("/documents/{document_id}/chunk", response_model=ChunkListResponse)
async def chunk_document_with_status(
    document_id: int,
    request: ProcessDocumentRequest,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Process document chunking with status tracking.

    Only teachers can process documents from their own courses.
    This endpoint provides status tracking throughout the chunking process.

    - **strategy**: Chunking strategy (recursive, semantic)
    - **chunk_size**: Size of each chunk in characters
    - **overlap**: Overlap between chunks in characters
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    _verify_document_ownership(db, document, current_user)

    if not document.content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document has no content to process",
        )

    # Validate parameters
    if request.overlap >= request.chunk_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=OVERLAP_ERROR,
        )

    # Process chunking with status tracking
    db_chunks = process_document_chunking(
        db, document_id, request.strategy.value, request.chunk_size, request.overlap
    )

    chunk_responses = [ChunkDBResponse.model_validate(chunk) for chunk in db_chunks]

    return ChunkListResponse(
        chunks=chunk_responses, total=len(chunk_responses), document_id=document_id
    )


@router.put("/documents/{document_id}/chunking-config", response_model=DocumentResponse)
async def update_chunking_config(
    document_id: int,
    config: ChunkingConfigUpdate,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Update chunking configuration for a document and reprocess if requested.

    Only teachers can update chunking configuration for documents in their courses.
    
    - **strategy**: Chunking strategy (recursive, semantic)
    - **chunk_size**: Size of each chunk in characters
    - **overlap**: Overlap between chunks in characters
    - **similarity_threshold**: Threshold for semantic chunking (0-1)
    - **embedding_model**: Model for semantic chunking embeddings
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    _verify_document_ownership(db, document, current_user)

    # Validate configuration
    if config.overlap and config.chunk_size and config.overlap >= config.chunk_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=OVERLAP_ERROR,
        )

    if config.min_chunk_size and config.max_chunk_size and config.min_chunk_size >= config.max_chunk_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="min_chunk_size must be less than max_chunk_size",
        )

    # Store the configuration (this could be enhanced to store in a separate config table)
    # For now, we'll trigger reprocessing with the new configuration
    if not document.content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document has no content to process with new configuration",
        )

    # Use current values as defaults if not provided
    strategy = config.strategy or ChunkingStrategy.RECURSIVE
    chunk_size = config.chunk_size or 500
    overlap = config.overlap or 50

    # Reprocess with new configuration
    updated_document = retry_document_processing(
        db, document_id, strategy.value, chunk_size, overlap
    )

    return DocumentResponse(**get_document_with_chunk_count(db, updated_document))


@router.get("/documents/{document_id}/chunk-quality", response_model=ChunkQualityMetricsResponse)
async def get_chunk_quality_metrics(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get chunk quality metrics for a document.

    All authenticated users can view chunk quality metrics.
    """
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOC_NOT_FOUND,
        )

    _verify_document_access(db, document, current_user)

    # Get chunk quality metrics from diagnostic service
    diagnostic_service = DiagnosticService(db)
    
    metrics = diagnostic_service.get_chunk_quality_metrics(document_id)
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk quality metrics not found",
        )

    return metrics


@router.get("/system/diagnostics", response_model=SystemDiagnosticsResponse)
async def get_system_diagnostics(
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Get system-wide diagnostic information.

    Only teachers can view system diagnostics.
    """
    # Get system diagnostics from diagnostic service
    diagnostic_service = DiagnosticService(db)
    
    diagnostics = diagnostic_service.run_system_diagnostics()
    return diagnostics


@router.get("/system/performance", response_model=dict)
async def get_performance_metrics(
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Get system performance metrics.

    Only teachers can view performance metrics.
    """
    # Get performance metrics from diagnostic service
    diagnostic_service = DiagnosticService(db)
    
    metrics = diagnostic_service.get_performance_metrics()
    return metrics


@router.post("/system/validate-pipeline", response_model=dict)
async def validate_processing_pipeline(
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Validate entire processing pipeline health.

    Only teachers can run pipeline validation.
    """
    # Validate processing pipeline
    diagnostic_service = DiagnosticService(db)
    
    validation_results = diagnostic_service.validate_processing_pipeline()
    return validation_results


@router.get("/users/documents", response_model=DocumentListResponse)
async def list_user_documents(
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    List all documents for the current user (across all courses).

    Only teachers can view their own documents.
    """
    documents = get_documents_by_user(db, current_user.id)
    
    doc_responses = [
        DocumentResponse(**get_document_with_chunk_count(db, doc))
        for doc in documents
    ]
    
    return DocumentListResponse(documents=doc_responses, total=len(doc_responses))


@router.post("/users/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_user_document_endpoint(
    file: UploadFile = File(...),
    course_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Upload a document for the user (not tied to a specific course).

    Only teachers can upload documents.
    
    Supported formats: PDF, Markdown (.md), Word (.docx), Text (.txt)
    """
    document = await upload_user_document(db, file, current_user, course_id)
    return DocumentResponse(**get_document_with_chunk_count(db, document))


@router.delete("/courses/{course_id}/weaviate-collection", status_code=status.HTTP_204_NO_CONTENT)
async def reset_course_weaviate_collection(
    course_id: int,
    current_user: User = Depends(get_current_teacher),
    db: Session = Depends(get_db),
):
    """
    Reset Weaviate collection for a course.
    
    This will delete all vectors from Weaviate for this course.
    Use this when changing embedding models to avoid vector dimension mismatches.
    
    Only teachers can reset collections for their own courses.
    """
    verify_course_ownership(db, course_id, current_user)
    
    weaviate_service = get_weaviate_service()
    deleted_count = weaviate_service.delete_course_collection(course_id)
    
    return None