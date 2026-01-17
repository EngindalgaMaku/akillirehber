"""Document management service with database integration."""

import uuid
import logging
from typing import List, Optional

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.db_models import Document, Chunk, Course, User, ProcessingStatusEnum
from app.models.schemas import DocumentResponse
from app.services.document_processor import DocumentProcessor, DocumentProcessingError
from app.services.processing_status_manager import ProcessingStatusManager


document_processor = DocumentProcessor()
logger = logging.getLogger(__name__)


def get_document_by_id(db: Session, document_id: int) -> Optional[Document]:
    """Get a document by ID."""
    return db.query(Document).filter(Document.id == document_id).first()


def get_documents_by_course(db: Session, course_id: int) -> List[Document]:
    """Get all documents for a course."""
    return (
        db.query(Document)
        .filter(Document.course_id == course_id)
        .order_by(Document.created_at.desc())
        .all()
    )


def get_document_with_chunk_count(db: Session, document: Document) -> dict:
    """Get document data with chunk count."""
    chunk_count = (
        db.query(func.count(Chunk.id))
        .filter(Chunk.document_id == document.id)
        .scalar()
    )
    return {
        "id": document.id,
        "filename": document.filename,
        "original_filename": document.original_filename,
        "file_type": document.file_type,
        "file_size": document.file_size,
        "char_count": document.char_count,
        "is_processed": document.is_processed,
        "course_id": document.course_id,
        "created_at": document.created_at,
        "chunk_count": chunk_count or 0,
        "embedding_status": document.embedding_status,
        "embedding_model": document.embedding_model,
        "embedded_at": document.embedded_at,
        "vector_count": document.vector_count,
    }


async def upload_document(
    db: Session, course: Course, file: UploadFile, user: User
) -> Document:
    """Upload and process a document for a course."""
    # Read file content
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty",
        )

    # Create document record first
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    
    db_document = Document(
        filename=unique_filename,
        original_filename=file.filename,
        file_type="unknown",  # Will be updated after processing
        file_size=len(content),
        char_count=0,  # Will be updated after processing
        content="",  # Will be updated after processing
        course_id=course.id,
        is_processed=False,
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    # Initialize processing status manager
    status_manager = ProcessingStatusManager(db)
    
    try:
        # Create initial processing status
        status_manager.create_status(db_document.id, ProcessingStatusEnum.PENDING)
        
        # Update status to extracting
        status_manager.update_status(db_document.id, ProcessingStatusEnum.EXTRACTING)
        
        # Process document to extract text
        text, file_type = document_processor.process_document(
            content, file.filename, file.content_type
        )
        
        # Update document with extracted information
        db_document.file_type = file_type.value
        db_document.char_count = len(text)
        db_document.content = text
        
        # Update status to completed (chunking will be handled separately)
        status_manager.update_status(db_document.id, ProcessingStatusEnum.COMPLETED)
        
        db.commit()
        db.refresh(db_document)
        
        return db_document
        
    except DocumentProcessingError as e:
        # Update status to error with details
        status_manager.update_status(
            db_document.id, 
            ProcessingStatusEnum.ERROR,
            error_message=str(e),
            error_details={
                "error_type": e.error_type,
                "context": e.context
            }
        )
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        # Handle unexpected errors
        status_manager.update_status(
            db_document.id,
            ProcessingStatusEnum.ERROR,
            error_message=f"Unexpected error: {str(e)}",
            error_details={"exception_type": type(e).__name__}
        )
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during document processing",
        )


def delete_document(db: Session, document_id: int, user: User) -> bool:
    """Delete a document and its chunks."""
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Verify user owns the course
    course = db.query(Course).filter(Course.id == document.course_id).first()
    if course.teacher_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete documents from your own courses",
        )

    # Delete vectors from Weaviate
    try:
        from app.services.weaviate_service import get_weaviate_service
        weaviate_service = get_weaviate_service()
        weaviate_service.delete_by_document(document.course_id, document_id)
    except Exception:
        pass  # Continue even if Weaviate deletion fails

    # Delete document (chunks will be cascade deleted)
    db.delete(document)
    db.commit()
    return True


def get_document_chunks(db: Session, document_id: int) -> List[Chunk]:
    """Get all chunks for a document."""
    return (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.index)
        .all()
    )


def delete_document_chunks(db: Session, document_id: int) -> int:
    """Delete all chunks for a document."""
    document = get_document_by_id(db, document_id)
    if document:
        document.is_processed = False
    count = db.query(Chunk).filter(Chunk.document_id == document_id).delete()
    db.commit()
    return count


def save_chunks_to_db(
    db: Session, document: Document, chunks_data: List[dict]
) -> List[Chunk]:
    """Save chunks to database."""
    # Delete existing chunks
    db.query(Chunk).filter(Chunk.document_id == document.id).delete()

    # Create new chunks
    db_chunks = []
    for chunk_data in chunks_data:
        db_chunk = Chunk(
            content=chunk_data["content"],
            index=chunk_data["index"],
            start_position=chunk_data["start_position"],
            end_position=chunk_data["end_position"],
            char_count=chunk_data["char_count"],
            has_overlap=chunk_data.get("has_overlap", False),
            document_id=document.id,
        )
        db.add(db_chunk)
        db_chunks.append(db_chunk)

    # Mark document as processed
    document.is_processed = True
    db.commit()

    # Refresh chunks to get IDs
    for chunk in db_chunks:
        db.refresh(chunk)

    return db_chunks


def process_document_chunking(
    db: Session, document_id: int, chunking_strategy: str = "recursive",
    chunk_size: int = 500, overlap: int = 50
) -> List[Chunk]:
    """Process document chunking with status tracking and diagnostics.
    
    Args:
        db: Database session
        document_id: ID of the document to chunk
        chunking_strategy: Strategy to use for chunking
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of created chunks
        
    Raises:
        HTTPException: If document not found or processing fails
    """
    # Get document
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Initialize status manager
    status_manager = ProcessingStatusManager(db)
    
    try:
        # Update status to chunking
        status_manager.update_status(document_id, ProcessingStatusEnum.CHUNKING)
        
        # Import chunker service and configuration
        from app.services.chunker import ChunkerService, ChunkingConfig
        from app.models.chunking import ChunkingStrategy
        
        chunker_service = ChunkerService()
        
        # Create chunking configuration
        try:
            strategy_enum = ChunkingStrategy(chunking_strategy)
        except ValueError:
            strategy_enum = ChunkingStrategy.RECURSIVE  # Default fallback
        
        config = ChunkingConfig(
            strategy=strategy_enum,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        # Perform chunking with diagnostics
        chunks, diagnostics = chunker_service.chunk_with_diagnostics(
            text=document.content,
            config=config
        )
        
        # Check for chunking errors
        if diagnostics.error_message:
            raise Exception(f"Chunking failed: {diagnostics.error_message}")
        
        # Log diagnostics for monitoring
        logger.info(
            f"Document {document_id} chunked successfully: "
            f"{diagnostics.total_chunks} chunks, "
            f"processing time: {diagnostics.processing_time:.2f}s, "
            f"quality score: {diagnostics.quality_score:.2f}"
        )
        
        # Log performance warnings if any
        if diagnostics.performance_warnings:
            for warning in diagnostics.performance_warnings:
                logger.warning(f"Document {document_id} chunking warning: {warning}")
        
        # Convert chunks to dict format for database
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "content": chunk.content,
                "index": chunk.index,
                "start_position": chunk.start_position,
                "end_position": chunk.end_position,
                "char_count": chunk.char_count,
                "has_overlap": chunk.has_overlap
            })
        
        # Save chunks to database
        db_chunks = save_chunks_to_db(db, document, chunks_data)
        
        # Generate and save chunk quality metrics using diagnostic service
        try:
            from app.services.diagnostic_service import DiagnosticService
            diagnostic_service = DiagnosticService(db)
            diagnostic_service._generate_chunk_quality_metrics(document_id, chunks)
        except Exception as e:
            logger.warning(f"Failed to generate chunk quality metrics for document {document_id}: {e}")
        
        # Update status to completed
        status_manager.update_status(document_id, ProcessingStatusEnum.COMPLETED)
        
        return db_chunks
        
    except Exception as e:
        # Update status to error with detailed diagnostics
        error_details = {
            "stage": "chunking",
            "strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "exception_type": type(e).__name__,
            "text_length": len(document.content) if document.content else 0
        }
        
        # Add diagnostics if available
        if 'diagnostics' in locals() and diagnostics:
            error_details.update({
                "processing_time": diagnostics.processing_time,
                "diagnostics_error": diagnostics.error_message,
                "diagnostics_details": diagnostics.error_details
            })
        
        status_manager.update_status(
            document_id,
            ProcessingStatusEnum.ERROR,
            error_message=f"Chunking failed: {str(e)}",
            error_details=error_details
        )
        
        logger.error(
            f"Document {document_id} chunking failed: {str(e)}", 
            extra={"error_details": error_details}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking failed: {str(e)}"
        )


def retry_document_processing(
    db: Session, document_id: int, chunking_strategy: str = "recursive",
    chunk_size: int = 500, overlap: int = 50
) -> Document:
    """Retry document processing with new parameters.
    
    Args:
        db: Database session
        document_id: ID of the document to retry
        chunking_strategy: Strategy to use for chunking
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        Updated document
        
    Raises:
        HTTPException: If document not found or not in error state
    """
    # Get document
    document = get_document_by_id(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Initialize status manager
    status_manager = ProcessingStatusManager(db)
    
    # Mark for retry (this validates the document is in ERROR state)
    try:
        status_manager.mark_for_retry(document_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Process chunking with new parameters
    process_document_chunking(db, document_id, chunking_strategy, chunk_size, overlap)
    
    # Refresh document
    db.refresh(document)
    return document


def get_document_processing_status(db: Session, document_id: int):
    """Get processing status for a document.
    
    Args:
        db: Database session
        document_id: ID of the document
        
    Returns:
        Processing status response or None
    """
    status_manager = ProcessingStatusManager(db)
    return status_manager.get_status(document_id)
