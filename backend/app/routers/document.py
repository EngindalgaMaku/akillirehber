"""API router for document upload and processing."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.document import (
    DocumentMetadata,
    DocumentUploadResponse,
)
from app.services.document_processor import DocumentProcessor, DocumentProcessingError

router = APIRouter(prefix="/api", tags=["document"])

# Initialize the document processor
document_processor = DocumentProcessor()

# Maximum file size: 10MB
MAX_FILE_SIZE = 100 * 1024 * 1024


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...)
) -> DocumentUploadResponse:
    """Upload a document and extract text content.

    Supports PDF, Markdown (.md), Word (.docx), and plain text (.txt).

    Args:
        file: Uploaded file

    Returns:
        Extracted text and document metadata

    Raises:
        HTTPException: If file type is unsupported or processing fails
    """
    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=422,
            detail="Filename is required"
        )

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read uploaded file: {str(e)}"
        ) from e

    # Check file size
    file_size = len(content)
    if file_size == 0:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is empty"
        )

    max_mb = MAX_FILE_SIZE // (1024 * 1024)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_mb}MB"
        )

    # Process document
    try:
        text, file_type = document_processor.process_document(
            content=content,
            filename=file.filename,
            content_type=file.content_type
        )
    except DocumentProcessingError as e:
        # Handle our custom processing errors
        if e.error_type in ["UnsupportedFormat", "ValidationError"]:
            raise HTTPException(
                status_code=422,
                detail=str(e)
            ) from e
        elif e.error_type == "MissingDependency":
            raise HTTPException(
                status_code=500,
                detail=str(e)
            ) from e
        else:
            # Other processing errors
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {str(e)}"
            ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        ) from e
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        ) from e

    # Build response
    metadata = DocumentMetadata(
        file_name=file.filename,
        file_size=file_size,
        char_count=len(text),
        file_type=file_type
    )

    return DocumentUploadResponse(
        text=text,
        metadata=metadata
    )
