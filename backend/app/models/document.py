"""Pydantic models for document upload and processing."""

from enum import Enum

from pydantic import BaseModel, Field


class SupportedFileType(str, Enum):
    """Supported file types for document upload."""

    PDF = "pdf"
    MARKDOWN = "md"
    DOCX = "docx"
    TEXT = "txt"


class DocumentMetadata(BaseModel):
    """Metadata about an uploaded document."""

    file_name: str = Field(..., description="Original file name")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    char_count: int = Field(
        ..., ge=0, description="Character count of extracted text"
    )
    file_type: SupportedFileType = Field(..., description="Detected file type")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload API."""

    text: str = Field(..., description="Extracted text content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")


class DocumentUploadError(BaseModel):
    """Error response for document upload failures."""

    detail: str = Field(..., description="Error message")
    supported_formats: list[str] = Field(
        default=["pdf", "md", "docx", "txt"],
        description="List of supported file formats"
    )
