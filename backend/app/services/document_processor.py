"""Document processing service for text extraction."""

import io
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

from app.models.document import SupportedFileType

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    
    def __init__(self, message: str, error_type: str = "ProcessingError", 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.context = context or {}


class DocumentProcessor:
    """Service for extracting text from uploaded documents."""

    SUPPORTED_EXTENSIONS = {
        ".pdf": SupportedFileType.PDF,
        ".md": SupportedFileType.MARKDOWN,
        ".markdown": SupportedFileType.MARKDOWN,
        ".docx": SupportedFileType.DOCX,
        ".txt": SupportedFileType.TEXT,
    }

    SUPPORTED_MIME_TYPES = {
        "application/pdf": SupportedFileType.PDF,
        "text/markdown": SupportedFileType.MARKDOWN,
        "text/x-markdown": SupportedFileType.MARKDOWN,
        "application/vnd.openxmlformats-officedocument"
        ".wordprocessingml.document": SupportedFileType.DOCX,
        "text/plain": SupportedFileType.TEXT,
    }

    def detect_file_type(
        self, filename: str, content_type: str | None = None
    ) -> SupportedFileType | None:
        """Detect file type from filename extension or content type.

        Args:
            filename: Original filename with extension
            content_type: MIME type from upload headers

        Returns:
            Detected file type or None if unsupported
        """
        # Try extension first
        ext = ""
        if "." in filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower()
        if ext in self.SUPPORTED_EXTENSIONS:
            return self.SUPPORTED_EXTENSIONS[ext]

        # Fall back to content type
        if content_type and content_type in self.SUPPORTED_MIME_TYPES:
            return self.SUPPORTED_MIME_TYPES[content_type]

        return None

    def extract_text(
        self, content: bytes, file_type: SupportedFileType
    ) -> str:
        """Extract text from document content.

        Args:
            content: Raw file bytes
            file_type: Detected file type

        Returns:
            Extracted text content

        Raises:
            DocumentProcessingError: If extraction fails
        """
        try:
            if file_type == SupportedFileType.PDF:
                return self._extract_pdf(content)
            if file_type == SupportedFileType.MARKDOWN:
                return self._extract_markdown(content)
            if file_type == SupportedFileType.DOCX:
                return self._extract_docx(content)
            if file_type == SupportedFileType.TEXT:
                return self._extract_text(content)
            
            raise DocumentProcessingError(
                f"Unsupported file type: {file_type}",
                error_type="UnsupportedFileType",
                context={"file_type": file_type.value}
            )
        except DocumentProcessingError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Wrap other exceptions
            raise DocumentProcessingError(
                f"Text extraction failed: {str(e)}",
                error_type="ExtractionError",
                context={
                    "file_type": file_type.value,
                    "original_error": str(e),
                    "exception_type": type(e).__name__
                }
            ) from e

    def _extract_pdf(self, content: bytes) -> str:
        """Extract text from PDF using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise DocumentProcessingError(
                "PyMuPDF is required for PDF processing. "
                "Install it with: pip install pymupdf",
                error_type="MissingDependency",
                context={"required_package": "pymupdf"}
            ) from e

        text_parts = []
        try:
            # Open PDF from bytes
            with fitz.open(stream=content, filetype="pdf") as doc:
                if len(doc) == 0:
                    raise DocumentProcessingError(
                        "PDF document contains no pages",
                        error_type="EmptyDocument",
                        context={"page_count": 0}
                    )

                for page_num, page in enumerate(doc):
                    try:
                        page_text = page.get_text()
                        text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")

        except fitz.FileDataError as e:
            raise DocumentProcessingError(
                f"Invalid or corrupted PDF file: {str(e)}",
                error_type="CorruptedFile",
                context={"fitz_error": str(e)}
            ) from e
        except fitz.EmptyFileError as e:
            raise DocumentProcessingError(
                "PDF file is empty or has no content",
                error_type="EmptyFile",
                context={"fitz_error": str(e)}
            ) from e
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to extract text from PDF: {str(e)}",
                error_type="PDFExtractionError",
                context={
                    "fitz_error": str(e),
                    "error_type": type(e).__name__
                }
            ) from e

        extracted_text = "\n".join(text_parts)

        # Validate extraction results
        if not extracted_text.strip():
            raise DocumentProcessingError(
                "PDF text extraction resulted in empty content",
                error_type="EmptyExtraction",
                context={"page_count": len(text_parts)}
            )

        return extracted_text

    def _extract_markdown(self, content: bytes) -> str:
        """Extract text from Markdown file.

        For markdown, we return the raw text as-is since it's text.
        The markdown formatting is preserved for downstream processing.
        """
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return content.decode("latin-1")
            except Exception as e:
                raise DocumentProcessingError(
                    f"Failed to decode markdown file: {str(e)}",
                    error_type="EncodingError",
                    context={
                        "attempted_encodings": ["utf-8", "latin-1"],
                        "content_size": len(content)
                    }
                ) from e

    def _extract_docx(self, content: bytes) -> str:
        """Extract text from Word document using python-docx."""
        try:
            from docx import Document
        except ImportError as e:
            raise DocumentProcessingError(
                "python-docx is required for Word document processing. "
                "Install it with: pip install python-docx",
                error_type="MissingDependency",
                context={"required_package": "python-docx"}
            ) from e

        try:
            doc = Document(io.BytesIO(content))
            paragraphs = [para.text for para in doc.paragraphs]
            extracted_text = "\n".join(paragraphs)
            
            # Validate extraction
            if not extracted_text.strip():
                raise DocumentProcessingError(
                    "Word document text extraction resulted in empty content",
                    error_type="EmptyExtraction",
                    context={"paragraph_count": len(paragraphs)}
                )
            
            return extracted_text
        except Exception as e:
            if isinstance(e, DocumentProcessingError):
                raise
            raise DocumentProcessingError(
                f"Failed to extract text from Word document: {str(e)}",
                error_type="DocxExtractionError",
                context={
                    "docx_error": str(e),
                    "error_type": type(e).__name__
                }
            ) from e

    def _extract_text(self, content: bytes) -> str:
        """Extract text from plain text file."""
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return content.decode("latin-1")
            except Exception as e:
                raise DocumentProcessingError(
                    f"Failed to decode text file: {str(e)}",
                    error_type="EncodingError",
                    context={
                        "attempted_encodings": ["utf-8", "latin-1"],
                        "content_size": len(content)
                    }
                ) from e

    def process_document(
        self, content: bytes, filename: str, content_type: Optional[str] = None
    ) -> Tuple[str, SupportedFileType]:
        """Process an uploaded document and extract text (legacy method).

        Args:
            content: Raw file bytes
            filename: Original filename
            content_type: MIME type from upload headers

        Returns:
            Tuple of (extracted_text, file_type)

        Raises:
            DocumentProcessingError: If processing fails
        """
        # Use the new method but return only the basic results for backward compatibility
        text, file_type, _ = self.process_document_with_diagnostics(
            content, filename, content_type
        )
        return text, file_type

    def validate_pdf_content(self, content: bytes) -> Dict[str, Any]:
        """Validate PDF content before processing.
        
        Args:
            content: Raw PDF bytes
            
        Returns:
            Dictionary with validation results
            
        Raises:
            DocumentProcessingError: If validation fails
        """
        validation_result = {
            "is_valid": False,
            "file_size": len(content),
            "error_details": None,
            "warnings": []
        }
        
        # Check minimum file size
        if len(content) < 100:  # Minimum viable PDF size
            raise DocumentProcessingError(
                "PDF file is too small to be valid",
                error_type="ValidationError",
                context={"file_size": len(content)}
            )
        
        # Check PDF header
        if not content.startswith(b'%PDF-'):
            raise DocumentProcessingError(
                "File does not have valid PDF header",
                error_type="ValidationError",
                context={"header": content[:10].hex()}
            )
        
        # Check for PDF trailer
        if b'%%EOF' not in content[-1024:]:
            validation_result["warnings"].append(
                "PDF trailer not found in expected location"
            )
        
        # Check file size limits (10MB default)
        max_size = 100 * 1024 * 1024  # 10MB
        if len(content) > max_size:
            raise DocumentProcessingError(
                f"PDF file too large: {len(content)} bytes (max: {max_size})",
                error_type="ValidationError",
                context={"file_size": len(content), "max_size": max_size}
            )
        
        validation_result["is_valid"] = True
        return validation_result

    def extract_text_with_retry(self, content: bytes, file_type: SupportedFileType, 
                               max_retries: int = 3) -> str:
        """Extract text with retry logic for transient failures.
        
        Args:
            content: Raw file bytes
            file_type: Detected file type
            max_retries: Maximum number of retry attempts
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentProcessingError: If extraction fails after all retries
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff for retries
                    wait_time = 2 ** (attempt - 1)
                    logger.info(f"Retrying text extraction (attempt {attempt + 1}/{max_retries + 1}) "
                               f"after {wait_time}s delay")
                    time.sleep(wait_time)
                
                return self.extract_text(content, file_type)
                
            except Exception as e:
                last_error = e
                error_context = {
                    "attempt": attempt + 1,
                    "max_retries": max_retries + 1,
                    "file_type": file_type.value,
                    "content_size": len(content)
                }
                
                logger.warning(
                    f"Text extraction attempt {attempt + 1} failed: {str(e)}",
                    extra={"context": error_context}
                )
                
                # Don't retry for certain error types
                if isinstance(e, (ImportError, ValueError)) and "required" in str(e):
                    # Library missing or fundamental validation error
                    break
        
        # All retries exhausted
        raise DocumentProcessingError(
            f"Text extraction failed after {max_retries + 1} attempts: {str(last_error)}",
            error_type="ExtractionError",
            context={
                "final_error": str(last_error),
                "attempts": max_retries + 1,
                "file_type": file_type.value
            }
        )

    def process_document_with_diagnostics(
        self, 
        content: bytes, 
        filename: str, 
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, SupportedFileType, Dict[str, Any]]:
        """Process document with comprehensive diagnostics and error handling.
        
        Args:
            content: Raw file bytes
            filename: Original filename
            content_type: MIME type from upload headers
            metadata: Additional metadata for processing
            
        Returns:
            Tuple of (extracted_text, file_type, diagnostics)
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        start_time = time.time()
        diagnostics = {
            "processing_start": datetime.utcnow().isoformat(),
            "file_info": {
                "filename": filename,
                "content_type": content_type,
                "file_size": len(content)
            },
            "validation_results": None,
            "extraction_info": None,
            "performance_metrics": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # Step 1: File type detection
            logger.info(f"Starting document processing for {filename}")
            file_type = self.detect_file_type(filename, content_type)
            
            if file_type is None:
                supported = list(self.SUPPORTED_EXTENSIONS.keys())
                raise DocumentProcessingError(
                    f"Unsupported file format for {filename}. "
                    f"Supported formats: {', '.join(supported)}",
                    error_type="UnsupportedFormat",
                    context={
                        "filename": filename,
                        "content_type": content_type,
                        "supported_formats": supported
                    }
                )
            
            diagnostics["file_info"]["detected_type"] = file_type.value
            
            # Step 2: File validation (PDF-specific for now)
            if file_type == SupportedFileType.PDF:
                try:
                    validation_results = self.validate_pdf_content(content)
                    diagnostics["validation_results"] = validation_results
                    
                    if validation_results["warnings"]:
                        diagnostics["warnings"].extend(validation_results["warnings"])
                        
                except DocumentProcessingError as e:
                    diagnostics["errors"].append({
                        "stage": "validation",
                        "error_type": e.error_type,
                        "message": str(e),
                        "context": e.context
                    })
                    raise
            
            # Step 3: Text extraction with retry logic
            extraction_start = time.time()
            try:
                text = self.extract_text_with_retry(content, file_type)
                extraction_duration = time.time() - extraction_start
                
                diagnostics["extraction_info"] = {
                    "success": True,
                    "method": f"_{file_type.value}_extraction",
                    "char_count": len(text),
                    "extraction_duration": extraction_duration,
                    "retry_count": 0  # Will be updated if retries occur
                }
                
                # Validate extracted text
                if not text or not text.strip():
                    diagnostics["warnings"].append(
                        "Extracted text is empty or contains only whitespace"
                    )
                
                if len(text) < 10:
                    diagnostics["warnings"].append(
                        f"Extracted text is very short ({len(text)} characters)"
                    )
                
            except DocumentProcessingError as e:
                diagnostics["errors"].append({
                    "stage": "extraction",
                    "error_type": e.error_type,
                    "message": str(e),
                    "context": e.context
                })
                raise
            
            # Step 4: Final processing metrics
            total_duration = time.time() - start_time
            diagnostics["performance_metrics"] = {
                "total_processing_time": total_duration,
                "extraction_time": extraction_duration,
                "processing_rate_chars_per_sec": len(text) / total_duration if total_duration > 0 else 0
            }
            
            diagnostics["processing_end"] = datetime.utcnow().isoformat()
            
            logger.info(
                f"Successfully processed {filename}: {len(text)} characters extracted "
                f"in {total_duration:.2f}s"
            )
            
            return text, file_type, diagnostics
            
        except DocumentProcessingError:
            # Re-raise our custom errors
            diagnostics["processing_end"] = datetime.utcnow().isoformat()
            diagnostics["performance_metrics"]["total_processing_time"] = time.time() - start_time
            raise
            
        except Exception as e:
            # Catch any unexpected errors
            diagnostics["errors"].append({
                "stage": "unknown",
                "error_type": "UnexpectedError",
                "message": str(e),
                "context": {"exception_type": type(e).__name__}
            })
            diagnostics["processing_end"] = datetime.utcnow().isoformat()
            diagnostics["performance_metrics"]["total_processing_time"] = time.time() - start_time
            
            logger.error(f"Unexpected error processing {filename}: {str(e)}", exc_info=True)
            
            raise DocumentProcessingError(
                f"Unexpected error during document processing: {str(e)}",
                error_type="UnexpectedError",
                context={"original_error": str(e), "exception_type": type(e).__name__}
            )

    def get_processing_status(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get processing status for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Processing status information or None
        """
        # This method would need a database session to work properly
        # For now, it returns None but could be enhanced to work with dependency injection
        logger.info(f"Processing status requested for document {document_id}")
        return None
