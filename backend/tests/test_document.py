"""Tests for document upload and processing functionality."""

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.document import SupportedFileType
from app.services.document_processor import DocumentProcessor, DocumentProcessingError


client = TestClient(app)


class TestDocumentProcessor:
    """Tests for DocumentProcessor service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()

    def test_detect_file_type_pdf(self):
        """Test PDF file type detection."""
        result = self.processor.detect_file_type("document.pdf")
        assert result == SupportedFileType.PDF

    def test_detect_file_type_markdown(self):
        """Test Markdown file type detection."""
        result = self.processor.detect_file_type("readme.md")
        assert result == SupportedFileType.MARKDOWN

        result = self.processor.detect_file_type("readme.markdown")
        assert result == SupportedFileType.MARKDOWN

    def test_detect_file_type_docx(self):
        """Test Word document file type detection."""
        result = self.processor.detect_file_type("document.docx")
        assert result == SupportedFileType.DOCX

    def test_detect_file_type_txt(self):
        """Test plain text file type detection."""
        result = self.processor.detect_file_type("notes.txt")
        assert result == SupportedFileType.TEXT

    def test_detect_file_type_unsupported(self):
        """Test unsupported file type detection."""
        result = self.processor.detect_file_type("image.png")
        assert result is None

    def test_detect_file_type_by_content_type(self):
        """Test file type detection by MIME type."""
        result = self.processor.detect_file_type(
            "unknown", content_type="application/pdf"
        )
        assert result == SupportedFileType.PDF

    def test_extract_text_plain(self):
        """Test plain text extraction."""
        content = b"Hello, World!"
        result = self.processor.extract_text(content, SupportedFileType.TEXT)
        assert result == "Hello, World!"

    def test_extract_markdown(self):
        """Test markdown text extraction."""
        content = b"# Heading\n\nParagraph text."
        result = self.processor.extract_text(
            content, SupportedFileType.MARKDOWN
        )
        assert result == "# Heading\n\nParagraph text."

    def test_process_document_txt(self):
        """Test processing a plain text document."""
        content = b"Test content"
        text, file_type = self.processor.process_document(
            content, "test.txt"
        )
        assert text == "Test content"
        assert file_type == SupportedFileType.TEXT

    def test_process_document_unsupported(self):
        """Test processing an unsupported file type."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.processor.process_document(b"content", "image.png")
        assert "Unsupported file format" in str(exc_info.value)
        assert exc_info.value.error_type == "UnsupportedFormat"

    def test_process_document_with_diagnostics(self):
        """Test processing with diagnostics enabled."""
        content = b"Test content for diagnostics"
        text, file_type, diagnostics = self.processor.process_document_with_diagnostics(
            content, "test.txt"
        )
        
        assert text == "Test content for diagnostics"
        assert file_type == SupportedFileType.TEXT
        assert "processing_start" in diagnostics
        assert "processing_end" in diagnostics
        assert "file_info" in diagnostics
        assert "extraction_info" in diagnostics
        assert "performance_metrics" in diagnostics
        
        # Check file info
        assert diagnostics["file_info"]["filename"] == "test.txt"
        assert diagnostics["file_info"]["file_size"] == len(content)
        assert diagnostics["file_info"]["detected_type"] == "txt"
        
        # Check extraction info
        assert diagnostics["extraction_info"]["success"] is True
        assert diagnostics["extraction_info"]["char_count"] == len(text)

    def test_validate_pdf_content_invalid_header(self):
        """Test PDF validation with invalid header."""
        content = b"Not a PDF file" + b"x" * 200  # Make it large enough to pass size check
        
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.processor.validate_pdf_content(content)
        
        assert exc_info.value.error_type == "ValidationError"
        assert "valid PDF header" in str(exc_info.value)

    def test_validate_pdf_content_too_small(self):
        """Test PDF validation with file too small."""
        content = b"%PDF-"  # Too small to be valid
        
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.processor.validate_pdf_content(content)
        
        assert exc_info.value.error_type == "ValidationError"
        assert "too small" in str(exc_info.value)

    def test_extract_text_with_retry_success(self):
        """Test text extraction with retry logic - success case."""
        content = b"Test content"
        result = self.processor.extract_text_with_retry(
            content, SupportedFileType.TEXT, max_retries=2
        )
        assert result == "Test content"

    def test_extract_text_encoding_fallback(self):
        """Test text extraction with encoding fallback to latin-1."""
        # Invalid UTF-8 bytes that should fall back to latin-1
        content = b"\xff\xfe\x00\x00Test"
        
        # This should succeed with latin-1 fallback
        result = self.processor.extract_text(content, SupportedFileType.TEXT)
        assert isinstance(result, str)  # Should successfully decode with latin-1


class TestUploadEndpoint:
    """Tests for /api/upload endpoint."""

    def test_upload_txt_file(self):
        """Test uploading a plain text file."""
        content = b"Hello, this is test content."
        files = {"file": ("test.txt", io.BytesIO(content), "text/plain")}

        response = client.post("/api/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello, this is test content."
        assert data["metadata"]["file_name"] == "test.txt"
        assert data["metadata"]["file_type"] == "txt"
        assert data["metadata"]["char_count"] == len(content)

    def test_upload_markdown_file(self):
        """Test uploading a markdown file."""
        content = b"# Title\n\nSome paragraph."
        files = {"file": ("readme.md", io.BytesIO(content), "text/markdown")}

        response = client.post("/api/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "# Title" in data["text"]
        assert data["metadata"]["file_type"] == "md"

    def test_upload_empty_file(self):
        """Test uploading an empty file."""
        files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}

        response = client.post("/api/upload", files=files)

        assert response.status_code == 422
        assert "empty" in response.json()["detail"].lower()

    def test_upload_unsupported_format(self):
        """Test uploading an unsupported file format."""
        files = {"file": ("image.png", io.BytesIO(b"fake"), "image/png")}

        response = client.post("/api/upload", files=files)

        assert response.status_code == 422
        assert "Unsupported" in response.json()["detail"]
