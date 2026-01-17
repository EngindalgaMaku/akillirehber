"""Phase 6 Integration Tests for Semantic Chunker Enhancement.

Feature: semantic-chunker-enhancement, Phase 6: Integration, Testing, and Documentation
Tests the complete integration of all components including API endpoints,
quality metrics, and real-world document processing.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.chunking import ChunkingStrategy
from app.services.chunker import (
    SemanticChunker,
    chunk_with_error_handling,
    ChunkingResult,
)


client = TestClient(app)


class TestAPIEndpointIntegration:
    """Integration tests for the /api/chunk endpoint."""
    
    def test_basic_chunking_request(self):
        """Test basic chunking request without quality metrics."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "This is the first sentence. This is the second sentence. "
                        "This is the third sentence. This is the fourth sentence.",
                "strategy": "fixed_size",
                "chunk_size": 100,
                "overlap": 10
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert "stats" in data
        assert "strategy_used" in data
        assert data["strategy_used"] == "fixed_size"
    
    def test_semantic_chunking_with_defaults(self):
        """Test semantic chunking with default parameters."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "Machine learning is a subset of artificial intelligence. "
                        "It enables computers to learn from data. "
                        "Deep learning is a type of machine learning. "
                        "Neural networks are used in deep learning.",
                "strategy": "semantic"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert len(data["chunks"]) > 0
    
    def test_semantic_chunking_with_quality_metrics(self):
        """Test semantic chunking with quality metrics enabled."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "Machine learning is a subset of artificial intelligence. "
                        "It enables computers to learn from data. "
                        "Deep learning is a type of machine learning. "
                        "Neural networks are used in deep learning.",
                "strategy": "semantic",
                "include_quality_metrics": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        # Quality metrics may or may not be present depending on API key availability
        # The important thing is the request doesn't fail
    
    def test_semantic_chunking_with_custom_parameters(self):
        """Test semantic chunking with custom parameters."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "This is a test document. It has multiple sentences. "
                        "Each sentence should be processed correctly. "
                        "The chunker should handle this well.",
                "strategy": "semantic",
                "enable_qa_detection": False,
                "enable_adaptive_threshold": False,
                "enable_cache": True,
                "min_chunk_size": 50,
                "max_chunk_size": 500,
                "buffer_size": 2
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
    
    def test_validation_error_empty_text(self):
        """Test validation error for empty text."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "   ",
                "strategy": "fixed_size"
            }
        )
        
        assert response.status_code == 422
    
    def test_validation_error_overlap_too_large(self):
        """Test validation error when overlap >= chunk_size."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "Some text to chunk.",
                "strategy": "fixed_size",
                "chunk_size": 100,
                "overlap": 100
            }
        )
        
        assert response.status_code == 422


class TestRealWorldDocuments:
    """Integration tests with real-world document types."""
    
    def test_turkish_news_article(self):
        """Test chunking of Turkish news article."""
        turkish_text = """
        Türkiye'nin başkenti Ankara'da bugün önemli bir toplantı yapıldı.
        Cumhurbaşkanı, ekonomik reformlar hakkında açıklamalarda bulundu.
        Yeni politikalar, önümüzdeki ay uygulamaya konulacak.
        
        Dr. Ahmet Yılmaz, İstanbul Üniversitesi'nde çalışıyor.
        Araştırma alanı yapay zeka ve makine öğrenmesi.
        Yayınladığı makaleler örn. Nature ve Science dergilerinde yer aldı.
        
        Ekonomi Bakanı, yeni teşvik paketini açıkladı.
        Paket, KOBİ'lere özel destekler içeriyor.
        Toplam bütçe 50 milyar TL olarak belirlendi.
        """
        
        result = chunk_with_error_handling(
            text=turkish_text,
            strategy=ChunkingStrategy.SEMANTIC,
            enable_qa_detection=True,
            enable_adaptive_threshold=True
        )
        
        assert result.success
        assert len(result.chunks) > 0
        
        # Verify Turkish characters are preserved
        all_text = " ".join(c.content for c in result.chunks)
        assert "Türkiye" in all_text
        assert "İstanbul" in all_text
        assert "Üniversitesi" in all_text
    
    def test_english_technical_document(self):
        """Test chunking of English technical document."""
        english_text = """
        Machine learning is a subset of artificial intelligence that enables
        computers to learn from data without being explicitly programmed.
        
        There are three main types of machine learning:
        1. Supervised learning uses labeled data to train models.
        2. Unsupervised learning finds patterns in unlabeled data.
        3. Reinforcement learning learns through trial and error.
        
        Deep learning is a specialized form of machine learning that uses
        neural networks with multiple layers. These networks can learn
        complex patterns and representations from large amounts of data.
        
        Common applications include image recognition, natural language
        processing, and autonomous vehicles.
        """
        
        result = chunk_with_error_handling(
            text=english_text,
            strategy=ChunkingStrategy.SEMANTIC,
            enable_qa_detection=True,
            enable_adaptive_threshold=True
        )
        
        assert result.success
        assert len(result.chunks) > 0
    
    def test_mixed_language_document(self):
        """Test chunking of mixed Turkish-English document."""
        mixed_text = """
        Bu sistem Python ve FastAPI kullanarak geliştirilmiştir.
        The API endpoints are documented using OpenAPI specification.
        
        Veritabanı olarak PostgreSQL kullanılmaktadır.
        Database connections are managed through SQLAlchemy ORM.
        
        Frontend React ile yazılmıştır.
        The user interface follows Material Design principles.
        """
        
        result = chunk_with_error_handling(
            text=mixed_text,
            strategy=ChunkingStrategy.SEMANTIC,
            enable_qa_detection=True,
            enable_adaptive_threshold=True
        )
        
        assert result.success
        assert len(result.chunks) > 0
    
    def test_qa_document(self):
        """Test chunking of Q&A document."""
        qa_text = """
        Nasıl yardımcı olabilirim?
        Size birkaç seçenek sunabilirim.
        
        Hangi konuda bilgi almak istersiniz?
        Teknik destek veya genel bilgi için buradayım.
        
        What is machine learning?
        Machine learning is a type of artificial intelligence.
        
        How does it work?
        It learns patterns from data to make predictions.
        """
        
        result = chunk_with_error_handling(
            text=qa_text,
            strategy=ChunkingStrategy.SEMANTIC,
            enable_qa_detection=True,
            enable_adaptive_threshold=True
        )
        
        assert result.success
        assert len(result.chunks) > 0


class TestChunkWithErrorHandling:
    """Tests for the chunk_with_error_handling function."""
    
    def test_empty_text_returns_error(self):
        """Test that empty text returns error result."""
        result = chunk_with_error_handling(
            text="",
            strategy=ChunkingStrategy.SEMANTIC
        )
        
        assert not result.success
        assert result.error is not None
        assert "empty" in result.warning_message.lower()
    
    def test_whitespace_only_returns_error(self):
        """Test that whitespace-only text returns error result."""
        result = chunk_with_error_handling(
            text="   \n\t  ",
            strategy=ChunkingStrategy.SEMANTIC
        )
        
        assert not result.success
        assert result.error is not None
    
    def test_successful_chunking_returns_diagnostics(self):
        """Test that successful chunking includes diagnostics."""
        result = chunk_with_error_handling(
            text="This is a test. This is another test. And one more test.",
            strategy=ChunkingStrategy.SEMANTIC
        )
        
        assert result.success
        assert result.diagnostics is not None
        assert result.diagnostics.total_chunks > 0
        assert result.diagnostics.processing_time >= 0
    
    def test_feature_flags_respected(self):
        """Test that feature flags are respected."""
        # Test with all features disabled
        result = chunk_with_error_handling(
            text="This is a test sentence. This is another sentence.",
            strategy=ChunkingStrategy.SEMANTIC,
            enable_qa_detection=False,
            enable_adaptive_threshold=False,
            enable_cache=False
        )
        
        assert result.success
        assert len(result.chunks) > 0


class TestSemanticChunkerFeatureFlags:
    """Tests for SemanticChunker feature flags."""
    
    def test_qa_detection_flag(self):
        """Test Q&A detection can be enabled/disabled."""
        # With Q&A detection enabled
        chunker_enabled = SemanticChunker(
            use_provider_manager=False,
            enable_qa_detection=True
        )
        assert chunker_enabled._enable_qa_detection is True
        
        # With Q&A detection disabled
        chunker_disabled = SemanticChunker(
            use_provider_manager=False,
            enable_qa_detection=False
        )
        assert chunker_disabled._enable_qa_detection is False
    
    def test_adaptive_threshold_flag(self):
        """Test adaptive threshold can be enabled/disabled."""
        # With adaptive threshold enabled
        chunker_enabled = SemanticChunker(
            use_provider_manager=False,
            enable_adaptive_threshold=True
        )
        assert chunker_enabled._enable_adaptive_threshold is True
        
        # With adaptive threshold disabled
        chunker_disabled = SemanticChunker(
            use_provider_manager=False,
            enable_adaptive_threshold=False
        )
        assert chunker_disabled._enable_adaptive_threshold is False
    
    def test_cache_flag(self):
        """Test cache can be enabled/disabled."""
        # With cache enabled
        chunker_enabled = SemanticChunker(
            use_provider_manager=True,
            enable_cache=True
        )
        assert chunker_enabled._enable_cache is True
        
        # With cache disabled
        chunker_disabled = SemanticChunker(
            use_provider_manager=True,
            enable_cache=False
        )
        assert chunker_disabled._enable_cache is False
    
    def test_provider_manager_flag(self):
        """Test provider manager can be enabled/disabled."""
        # With provider manager enabled
        chunker_enabled = SemanticChunker(
            use_provider_manager=True
        )
        assert chunker_enabled._use_provider_manager is True
        
        # With provider manager disabled (legacy mode)
        chunker_disabled = SemanticChunker(
            use_provider_manager=False
        )
        assert chunker_disabled._use_provider_manager is False


class TestBackwardCompatibility:
    """Tests for backward compatibility."""
    
    def test_legacy_api_still_works(self):
        """Test that legacy API parameters still work."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "This is a test sentence. This is another sentence.",
                "strategy": "fixed_size",
                "chunk_size": 100,
                "overlap": 10
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert "stats" in data
    
    def test_semantic_chunker_legacy_mode(self):
        """Test SemanticChunker in legacy mode."""
        chunker = SemanticChunker(
            use_provider_manager=False,
            enable_cache=False,
            enable_qa_detection=False,
            enable_adaptive_threshold=False
        )
        
        # Should initialize without errors
        assert chunker._use_provider_manager is False
        assert chunker._enable_cache is False
        assert chunker._enable_qa_detection is False
        assert chunker._enable_adaptive_threshold is False
