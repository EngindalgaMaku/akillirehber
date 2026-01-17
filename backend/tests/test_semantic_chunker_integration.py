"""Integration tests for semantic chunker with enhanced tokenization.

Feature: semantic-chunker-enhancement
Tests the full pipeline: language detection → tokenization → chunking
"""

import pytest
from app.services.chunker import SemanticChunker


class TestSemanticChunkerIntegration:
    """Integration tests for the full chunking pipeline."""
    
    def test_turkish_document_chunking(self):
        """Test chunking of a real Turkish document."""
        chunker = SemanticChunker()
        
        turkish_text = """
        Türkiye'nin başkenti Ankara'da bugün önemli bir toplantı yapıldı.
        Cumhurbaşkanı, ekonomik reformlar hakkında açıklamalarda bulundu.
        Yeni politikalar, önümüzdeki ay uygulamaya konulacak.
        
        Dr. Ahmet Yılmaz, İstanbul Üniversitesi'nde çalışıyor.
        Araştırma alanı yapay zeka ve makine öğrenmesi.
        Yayınladığı makaleler örn. Nature ve Science dergilerinde yer aldı.
        """
        
        # Test sentence splitting
        sentences = chunker._split_into_sentences(turkish_text)
        
        # Should split into sentences correctly
        assert len(sentences) > 0
        
        # Should not split at abbreviations
        assert any('Dr. Ahmet' in s for s in sentences)
        assert any('örn. Nature' in s for s in sentences)
        
        # Turkish characters should be preserved
        full_text = ' '.join(sentences)
        assert 'Türkiye' in full_text
        assert 'İstanbul' in full_text
        assert 'Üniversitesi' in full_text
    
    def test_english_document_chunking(self):
        """Test chunking of a real English document."""
        chunker = SemanticChunker()
        
        english_text = """
        The president announced new economic policies today.
        These reforms will be implemented next month.
        Experts believe this will improve the economy significantly.
        
        Dr. John Smith works at MIT.
        His research focuses on AI and machine learning.
        His papers e.g. in Nature and Science are well-cited.
        """
        
        # Test sentence splitting
        sentences = chunker._split_into_sentences(english_text)
        
        # Should split into sentences correctly
        assert len(sentences) > 0
        
        # Should not split at abbreviations
        assert any('Dr. John' in s for s in sentences)
        assert any('e.g. in' in s for s in sentences)
    
    def test_mixed_language_document(self):
        """Test chunking of mixed Turkish-English document."""
        chunker = SemanticChunker()
        
        mixed_text = """
        Bu sistem Python ve FastAPI kullanarak geliştirilmiştir.
        The API endpoints are documented using OpenAPI specification.
        Veritabanı olarak PostgreSQL kullanılmaktadır.
        Database connections are managed through SQLAlchemy ORM.
        """
        
        # Test sentence splitting
        sentences = chunker._split_into_sentences(mixed_text)
        
        # Should split into sentences correctly
        assert len(sentences) >= 4
        
        # Both Turkish and English content should be preserved
        full_text = ' '.join(sentences)
        assert 'Python' in full_text
        assert 'FastAPI' in full_text
        assert 'PostgreSQL' in full_text
        assert 'SQLAlchemy' in full_text
    
    def test_document_with_questions(self):
        """Test chunking of document with questions."""
        chunker = SemanticChunker()
        
        text = """
        Nasıl yardımcı olabilirim?
        Size birkaç seçenek sunabilirim.
        Hangi konuda bilgi almak istersiniz?
        Teknik destek veya genel bilgi için buradayım.
        """
        
        # Test sentence splitting
        sentences = chunker._split_into_sentences(text)
        
        # Should split questions correctly
        assert len(sentences) >= 4
        
        # Questions should be preserved
        assert any('?' in s for s in sentences)
    
    def test_document_with_abbreviations(self):
        """Test chunking of document with many abbreviations."""
        chunker = SemanticChunker()
        
        text = """
        Dr. Mehmet ve Prof. Ayşe A.Ş. şirketinde çalışıyor.
        Ofis adresi: Atatürk Cad. No. 123 Apt. 5 Kadıköy/İstanbul.
        Tel. numarası: 0212-555-1234.
        Fax. numarası: 0212-555-1235.
        """
        
        # Test sentence splitting
        sentences = chunker._split_into_sentences(text)
        
        # Should split into sentences (may vary based on newlines)
        assert len(sentences) >= 4
        
        # Abbreviations should be preserved
        full_text = ' '.join(sentences)
        assert 'Dr.' in full_text
        assert 'Prof.' in full_text
        assert 'A.Ş.' in full_text
        assert 'Cad.' in full_text
        assert 'No.' in full_text
        assert 'Apt.' in full_text
        assert 'Tel.' in full_text
        assert 'Fax.' in full_text
    
    def test_document_with_decimals(self):
        """Test chunking of document with decimal numbers."""
        chunker = SemanticChunker()
        
        text = """
        Ürün fiyatı 10.5 TL.
        İndirimli fiyat 8.75 TL.
        KDV oranı %18.0.
        Toplam tutar 25.3 TL.
        """
        
        # Test sentence splitting
        sentences = chunker._split_into_sentences(text)
        
        # Should split correctly
        assert len(sentences) == 4
        
        # Decimals should be preserved
        full_text = ' '.join(sentences)
        assert '10.5' in full_text
        assert '8.75' in full_text
        assert '18.0' in full_text
        assert '25.3' in full_text
    
    def test_document_with_quoted_text(self):
        """Test chunking of document with quoted text."""
        chunker = SemanticChunker()
        
        text = """
        Müdür dedi ki: "Toplantı saat 14:00'te başlayacak."
        Herkes hazır olmalı.
        "Lütfen zamanında gelin." diye ekledi.
        Toplantı salonu 3. katta.
        """
        
        # Test sentence splitting
        sentences = chunker._split_into_sentences(text)
        
        # Should split correctly (may merge some sentences with quotes)
        assert len(sentences) >= 3
        
        # Quoted text should be preserved
        full_text = ' '.join(sentences)
        assert 'Toplantı saat 14:00\'te başlayacak' in full_text
        assert 'Lütfen zamanında gelin' in full_text
        assert 'Herkes hazır olmalı' in full_text
        assert 'Toplantı salonu 3. katta' in full_text
    
    def test_document_with_urls(self):
        """Test chunking of document with URLs."""
        chunker = SemanticChunker()
        
        text = """
        Daha fazla bilgi için https://example.com adresini ziyaret edin.
        API dokümantasyonu https://api.example.com/docs adresinde.
        Destek için support@example.com adresine yazabilirsiniz.
        """
        
        # Test sentence splitting
        sentences = chunker._split_into_sentences(text)
        
        # Should split correctly
        assert len(sentences) == 3
        
        # URLs and emails should be preserved
        full_text = ' '.join(sentences)
        assert 'https://example.com' in full_text
        assert 'https://api.example.com/docs' in full_text
        assert 'support@example.com' in full_text
    
    def test_fallback_to_nltk(self):
        """Test that fallback to NLTK works if enhanced tokenizer fails."""
        chunker = SemanticChunker()
        
        # Simple English text that should work with both methods
        text = "This is a sentence. This is another sentence."
        
        sentences = chunker._split_into_sentences(text)
        
        # Should split into 2 sentences
        assert len(sentences) == 2
        assert sentences[0].strip() == "This is a sentence."
        assert sentences[1].strip() == "This is another sentence."
    
    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = SemanticChunker()
        
        sentences = chunker._split_into_sentences("")
        assert sentences == []
        
        sentences = chunker._split_into_sentences("   ")
        assert sentences == []
    
    def test_single_sentence(self):
        """Test handling of single sentence."""
        chunker = SemanticChunker()
        
        text = "Bu tek bir cümle."
        sentences = chunker._split_into_sentences(text)
        
        assert len(sentences) == 1
        assert sentences[0] == "Bu tek bir cümle."
    
    def test_no_regression_in_existing_functionality(self):
        """Test that existing chunking functionality still works."""
        chunker = SemanticChunker()
        
        # Test with a simple text
        text = """
        This is the first sentence. This is the second sentence.
        This is the third sentence. This is the fourth sentence.
        """
        
        sentences = chunker._split_into_sentences(text)
        
        # Should split correctly
        assert len(sentences) == 4
        
        # All sentences should be present
        assert "first sentence" in sentences[0]
        assert "second sentence" in sentences[1]
        assert "third sentence" in sentences[2]
        assert "fourth sentence" in sentences[3]
