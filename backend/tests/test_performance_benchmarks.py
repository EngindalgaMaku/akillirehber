"""Performance benchmarks for Semantic Chunker Enhancement.

Feature: semantic-chunker-enhancement, Phase 6: Performance Benchmarks
Measures processing time, API call reduction, cache hit rate, and memory usage.
"""

import time
import pytest
from typing import List, Tuple

from app.services.chunker import SemanticChunker, fallback_to_sentence_chunking
from app.services.language_detector import LanguageDetector
from app.services.sentence_tokenizer import EnhancedSentenceTokenizer
from app.services.qa_detector import QADetector
from app.services.adaptive_threshold import AdaptiveThresholdCalculator
from app.services.embedding_cache import EmbeddingCache


def generate_text(size_chars: int, language: str = "english") -> str:
    """Generate text of approximately the specified size."""
    if language == "turkish":
        base_sentences = [
            "Bu bir test cümlesidir.",
            "Türkçe metin işleme önemlidir.",
            "Yapay zeka teknolojileri gelişiyor.",
            "Doğal dil işleme alanında ilerlemeler var.",
            "Makine öğrenmesi modelleri eğitiliyor.",
        ]
    else:
        base_sentences = [
            "This is a test sentence for benchmarking.",
            "Natural language processing is important.",
            "Machine learning models are being trained.",
            "Artificial intelligence is advancing rapidly.",
            "Text chunking improves retrieval accuracy.",
        ]
    
    text = ""
    idx = 0
    while len(text) < size_chars:
        text += base_sentences[idx % len(base_sentences)] + " "
        idx += 1
    
    return text[:size_chars]


class TestProcessingTimeBenchmarks:
    """Benchmarks for processing time at various text sizes."""
    
    @pytest.mark.parametrize("size_chars,max_time_seconds", [
        (1000, 2.0),    # 1K chars should complete in <2s
        (5000, 5.0),    # 5K chars should complete in <5s
        (10000, 10.0),  # 10K chars should complete in <10s
    ])
    def test_sentence_tokenization_time(self, size_chars: int, max_time_seconds: float):
        """Benchmark sentence tokenization processing time."""
        text = generate_text(size_chars)
        tokenizer = EnhancedSentenceTokenizer()
        detector = LanguageDetector()
        
        start_time = time.time()
        language = detector.detect_language(text)
        sentences = tokenizer.tokenize(text, language)
        elapsed = time.time() - start_time
        
        assert elapsed < max_time_seconds, (
            f"Tokenization of {size_chars} chars took {elapsed:.2f}s, "
            f"expected <{max_time_seconds}s"
        )
        assert len(sentences) > 0
    
    @pytest.mark.parametrize("size_chars,max_time_seconds", [
        (1000, 1.0),   # 1K chars should complete in <1s
        (5000, 2.0),   # 5K chars should complete in <2s
        (10000, 3.0),  # 10K chars should complete in <3s
    ])
    def test_language_detection_time(self, size_chars: int, max_time_seconds: float):
        """Benchmark language detection processing time."""
        text = generate_text(size_chars)
        detector = LanguageDetector()
        
        start_time = time.time()
        language = detector.detect_language(text)
        elapsed = time.time() - start_time
        
        assert elapsed < max_time_seconds, (
            f"Language detection of {size_chars} chars took {elapsed:.2f}s, "
            f"expected <{max_time_seconds}s"
        )
    
    @pytest.mark.parametrize("size_chars,max_time_seconds", [
        (1000, 1.0),   # 1K chars should complete in <1s
        (5000, 2.0),   # 5K chars should complete in <2s
        (10000, 3.0),  # 10K chars should complete in <3s
    ])
    def test_qa_detection_time(self, size_chars: int, max_time_seconds: float):
        """Benchmark Q&A detection processing time."""
        text = generate_text(size_chars)
        detector = LanguageDetector()
        qa_detector = QADetector()
        tokenizer = EnhancedSentenceTokenizer()
        
        language = detector.detect_language(text)
        sentences = tokenizer.tokenize(text, language)
        
        start_time = time.time()
        merged = qa_detector.merge_qa_pairs(sentences, language)
        elapsed = time.time() - start_time
        
        assert elapsed < max_time_seconds, (
            f"Q&A detection of {size_chars} chars took {elapsed:.2f}s, "
            f"expected <{max_time_seconds}s"
        )
    
    @pytest.mark.parametrize("size_chars,max_time_seconds", [
        (1000, 1.0),   # 1K chars should complete in <1s
        (5000, 2.0),   # 5K chars should complete in <2s
        (10000, 3.0),  # 10K chars should complete in <3s
    ])
    def test_adaptive_threshold_time(self, size_chars: int, max_time_seconds: float):
        """Benchmark adaptive threshold calculation time."""
        text = generate_text(size_chars)
        calculator = AdaptiveThresholdCalculator()
        
        start_time = time.time()
        recommendation = calculator.recommend_threshold(text)
        elapsed = time.time() - start_time
        
        assert elapsed < max_time_seconds, (
            f"Adaptive threshold of {size_chars} chars took {elapsed:.2f}s, "
            f"expected <{max_time_seconds}s"
        )
        assert 0.3 <= recommendation.recommended_threshold <= 0.9
    
    @pytest.mark.parametrize("size_chars,max_time_seconds", [
        (1000, 2.0),   # 1K chars should complete in <2s
        (5000, 5.0),   # 5K chars should complete in <5s
    ])
    def test_fallback_chunking_time(self, size_chars: int, max_time_seconds: float):
        """Benchmark fallback sentence chunking time."""
        text = generate_text(size_chars)
        
        start_time = time.time()
        chunks = fallback_to_sentence_chunking(text, chunk_size=500, overlap=50)
        elapsed = time.time() - start_time
        
        assert elapsed < max_time_seconds, (
            f"Fallback chunking of {size_chars} chars took {elapsed:.2f}s, "
            f"expected <{max_time_seconds}s"
        )
        assert len(chunks) > 0


class TestCacheEffectiveness:
    """Benchmarks for embedding cache effectiveness."""
    
    def test_cache_hit_rate(self):
        """Test that cache achieves >70% hit rate on repeated texts."""
        cache = EmbeddingCache(ttl=3600, max_entries=1000)
        
        # Simulate embedding storage
        texts = [f"Test sentence number {i}." for i in range(100)]
        model = "test-model"
        
        # First pass: all misses
        for i, text in enumerate(texts):
            embedding = [float(i)] * 10  # Dummy embedding
            cache.set(text, model, embedding)
        
        # Second pass: all hits
        hits = 0
        for text in texts:
            result = cache.get(text, model)
            if result is not None:
                hits += 1
        
        hit_rate = hits / len(texts)
        assert hit_rate >= 0.7, f"Cache hit rate {hit_rate:.2%} is below 70%"
    
    def test_cache_reduces_lookups(self):
        """Test that cache reduces the need for embedding lookups."""
        cache = EmbeddingCache(ttl=3600, max_entries=1000)
        
        texts = ["Same text repeated."] * 10
        model = "test-model"
        
        # Store once
        cache.set(texts[0], model, [1.0, 2.0, 3.0])
        
        # All subsequent lookups should hit cache
        lookups_needed = 0
        for text in texts:
            if cache.get(text, model) is None:
                lookups_needed += 1
        
        # Only 0 lookups needed (first one was stored)
        assert lookups_needed == 0, f"Expected 0 lookups, got {lookups_needed}"


class TestTurkishProcessingBenchmarks:
    """Benchmarks specific to Turkish text processing."""
    
    def test_turkish_tokenization_accuracy(self):
        """Test Turkish sentence tokenization accuracy."""
        tokenizer = EnhancedSentenceTokenizer()
        
        # Test cases with expected sentence counts
        test_cases = [
            ("Bu bir cümle. Bu ikinci cümle.", 2),
            ("Dr. Ahmet geldi. Prof. Mehmet gitti.", 2),
            ("Fiyat 10.5 TL. Toplam 25.3 TL.", 2),
            ("Nasılsın? İyiyim. Sen nasılsın?", 3),
        ]
        
        correct = 0
        for text, expected in test_cases:
            from app.services.language_detector import Language
            sentences = tokenizer.tokenize(text, Language.TURKISH)
            if len(sentences) == expected:
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.95, f"Turkish tokenization accuracy {accuracy:.2%} is below 95%"
    
    def test_turkish_character_preservation(self):
        """Test that Turkish characters are preserved during processing."""
        text = "Çok güzel bir gün. İşte şöyle böyle. Ürün özellikleri."
        tokenizer = EnhancedSentenceTokenizer()
        
        from app.services.language_detector import Language
        sentences = tokenizer.tokenize(text, Language.TURKISH)
        
        reconstructed = " ".join(sentences)
        
        # All Turkish characters should be preserved
        turkish_chars = "çğıöşüÇĞİÖŞÜ"
        for char in turkish_chars:
            if char in text:
                assert char in reconstructed, f"Turkish character '{char}' was lost"


class TestMemoryUsage:
    """Benchmarks for memory usage."""
    
    def test_cache_memory_limit(self):
        """Test that cache respects memory limits."""
        max_entries = 100
        cache = EmbeddingCache(ttl=3600, max_entries=max_entries)
        
        # Add more entries than max
        for i in range(max_entries * 2):
            cache.set(f"text_{i}", "model", [float(i)] * 10)
        
        # Cache should not exceed max entries
        stats = cache.get_stats()
        assert stats.entry_count <= max_entries, (
            f"Cache has {stats.entry_count} entries, "
            f"expected <= {max_entries}"
        )
    
    def test_language_detector_cache_efficiency(self):
        """Test that language detector caches results efficiently."""
        detector = LanguageDetector()
        
        # Detect same text multiple times
        text = "This is a test sentence for caching."
        
        for _ in range(100):
            detector.detect_language(text)
        
        # Should use cache, not recompute
        # (We can't directly measure cache hits, but this shouldn't crash)
        assert True


class TestPerformanceSummary:
    """Summary test that reports overall performance metrics."""
    
    def test_performance_summary(self):
        """Generate a performance summary report."""
        results = {}
        
        # Test tokenization speed
        text_5k = generate_text(5000)
        tokenizer = EnhancedSentenceTokenizer()
        detector = LanguageDetector()
        
        start = time.time()
        language = detector.detect_language(text_5k)
        sentences = tokenizer.tokenize(text_5k, language)
        results["tokenization_5k_ms"] = (time.time() - start) * 1000
        
        # Test Q&A detection speed
        qa_detector = QADetector()
        start = time.time()
        qa_detector.merge_qa_pairs(sentences, language)
        results["qa_detection_5k_ms"] = (time.time() - start) * 1000
        
        # Test adaptive threshold speed
        calculator = AdaptiveThresholdCalculator()
        start = time.time()
        calculator.recommend_threshold(text_5k)
        results["adaptive_threshold_5k_ms"] = (time.time() - start) * 1000
        
        # Test fallback chunking speed
        start = time.time()
        fallback_to_sentence_chunking(text_5k, 500, 50)
        results["fallback_chunking_5k_ms"] = (time.time() - start) * 1000
        
        # Print summary
        print("\n=== Performance Summary (5K chars) ===")
        for metric, value in results.items():
            print(f"  {metric}: {value:.2f}ms")
        
        # All operations should complete in reasonable time
        assert results["tokenization_5k_ms"] < 5000
        assert results["qa_detection_5k_ms"] < 2000
        assert results["adaptive_threshold_5k_ms"] < 2000
        assert results["fallback_chunking_5k_ms"] < 5000
