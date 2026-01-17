"""Property-based tests for chunk quality metrics.

Feature: semantic-chunker-enhancement, Phase 4: Quality Metrics
This module contains property-based tests for chunk quality metrics,
including coherence calculation, inter-chunk similarity, and quality reports.
"""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from unittest.mock import Mock, patch

from app.services.chunk_quality import (
    ChunkQualityAnalyzer,
    ChunkQualityMetrics,
    QualityReport,
)


class TestSemanticCoherenceProperty:
    """Property-based tests for semantic coherence calculation.
    
    Feature: semantic-chunker-enhancement, Property 14: Semantic Coherence Calculation
    Validates: Requirements 7.1
    
    For any generated chunk, a semantic coherence score (0-1) SHALL be calculated
    based on intra-chunk sentence similarity.
    """
    
    def test_coherence_score_in_valid_range(self):
        """Test that coherence scores are always in [0, 1] range.
        
        Feature: semantic-chunker-enhancement, Property 14
        Validates: Requirements 7.1
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Mock embedding provider to avoid API calls
        mock_embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7],
        ]
        
        with patch.object(analyzer, '_ensure_embedding_provider'):
            analyzer._embedding_provider = Mock()
            analyzer._embedding_provider.get_embeddings.return_value = mock_embeddings
            
            text = "First sentence here. Second sentence here. Third sentence here."
            coherence = analyzer.calculate_semantic_coherence(text)
            
            assert 0.0 <= coherence <= 1.0, (
                f"Coherence score {coherence} is outside valid range [0, 1]"
            )
    
    def test_single_sentence_has_perfect_coherence(self):
        """Test that single sentence chunks have coherence of 1.0.
        
        Feature: semantic-chunker-enhancement, Property 14
        Validates: Requirements 7.1
        """
        analyzer = ChunkQualityAnalyzer()
        
        text = "This is a single sentence without any periods inside"
        coherence = analyzer.calculate_semantic_coherence(text)
        
        assert coherence == 1.0, (
            f"Single sentence should have coherence 1.0, got {coherence}"
        )
    
    def test_empty_chunk_has_perfect_coherence(self):
        """Test that empty chunks have coherence of 1.0.
        
        Feature: semantic-chunker-enhancement, Property 14
        Validates: Requirements 7.1
        """
        analyzer = ChunkQualityAnalyzer()
        
        coherence = analyzer.calculate_semantic_coherence("")
        
        assert coherence == 1.0, (
            f"Empty chunk should have coherence 1.0, got {coherence}"
        )
    
    @given(
        sentence_count=st.integers(min_value=2, max_value=5)
    )
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_coherence_calculated_for_multiple_sentences(self, sentence_count):
        """Property test: Coherence is calculated for chunks with multiple sentences.
        
        Feature: semantic-chunker-enhancement, Property 14
        Validates: Requirements 7.1
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Create mock embeddings for each sentence
        mock_embeddings = [
            [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i]
            for i in range(1, sentence_count + 1)
        ]
        
        with patch.object(analyzer, '_ensure_embedding_provider'):
            analyzer._embedding_provider = Mock()
            analyzer._embedding_provider.get_embeddings.return_value = mock_embeddings
            
            # Create text with multiple sentences
            sentences = [f"Sentence number {i}." for i in range(sentence_count)]
            text = " ".join(sentences)
            
            coherence = analyzer.calculate_semantic_coherence(text)
            
            # Property: Coherence should be calculated and in valid range
            assert 0.0 <= coherence <= 1.0, (
                f"Coherence {coherence} should be in [0, 1]"
            )
    
    def test_identical_sentences_have_high_coherence(self):
        """Test that identical sentences result in high coherence.
        
        Feature: semantic-chunker-enhancement, Property 14
        Validates: Requirements 7.1
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Mock identical embeddings
        identical_embedding = [0.5, 0.5, 0.5, 0.5, 0.5]
        mock_embeddings = [identical_embedding, identical_embedding, identical_embedding]
        
        with patch.object(analyzer, '_ensure_embedding_provider'):
            analyzer._embedding_provider = Mock()
            analyzer._embedding_provider.get_embeddings.return_value = mock_embeddings
            
            text = "Same topic. Same topic. Same topic."
            coherence = analyzer.calculate_semantic_coherence(text)
            
            # Identical embeddings should give coherence very close to 1.0
            assert coherence == pytest.approx(1.0, rel=1e-6), (
                f"Identical sentences should have coherence ~1.0, got {coherence}"
            )
    
    def test_orthogonal_sentences_have_low_coherence(self):
        """Test that orthogonal (unrelated) sentences result in low coherence.
        
        Feature: semantic-chunker-enhancement, Property 14
        Validates: Requirements 7.1
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Mock orthogonal embeddings (dot product = 0)
        mock_embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        
        with patch.object(analyzer, '_ensure_embedding_provider'):
            analyzer._embedding_provider = Mock()
            analyzer._embedding_provider.get_embeddings.return_value = mock_embeddings
            
            text = "Topic A. Topic B. Topic C."
            coherence = analyzer.calculate_semantic_coherence(text)
            
            # Orthogonal embeddings should give coherence of 0.0
            assert coherence == 0.0, (
                f"Orthogonal sentences should have coherence 0.0, got {coherence}"
            )


class TestInterChunkSimilarityProperty:
    """Property-based tests for inter-chunk similarity measurement.
    
    Feature: semantic-chunker-enhancement, Property 15: Inter-Chunk Similarity Measurement
    Validates: Requirements 7.2
    
    For any pair of consecutive chunks, inter-chunk similarity SHALL be measured
    and reported.
    """
    
    def test_inter_chunk_similarity_in_valid_range(self):
        """Test that inter-chunk similarity is in [0, 1] range.
        
        Feature: semantic-chunker-enhancement, Property 15
        Validates: Requirements 7.2
        """
        analyzer = ChunkQualityAnalyzer()
        
        mock_embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
        ]
        
        with patch.object(analyzer, '_ensure_embedding_provider'):
            analyzer._embedding_provider = Mock()
            analyzer._embedding_provider.get_embeddings.return_value = mock_embeddings
            
            chunk1 = "First chunk content. With multiple sentences."
            chunk2 = "Second chunk content. Also with sentences."
            
            similarity = analyzer.calculate_inter_chunk_similarity(chunk1, chunk2)
            
            assert 0.0 <= similarity <= 1.0, (
                f"Inter-chunk similarity {similarity} should be in [0, 1]"
            )
    
    def test_empty_chunks_return_zero_similarity(self):
        """Test that empty chunks return 0 similarity.
        
        Feature: semantic-chunker-enhancement, Property 15
        Validates: Requirements 7.2
        """
        analyzer = ChunkQualityAnalyzer()
        
        similarity = analyzer.calculate_inter_chunk_similarity("", "Some content.")
        assert similarity == 0.0
        
        similarity = analyzer.calculate_inter_chunk_similarity("Some content.", "")
        assert similarity == 0.0
        
        similarity = analyzer.calculate_inter_chunk_similarity("", "")
        assert similarity == 0.0
    
    @given(
        chunk_count=st.integers(min_value=2, max_value=5)
    )
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_all_consecutive_pairs_measured(self, chunk_count):
        """Property test: Similarity is calculated for all consecutive pairs.
        
        Feature: semantic-chunker-enhancement, Property 15
        Validates: Requirements 7.2
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Create mock embeddings
        mock_embeddings = [
            [0.1 * i, 0.2 * i, 0.3 * i]
            for i in range(1, 3)  # 2 embeddings per call (last + first sentence)
        ]
        
        with patch.object(analyzer, '_ensure_embedding_provider'):
            analyzer._embedding_provider = Mock()
            analyzer._embedding_provider.get_embeddings.return_value = mock_embeddings
            
            # Create chunks
            chunks = [f"Chunk {i} content. With sentences." for i in range(chunk_count)]
            
            similarities = analyzer.calculate_all_inter_chunk_similarities(chunks)
            
            # Property: Should have chunk_count - 1 similarity values
            expected_count = chunk_count - 1
            assert len(similarities) == expected_count, (
                f"Expected {expected_count} similarities, got {len(similarities)}"
            )
            
            # All similarities should be in valid range
            for sim in similarities:
                assert 0.0 <= sim <= 1.0, (
                    f"Similarity {sim} should be in [0, 1]"
                )
    
    def test_single_chunk_returns_empty_list(self):
        """Test that single chunk returns empty similarity list.
        
        Feature: semantic-chunker-enhancement, Property 15
        Validates: Requirements 7.2
        """
        analyzer = ChunkQualityAnalyzer()
        
        similarities = analyzer.calculate_all_inter_chunk_similarities(["Single chunk."])
        
        assert similarities == [], (
            f"Single chunk should return empty list, got {similarities}"
        )
    
    def test_no_chunks_returns_empty_list(self):
        """Test that no chunks returns empty similarity list.
        
        Feature: semantic-chunker-enhancement, Property 15
        Validates: Requirements 7.2
        """
        analyzer = ChunkQualityAnalyzer()
        
        similarities = analyzer.calculate_all_inter_chunk_similarities([])
        
        assert similarities == [], (
            f"No chunks should return empty list, got {similarities}"
        )


class TestLowCoherenceDetectionProperty:
    """Property-based tests for low coherence detection.
    
    Feature: semantic-chunker-enhancement, Property 16: Low Coherence Detection
    Validates: Requirements 7.4
    
    For any chunk with semantic coherence < 0.5, the system SHALL flag it
    as low-quality.
    """
    
    def test_low_coherence_chunks_flagged(self):
        """Test that chunks with coherence < 0.5 are flagged.
        
        Feature: semantic-chunker-enhancement, Property 16
        Validates: Requirements 7.4
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Create metrics with varying coherence
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.8,
                sentence_count=3,
                avg_sentence_similarity=0.8,
                topic_consistency=0.8
            ),
            ChunkQualityMetrics(
                chunk_index=1,
                semantic_coherence=0.3,  # Low coherence
                sentence_count=3,
                avg_sentence_similarity=0.3,
                topic_consistency=0.3
            ),
            ChunkQualityMetrics(
                chunk_index=2,
                semantic_coherence=0.6,
                sentence_count=3,
                avg_sentence_similarity=0.6,
                topic_consistency=0.6
            ),
            ChunkQualityMetrics(
                chunk_index=3,
                semantic_coherence=0.4,  # Low coherence
                sentence_count=3,
                avg_sentence_similarity=0.4,
                topic_consistency=0.4
            ),
        ]
        
        low_coherence = analyzer.detect_low_coherence_chunks(metrics)
        
        # Chunks 1 and 3 should be flagged (coherence < 0.5)
        assert 1 in low_coherence, "Chunk 1 (coherence 0.3) should be flagged"
        assert 3 in low_coherence, "Chunk 3 (coherence 0.4) should be flagged"
        assert 0 not in low_coherence, "Chunk 0 (coherence 0.8) should not be flagged"
        assert 2 not in low_coherence, "Chunk 2 (coherence 0.6) should not be flagged"
    
    def test_exactly_threshold_not_flagged(self):
        """Test that chunks with coherence exactly 0.5 are not flagged.
        
        Feature: semantic-chunker-enhancement, Property 16
        Validates: Requirements 7.4
        """
        analyzer = ChunkQualityAnalyzer()
        
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.5,  # Exactly at threshold
                sentence_count=3,
                avg_sentence_similarity=0.5,
                topic_consistency=0.5
            ),
        ]
        
        low_coherence = analyzer.detect_low_coherence_chunks(metrics)
        
        assert 0 not in low_coherence, (
            "Chunk with coherence exactly 0.5 should not be flagged"
        )
    
    @given(
        coherence=st.floats(min_value=0.0, max_value=0.49, allow_nan=False)
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_any_coherence_below_threshold_flagged(self, coherence):
        """Property test: Any coherence below 0.5 is flagged.
        
        Feature: semantic-chunker-enhancement, Property 16
        Validates: Requirements 7.4
        """
        analyzer = ChunkQualityAnalyzer()
        
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=coherence,
                sentence_count=3,
                avg_sentence_similarity=coherence,
                topic_consistency=coherence
            ),
        ]
        
        low_coherence = analyzer.detect_low_coherence_chunks(metrics)
        
        assert 0 in low_coherence, (
            f"Chunk with coherence {coherence} should be flagged"
        )
    
    @given(
        coherence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False)
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_any_coherence_at_or_above_threshold_not_flagged(self, coherence):
        """Property test: Any coherence >= 0.5 is not flagged.
        
        Feature: semantic-chunker-enhancement, Property 16
        Validates: Requirements 7.4
        """
        analyzer = ChunkQualityAnalyzer()
        
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=coherence,
                sentence_count=3,
                avg_sentence_similarity=coherence,
                topic_consistency=coherence
            ),
        ]
        
        low_coherence = analyzer.detect_low_coherence_chunks(metrics)
        
        assert 0 not in low_coherence, (
            f"Chunk with coherence {coherence} should not be flagged"
        )
    
    def test_custom_threshold(self):
        """Test that custom threshold works correctly.
        
        Feature: semantic-chunker-enhancement, Property 16
        Validates: Requirements 7.4
        """
        analyzer = ChunkQualityAnalyzer()
        
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.6,
                sentence_count=3,
                avg_sentence_similarity=0.6,
                topic_consistency=0.6
            ),
        ]
        
        # With default threshold (0.5), should not be flagged
        low_coherence = analyzer.detect_low_coherence_chunks(metrics)
        assert 0 not in low_coherence
        
        # With custom threshold (0.7), should be flagged
        low_coherence = analyzer.detect_low_coherence_chunks(metrics, threshold=0.7)
        assert 0 in low_coherence
    
    def test_empty_metrics_returns_empty_list(self):
        """Test that empty metrics returns empty list.
        
        Feature: semantic-chunker-enhancement, Property 16
        Validates: Requirements 7.4
        """
        analyzer = ChunkQualityAnalyzer()
        
        low_coherence = analyzer.detect_low_coherence_chunks([])
        
        assert low_coherence == [], (
            f"Empty metrics should return empty list, got {low_coherence}"
        )


class TestQualityReportGeneration:
    """Tests for quality report generation.
    
    Feature: semantic-chunker-enhancement, Task 7.9
    Validates: Requirements 7.5
    """
    
    def test_empty_chunks_returns_empty_report(self):
        """Test that empty chunks returns a valid empty report.
        
        Feature: semantic-chunker-enhancement, Task 7.9
        Validates: Requirements 7.5
        """
        analyzer = ChunkQualityAnalyzer()
        
        report = analyzer.generate_quality_report([])
        
        assert report.total_chunks == 0
        assert report.avg_coherence == 0.0
        assert report.min_coherence == 0.0
        assert report.max_coherence == 0.0
        assert report.overall_quality_score == 0.0
    
    def test_report_contains_all_fields(self):
        """Test that report contains all required fields.
        
        Feature: semantic-chunker-enhancement, Task 7.9
        Validates: Requirements 7.5
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Create pre-computed metrics to avoid API calls
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.8,
                sentence_count=3,
                avg_sentence_similarity=0.8,
                topic_consistency=0.8
            ),
            ChunkQualityMetrics(
                chunk_index=1,
                semantic_coherence=0.6,
                sentence_count=2,
                avg_sentence_similarity=0.6,
                topic_consistency=0.6
            ),
        ]
        
        # Mock inter-chunk similarity calculation
        with patch.object(analyzer, 'calculate_all_inter_chunk_similarities') as mock_sim:
            mock_sim.return_value = [0.7]
            
            report = analyzer.generate_quality_report(
                ["Chunk 1.", "Chunk 2."],
                chunk_metrics=metrics
            )
        
        # Verify all fields are present
        assert hasattr(report, 'total_chunks')
        assert hasattr(report, 'avg_coherence')
        assert hasattr(report, 'min_coherence')
        assert hasattr(report, 'max_coherence')
        assert hasattr(report, 'chunks_below_threshold')
        assert hasattr(report, 'inter_chunk_similarities')
        assert hasattr(report, 'merge_recommendations')
        assert hasattr(report, 'split_recommendations')
        assert hasattr(report, 'overall_quality_score')
        assert hasattr(report, 'recommendations')
    
    def test_coherence_statistics_calculated_correctly(self):
        """Test that coherence statistics are calculated correctly.
        
        Feature: semantic-chunker-enhancement, Task 7.9
        Validates: Requirements 7.5
        """
        analyzer = ChunkQualityAnalyzer()
        
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.8,
                sentence_count=3,
                avg_sentence_similarity=0.8,
                topic_consistency=0.8
            ),
            ChunkQualityMetrics(
                chunk_index=1,
                semantic_coherence=0.4,
                sentence_count=2,
                avg_sentence_similarity=0.4,
                topic_consistency=0.4
            ),
            ChunkQualityMetrics(
                chunk_index=2,
                semantic_coherence=0.6,
                sentence_count=2,
                avg_sentence_similarity=0.6,
                topic_consistency=0.6
            ),
        ]
        
        with patch.object(analyzer, 'calculate_all_inter_chunk_similarities') as mock_sim:
            mock_sim.return_value = [0.5, 0.5]
            
            report = analyzer.generate_quality_report(
                ["Chunk 1.", "Chunk 2.", "Chunk 3."],
                chunk_metrics=metrics
            )
        
        assert report.total_chunks == 3
        assert report.avg_coherence == pytest.approx(0.6, rel=0.01)
        assert report.min_coherence == 0.4
        assert report.max_coherence == 0.8
        assert 1 in report.chunks_below_threshold  # Chunk 1 has coherence 0.4


class TestMergeRecommendations:
    """Tests for merge recommendations.
    
    Feature: semantic-chunker-enhancement, Task 7.10
    Validates: Requirements 7.5
    """
    
    def test_high_similarity_pairs_recommended_for_merge(self):
        """Test that high similarity pairs are recommended for merge.
        
        Feature: semantic-chunker-enhancement, Task 7.10
        Validates: Requirements 7.5
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Similarities: [0.9, 0.5, 0.85]
        # Pairs (0,1) and (2,3) should be recommended (> 0.8)
        similarities = [0.9, 0.5, 0.85]
        
        recommendations = analyzer.generate_merge_recommendations(similarities)
        
        assert (0, 1) in recommendations, "Pair (0,1) with similarity 0.9 should be recommended"
        assert (2, 3) in recommendations, "Pair (2,3) with similarity 0.85 should be recommended"
        assert (1, 2) not in recommendations, "Pair (1,2) with similarity 0.5 should not be recommended"
    
    def test_exactly_threshold_not_recommended(self):
        """Test that pairs with similarity exactly 0.8 are not recommended.
        
        Feature: semantic-chunker-enhancement, Task 7.10
        Validates: Requirements 7.5
        """
        analyzer = ChunkQualityAnalyzer()
        
        similarities = [0.8]
        
        recommendations = analyzer.generate_merge_recommendations(similarities)
        
        assert (0, 1) not in recommendations, (
            "Pair with similarity exactly 0.8 should not be recommended"
        )
    
    def test_custom_threshold(self):
        """Test that custom threshold works correctly.
        
        Feature: semantic-chunker-enhancement, Task 7.10
        Validates: Requirements 7.5
        """
        analyzer = ChunkQualityAnalyzer()
        
        similarities = [0.7]
        
        # With default threshold (0.8), should not be recommended
        recommendations = analyzer.generate_merge_recommendations(similarities)
        assert (0, 1) not in recommendations
        
        # With custom threshold (0.6), should be recommended
        recommendations = analyzer.generate_merge_recommendations(similarities, threshold=0.6)
        assert (0, 1) in recommendations
    
    def test_empty_similarities_returns_empty_list(self):
        """Test that empty similarities returns empty list.
        
        Feature: semantic-chunker-enhancement, Task 7.10
        Validates: Requirements 7.5
        """
        analyzer = ChunkQualityAnalyzer()
        
        recommendations = analyzer.generate_merge_recommendations([])
        
        assert recommendations == []


class TestSplitRecommendations:
    """Tests for split recommendations.
    
    Feature: semantic-chunker-enhancement, Task 7.11
    Validates: Requirements 7.5
    """
    
    def test_low_coherence_chunks_recommended_for_split(self):
        """Test that low coherence chunks are recommended for split.
        
        Feature: semantic-chunker-enhancement, Task 7.11
        Validates: Requirements 7.5
        """
        analyzer = ChunkQualityAnalyzer()
        
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.8,
                sentence_count=3,
                avg_sentence_similarity=0.8,
                topic_consistency=0.8
            ),
            ChunkQualityMetrics(
                chunk_index=1,
                semantic_coherence=0.3,  # Low coherence, multiple sentences
                sentence_count=3,
                avg_sentence_similarity=0.3,
                topic_consistency=0.3
            ),
            ChunkQualityMetrics(
                chunk_index=2,
                semantic_coherence=0.4,  # Low coherence, multiple sentences
                sentence_count=2,
                avg_sentence_similarity=0.4,
                topic_consistency=0.4
            ),
        ]
        
        recommendations = analyzer.generate_split_recommendations(metrics)
        
        assert 1 in recommendations, "Chunk 1 (coherence 0.3) should be recommended for split"
        assert 2 in recommendations, "Chunk 2 (coherence 0.4) should be recommended for split"
        assert 0 not in recommendations, "Chunk 0 (coherence 0.8) should not be recommended"
    
    def test_single_sentence_not_recommended_for_split(self):
        """Test that single sentence chunks are not recommended for split.
        
        Feature: semantic-chunker-enhancement, Task 7.11
        Validates: Requirements 7.5
        """
        analyzer = ChunkQualityAnalyzer()
        
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.3,  # Low coherence but single sentence
                sentence_count=1,
                avg_sentence_similarity=0.3,
                topic_consistency=0.3
            ),
        ]
        
        recommendations = analyzer.generate_split_recommendations(metrics)
        
        assert 0 not in recommendations, (
            "Single sentence chunk should not be recommended for split"
        )
    
    def test_empty_metrics_returns_empty_list(self):
        """Test that empty metrics returns empty list.
        
        Feature: semantic-chunker-enhancement, Task 7.11
        Validates: Requirements 7.5
        """
        analyzer = ChunkQualityAnalyzer()
        
        recommendations = analyzer.generate_split_recommendations([])
        
        assert recommendations == []


class TestOverallQualityScore:
    """Tests for overall quality score calculation."""
    
    def test_high_quality_chunks_have_high_score(self):
        """Test that high quality chunks result in high overall score."""
        analyzer = ChunkQualityAnalyzer()
        
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.9,
                sentence_count=3,
                avg_sentence_similarity=0.9,
                topic_consistency=0.9
            ),
            ChunkQualityMetrics(
                chunk_index=1,
                semantic_coherence=0.85,
                sentence_count=3,
                avg_sentence_similarity=0.85,
                topic_consistency=0.85
            ),
        ]
        
        with patch.object(analyzer, 'calculate_all_inter_chunk_similarities') as mock_sim:
            mock_sim.return_value = [0.5]  # Not too high, no merge recommendation
            
            report = analyzer.generate_quality_report(
                ["Chunk 1.", "Chunk 2."],
                chunk_metrics=metrics
            )
        
        # High coherence, no low coherence chunks, no recommendations
        assert report.overall_quality_score > 0.7, (
            f"High quality chunks should have score > 0.7, got {report.overall_quality_score}"
        )
    
    def test_low_quality_chunks_have_low_score(self):
        """Test that low quality chunks result in low overall score."""
        analyzer = ChunkQualityAnalyzer()
        
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.3,
                sentence_count=3,
                avg_sentence_similarity=0.3,
                topic_consistency=0.3
            ),
            ChunkQualityMetrics(
                chunk_index=1,
                semantic_coherence=0.2,
                sentence_count=3,
                avg_sentence_similarity=0.2,
                topic_consistency=0.2
            ),
        ]
        
        with patch.object(analyzer, 'calculate_all_inter_chunk_similarities') as mock_sim:
            mock_sim.return_value = [0.9]  # High similarity, merge recommendation
            
            report = analyzer.generate_quality_report(
                ["Chunk 1.", "Chunk 2."],
                chunk_metrics=metrics
            )
        
        # Low coherence, all chunks below threshold, merge recommendation
        assert report.overall_quality_score < 0.5, (
            f"Low quality chunks should have score < 0.5, got {report.overall_quality_score}"
        )
    
    def test_score_always_in_valid_range(self):
        """Test that overall score is always in [0, 1] range."""
        analyzer = ChunkQualityAnalyzer()
        
        # Test with various metric combinations
        test_cases = [
            (0.0, 0.0),  # Worst case
            (1.0, 1.0),  # Best case
            (0.5, 0.5),  # Average case
        ]
        
        for coherence, similarity in test_cases:
            metrics = [
                ChunkQualityMetrics(
                    chunk_index=0,
                    semantic_coherence=coherence,
                    sentence_count=3,
                    avg_sentence_similarity=coherence,
                    topic_consistency=coherence
                ),
            ]
            
            with patch.object(analyzer, 'calculate_all_inter_chunk_similarities') as mock_sim:
                mock_sim.return_value = []
                
                report = analyzer.generate_quality_report(
                    ["Chunk 1."],
                    chunk_metrics=metrics
                )
            
            assert 0.0 <= report.overall_quality_score <= 1.0, (
                f"Score {report.overall_quality_score} should be in [0, 1]"
            )


class TestIntegrationQualityMetrics:
    """Integration tests for quality metrics.
    
    Feature: semantic-chunker-enhancement, Task 7.13
    Tests full pipeline with quality calculation.
    """
    
    def test_full_pipeline_with_quality_calculation(self):
        """Test full pipeline with quality calculation.
        
        Feature: semantic-chunker-enhancement, Task 7.13
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Create sample chunks
        chunks = [
            "This is the first chunk. It talks about topic A. More about topic A here.",
            "Second chunk discusses topic B. Topic B is interesting. More B content.",
            "Third chunk covers topic C. Topic C details. Final C information.",
        ]
        
        # Create mock metrics
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.85,
                sentence_count=3,
                avg_sentence_similarity=0.85,
                topic_consistency=0.9
            ),
            ChunkQualityMetrics(
                chunk_index=1,
                semantic_coherence=0.75,
                sentence_count=3,
                avg_sentence_similarity=0.75,
                topic_consistency=0.8
            ),
            ChunkQualityMetrics(
                chunk_index=2,
                semantic_coherence=0.80,
                sentence_count=3,
                avg_sentence_similarity=0.80,
                topic_consistency=0.85
            ),
        ]
        
        # Mock inter-chunk similarity
        with patch.object(analyzer, 'calculate_all_inter_chunk_similarities') as mock_sim:
            mock_sim.return_value = [0.6, 0.65]
            
            report = analyzer.generate_quality_report(chunks, chunk_metrics=metrics)
        
        # Verify report is generated correctly
        assert report.total_chunks == 3
        assert report.avg_coherence == pytest.approx(0.8, rel=0.01)
        assert report.min_coherence == 0.75
        assert report.max_coherence == 0.85
        assert len(report.chunks_below_threshold) == 0  # All above 0.5
        assert len(report.inter_chunk_similarities) == 2
        assert len(report.merge_recommendations) == 0  # None above 0.8
        assert len(report.split_recommendations) == 0  # None below 0.5
        assert 0.0 <= report.overall_quality_score <= 1.0
        assert len(report.recommendations) > 0
    
    def test_metrics_accuracy_with_known_values(self):
        """Test that metrics are accurate with known values.
        
        Feature: semantic-chunker-enhancement, Task 7.13
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Create metrics with known values
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.9,
                sentence_count=3,
                avg_sentence_similarity=0.9,
                topic_consistency=0.9
            ),
            ChunkQualityMetrics(
                chunk_index=1,
                semantic_coherence=0.3,  # Low coherence
                sentence_count=3,
                avg_sentence_similarity=0.3,
                topic_consistency=0.3
            ),
        ]
        
        with patch.object(analyzer, 'calculate_all_inter_chunk_similarities') as mock_sim:
            mock_sim.return_value = [0.85]  # High similarity
            
            report = analyzer.generate_quality_report(
                ["Chunk 1.", "Chunk 2."],
                chunk_metrics=metrics
            )
        
        # Verify accuracy
        assert report.avg_coherence == pytest.approx(0.6, rel=0.01)
        assert report.min_coherence == 0.3
        assert report.max_coherence == 0.9
        assert 1 in report.chunks_below_threshold  # Chunk 1 has low coherence
        assert (0, 1) in report.merge_recommendations  # High similarity
        assert 1 in report.split_recommendations  # Low coherence
    
    def test_report_generation_with_various_chunk_counts(self):
        """Test report generation with various chunk counts.
        
        Feature: semantic-chunker-enhancement, Task 7.13
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Test with 1 chunk
        metrics_1 = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.8,
                sentence_count=3,
                avg_sentence_similarity=0.8,
                topic_consistency=0.8
            ),
        ]
        
        with patch.object(analyzer, 'calculate_all_inter_chunk_similarities') as mock_sim:
            mock_sim.return_value = []
            
            report_1 = analyzer.generate_quality_report(["Chunk 1."], chunk_metrics=metrics_1)
        
        assert report_1.total_chunks == 1
        assert len(report_1.inter_chunk_similarities) == 0
        assert len(report_1.merge_recommendations) == 0
        
        # Test with 5 chunks
        metrics_5 = [
            ChunkQualityMetrics(
                chunk_index=i,
                semantic_coherence=0.7,
                sentence_count=3,
                avg_sentence_similarity=0.7,
                topic_consistency=0.7
            )
            for i in range(5)
        ]
        
        with patch.object(analyzer, 'calculate_all_inter_chunk_similarities') as mock_sim:
            mock_sim.return_value = [0.5, 0.5, 0.5, 0.5]
            
            report_5 = analyzer.generate_quality_report(
                [f"Chunk {i}." for i in range(5)],
                chunk_metrics=metrics_5
            )
        
        assert report_5.total_chunks == 5
        assert len(report_5.inter_chunk_similarities) == 4
    
    def test_recommendations_are_actionable(self):
        """Test that recommendations are actionable.
        
        Feature: semantic-chunker-enhancement, Task 7.13
        """
        analyzer = ChunkQualityAnalyzer()
        
        # Create metrics with issues
        metrics = [
            ChunkQualityMetrics(
                chunk_index=0,
                semantic_coherence=0.3,  # Low coherence
                sentence_count=3,
                avg_sentence_similarity=0.3,
                topic_consistency=0.3
            ),
            ChunkQualityMetrics(
                chunk_index=1,
                semantic_coherence=0.4,  # Low coherence
                sentence_count=3,
                avg_sentence_similarity=0.4,
                topic_consistency=0.4
            ),
        ]
        
        with patch.object(analyzer, 'calculate_all_inter_chunk_similarities') as mock_sim:
            mock_sim.return_value = [0.9]  # High similarity
            
            report = analyzer.generate_quality_report(
                ["Chunk 1.", "Chunk 2."],
                chunk_metrics=metrics
            )
        
        # Verify recommendations are present and actionable
        assert len(report.recommendations) > 0
        
        # Should mention low coherence
        has_coherence_recommendation = any(
            'coherence' in r.lower() for r in report.recommendations
        )
        assert has_coherence_recommendation, "Should have coherence recommendation"
        
        # Should mention merge
        has_merge_recommendation = any(
            'merg' in r.lower() for r in report.recommendations
        )
        assert has_merge_recommendation, "Should have merge recommendation"
