"""Tests for SemanticSimilarityService."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.semantic_similarity_service import (
    cosine_similarity,
    SemanticSimilarityService,
)


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Test that identical vectors have similarity of 1.0."""
        vec = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity of 0.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Test that opposite vectors are clamped to 0.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        # Opposite vectors have cosine of -1, but we clamp to 0
        assert similarity == 0.0

    def test_empty_vectors(self):
        """Test that empty vectors return 0.0."""
        similarity = cosine_similarity([], [1.0, 2.0])
        assert similarity == 0.0

        similarity = cosine_similarity([1.0, 2.0], [])
        assert similarity == 0.0

    def test_zero_length_vectors(self):
        """Test that zero-length vectors return 0.0."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_similar_vectors(self):
        """Test that similar vectors have high similarity."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.1, 2.1, 3.1]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity > 0.99  # Very similar

    def test_bounds(self):
        """Test that similarity is always in [0, 1] range."""
        # Generate random vectors
        np.random.seed(42)
        for _ in range(100):
            vec1 = np.random.randn(10).tolist()
            vec2 = np.random.randn(10).tolist()
            similarity = cosine_similarity(vec1, vec2)
            assert 0.0 <= similarity <= 1.0


class TestSemanticSimilarityService:
    """Tests for SemanticSimilarityService."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return Mock()

    @pytest.fixture
    def service(self, mock_db):
        """Create a SemanticSimilarityService instance."""
        return SemanticSimilarityService(mock_db)

    def test_compute_similarity(self, service):
        """Test compute_similarity method."""
        with patch.object(
            service.embedding_service, 'get_embedding'
        ) as mock_get_embedding:
            # Mock embeddings
            mock_get_embedding.side_effect = [
                [1.0, 0.0, 0.0],  # First call
                [1.0, 0.0, 0.0],  # Second call
            ]

            score = service.compute_similarity(
                "text1", "text2", "test-model"
            )

            assert score == pytest.approx(1.0)
            assert mock_get_embedding.call_count == 2

    def test_find_best_match_single(self, service):
        """Test find_best_match with single ground truth."""
        with patch.object(
            service.embedding_service, 'get_embedding'
        ) as mock_get_embedding:
            # Mock embeddings
            mock_get_embedding.side_effect = [
                [1.0, 0.0, 0.0],  # Generated answer
                [1.0, 0.0, 0.0],  # Ground truth
            ]

            max_score, best_match, all_scores = service.find_best_match(
                "generated", ["ground truth"], "test-model"
            )

            assert max_score == pytest.approx(1.0)
            assert best_match == "ground truth"
            assert len(all_scores) == 1
            assert all_scores[0]["ground_truth"] == "ground truth"
            assert all_scores[0]["score"] == pytest.approx(1.0)

    def test_find_best_match_multiple(self, service):
        """Test find_best_match with multiple ground truths."""
        with patch.object(
            service.embedding_service, 'get_embedding'
        ) as mock_get_embedding:
            # Mock embeddings - generated answer matches second GT better
            mock_get_embedding.side_effect = [
                [1.0, 0.0, 0.0],  # Generated answer
                [0.5, 0.5, 0.0],  # Ground truth 1 (lower similarity)
                [1.0, 0.0, 0.0],  # Generated answer (recomputed)
                [1.0, 0.0, 0.0],  # Ground truth 2 (perfect match)
            ]

            max_score, best_match, all_scores = service.find_best_match(
                "generated",
                ["ground truth 1", "ground truth 2"],
                "test-model"
            )

            assert max_score == pytest.approx(1.0)
            assert best_match == "ground truth 2"
            assert len(all_scores) == 2

    def test_find_best_match_empty(self, service):
        """Test find_best_match with empty ground truths."""
        max_score, best_match, all_scores = service.find_best_match(
            "generated", [], "test-model"
        )

        assert max_score == 0.0
        assert best_match == ""
        assert all_scores == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
