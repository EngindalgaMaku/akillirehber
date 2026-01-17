"""Tests for chunking API endpoint."""

from hypothesis import given, settings, strategies as st, HealthCheck


class TestChunkEndpoint:
    """Tests for POST /api/chunk endpoint."""

    def test_chunk_fixed_size_basic(self, client):
        """Test basic fixed-size chunking."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "This is a test text for chunking. " * 10,
                "strategy": "fixed_size",
                "chunk_size": 100,
                "overlap": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
        assert "stats" in data
        assert data["strategy_used"] == "fixed_size"
        assert len(data["chunks"]) > 0

    def test_chunk_stats_accuracy(self, client):
        """Test that statistics are calculated correctly."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "Hello world. This is a test.",
                "strategy": "fixed_size",
                "chunk_size": 50,
                "overlap": 0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        stats = data["stats"]

        # Verify stats match actual chunks
        chunks = data["chunks"]
        char_counts = [c["char_count"] for c in chunks]

        assert stats["total_chunks"] == len(chunks)
        assert stats["total_characters"] == sum(char_counts)
        assert stats["min_chunk_size"] == min(char_counts)
        assert stats["max_chunk_size"] == max(char_counts)

    def test_chunk_recursive_strategy(self, client):
        """Test recursive chunking strategy."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "First paragraph.\n\nSecond paragraph.\n\nThird.",
                "strategy": "recursive",
                "chunk_size": 100,
                "overlap": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_used"] == "recursive"

    def test_chunk_sentence_strategy(self, client):
        """Test sentence-based chunking strategy."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "First sentence. Second sentence. Third sentence.",
                "strategy": "sentence",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_used"] == "sentence"

    def test_chunk_empty_text_rejected(self, client):
        """Test that empty text is rejected."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "",
                "strategy": "fixed_size",
            },
        )
        assert response.status_code == 422

    def test_chunk_overlap_validation(self, client):
        """Test that overlap >= chunk_size is rejected."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "Some text to chunk",
                "strategy": "fixed_size",
                "chunk_size": 50,
                "overlap": 60,
            },
        )
        assert response.status_code == 422

    def test_chunk_response_structure(self, client):
        """Test that response has correct structure."""
        response = client.post(
            "/api/chunk",
            json={
                "text": "Test text for structure validation.",
                "strategy": "fixed_size",
                "chunk_size": 100,
                "overlap": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "chunks" in data
        assert "stats" in data
        assert "strategy_used" in data

        # Check chunk structure
        if data["chunks"]:
            chunk = data["chunks"][0]
            assert "index" in chunk
            assert "content" in chunk
            assert "start_position" in chunk
            assert "end_position" in chunk
            assert "char_count" in chunk
            assert "has_overlap" in chunk

        # Check stats structure
        stats = data["stats"]
        assert "total_chunks" in stats
        assert "total_characters" in stats
        assert "avg_chunk_size" in stats
        assert "min_chunk_size" in stats
        assert "max_chunk_size" in stats


class TestStatisticsAccuracyProperty:
    """Property-based tests for statistics accuracy.

    Feature: rag-llm-admin-panel, Property 3: Statistics Accuracy
    Validates: Requirements 3.4

    For any set of generated chunks, the reported statistics
    (total_chunks, avg_chunk_size, min_chunk_size, max_chunk_size)
    SHALL be mathematically correct based on the actual chunk data.
    """

    @given(
        text=st.text(min_size=20, max_size=2000).filter(lambda x: x.strip()),
        chunk_size=st.integers(min_value=10, max_value=500),
        overlap_ratio=st.floats(min_value=0.0, max_value=0.8),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    def test_statistics_accuracy_property(
        self, client, text, chunk_size, overlap_ratio
    ):
        """Property test: Statistics are mathematically correct.

        Feature: rag-llm-admin-panel, Property 3: Statistics Accuracy
        Validates: Requirements 3.4
        """
        overlap = int(chunk_size * overlap_ratio)

        response = client.post(
            "/api/chunk",
            json={
                "text": text,
                "strategy": "fixed_size",
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
        )

        assert response.status_code == 200
        data = response.json()

        chunks = data["chunks"]
        stats = data["stats"]

        # Property: total_chunks equals actual chunk count
        assert stats["total_chunks"] == len(chunks)

        if chunks:
            char_counts = [c["char_count"] for c in chunks]

            # Property: total_characters equals sum of all char_counts
            assert stats["total_characters"] == sum(char_counts)

            # Property: avg_chunk_size is mathematically correct
            expected_avg = sum(char_counts) / len(char_counts)
            assert abs(stats["avg_chunk_size"] - expected_avg) < 0.001

            # Property: min_chunk_size equals minimum char_count
            assert stats["min_chunk_size"] == min(char_counts)

            # Property: max_chunk_size equals maximum char_count
            assert stats["max_chunk_size"] == max(char_counts)
        else:
            # Empty chunks case
            assert stats["total_characters"] == 0
            assert stats["avg_chunk_size"] == 0.0
            assert stats["min_chunk_size"] == 0
            assert stats["max_chunk_size"] == 0


class TestCharacterCountAccuracyProperty:
    """Property-based tests for character count accuracy.

    Feature: rag-llm-admin-panel, Property 4: Character Count Accuracy
    Validates: Requirements 4.3

    For any individual chunk, the reported char_count SHALL equal
    the actual length of the chunk content.
    """

    @given(
        text=st.text(min_size=20, max_size=2000).filter(lambda x: x.strip()),
        chunk_size=st.integers(min_value=10, max_value=500),
        overlap_ratio=st.floats(min_value=0.0, max_value=0.8),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
        ]
    )
    def test_char_count_equals_content_length_fixed_size(
        self, client, text, chunk_size, overlap_ratio
    ):
        """Property test: char_count equals len(content) for fixed_size strategy.

        Feature: rag-llm-admin-panel, Property 4: Character Count Accuracy
        Validates: Requirements 4.3
        """
        overlap = int(chunk_size * overlap_ratio)

        response = client.post(
            "/api/chunk",
            json={
                "text": text,
                "strategy": "fixed_size",
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Property: For every chunk, char_count must equal len(content)
        for chunk in data["chunks"]:
            assert chunk["char_count"] == len(chunk["content"]), (
                f"Chunk {chunk['index']}: char_count ({chunk['char_count']}) "
                f"!= len(content) ({len(chunk['content'])})"
            )

    @given(
        text=st.text(min_size=20, max_size=2000).filter(lambda x: x.strip()),
        chunk_size=st.integers(min_value=10, max_value=500),
        overlap_ratio=st.floats(min_value=0.0, max_value=0.8),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
        ]
    )
    def test_char_count_equals_content_length_recursive(
        self, client, text, chunk_size, overlap_ratio
    ):
        """Property test: char_count equals len(content) for recursive strategy.

        Feature: rag-llm-admin-panel, Property 4: Character Count Accuracy
        Validates: Requirements 4.3
        """
        overlap = int(chunk_size * overlap_ratio)

        response = client.post(
            "/api/chunk",
            json={
                "text": text,
                "strategy": "recursive",
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Property: For every chunk, char_count must equal len(content)
        for chunk in data["chunks"]:
            assert chunk["char_count"] == len(chunk["content"]), (
                f"Chunk {chunk['index']}: char_count ({chunk['char_count']}) "
                f"!= len(content) ({len(chunk['content'])})"
            )

    @given(
        text=st.from_regex(r'[A-Za-z0-9 ]+\. [A-Za-z0-9 ]+\.', fullmatch=True),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
        ]
    )
    def test_char_count_equals_content_length_sentence(
        self, client, text
    ):
        """Property test: char_count equals len(content) for sentence strategy.

        Feature: rag-llm-admin-panel, Property 4: Character Count Accuracy
        Validates: Requirements 4.3
        """
        response = client.post(
            "/api/chunk",
            json={
                "text": text,
                "strategy": "sentence",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Property: For every chunk, char_count must equal len(content)
        for chunk in data["chunks"]:
            assert chunk["char_count"] == len(chunk["content"]), (
                f"Chunk {chunk['index']}: char_count ({chunk['char_count']}) "
                f"!= len(content) ({len(chunk['content'])})"
            )


class TestErrorResponseConsistencyProperty:
    """Property-based tests for error response consistency.

    Feature: rag-llm-admin-panel, Property 6: Error Response Consistency
    Validates: Requirements 3.6, 6.4

    For any invalid request (empty text, invalid parameters), the API
    SHALL return an appropriate HTTP error status code (4xx) with a
    descriptive error message.
    """

    @given(
        text=st.text(max_size=100).filter(lambda x: not x.strip()),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
        ]
    )
    def test_empty_or_whitespace_text_returns_422(self, client, text):
        """Property test: Empty or whitespace-only text returns 422.

        Feature: rag-llm-admin-panel, Property 6: Error Response Consistency
        Validates: Requirements 3.6, 6.4
        """
        response = client.post(
            "/api/chunk",
            json={
                "text": text,
                "strategy": "fixed_size",
                "chunk_size": 100,
                "overlap": 10,
            },
        )

        # Property: Empty/whitespace text must return 422
        assert response.status_code == 422
        data = response.json()
        # Property: Response must contain error detail
        assert "detail" in data

    @given(
        chunk_size=st.integers(min_value=10, max_value=500),
        overlap_multiplier=st.floats(min_value=1.0, max_value=5.0),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ]
    )
    def test_overlap_gte_chunk_size_returns_422(
        self, client, chunk_size, overlap_multiplier
    ):
        """Property test: overlap >= chunk_size returns 422.

        Feature: rag-llm-admin-panel, Property 6: Error Response Consistency
        Validates: Requirements 3.6, 6.4
        """
        # Generate overlap that is >= chunk_size
        overlap = int(chunk_size * overlap_multiplier)

        response = client.post(
            "/api/chunk",
            json={
                "text": "Valid text for testing error responses.",
                "strategy": "fixed_size",
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
        )

        # Property: overlap >= chunk_size must return 422
        assert response.status_code == 422
        data = response.json()
        # Property: Response must contain error detail
        assert "detail" in data

    @given(
        chunk_size=st.integers(max_value=9),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ]
    )
    def test_chunk_size_below_minimum_returns_422(self, client, chunk_size):
        """Property test: chunk_size < 10 returns 422.

        Feature: rag-llm-admin-panel, Property 6: Error Response Consistency
        Validates: Requirements 3.6, 6.4
        """
        response = client.post(
            "/api/chunk",
            json={
                "text": "Valid text for testing error responses.",
                "strategy": "fixed_size",
                "chunk_size": chunk_size,
                "overlap": 0,
            },
        )

        # Property: chunk_size < 10 must return 422
        assert response.status_code == 422
        data = response.json()
        # Property: Response must contain error detail
        assert "detail" in data

    @given(
        chunk_size=st.integers(min_value=10001),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ]
    )
    def test_chunk_size_above_maximum_returns_422(self, client, chunk_size):
        """Property test: chunk_size > 10000 returns 422.

        Feature: rag-llm-admin-panel, Property 6: Error Response Consistency
        Validates: Requirements 3.6, 6.4
        """
        response = client.post(
            "/api/chunk",
            json={
                "text": "Valid text for testing error responses.",
                "strategy": "fixed_size",
                "chunk_size": chunk_size,
                "overlap": 0,
            },
        )

        # Property: chunk_size > 10000 must return 422
        assert response.status_code == 422
        data = response.json()
        # Property: Response must contain error detail
        assert "detail" in data

    @given(
        overlap=st.integers(max_value=-1),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ]
    )
    def test_negative_overlap_returns_422(self, client, overlap):
        """Property test: negative overlap returns 422.

        Feature: rag-llm-admin-panel, Property 6: Error Response Consistency
        Validates: Requirements 3.6, 6.4
        """
        response = client.post(
            "/api/chunk",
            json={
                "text": "Valid text for testing error responses.",
                "strategy": "fixed_size",
                "chunk_size": 100,
                "overlap": overlap,
            },
        )

        # Property: negative overlap must return 422
        assert response.status_code == 422
        data = response.json()
        # Property: Response must contain error detail
        assert "detail" in data

    @given(
        strategy=st.text(min_size=1, max_size=50).filter(
            lambda x: x not in [
                "fixed_size", "recursive", "sentence",
                "semantic", "late_chunking", "agentic"
            ]
        ),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
        ]
    )
    def test_invalid_strategy_returns_422(self, client, strategy):
        """Property test: invalid strategy returns 422.

        Feature: rag-llm-admin-panel, Property 6: Error Response Consistency
        Validates: Requirements 3.6, 6.4
        """
        response = client.post(
            "/api/chunk",
            json={
                "text": "Valid text for testing error responses.",
                "strategy": strategy,
                "chunk_size": 100,
                "overlap": 10,
            },
        )

        # Property: invalid strategy must return 422
        assert response.status_code == 422
        data = response.json()
        # Property: Response must contain error detail
        assert "detail" in data

    @given(
        similarity_threshold=st.floats(
            allow_nan=False, allow_infinity=False
        ).filter(lambda x: x < 0.0 or x > 1.0),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
        ]
    )
    def test_invalid_similarity_threshold_returns_422(
        self, client, similarity_threshold
    ):
        """Property test: similarity_threshold outside [0, 1] returns 422.

        Feature: rag-llm-admin-panel, Property 6: Error Response Consistency
        Validates: Requirements 3.6, 6.4
        """
        response = client.post(
            "/api/chunk",
            json={
                "text": "Valid text for testing error responses.",
                "strategy": "semantic",
                "chunk_size": 100,
                "overlap": 10,
                "similarity_threshold": similarity_threshold,
            },
        )

        # Property: invalid similarity_threshold must return 422
        assert response.status_code == 422
        data = response.json()
        # Property: Response must contain error detail
        assert "detail" in data
