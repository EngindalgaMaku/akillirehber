"""Integration tests for Semantic Similarity Test feature.

These tests verify end-to-end workflows including:
- Quick test flow with result saving
- Batch test flow with multiple cases
- Error scenarios (missing fields, invalid JSON, service failures)
- Saved results management (filtering, pagination, deletion)
"""

import pytest
import json
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from app.models.db_models import User, Course, CourseSettings, SemanticSimilarityResult


@pytest.fixture
def test_user(db_session):
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password="hashed_password",
        full_name="Test User",
        role="teacher"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_course(db_session, test_user):
    """Create a test course."""
    course = Course(
        name="Test Course",
        description="Test course for integration tests",
        teacher_id=test_user.id
    )
    db_session.add(course)
    db_session.commit()
    db_session.refresh(course)
    return course


@pytest.fixture
def test_course_settings(db_session, test_course):
    """Create test course settings."""
    settings = CourseSettings(
        course_id=test_course.id,
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        llm_temperature=0.7,
        llm_max_tokens=500,
        default_embedding_model="text-embedding-ada-002",
        system_prompt="You are a helpful assistant."
    )
    db_session.add(settings)
    db_session.commit()
    db_session.refresh(settings)
    return settings


@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers for test user."""
    # Mock JWT token
    return {"Authorization": f"Bearer test_token_{test_user.id}"}


@pytest.fixture
def mock_auth(test_user, client):
    """Mock authentication to return test user."""
    from app.routers.semantic_similarity import router
    from app.services.auth_service import get_current_user
    
    def override_get_current_user():
        return test_user
    
    client.app.dependency_overrides[get_current_user] = override_get_current_user
    yield
    client.app.dependency_overrides.clear()


@pytest.fixture
def mock_teacher_auth(test_user, client):
    """Mock teacher authentication to return test user."""
    from app.routers.semantic_similarity import router
    from app.services.auth_service import get_current_teacher
    
    def override_get_current_teacher():
        return test_user
    
    client.app.dependency_overrides[get_current_teacher] = override_get_current_teacher
    yield
    client.app.dependency_overrides.clear()


class TestQuickTestFlow:
    """Test 12.1: Quick test flow end-to-end.
    
    Requirements: 1.1, 2.1, 4.1
    """

    def test_quick_test_with_all_fields(
        self, client, test_course, test_course_settings, mock_auth
    ):
        """Submit quick test with all fields and verify similarity calculation."""
        # Mock the SemanticSimilarityService
        with patch(
            "app.routers.semantic_similarity.SemanticSimilarityService"
        ) as MockService:
            mock_service_instance = Mock()
            # Mock the find_best_match method
            mock_service_instance.find_best_match.return_value = (
                1.0,  # max_score
                "Python is a programming language.",  # best_match_text
                [{"ground_truth": "Python is a programming language.", "score": 1.0}]  # all_scores
            )
            MockService.return_value = mock_service_instance

            # Submit quick test
            response = client.post(
                "/api/semantic-similarity/quick-test",
                json={
                    "course_id": test_course.id,
                    "question": "What is Python?",
                    "ground_truth": "Python is a programming language.",
                    "generated_answer": "Python is a high-level programming language.",
                    "alternative_ground_truths": []
                }
            )

            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "question" in data
            assert "ground_truth" in data
            assert "generated_answer" in data
            assert "similarity_score" in data
            assert "best_match_ground_truth" in data
            assert "embedding_model_used" in data
            assert "latency_ms" in data
            
            # Verify similarity calculation
            assert data["similarity_score"] == pytest.approx(1.0)
            assert data["best_match_ground_truth"] == "Python is a programming language."
            assert data["embedding_model_used"] == "text-embedding-ada-002"

    def test_quick_test_save_and_retrieve(
        self, client, test_course, test_course_settings, mock_auth
    ):
        """Submit quick test, save result, and verify it appears in saved results."""
        # Mock the SemanticSimilarityService
        with patch(
            "app.routers.semantic_similarity.SemanticSimilarityService"
        ) as MockService:
            mock_service_instance = Mock()
            mock_service_instance.find_best_match.return_value = (
                0.95,  # max_score
                "Python is a programming language.",  # best_match_text
                [{"ground_truth": "Python is a programming language.", "score": 0.95}]  # all_scores
            )
            MockService.return_value = mock_service_instance

            # Submit quick test
            quick_test_response = client.post(
                "/api/semantic-similarity/quick-test",
                json={
                    "course_id": test_course.id,
                    "question": "What is Python?",
                    "ground_truth": "Python is a programming language.",
                    "generated_answer": "Python is a high-level language.",
                }
            )

            assert quick_test_response.status_code == 200
            quick_test_data = quick_test_response.json()

            # Convert all_scores to plain dicts for JSON serialization
            all_scores_dicts = [
                {"ground_truth": score["ground_truth"], "score": score["score"]}
                for score in quick_test_data["all_scores"]
            ]

            # Save the result
            save_response = client.post(
                "/api/semantic-similarity/results",
                json={
                    "course_id": test_course.id,
                    "group_name": "Integration Test Group",
                    "question": quick_test_data["question"],
                    "ground_truth": quick_test_data["ground_truth"],
                    "generated_answer": quick_test_data["generated_answer"],
                    "similarity_score": quick_test_data["similarity_score"],
                    "best_match_ground_truth": quick_test_data["best_match_ground_truth"],
                    "all_scores": all_scores_dicts,
                    "embedding_model_used": quick_test_data["embedding_model_used"],
                    "latency_ms": quick_test_data["latency_ms"],
                }
            )

            assert save_response.status_code == 201
            saved_data = save_response.json()
            assert "id" in saved_data
            result_id = saved_data["id"]

            # Retrieve saved results
            list_response = client.get(
                f"/api/semantic-similarity/results?course_id={test_course.id}"
            )

            assert list_response.status_code == 200
            list_data = list_response.json()
            assert list_data["total"] >= 1
            
            # Verify the saved result appears in the list
            result_found = False
            for result in list_data["results"]:
                if result["id"] == result_id:
                    result_found = True
                    assert result["question"] == "What is Python?"
                    assert result["group_name"] == "Integration Test Group"
                    break
            
            assert result_found, "Saved result not found in list"


class TestBatchTestFlow:
    """Test 12.2: Batch test flow end-to-end.
    
    Requirements: 3.1, 3.4, 3.6
    """

    def test_batch_test_multiple_cases(
        self, client, test_course, test_course_settings, mock_auth
    ):
        """Submit batch test with multiple cases and verify all are processed."""
        # Mock the SemanticSimilarityService
        with patch(
            "app.routers.semantic_similarity.SemanticSimilarityService"
        ) as MockService:
            mock_service_instance = Mock()
            # Mock find_best_match to return different scores for each call
            mock_service_instance.find_best_match.side_effect = [
                (1.0, "Python is a programming language.", [{"ground_truth": "Python is a programming language.", "score": 1.0}]),
                (0.8, "Java is a programming language.", [{"ground_truth": "Java is a programming language.", "score": 0.8}]),
                (0.6, "C++ is a programming language.", [{"ground_truth": "C++ is a programming language.", "score": 0.6}]),
            ]
            MockService.return_value = mock_service_instance

            # Submit batch test
            response = client.post(
                "/api/semantic-similarity/batch-test",
                json={
                    "course_id": test_course.id,
                    "test_cases": [
                        {
                            "question": "What is Python?",
                            "ground_truth": "Python is a programming language.",
                            "generated_answer": "Python is a programming language.",
                        },
                        {
                            "question": "What is Java?",
                            "ground_truth": "Java is a programming language.",
                            "generated_answer": "Java is an OOP language.",
                        },
                        {
                            "question": "What is C++?",
                            "ground_truth": "C++ is a programming language.",
                            "generated_answer": "C++ is used for systems programming.",
                        },
                    ]
                }
            )

            assert response.status_code == 200
            data = response.json()
            
            # Verify all cases processed
            assert "aggregate" in data
            assert data["aggregate"]["test_count"] == 3
            assert data["aggregate"]["successful_count"] == 3
            assert data["aggregate"]["failed_count"] == 0
            assert len(data["results"]) == 3
            
            # Verify each result has required fields
            for result in data["results"]:
                assert "question" in result
                assert "ground_truth" in result
                assert "generated_answer" in result
                assert "similarity_score" in result
                assert result["error_message"] is None

    def test_batch_test_aggregate_statistics(
        self, client, test_course, test_course_settings, mock_auth
    ):
        """Submit batch test and verify aggregate statistics are correct."""
        # Mock the SemanticSimilarityService
        with patch(
            "app.routers.semantic_similarity.SemanticSimilarityService"
        ) as MockService:
            mock_service_instance = Mock()
            # Mock find_best_match to return different scores
            mock_service_instance.find_best_match.side_effect = [
                (1.0, "A1", [{"ground_truth": "A1", "score": 1.0}]),
                (0.8, "A2", [{"ground_truth": "A2", "score": 0.8}]),
                (0.5, "A3", [{"ground_truth": "A3", "score": 0.5}]),
            ]
            MockService.return_value = mock_service_instance

            response = client.post(
                "/api/semantic-similarity/batch-test",
                json={
                    "course_id": test_course.id,
                    "test_cases": [
                        {
                            "question": "Q1",
                            "ground_truth": "A1",
                            "generated_answer": "A1",
                        },
                        {
                            "question": "Q2",
                            "ground_truth": "A2",
                            "generated_answer": "A2 similar",
                        },
                        {
                            "question": "Q3",
                            "ground_truth": "A3",
                            "generated_answer": "A3 different",
                        },
                    ]
                }
            )

            assert response.status_code == 200
            data = response.json()
            
            # Verify aggregate statistics exist
            assert "aggregate" in data
            aggregate = data["aggregate"]
            assert "avg_similarity" in aggregate
            assert "min_similarity" in aggregate
            assert "max_similarity" in aggregate
            assert "total_latency_ms" in aggregate
            
            # Verify statistics are reasonable
            assert aggregate["avg_similarity"] is not None
            assert 0.0 <= aggregate["avg_similarity"] <= 1.0
            assert 0.0 <= aggregate["min_similarity"] <= 1.0
            assert 0.0 <= aggregate["max_similarity"] <= 1.0
            assert aggregate["min_similarity"] <= aggregate["avg_similarity"] <= aggregate["max_similarity"]
            assert aggregate["total_latency_ms"] >= 0  # Can be 0 when mocked


class TestErrorScenarios:
    """Test 12.3: Error scenarios.
    
    Requirements: 7.1, 7.2, 7.3, 7.4
    """

    def test_missing_required_fields(
        self, client, test_course, mock_auth
    ):
        """Test with missing required fields."""
        # Missing question
        response = client.post(
            "/api/semantic-similarity/quick-test",
            json={
                "course_id": test_course.id,
                "ground_truth": "Answer",
                "generated_answer": "Answer",
            }
        )
        assert response.status_code == 422  # Validation error

        # Missing ground_truth
        response = client.post(
            "/api/semantic-similarity/quick-test",
            json={
                "course_id": test_course.id,
                "question": "Question",
                "generated_answer": "Answer",
            }
        )
        assert response.status_code == 422

        # Missing course_id
        response = client.post(
            "/api/semantic-similarity/quick-test",
            json={
                "question": "Question",
                "ground_truth": "Answer",
                "generated_answer": "Answer",
            }
        )
        assert response.status_code == 422

    def test_invalid_json_batch(
        self, client, test_course, mock_auth
    ):
        """Test batch test with invalid JSON structure."""
        # Invalid test_cases structure (not a list)
        response = client.post(
            "/api/semantic-similarity/batch-test",
            json={
                "course_id": test_course.id,
                "test_cases": "not a list"
            }
        )
        assert response.status_code == 422

        # Missing required fields in test case
        response = client.post(
            "/api/semantic-similarity/batch-test",
            json={
                "course_id": test_course.id,
                "test_cases": [
                    {
                        "question": "Q1",
                        # Missing ground_truth and generated_answer
                    }
                ]
            }
        )
        assert response.status_code == 422

    def test_embedding_service_failure(
        self, client, test_course, test_course_settings, mock_auth
    ):
        """Test handling of embedding service failures."""
        # Mock the SemanticSimilarityService to raise an exception
        with patch(
            "app.routers.semantic_similarity.SemanticSimilarityService"
        ) as MockService:
            mock_service_instance = Mock()
            mock_service_instance.find_best_match.side_effect = Exception(
                "Embedding service unavailable"
            )
            MockService.return_value = mock_service_instance

            response = client.post(
                "/api/semantic-similarity/quick-test",
                json={
                    "course_id": test_course.id,
                    "question": "What is Python?",
                    "ground_truth": "Python is a language.",
                    "generated_answer": "Python is a programming language.",
                }
            )

            # Should return 500 error
            assert response.status_code == 500
            assert "Embedding service unavailable" in response.json()["detail"]

    def test_batch_partial_failure(
        self, client, test_course, test_course_settings, mock_auth
    ):
        """Test that batch processing continues after individual failures."""
        # Mock the SemanticSimilarityService
        with patch(
            "app.routers.semantic_similarity.SemanticSimilarityService"
        ) as MockService:
            mock_service_instance = Mock()
            
            # Mock find_best_match to succeed twice and fail once
            mock_service_instance.find_best_match.side_effect = [
                (0.9, "A1", [{"ground_truth": "A1", "score": 0.9}]),  # Success
                Exception("Temporary failure"),  # Failure
                (0.8, "A3", [{"ground_truth": "A3", "score": 0.8}]),  # Success
            ]
            MockService.return_value = mock_service_instance

            response = client.post(
                "/api/semantic-similarity/batch-test",
                json={
                    "course_id": test_course.id,
                    "test_cases": [
                        {
                            "question": "Q1",
                            "ground_truth": "A1",
                            "generated_answer": "A1",
                        },
                        {
                            "question": "Q2",
                            "ground_truth": "A2",
                            "generated_answer": "A2",
                        },
                        {
                            "question": "Q3",
                            "ground_truth": "A3",
                            "generated_answer": "A3",
                        },
                    ]
                }
            )

            assert response.status_code == 200
            data = response.json()
            
            # Verify processing continued
            assert "aggregate" in data
            assert data["aggregate"]["test_count"] == 3
            assert data["aggregate"]["successful_count"] == 2
            assert data["aggregate"]["failed_count"] == 1
            
            # Verify failed case has error message
            failed_case = data["results"][1]
            assert failed_case["error_message"] is not None
            assert "Temporary failure" in failed_case["error_message"]


class TestSavedResultsManagement:
    """Test 12.4: Saved results management.
    
    Requirements: 4.4, 4.5, 4.6, 4.7
    """

    def test_filter_by_group_name(
        self, client, db_session, test_course, test_user, mock_auth
    ):
        """Test filtering saved results by group name."""
        # Create results with different group names
        result1 = SemanticSimilarityResult(
            course_id=test_course.id,
            group_name="Group A",
            question="Q1",
            ground_truth="A1",
            generated_answer="A1",
            similarity_score=0.9,
            best_match_ground_truth="A1",
            all_scores=[{"ground_truth": "A1", "score": 0.9}],
            embedding_model_used="test-model",
            latency_ms=100,
            created_by=test_user.id,
        )
        result2 = SemanticSimilarityResult(
            course_id=test_course.id,
            group_name="Group B",
            question="Q2",
            ground_truth="A2",
            generated_answer="A2",
            similarity_score=0.8,
            best_match_ground_truth="A2",
            all_scores=[{"ground_truth": "A2", "score": 0.8}],
            embedding_model_used="test-model",
            latency_ms=100,
            created_by=test_user.id,
        )
        result3 = SemanticSimilarityResult(
            course_id=test_course.id,
            group_name="Group A",
            question="Q3",
            ground_truth="A3",
            generated_answer="A3",
            similarity_score=0.7,
            best_match_ground_truth="A3",
            all_scores=[{"ground_truth": "A3", "score": 0.7}],
            embedding_model_used="test-model",
            latency_ms=100,
            created_by=test_user.id,
        )
        db_session.add_all([result1, result2, result3])
        db_session.commit()

        # Filter by Group A
        response = client.get(
            f"/api/semantic-similarity/results?course_id={test_course.id}&group_name=Group A"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2
        
        # Verify all results are from Group A
        for result in data["results"]:
            assert result["group_name"] == "Group A"

    def test_pagination(
        self, client, db_session, test_course, test_user, mock_auth
    ):
        """Test pagination of saved results."""
        # Create 15 results
        for i in range(15):
            result = SemanticSimilarityResult(
                course_id=test_course.id,
                group_name="Test Group",
                question=f"Q{i}",
                ground_truth=f"A{i}",
                generated_answer=f"A{i}",
                similarity_score=0.9,
                best_match_ground_truth=f"A{i}",
                all_scores=[{"ground_truth": f"A{i}", "score": 0.9}],
                embedding_model_used="test-model",
                latency_ms=100,
                created_by=test_user.id,
            )
            db_session.add(result)
        db_session.commit()

        # Get first page (10 results)
        response1 = client.get(
            f"/api/semantic-similarity/results?course_id={test_course.id}&skip=0&limit=10"
        )
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["total"] == 15
        assert len(data1["results"]) == 10

        # Get second page (5 results)
        response2 = client.get(
            f"/api/semantic-similarity/results?course_id={test_course.id}&skip=10&limit=10"
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["total"] == 15
        assert len(data2["results"]) == 5

        # Verify no overlap between pages
        ids_page1 = {r["id"] for r in data1["results"]}
        ids_page2 = {r["id"] for r in data2["results"]}
        assert len(ids_page1.intersection(ids_page2)) == 0

    def test_view_result_details(
        self, client, db_session, test_course, test_user, mock_auth
    ):
        """Test viewing details of a single saved result."""
        # Create a result
        result = SemanticSimilarityResult(
            course_id=test_course.id,
            group_name="Test Group",
            question="What is Python?",
            ground_truth="Python is a programming language.",
            alternative_ground_truths=["Python is a high-level language."],
            generated_answer="Python is a versatile programming language.",
            similarity_score=0.95,
            best_match_ground_truth="Python is a programming language.",
            all_scores=[
                {"ground_truth": "Python is a programming language.", "score": 0.95},
                {"ground_truth": "Python is a high-level language.", "score": 0.90}
            ],
            embedding_model_used="text-embedding-ada-002",
            latency_ms=150,
            created_by=test_user.id,
        )
        db_session.add(result)
        db_session.commit()
        db_session.refresh(result)

        # Get result details
        response = client.get(
            f"/api/semantic-similarity/results/{result.id}"
        )

        assert response.status_code == 200
        data = response.json()
        
        # Verify all fields are present
        assert data["id"] == result.id
        assert data["course_id"] == test_course.id
        assert data["group_name"] == "Test Group"
        assert data["question"] == "What is Python?"
        assert data["ground_truth"] == "Python is a programming language."
        assert data["alternative_ground_truths"] == ["Python is a high-level language."]
        assert data["generated_answer"] == "Python is a versatile programming language."
        assert data["similarity_score"] == 0.95
        assert data["best_match_ground_truth"] == "Python is a programming language."
        assert data["embedding_model_used"] == "text-embedding-ada-002"
        assert data["latency_ms"] == 150

    def test_delete_result(
        self, client, db_session, test_course, test_user, mock_teacher_auth
    ):
        """Test deleting a saved result."""
        # Create a result
        result = SemanticSimilarityResult(
            course_id=test_course.id,
            group_name="Test Group",
            question="Q1",
            ground_truth="A1",
            generated_answer="A1",
            similarity_score=0.9,
            best_match_ground_truth="A1",
            all_scores=[{"ground_truth": "A1", "score": 0.9}],
            embedding_model_used="test-model",
            latency_ms=100,
            created_by=test_user.id,
        )
        db_session.add(result)
        db_session.commit()
        db_session.refresh(result)
        result_id = result.id

        # Delete the result
        response = client.delete(
            f"/api/semantic-similarity/results/{result_id}"
        )

        assert response.status_code == 204

        # Verify result is deleted
        deleted_result = db_session.query(SemanticSimilarityResult).filter(
            SemanticSimilarityResult.id == result_id
        ).first()
        assert deleted_result is None

    def test_delete_nonexistent_result(
        self, client, mock_teacher_auth
    ):
        """Test deleting a result that doesn't exist."""
        response = client.delete(
            "/api/semantic-similarity/results/99999"
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
