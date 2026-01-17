"""Tests for course settings API endpoints.

Feature: course-settings-enhancement
Property-based tests for system prompt validation.
"""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from pydantic import ValidationError

from app.models.schemas import CourseSettingsUpdate


class TestSystemPromptCharacterLimitProperty:
    """Property-based tests for system prompt character limit enforcement.

    Feature: course-settings-enhancement, Property 3: Character Limit Enforcement
    Validates: Requirements 5.1

    For any system prompt input exceeding 2000 characters, the system
    SHALL reject the input and display an appropriate error message.
    """

    @given(
        prompt_length=st.integers(min_value=2001, max_value=5000),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.too_slow,
        ]
    )
    def test_system_prompt_exceeding_2000_chars_rejected(self, prompt_length):
        """Property test: System prompts exceeding 2000 characters are rejected.

        Feature: course-settings-enhancement, Property 3: Character Limit Enforcement
        Validates: Requirements 5.1
        """
        # Generate a system prompt that exceeds 2000 characters
        long_prompt = "a" * prompt_length

        # Property: System prompt exceeding 2000 characters must raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            CourseSettingsUpdate(system_prompt=long_prompt)

        # Property: Error message should indicate string length constraint
        error_details = exc_info.value.errors()
        assert len(error_details) > 0
        assert any(
            "string_too_long" in str(err.get("type", "")) or
            "max_length" in str(err).lower() or
            "2000" in str(err)
            for err in error_details
        )

    @given(
        prompt_length=st.integers(min_value=0, max_value=2000),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.too_slow,
        ]
    )
    def test_system_prompt_within_limit_accepted(self, prompt_length):
        """Property test: System prompts within 2000 characters are accepted.

        Feature: course-settings-enhancement, Property 3: Character Limit Enforcement
        Validates: Requirements 5.1
        """
        # Generate a system prompt within the limit
        valid_prompt = "a" * prompt_length

        # Property: System prompt within 2000 characters must be accepted
        settings_update = CourseSettingsUpdate(system_prompt=valid_prompt)
        assert settings_update.system_prompt == valid_prompt
        assert len(settings_update.system_prompt) <= 2000

    @given(
        prompt=st.text(min_size=2001, max_size=3000),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
        ]
    )
    def test_system_prompt_with_various_characters_exceeding_limit_rejected(self, prompt):
        """Property test: System prompts with various characters exceeding limit are rejected.

        Feature: course-settings-enhancement, Property 3: Character Limit Enforcement
        Validates: Requirements 5.1
        """
        # Property: Any system prompt exceeding 2000 characters must be rejected
        with pytest.raises(ValidationError):
            CourseSettingsUpdate(system_prompt=prompt)

    def test_system_prompt_exactly_2000_chars_accepted(self):
        """Test that system prompt with exactly 2000 characters is accepted.

        Feature: course-settings-enhancement, Property 3: Character Limit Enforcement
        Validates: Requirements 5.1
        """
        # Edge case: exactly 2000 characters should be accepted
        exact_limit_prompt = "a" * 2000
        settings_update = CourseSettingsUpdate(system_prompt=exact_limit_prompt)
        assert len(settings_update.system_prompt) == 2000

    def test_system_prompt_2001_chars_rejected(self):
        """Test that system prompt with 2001 characters is rejected.

        Feature: course-settings-enhancement, Property 3: Character Limit Enforcement
        Validates: Requirements 5.1
        """
        # Edge case: 2001 characters should be rejected
        over_limit_prompt = "a" * 2001
        with pytest.raises(ValidationError):
            CourseSettingsUpdate(system_prompt=over_limit_prompt)

    def test_system_prompt_none_accepted(self):
        """Test that None system prompt is accepted.

        Feature: course-settings-enhancement, Property 3: Character Limit Enforcement
        Validates: Requirements 5.1
        """
        # None should be accepted (optional field)
        settings_update = CourseSettingsUpdate(system_prompt=None)
        assert settings_update.system_prompt is None

    def test_system_prompt_empty_string_accepted(self):
        """Test that empty string system prompt is accepted.

        Feature: course-settings-enhancement, Property 3: Character Limit Enforcement
        Validates: Requirements 5.1
        """
        # Empty string should be accepted
        settings_update = CourseSettingsUpdate(system_prompt="")
        assert settings_update.system_prompt == ""



class TestEmbeddingModelPersistenceProperty:
    """Property-based tests for embedding model persistence.

    Feature: alibaba-embedding-integration, Property 2: Embedding Model Persistence
    Validates: Requirements 1.2

    For any course, when the embedding model is updated to a specific value,
    retrieving the course settings should return that same embedding model value.
    """

    @given(
        model_name=st.one_of(
            st.just("openai/text-embedding-3-small"),
            st.just("openai/text-embedding-3-large"),
            st.just("openai/text-embedding-ada-002"),
            st.just("alibaba/text-embedding-v4"),
            st.text(min_size=1, max_size=255, alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='/-_.'
            ))
        )
    )
    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
        ]
    )
    def test_embedding_model_persistence(self, model_name):
        """Property test: Embedding model value persists after update.

        Feature: alibaba-embedding-integration, Property 2: Embedding Model Persistence
        Validates: Requirements 1.2
        """
        # Property: Setting an embedding model should persist that exact value
        settings_update = CourseSettingsUpdate(default_embedding_model=model_name)
        assert settings_update.default_embedding_model == model_name

    @given(
        model_name=st.sampled_from([
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large",
            "openai/text-embedding-ada-002",
            "alibaba/text-embedding-v4",
        ])
    )
    @settings(max_examples=100)
    def test_supported_embedding_models_accepted(self, model_name):
        """Property test: All supported embedding models are accepted.

        Feature: alibaba-embedding-integration, Property 2: Embedding Model Persistence
        Validates: Requirements 1.2
        """
        # Property: All supported embedding models should be accepted
        settings_update = CourseSettingsUpdate(default_embedding_model=model_name)
        assert settings_update.default_embedding_model == model_name
        assert isinstance(settings_update.default_embedding_model, str)
        assert len(settings_update.default_embedding_model) > 0

    def test_alibaba_model_specifically_accepted(self):
        """Test that Alibaba model is specifically accepted.

        Feature: alibaba-embedding-integration, Property 2: Embedding Model Persistence
        Validates: Requirements 1.2
        """
        # Specific test for Alibaba model
        settings_update = CourseSettingsUpdate(
            default_embedding_model="alibaba/text-embedding-v4"
        )
        assert settings_update.default_embedding_model == "alibaba/text-embedding-v4"

    def test_embedding_model_none_uses_default(self):
        """Test that None embedding model uses default.

        Feature: alibaba-embedding-integration, Property 2: Embedding Model Persistence
        Validates: Requirements 1.2
        """
        # None should be accepted (will use default from database)
        settings_update = CourseSettingsUpdate(default_embedding_model=None)
        assert settings_update.default_embedding_model is None



class TestBackwardCompatibilityProperty:
    """Property-based tests for backward compatibility with existing settings.

    Feature: alibaba-embedding-integration, Property 7: Backward Compatibility
    Validates: Requirements 6.2

    For any existing course with a configured embedding model, loading the
    course settings should return the original model without modification.
    """

    @given(
        existing_model=st.sampled_from([
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large",
            "openai/text-embedding-ada-002",
            "text-embedding-3-small",  # Legacy format without prefix
            "text-embedding-ada-002",  # Legacy format without prefix
        ])
    )
    @settings(max_examples=100)
    def test_existing_models_remain_unchanged(self, existing_model):
        """Property test: Existing embedding models remain unchanged.

        Feature: alibaba-embedding-integration, Property 7: Backward Compatibility
        Validates: Requirements 6.2
        """
        # Property: Existing model values should be preserved exactly
        settings_update = CourseSettingsUpdate(default_embedding_model=existing_model)
        assert settings_update.default_embedding_model == existing_model

    @given(
        model_name=st.one_of(
            st.just("openai/text-embedding-3-small"),
            st.just("openai/text-embedding-3-large"),
            st.just("openai/text-embedding-ada-002"),
        )
    )
    @settings(max_examples=100)
    def test_openai_models_continue_working(self, model_name):
        """Property test: OpenAI models continue working after Alibaba integration.

        Feature: alibaba-embedding-integration, Property 7: Backward Compatibility
        Validates: Requirements 6.2
        """
        # Property: OpenAI models should continue to work
        settings_update = CourseSettingsUpdate(default_embedding_model=model_name)
        assert settings_update.default_embedding_model == model_name
        assert settings_update.default_embedding_model.startswith("openai/")

    def test_default_model_unchanged(self):
        """Test that default model remains openai/text-embedding-3-small.

        Feature: alibaba-embedding-integration, Property 7: Backward Compatibility
        Validates: Requirements 6.2
        """
        # The default should remain unchanged for backward compatibility
        # When no model is specified, the database default should be used
        settings_update = CourseSettingsUpdate()
        # When no default_embedding_model is provided, it should be None
        # (database will use its default)
        assert settings_update.default_embedding_model is None

    def test_legacy_model_format_accepted(self):
        """Test that legacy model formats are still accepted.

        Feature: alibaba-embedding-integration, Property 7: Backward Compatibility
        Validates: Requirements 6.2
        """
        # Legacy formats without provider prefix should still work
        legacy_models = [
            "text-embedding-3-small",
            "text-embedding-ada-002",
        ]
        for model in legacy_models:
            settings_update = CourseSettingsUpdate(default_embedding_model=model)
            assert settings_update.default_embedding_model == model



class TestDefaultEmbeddingModel:
    """Unit tests for default embedding model.

    Feature: alibaba-embedding-integration
    Validates: Requirements 6.1
    """

    def test_default_embedding_model_is_openai_small(self):
        """Test that new course settings default to openai/text-embedding-3-small.

        Feature: alibaba-embedding-integration
        Validates: Requirements 6.1
        """
        from app.models.db_models import CourseSettings
        from sqlalchemy import inspect
        
        # Check the column default at the database level
        mapper = inspect(CourseSettings)
        default_embedding_model_col = mapper.columns['default_embedding_model']
        
        # The default should be openai/text-embedding-3-small
        assert default_embedding_model_col.default is not None
        assert default_embedding_model_col.default.arg == "openai/text-embedding-3-small"

    def test_course_settings_schema_default(self):
        """Test that CourseSettingsUpdate schema has correct default.

        Feature: alibaba-embedding-integration
        Validates: Requirements 6.1
        """
        from app.models.schemas import CourseSettingsBase
        
        # Create a base settings schema with defaults
        settings = CourseSettingsBase()
        
        # The default should be openai/text-embedding-3-small
        assert settings.default_embedding_model == "openai/text-embedding-3-small"

    def test_update_schema_allows_none(self):
        """Test that update schema allows None (to keep existing value).

        Feature: alibaba-embedding-integration
        Validates: Requirements 6.1
        """
        # Update schema should allow None to keep existing value
        settings_update = CourseSettingsUpdate(default_embedding_model=None)
        assert settings_update.default_embedding_model is None

    def test_update_schema_allows_alibaba_model(self):
        """Test that update schema allows Alibaba model.

        Feature: alibaba-embedding-integration
        Validates: Requirements 6.1
        """
        # Update schema should allow Alibaba model
        settings_update = CourseSettingsUpdate(
            default_embedding_model="alibaba/text-embedding-v4"
        )
        assert settings_update.default_embedding_model == "alibaba/text-embedding-v4"
