# Implementation Plan: Alibaba Embedding Integration

## Overview

This implementation plan breaks down the integration of Alibaba's text-embedding-v4 model into discrete, manageable tasks. The approach follows a provider-routing pattern where the embedding service detects model prefixes and routes to the appropriate API (OpenRouter or DashScope).

## Tasks

- [x] 1. Update environment configuration
  - Add DASHSCOPE_API_KEY to .env.example file
  - Document the new environment variable in README or deployment docs
  - _Requirements: 2.1_

- [ ] 2. Enhance EmbeddingService with multi-provider support
  - [x] 2.1 Add Alibaba DashScope constants and configuration
    - Add DASHSCOPE_BASE_URL constant
    - Add model prefix constants (ALIBABA_MODEL_PREFIX, OPENAI_MODEL_PREFIX)
    - Add ALIBABA_BATCH_SIZE constant (10 items)
    - _Requirements: 3.1, 5.2_

  - [x] 2.2 Implement provider detection logic
    - Add _get_provider_for_model() method to determine provider from model name
    - Add _get_client_for_model() method to get appropriate client
    - _Requirements: 3.1_

  - [ ]* 2.3 Write property test for provider routing
    - **Property 1: Provider Routing Based on Model Prefix**
    - **Validates: Requirements 3.1, 1.3, 3.2, 3.3**

  - [x] 2.4 Implement Alibaba DashScope client initialization
    - Add _get_alibaba_client() method
    - Handle DASHSCOPE_API_KEY environment variable
    - Raise clear error if API key is missing
    - _Requirements: 2.1, 1.4, 2.4_

  - [ ]* 2.5 Write unit tests for client initialization
    - Test OpenRouter client creation
    - Test Alibaba client creation
    - Test missing API key error messages
    - _Requirements: 1.4, 2.4_

- [x] 3. Update embedding generation methods
  - [x] 3.1 Modify get_embedding() to support multiple providers
    - Use _get_client_for_model() to get appropriate client
    - Maintain existing interface and behavior
    - _Requirements: 3.1, 3.4_

  - [x] 3.2 Modify get_embeddings() to support multiple providers
    - Use _get_client_for_model() to get appropriate client
    - Respect provider-specific batch sizes
    - _Requirements: 3.1, 5.1, 5.2_

  - [ ]* 3.3 Write property test for embedding format consistency
    - **Property 3: Consistent Embedding Format**
    - **Validates: Requirements 3.4**

  - [ ]* 3.4 Write property test for batch processing
    - **Property 5: Batch Processing Consistency**
    - **Validates: Requirements 5.1, 5.4**

  - [ ]* 3.5 Write property test for batch size constraints
    - **Property 6: Batch Size Constraints**
    - **Validates: Requirements 5.2**

- [x] 4. Update embedding dimension lookup
  - [x] 4.1 Add Alibaba model dimensions to get_embedding_dimension()
    - Add "alibaba/text-embedding-v4": 1024 to dimensions dict
    - _Requirements: 4.1_

  - [ ]* 4.2 Write unit test for dimension lookup
    - Test OpenAI model dimensions
    - Test Alibaba model dimensions
    - Test unknown model default
    - _Requirements: 4.1_

- [x] 5. Checkpoint - Ensure embedding service tests pass
  - Run all embedding service tests
  - Verify provider routing works correctly
  - Ensure all tests pass, ask the user if questions arise

- [x] 6. Add error handling and logging
  - [x] 6.1 Implement provider-specific error handling
    - Catch API errors and add provider context
    - Distinguish between configuration, API, and network errors
    - _Requirements: 3.5, 7.1, 7.2, 7.3_

  - [x] 6.2 Add logging for embedding operations
    - Log successful embedding generation with provider and model info
    - Log errors with provider, model, and error details
    - _Requirements: 7.1, 7.4_

  - [ ]* 6.3 Write unit tests for error handling
    - Test missing API key errors
    - Test API error responses
    - Test error logging
    - _Requirements: 1.4, 2.4, 3.5, 7.1, 7.2, 7.3_

- [x] 7. Update course settings integration
  - [x]* 7.1 Write property test for settings persistence
    - **Property 2: Embedding Model Persistence**
    - **Validates: Requirements 1.2**

  - [x]* 7.2 Write property test for backward compatibility
    - **Property 7: Backward Compatibility with Existing Settings**
    - **Validates: Requirements 6.2**

  - [x]* 7.3 Write unit test for default embedding model
    - Test that new course settings default to "openai/text-embedding-3-small"
    - _Requirements: 6.1_

- [x] 8. Update document embedding tracking
  - [x]* 8.1 Write property test for document model tracking
    - **Property 4: Document Embedding Model Tracking**
    - **Validates: Requirements 4.3**

  - [x]* 8.2 Write integration test for document embedding
    - Test embedding a document with Alibaba model
    - Verify model is stored in document record
    - _Requirements: 4.3_

- [x] 9. Add frontend support for Alibaba model
  - [x] 9.1 Update course settings UI to include Alibaba model option
    - Add "alibaba/text-embedding-v4" to embedding model dropdown
    - Display appropriate label with dimension info
    - _Requirements: 1.1_

  - [ ]* 9.2 Write unit test for settings UI
    - Test that Alibaba model appears in dropdown options
    - _Requirements: 1.1_
    - _Note: Skipped - no testing framework configured in frontend_

- [x] 10. Checkpoint - Integration testing
  - [x]* 10.1 Write integration test for end-to-end Alibaba embedding
    - Create course with Alibaba model
    - Upload and process document
    - Verify embeddings are generated
    - _Requirements: 1.2, 1.3, 3.1, 4.3_

  - [x]* 10.2 Write integration test for graceful degradation
    - **Property 8: Graceful Degradation Without Alibaba Key**
    - **Validates: Requirements 2.2, 6.4**

  - [x]* 10.3 Write integration test for provider switching
    - Create course with OpenRouter model
    - Generate embeddings
    - Switch to Alibaba model
    - Generate new embeddings
    - Verify both work correctly
    - _Requirements: 3.1, 6.2_

- [x] 11. Final checkpoint - Ensure all tests pass
  - Run full test suite
  - Verify all property tests pass (100+ iterations each)
  - Verify all unit tests pass
  - Verify all integration tests pass
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows
- The implementation uses Python with the existing OpenAI SDK
- Alibaba's OpenAI-compatible API simplifies integration significantly
