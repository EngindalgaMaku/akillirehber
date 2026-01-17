# Implementation Plan: Semantic Similarity Test

## Overview

This implementation plan breaks down the Semantic Similarity Test feature into discrete, manageable tasks. The implementation follows a bottom-up approach: database models → services → API endpoints → frontend components → integration.

## Tasks

- [x] 1. Set up database models and migrations
  - Create SemanticSimilarityResult model in db_models.py
  - Create Alembic migration for new table
  - Add relationship to Course model
  - _Requirements: 4.3_

- [ ]* 1.1 Write property test for result persistence
  - **Property 8: Result Persistence Round-Trip**
  - **Validates: Requirements 4.3**

- [x] 2. Create Pydantic schemas
  - Create SemanticSimilarityQuickTestRequest schema
  - Create SemanticSimilarityQuickTestResponse schema
  - Create SemanticSimilarityBatchTestRequest schema
  - Create SemanticSimilarityBatchTestResponse schema
  - Create SemanticSimilarityResultCreate schema
  - Create SemanticSimilarityResultResponse schema
  - Create SemanticSimilarityResultListResponse schema
  - _Requirements: 1.1, 2.1, 3.1, 4.3_

- [x] 3. Implement SemanticSimilarityService
  - [x] 3.1 Implement cosine_similarity function
    - Compute dot product of two vectors
    - Compute norms of both vectors
    - Return cosine similarity score
    - Handle zero-length vectors
    - _Requirements: 1.4_

  - [ ]* 3.2 Write property test for cosine similarity bounds
    - **Property 1: Cosine Similarity Bounds**
    - **Validates: Requirements 1.4, 1.5**

  - [ ]* 3.3 Write property test for embedding consistency
    - **Property 3: Embedding Computation Consistency**
    - **Validates: Requirements 1.1, 1.4**

  - [x] 3.4 Implement compute_similarity method
    - Get embeddings for both texts using EmbeddingService
    - Call cosine_similarity function
    - Return score between 0.0 and 1.0
    - _Requirements: 1.1, 1.4_

  - [x] 3.5 Implement find_best_match method
    - Compute similarity for each ground truth
    - Find maximum score
    - Return max score, best match, and all scores
    - _Requirements: 1.2, 1.3_

  - [ ]* 3.6 Write property test for maximum similarity selection
    - **Property 2: Maximum Similarity Selection**
    - **Validates: Requirements 1.2, 1.3**

  - [x] 3.7 Implement generate_answer method
    - Reuse existing RAG pipeline (WeaviateService + LLMService)
    - Get course settings for LLM configuration
    - Return generated answer
    - _Requirements: 2.3, 3.3_

  - [ ]* 3.8 Write property test for conditional answer generation
    - **Property 4: Conditional Answer Generation**
    - **Validates: Requirements 2.3, 2.4, 3.3**

- [ ] 4. Checkpoint - Ensure service tests pass
  - Ensure all tests pass, ask the user if questions arise.


- [x] 5. Implement API router endpoints
  - [x] 5.1 Create semantic_similarity.py router file
    - Set up FastAPI router with prefix /api/semantic-similarity
    - Import dependencies and services
    - _Requirements: All_

  - [x] 5.2 Implement POST /quick-test endpoint
    - Validate request body
    - Check course access permissions
    - Get course settings for embedding/LLM models
    - Generate answer if not provided
    - Compute similarity using SemanticSimilarityService
    - Return response with all scores and metadata
    - _Requirements: 1.1, 1.2, 1.3, 2.3, 2.4_

  - [ ]* 5.3 Write property test for input validation
    - **Property 11: Input Validation**
    - **Validates: Requirements 7.1**

  - [x] 5.4 Implement POST /batch-test endpoint
    - Validate request body and JSON structure
    - Process each test case sequentially
    - Handle individual test case failures gracefully
    - Compute aggregate statistics
    - Return batch response
    - _Requirements: 3.2, 3.3, 3.4, 3.6, 7.5_

  - [ ]* 5.5 Write property test for JSON batch parsing
    - **Property 5: JSON Batch Parsing**
    - **Validates: Requirements 3.2**

  - [ ]* 5.6 Write property test for batch processing completeness
    - **Property 6: Batch Processing Completeness**
    - **Validates: Requirements 3.4, 7.5**

  - [ ]* 5.7 Write property test for aggregate statistics
    - **Property 7: Aggregate Statistics Accuracy**
    - **Validates: Requirements 3.6**

  - [ ]* 5.8 Write property test for malformed JSON handling
    - **Property 12: Malformed JSON Handling**
    - **Validates: Requirements 7.4**

  - [x] 5.9 Implement POST /results endpoint
    - Validate request body
    - Check course access permissions
    - Save result to database
    - Return saved result with ID
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 5.10 Implement GET /results endpoint
    - Validate query parameters
    - Check course access permissions
    - Apply group name filter if provided
    - Apply pagination (skip/limit)
    - Get unique group names for filter dropdown
    - Return paginated results with total count
    - _Requirements: 4.4, 4.5, 4.6_

  - [ ]* 5.11 Write property test for group filtering
    - **Property 9: Group Filtering Correctness**
    - **Validates: Requirements 4.5, 4.6**

  - [x] 5.12 Implement GET /results/{result_id} endpoint
    - Validate result ID
    - Check course access permissions
    - Return full result details
    - _Requirements: 4.7_

  - [x] 5.13 Implement DELETE /results/{result_id} endpoint
    - Validate result ID
    - Check teacher permissions
    - Delete result from database
    - Return 204 No Content
    - _Requirements: 4.7_

- [ ]* 5.14 Write unit tests for API endpoints
  - Test authentication and authorization
  - Test error responses
  - Test edge cases

- [x] 6. Checkpoint - Ensure API tests pass
  - Ensure all tests pass, ask the user if questions arise.


- [x] 7. Update frontend API client
  - [x] 7.1 Add TypeScript interfaces to api.ts
    - Add SemanticSimilarityQuickTestRequest interface
    - Add SemanticSimilarityQuickTestResponse interface
    - Add SemanticSimilarityBatchTestRequest interface
    - Add SemanticSimilarityBatchTestResponse interface
    - Add SemanticSimilarityResult interface
    - Add SemanticSimilarityResultListResponse interface
    - _Requirements: All_

  - [x] 7.2 Add API client methods
    - Add quickTest method
    - Add batchTest method
    - Add saveResult method
    - Add getResults method
    - Add getResult method
    - Add deleteResult method
    - _Requirements: All_

- [x] 8. Implement frontend page component
  - [x] 8.1 Create semantic-similarity page directory
    - Create frontend/src/app/dashboard/semantic-similarity/page.tsx
    - Set up basic page structure with PageHeader
    - Add course selector
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 8.2 Implement QuickTestCard component
    - Create expandable card with ChevronUp/Down
    - Add question input (Textarea)
    - Add ground truth input (Textarea)
    - Add alternative ground truths with dynamic add/remove
    - Add generated answer input (optional Textarea)
    - Add test button with loading state
    - Display similarity score with color coding
    - Display generated answer if applicable
    - Display latency
    - Add save result button
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 8.3 Implement BatchTestCard component
    - Create expandable card
    - Add JSON textarea with syntax highlighting
    - Add example format helper text
    - Add test button with loading state
    - Display results table with columns
    - Display aggregate statistics
    - Add export results button
    - _Requirements: 3.1, 3.2, 3.5, 3.6_

  - [x] 8.4 Implement SavedResultsCard component
    - Create expandable card
    - Add group filter dropdown
    - Display results list with pagination
    - Add result detail view dialog
    - Add delete result button
    - Implement load more functionality
    - _Requirements: 4.4, 4.5, 4.6, 4.7_

  - [x] 8.5 Implement save result dialog
    - Create dialog for saving quick test results
    - Add group name input (optional)
    - Add save button with loading state
    - Handle save success/error
    - _Requirements: 4.1, 4.2_

  - [x] 8.6 Add error handling and toast notifications
    - Display validation errors inline
    - Show toast for async operation results
    - Handle service unavailable errors
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 8.7 Add visual indicators for similarity scores
    - Implement getMetricColor function (green/yellow/red)
    - Implement getMetricBgColor function
    - Apply color coding to score displays
    - _Requirements: 2.5_

- [ ]* 8.8 Write unit tests for frontend components
  - Test form validation
  - Test user interactions
  - Test error state display
  - Test conditional rendering

- [x] 9. Update navigation sidebar
  - Add "Semantic Similarity" link under RAGAS section
  - Use appropriate icon (e.g., Target or Gauge)
  - Ensure consistent styling with other nav items
  - _Requirements: 5.1_

- [x] 10. Register API router in main application
  - Import semantic_similarity router in backend/app/main.py
  - Add router to FastAPI app
  - Verify router is accessible
  - _Requirements: All_

- [x] 11. Create database migration
  - Run alembic revision --autogenerate
  - Review generated migration
  - Test migration up and down
  - _Requirements: 4.3_

- [x] 12. Integration testing
  - [x] 12.1 Test quick test flow end-to-end
    - Submit quick test with all fields
    - Verify similarity calculation
    - Save result
    - Verify result appears in saved results
    - _Requirements: 1.1, 2.1, 4.1_

  - [x] 12.2 Test batch test flow end-to-end
    - Submit batch test with multiple cases
    - Verify all cases processed
    - Verify aggregate statistics
    - _Requirements: 3.1, 3.4, 3.6_

  - [x] 12.3 Test error scenarios
    - Test with missing required fields
    - Test with invalid JSON
    - Test with service failures
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 12.4 Test saved results management
    - Filter by group name
    - Paginate through results
    - View result details
    - Delete result
    - _Requirements: 4.4, 4.5, 4.6, 4.7_

- [ ]* 12.5 Write integration tests
  - Test complete user workflows
  - Test cross-component interactions

- [ ] 13. Final checkpoint - Ensure all tests pass
  - Run full test suite
  - Verify all property tests pass with 100+ iterations
  - Check code coverage meets targets
  - Ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Integration tests verify end-to-end workflows

