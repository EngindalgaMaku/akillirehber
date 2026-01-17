# Implementation Plan: Reranker Integration

## Overview

This implementation plan breaks down the reranker integration into discrete, manageable tasks. The implementation follows a bottom-up approach: core service → database → API → frontend → testing.

## Tasks

- [x] 1. Setup and Dependencies
  - Install required packages (cohere SDK already installed)
  - Add reranker configuration constants
  - Set up environment variables
  - _Requirements: 1.1, 2.3, 4.3_

- [x] 2. Database Schema Updates
  - [x] 2.1 Create Alembic migration for reranker fields
    - Add `enable_reranker` boolean field (default: false)
    - Add `reranker_provider` string field (nullable)
    - Add `reranker_model` string field (nullable)
    - Add `reranker_top_k` integer field (default: 100)
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 2.2 Update CourseSettings model
    - Add reranker fields to SQLAlchemy model
    - Add default values
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 2.3 Update Pydantic schemas
    - Add reranker fields to CourseSettingsUpdate schema
    - Add validation for provider values
    - Add validation for reranker_top_k range (10-1000)
    - _Requirements: 6.7, 13.1_

  - [ ]* 2.4 Write property test for configuration validation
    - **Property 6: Configuration Validation**
    - **Validates: Requirements 6.6**

- [x] 3. Core Reranker Service
  - [x] 3.1 Create RerankService class structure
    - Initialize with API keys from environment
    - Set up client instances (lazy loading)
    - Implement cache (TTLCache with 5 min TTL)
    - _Requirements: 1.1, 8.1_

  - [x] 3.2 Implement cache key generation
    - Generate unique keys from query + document IDs + provider + model
    - Handle hash collisions
    - _Requirements: 10.2_

  - [ ]* 3.3 Write property test for cache key uniqueness
    - **Property 5: Cache Key Uniqueness**
    - **Validates: Requirements 10.2**

  - [x] 3.4 Implement provider routing logic
    - Route to correct provider based on provider parameter
    - Validate provider availability
    - _Requirements: 1.2, 8.1_

  - [ ]* 3.5 Write property test for provider routing
    - **Property 7: Provider Routing**
    - **Validates: Requirements 1.2**

- [x] 4. Cohere Reranker Implementation
  - [x] 4.1 Implement Cohere client initialization
    - Get API key from environment
    - Create Cohere client with error handling
    - _Requirements: 2.1, 2.3_

  - [x] 4.2 Implement _rerank_cohere method
    - Extract document texts
    - Call Cohere rerank API
    - Parse response and map back to documents
    - Add relevance_score and rerank_index fields
    - Handle API errors gracefully
    - _Requirements: 2.1, 2.2, 2.4, 8.2_

  - [x] 4.3 Add Cohere model configuration
    - Define available models (rerank-english-v3.0, rerank-multilingual-v3.0)
    - Add model metadata (languages, limits)
    - _Requirements: 2.2, 12.1_

  - [ ]* 4.4 Write unit tests for Cohere reranker
    - Test API call formatting
    - Test response parsing
    - Test error handling
    - _Requirements: 14.1_

  - [ ]* 4.5 Write property test for score ordering
    - **Property 1: Reranker Score Ordering**
    - **Validates: Requirements 7.4**

  - [ ]* 4.6 Write property test for document preservation
    - **Property 2: Document Preservation**
    - **Validates: Requirements 8.3**

- [x] 5. Alibaba Reranker Implementation
  - [x] 5.1 Implement Alibaba client initialization
    - Get DASHSCOPE_API_KEY from environment
    - Create Alibaba client with error handling
    - _Requirements: 4.1, 4.3_

  - [x] 5.2 Implement _rerank_alibaba method
    - Extract document texts
    - Call Alibaba rerank API
    - Parse response and map back to documents
    - Handle API errors gracefully
    - _Requirements: 4.1, 4.2, 4.4_

  - [x] 5.3 Add Alibaba model configuration
    - Define available models (gte-rerank, gte-rerank-hybrid)
    - Add model metadata
    - _Requirements: 4.2, 12.2_

  - [x]* 5.4 Write unit tests for Alibaba reranker
    - Test API call formatting
    - Test response parsing
    - Test Chinese content handling
    - _Requirements: 14.1_

- [x] 6. Weaviate Reranker Integration
  - [x] 6.1 Research Weaviate reranker module setup
    - Document Weaviate configuration requirements
    - Identify supported reranker providers
    - _Requirements: 5.1, 5.2_

  - [x] 6.2 Implement _rerank_weaviate method
    - Handle Weaviate-specific reranking
    - Document limitations (requires documents in Weaviate)
    - _Requirements: 5.1, 5.3_

  - [x] 6.3 Add Weaviate reranker configuration
    - Define available models
    - Add module requirements
    - _Requirements: 5.3, 12.3_

  - [ ]* 6.4 Write unit tests for Weaviate reranker
    - Test configuration
    - Test query formatting
    - _Requirements: 14.1_

- [ ] 7. Main Rerank Method Implementation
  - [x] 7.1 Implement main rerank() method
    - Check cache first
    - Route to appropriate provider
    - Handle errors with fallback
    - Store results in cache
    - Log performance metrics
    - _Requirements: 8.1, 8.2, 9.1, 10.2_

  - [x] 7.2 Implement input validation
    - Validate query is not empty
    - Validate documents list
    - Validate provider and model
    - _Requirements: 8.4, 8.5_

  - [ ]* 7.3 Write property test for empty input handling
    - **Property 10: Empty Input Handling**
    - **Validates: Requirements 8.4**

  - [ ]* 7.4 Write property test for top-k constraint
    - **Property 4: Top-K Constraint**
    - **Validates: Requirements 7.4**

  - [ ]* 7.5 Write property test for metadata preservation
    - **Property 9: Metadata Preservation**
    - **Validates: Requirements 8.3**

- [ ] 8. Error Handling and Fallback
  - [x] 8.1 Implement error handling patterns
    - Handle missing API keys
    - Handle invalid providers
    - Handle API rate limits
    - Handle timeouts (5 second limit)
    - Handle network errors
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 8.2 Implement fallback logic
    - Return original results on error
    - Log appropriate error levels
    - Track failure metrics
    - _Requirements: 9.1, 9.5_

  - [ ]* 8.3 Write property test for fallback consistency
    - **Property 3: Fallback Consistency**
    - **Validates: Requirements 9.1**

  - [ ]* 8.4 Write unit tests for error scenarios
    - Test each error type
    - Verify fallback behavior
    - Verify logging
    - _Requirements: 14.3_

- [ ] 9. RAG Service Integration
  - [x] 9.1 Update search_with_context method
    - Determine initial top_k based on reranker settings
    - Perform hybrid search with adjusted top_k
    - Call reranker if enabled
    - Handle reranker failures gracefully
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 9.2 Add reranker metrics logging
    - Log reranking success/failure
    - Log latency
    - Log score improvements
    - _Requirements: 1.5, 7.6, 10.4_

  - [ ]* 9.3 Write integration test for search with reranking
    - Test complete flow
    - Verify reranking improves results
    - Test fallback scenarios
    - _Requirements: 14.2_

- [ ] 10. API Endpoint Updates
  - [x] 10.1 Update course settings GET endpoint
    - Return reranker fields
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 10.2 Update course settings PUT endpoint
    - Accept reranker fields
    - Validate configuration
    - Return validation errors
    - _Requirements: 6.5, 6.6, 6.7_

  - [ ]* 10.3 Write API tests for settings endpoints
    - Test GET with reranker fields
    - Test PUT with valid configuration
    - Test PUT with invalid configuration
    - _Requirements: 14.1_

- [x] 11. Frontend - Settings UI
  - [x] 11.1 Add reranker section to Settings Tab
    - Add section header with icon
    - Add enable/disable toggle
    - Add helpful description
    - _Requirements: 11.1, 11.2_

  - [x] 11.2 Add provider selection dropdown
    - Show dropdown when reranker enabled
    - Options: Cohere, Alibaba, Weaviate
    - Add provider descriptions
    - _Requirements: 11.3_

  - [x] 11.3 Add model selection dropdown
    - Show models based on selected provider
    - Dynamic model list from backend
    - Add model descriptions
    - _Requirements: 11.4_

  - [x] 11.4 Add reranker_top_k slider
    - Range: 10-1000
    - Default: 100
    - Show current value
    - Add tooltip explaining purpose
    - _Requirements: 11.5_

  - [x] 11.5 Add tooltips and help text
    - Explain what reranking does
    - Explain each setting
    - Add links to documentation
    - _Requirements: 11.6_

  - [x] 11.6 Implement client-side validation
    - Validate provider selection
    - Validate model selection
    - Validate top_k range
    - Show validation errors
    - _Requirements: 11.7_

- [x] 12. Performance Optimization
  - [x] 12.1 Implement caching strategy
    - Set up TTLCache with 5 min TTL
    - Implement cache key generation
    - Add cache hit/miss metrics
    - _Requirements: 10.2, 10.3_

  - [x] 12.2 Add timeout handling
    - Set 5 second timeout for reranking
    - Return original results on timeout
    - Log timeout events
    - _Requirements: 9.3, 10.1_

  - [x] 12.3 Add performance monitoring
    - Track reranking latency
    - Track cache effectiveness
    - Track score improvements
    - _Requirements: 10.4, 10.5_

  - [ ]* 12.4 Write performance tests
    - Test latency with 100 documents
    - Test latency with 500 documents
    - Test cache effectiveness
    - _Requirements: 14.4_

- [ ] 13. Documentation
  - [ ] 13.1 Create reranker setup guide
    - Document API key setup for each provider
    - Document configuration options
    - Add usage examples
    - _Requirements: 15.1, 15.2_

  - [ ] 13.2 Update API documentation
    - Document new settings fields
    - Document reranker behavior
    - Add example requests/responses
    - _Requirements: 15.2_

  - [ ] 13.3 Create best practices guide
    - When to use reranking
    - Provider selection guide
    - Performance optimization tips
    - _Requirements: 15.5_

  - [ ] 13.4 Update environment variable documentation
    - Document COHERE_API_KEY
    - Document DASHSCOPE_API_KEY
    - Document optional configuration
    - _Requirements: 15.2_

- [ ] 14. Testing and Validation
  - [ ] 14.1 Run all unit tests
    - Verify all unit tests pass
    - Check test coverage
    - _Requirements: 14.1_

  - [ ] 14.2 Run all property tests
    - Verify all property tests pass (100+ iterations each)
    - Check for edge cases
    - _Requirements: 14.1_

  - [ ] 14.3 Run integration tests
    - Test end-to-end flows
    - Test provider switching
    - Test error scenarios
    - _Requirements: 14.2, 14.3_

  - [ ] 14.4 Run performance tests
    - Measure reranking latency
    - Verify cache effectiveness
    - Check memory usage
    - _Requirements: 14.4_

- [ ] 15. Final Checkpoint
  - Run full test suite
  - Verify backward compatibility
  - Test with real API keys
  - Verify documentation completeness
  - Ensure all tests pass, ask the user if questions arise
  - _Requirements: All_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and error cases
- Integration tests validate end-to-end workflows
- The implementation uses Python with existing Cohere SDK
- Weaviate native reranker requires module configuration
- Reranker is optional and disabled by default
- Graceful fallback is critical for reliability
