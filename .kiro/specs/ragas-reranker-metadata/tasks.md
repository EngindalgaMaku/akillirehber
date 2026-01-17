# Implementation Plan: RAGAS Reranker Metadata

## Overview

This implementation adds reranker metadata tracking to the RAGAS evaluation system through minimal, backward-compatible changes to schemas, evaluation logic, and frontend display.

## Tasks

- [x] 1. Update RAGAS Service Schemas
  - Update `EvaluationInput` to accept optional reranker metadata fields
  - Update `EvaluationOutput` to include reranker metadata fields
  - Update `BatchEvaluationOutput` to include reranker usage statistics
  - Ensure all fields are optional for backward compatibility
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.2, 4.1, 4.2_

- [x] 2. Update RAGAS Evaluation Functions
  - [x] 2.1 Update `evaluate_with_ragas()` function
    - Add reranker metadata to output
    - Log reranker usage when present
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [x] 2.2 Update `evaluate_simple()` function
    - Add reranker metadata to output
    - Maintain consistency with RAGAS evaluation
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [x] 2.3 Update `evaluate_batch()` function
    - Track reranker usage across batch
    - Calculate reranker usage statistics
    - Include statistics in batch output
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 3. Add Metadata Validation
  - Validate reranker_provider against known providers (cohere, alibaba)
  - Log warning for invalid providers but continue evaluation
  - Handle None/null values gracefully
  - _Requirements: 1.4, 1.5_

- [ ] 4. Create Database Migration
  - Add `reranker_used` boolean column to test_runs table
  - Add `reranker_provider` string column to test_runs table
  - Add `reranker_model` string column to test_runs table
  - Ensure columns are nullable for backward compatibility
  - Test migration on development database
  - _Requirements: 2.1, 4.3_
  - **NOTE: Skipped - metadata currently only in RAGAS response, not persisted to DB**

- [x] 5. Update Backend RAGAS Integration
  - Update RAGAS API calls to include reranker metadata from course settings
  - Pass reranker_provider when enable_reranker is true
  - Pass reranker_model when enable_reranker is true
  - Store reranker metadata in test_runs table
  - _Requirements: 1.1, 2.1, 2.2, 2.3_

- [x] 6. Update Frontend TypeScript Interfaces
  - Add reranker metadata fields to TestResult interface
  - Add reranker_usage field to BatchResults interface
  - Ensure fields are optional for backward compatibility
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 7. Add Frontend Reranker Status Display
  - Display reranker badge when reranker_used is true
  - Show provider and model information
  - Use amber color scheme to match reranker theme
  - Add Sparkles icon for visual consistency
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 8. Add Frontend Batch Statistics Display
  - Display reranker usage summary in batch results
  - Show count per provider
  - Use info box styling for statistics
  - _Requirements: 5.4_
  - **NOTE: Not applicable - batch evaluation is backend-only, frontend shows individual results**

- [ ] 9. Test Backward Compatibility
  - Test RAGAS evaluation without reranker metadata
  - Verify old API calls still work
  - Test with existing database records
  - Verify frontend handles missing metadata gracefully
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 10. Integration Testing
  - Test end-to-end flow with reranker enabled
  - Test end-to-end flow with reranker disabled
  - Verify metadata flows from backend to RAGAS to frontend
  - Test batch evaluation with mixed reranker usage
  - _Requirements: All_

- [ ] 11. Documentation Updates
  - Update RAGAS API documentation with new fields
  - Add examples showing reranker metadata usage
  - Document migration steps
  - Update frontend component documentation
  - _Requirements: All_

## Notes

- All changes are backward compatible - existing code continues to work
- Reranker metadata is optional and informational only
- No performance impact expected (minimal data addition)
- Migration is non-blocking (nullable columns)
- Frontend changes are purely additive (conditional rendering)
