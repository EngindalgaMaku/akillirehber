# Requirements Document

## Introduction

This document specifies the requirements for a Semantic Similarity Test feature that will be added to the RAG evaluation system. The feature will measure the semantic similarity between ground truth answers and generated answers using cosine similarity on embedding vectors. This test will complement the existing RAGAS evaluation metrics by providing a direct, quantitative measure of answer similarity.

## Glossary

- **Semantic_Similarity_Test**: A test that measures how semantically similar two text passages are using embedding vectors and cosine similarity
- **Embedding_Model**: A machine learning model that converts text into numerical vector representations
- **Cosine_Similarity**: A metric that measures the cosine of the angle between two vectors, ranging from -1 to 1, where 1 indicates identical direction
- **Ground_Truth**: The expected correct answer(s) for a given question
- **Generated_Answer**: The answer produced by the RAG system in response to a question
- **Quick_Test**: A single-question evaluation that provides immediate results
- **Batch_Test**: A multi-question evaluation that processes multiple test cases from JSON input
- **Session_Embedding_Model**: The embedding model configured in the course settings for the current session
- **Session_LLM**: The language model configured in the course settings for the current session
- **Test_Result**: A saved record of a semantic similarity test execution with all inputs and outputs

## Requirements

### Requirement 1: Semantic Similarity Calculation

**User Story:** As a teacher, I want to measure the semantic similarity between ground truth and generated answers, so that I can quantitatively assess answer quality.

#### Acceptance Criteria

1. WHEN a ground truth and generated answer are provided, THE Semantic_Similarity_Test SHALL compute embedding vectors for both texts using the Session_Embedding_Model
2. WHEN multiple alternative ground truths are provided, THE Semantic_Similarity_Test SHALL compute similarity scores for each alternative
3. WHEN multiple similarity scores exist, THE Semantic_Similarity_Test SHALL return the maximum similarity score as the final result
4. THE Semantic_Similarity_Test SHALL compute cosine similarity between embedding vectors
5. THE Semantic_Similarity_Test SHALL return similarity scores as percentages between 0% and 100%

### Requirement 2: Quick Test Interface

**User Story:** As a teacher, I want to quickly test a single question's semantic similarity, so that I can immediately evaluate answer quality without creating a full test set.

#### Acceptance Criteria

1. WHEN the quick test interface is displayed, THE System SHALL provide input fields for question, ground truth, and generated answer
2. WHEN alternative ground truths are needed, THE System SHALL allow adding multiple alternative ground truth inputs
3. WHEN the user provides a question without a generated answer, THE System SHALL use the Session_LLM to generate an answer using the RAG pipeline
4. WHEN the user provides a pre-generated answer, THE System SHALL skip answer generation and directly compute similarity
5. WHEN the test completes, THE System SHALL display the similarity score with visual indicators (color-coded by score range)
6. WHEN the test completes, THE System SHALL display execution latency in milliseconds

### Requirement 3: Batch Test Interface

**User Story:** As a teacher, I want to test multiple questions at once using JSON input, so that I can efficiently evaluate semantic similarity across many test cases.

#### Acceptance Criteria

1. WHEN the batch test interface is displayed, THE System SHALL provide a JSON input field for multiple test cases
2. THE System SHALL accept JSON format with array of objects containing: question, ground_truth, alternative_ground_truths (optional), generated_answer (optional)
3. WHEN generated_answer is omitted from a test case, THE System SHALL generate answers using the Session_LLM and RAG pipeline
4. WHEN batch test is executed, THE System SHALL process all test cases sequentially
5. WHEN batch test completes, THE System SHALL display results in a table with: question, ground truth, generated answer, similarity score, and latency
6. WHEN batch test completes, THE System SHALL display aggregate statistics: average similarity, min similarity, max similarity, total latency

### Requirement 4: Result Persistence

**User Story:** As a teacher, I want to save semantic similarity test results, so that I can review them later and track improvements over time.

#### Acceptance Criteria

1. WHEN a quick test completes, THE System SHALL provide an option to save the result
2. WHEN saving a result, THE System SHALL allow specifying an optional group name for organization
3. WHEN a result is saved, THE System SHALL store: course_id, question, ground_truth, alternative_ground_truths, generated_answer, similarity_score, latency_ms, group_name, session settings used, and timestamp
4. THE System SHALL provide a saved results view that displays all saved results for the selected course
5. WHEN viewing saved results, THE System SHALL allow filtering by group name
6. WHEN viewing saved results, THE System SHALL support pagination with 10 results per page
7. WHEN viewing a saved result, THE System SHALL display all stored information including the similarity score and test parameters

### Requirement 5: Integration with Existing UI

**User Story:** As a teacher, I want the semantic similarity test to be easily accessible alongside RAGAS tests, so that I can use both evaluation methods seamlessly.

#### Acceptance Criteria

1. THE System SHALL add a "Semantic Similarity" navigation item in the sidebar under the RAGAS section
2. WHEN the semantic similarity page loads, THE System SHALL display the course selector consistent with the RAGAS page
3. THE System SHALL use the same visual design patterns as the RAGAS page (cards, colors, layouts)
4. THE System SHALL display the quick test section as an expandable card similar to RAGAS quick test
5. THE System SHALL display the batch test section as a separate expandable card
6. THE System SHALL display the saved results section as an expandable card with filtering and pagination

### Requirement 6: Session Configuration Usage

**User Story:** As a teacher, I want the semantic similarity test to use my course's configured models, so that results are consistent with my RAG system setup.

#### Acceptance Criteria

1. WHEN generating answers, THE System SHALL use the Session_LLM configured in course settings
2. WHEN computing embeddings, THE System SHALL use the Session_Embedding_Model configured in course settings
3. WHEN using the system prompt, THE System SHALL use the prompt configured in course settings
4. THE System SHALL allow viewing which models are being used for the current test
5. WHEN course settings are updated, THE System SHALL reflect the changes in subsequent tests

### Requirement 7: Error Handling

**User Story:** As a teacher, I want clear error messages when tests fail, so that I can understand and fix issues quickly.

#### Acceptance Criteria

1. WHEN required fields are missing, THE System SHALL display validation errors before test execution
2. WHEN the embedding service fails, THE System SHALL display an error message indicating the embedding service is unavailable
3. WHEN the LLM service fails during answer generation, THE System SHALL display an error message with the failure reason
4. WHEN JSON input is malformed in batch test, THE System SHALL display a parsing error with the specific issue
5. WHEN a test case fails in batch test, THE System SHALL continue processing remaining cases and mark the failed case in results

### Requirement 8: Performance Display

**User Story:** As a teacher, I want to see how long tests take to execute, so that I can understand the performance characteristics of my RAG system.

#### Acceptance Criteria

1. WHEN a quick test completes, THE System SHALL display the total execution time in milliseconds
2. WHEN a batch test completes, THE System SHALL display individual latency for each test case
3. WHEN a batch test completes, THE System SHALL display total execution time for all test cases
4. THE System SHALL measure latency from test start to result availability
5. THE System SHALL display latency with millisecond precision
