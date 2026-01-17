# Requirements Document

## Introduction

RAGAS değerlendirme sistemi, şu anda reranker kullanılıp kullanılmadığını bilmiyor ve raporlamıyor. Bu özellik, RAGAS değerlendirmelerine reranker metadata'sı ekleyerek, kullanıcıların hangi sonuçların reranked olduğunu görmesini sağlayacak.

## Glossary

- **RAGAS**: RAG (Retrieval-Augmented Generation) sistemlerini değerlendirmek için kullanılan metrik kütüphanesi
- **Reranker**: Arama sonuçlarını sorguya göre yeniden sıralayan servis
- **Metadata**: Değerlendirme sonuçlarıyla birlikte dönen ek bilgiler
- **Context**: RAG sisteminde LLM'e verilen doküman parçaları
- **Evaluation_Input**: RAGAS değerlendirmesi için gönderilen veri yapısı
- **Evaluation_Output**: RAGAS değerlendirmesinden dönen sonuç yapısı

## Requirements

### Requirement 1: Reranker Metadata İletimi

**User Story:** As a developer, I want to pass reranker metadata to RAGAS evaluation, so that I can track which evaluations used reranked contexts.

#### Acceptance Criteria

1. WHEN a RAGAS evaluation request is made, THE System SHALL accept optional reranker metadata
2. THE System SHALL accept reranker_provider field (cohere/alibaba/null)
3. THE System SHALL accept reranker_model field (model name or null)
4. WHEN reranker metadata is provided, THE System SHALL validate provider and model values
5. WHEN reranker metadata is invalid, THE System SHALL accept but log a warning

### Requirement 2: Reranker Metadata Storage

**User Story:** As a developer, I want reranker metadata to be stored with evaluation results, so that I can analyze the impact of reranking on evaluation metrics.

#### Acceptance Criteria

1. WHEN evaluation is complete, THE System SHALL include reranker metadata in the response
2. THE System SHALL return reranker_used boolean field (true if reranker was used)
3. WHEN reranker was used, THE System SHALL return reranker_provider field
4. WHEN reranker was used, THE System SHALL return reranker_model field
5. WHEN reranker was not used, THE System SHALL return null for provider and model fields

### Requirement 3: Batch Evaluation Metadata

**User Story:** As a developer, I want reranker metadata in batch evaluations, so that I can track reranker usage across multiple test cases.

#### Acceptance Criteria

1. WHEN batch evaluation is performed, THE System SHALL accept reranker metadata for each item
2. THE System SHALL return reranker metadata for each evaluation result
3. WHEN different items use different reranker settings, THE System SHALL track each separately
4. THE System SHALL include reranker usage summary in batch results

### Requirement 4: Backward Compatibility

**User Story:** As a developer, I want existing RAGAS evaluation code to continue working, so that I don't break existing integrations.

#### Acceptance Criteria

1. WHEN reranker metadata is not provided, THE System SHALL default to null values
2. THE System SHALL not require reranker metadata fields
3. WHEN old API format is used, THE System SHALL work without errors
4. THE System SHALL maintain existing response structure with added optional fields

### Requirement 5: Frontend Display

**User Story:** As a user, I want to see reranker information in RAGAS results, so that I understand which evaluations used reranking.

#### Acceptance Criteria

1. WHEN viewing RAGAS results, THE System SHALL display reranker status
2. WHEN reranker was used, THE System SHALL show provider and model information
3. THE System SHALL visually distinguish reranked vs non-reranked evaluations
4. THE System SHALL show reranker usage statistics in batch results

## Non-Functional Requirements

### Performance
- Metadata addition should not impact evaluation performance
- Metadata should add less than 1KB to response size

### Compatibility
- Must work with existing RAGAS evaluation flow
- Must not break existing API contracts
- Must support both single and batch evaluations

### Usability
- Metadata should be clearly documented
- Frontend should clearly show reranker status
- Logs should indicate when reranker metadata is present
