# Requirements Document

## Introduction

This document specifies the requirements for integrating Alibaba's text-embedding-v4 model as an additional embedding option in the course settings. Currently, the system only supports OpenRouter's OpenAI embedding models. This enhancement will provide users with an alternative embedding provider, offering potential benefits in terms of cost, performance, or language support.

## Glossary

- **Embedding_Service**: The service responsible for generating vector embeddings from text
- **Course_Settings**: Configuration parameters specific to each course, including embedding model selection
- **Embedding_Model**: A machine learning model that converts text into numerical vector representations
- **OpenRouter**: Current embedding provider that proxies OpenAI models
- **Alibaba_Cloud**: Cloud service provider offering the text-embedding-v4 model
- **DashScope**: Alibaba Cloud's AI model service platform
- **Vector_Dimension**: The size of the embedding vector produced by a model

## Requirements

### Requirement 1: Alibaba Embedding Model Support

**User Story:** As a course administrator, I want to select Alibaba's text-embedding-v4 as my embedding model, so that I can use an alternative provider for generating embeddings.

#### Acceptance Criteria

1. WHEN a user views course settings, THE System SHALL display "alibaba/text-embedding-v4" as an available embedding model option
2. WHEN a user selects "alibaba/text-embedding-v4", THE System SHALL store this selection in the course settings
3. WHEN generating embeddings with Alibaba model selected, THE Embedding_Service SHALL use Alibaba's DashScope API
4. WHEN the Alibaba API key is not configured, THE System SHALL return a clear error message indicating the missing configuration

### Requirement 2: API Configuration

**User Story:** As a system administrator, I want to configure Alibaba API credentials, so that the system can authenticate with Alibaba's DashScope service.

#### Acceptance Criteria

1. THE System SHALL support DASHSCOPE_API_KEY environment variable for Alibaba API authentication
2. WHEN the DASHSCOPE_API_KEY is not set, THE System SHALL allow configuration but prevent embedding generation with Alibaba models
3. THE System SHALL validate the API key format before attempting API calls
4. WHEN an invalid API key is used, THE System SHALL return a descriptive error message

### Requirement 3: Embedding Generation

**User Story:** As a developer, I want the embedding service to support multiple providers, so that different embedding models can be used interchangeably.

#### Acceptance Criteria

1. WHEN generating embeddings, THE Embedding_Service SHALL route requests to the appropriate provider based on the model name
2. WHEN using "alibaba/text-embedding-v4", THE Embedding_Service SHALL call Alibaba's DashScope API
3. WHEN using "openai/*" models, THE Embedding_Service SHALL continue using OpenRouter API
4. THE Embedding_Service SHALL return embeddings in a consistent format regardless of provider
5. WHEN a provider API call fails, THE System SHALL return an error with provider-specific details

### Requirement 4: Vector Dimension Compatibility

**User Story:** As a system architect, I want to ensure vector dimensions are tracked correctly, so that embeddings from different models can be stored and retrieved properly.

#### Acceptance Criteria

1. THE Embedding_Service SHALL return the correct vector dimension for "alibaba/text-embedding-v4" (1024 dimensions)
2. WHEN switching embedding models for a course, THE System SHALL warn users about dimension incompatibility
3. THE System SHALL store the embedding model used for each document
4. WHEN querying with a different embedding model than used for indexing, THE System SHALL return a warning

### Requirement 5: Batch Processing Support

**User Story:** As a developer, I want batch embedding generation to work with Alibaba models, so that multiple texts can be processed efficiently.

#### Acceptance Criteria

1. WHEN generating embeddings for multiple texts, THE Embedding_Service SHALL support batch processing for Alibaba models
2. THE Embedding_Service SHALL respect Alibaba's API rate limits and batch size constraints
3. WHEN a batch request fails, THE System SHALL provide details about which texts failed
4. THE Embedding_Service SHALL maintain the same batch processing interface for all providers

### Requirement 6: Backward Compatibility

**User Story:** As an existing user, I want my current courses to continue working without changes, so that the new feature doesn't disrupt existing functionality.

#### Acceptance Criteria

1. WHEN a course has no embedding model specified, THE System SHALL default to "openai/text-embedding-3-small"
2. WHEN existing courses are loaded, THE System SHALL continue using their configured embedding models
3. THE System SHALL not require migration of existing embeddings
4. WHEN the Alibaba provider is unavailable, THE System SHALL continue functioning with OpenRouter models

### Requirement 7: Error Handling and Logging

**User Story:** As a system administrator, I want clear error messages and logs, so that I can troubleshoot embedding generation issues.

#### Acceptance Criteria

1. WHEN an embedding API call fails, THE System SHALL log the provider, model, and error details
2. WHEN rate limits are exceeded, THE System SHALL return a specific error indicating rate limiting
3. WHEN network errors occur, THE System SHALL distinguish between network and API errors
4. THE System SHALL log successful embedding generation with provider and model information
