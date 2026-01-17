# Privacy Protection System - Requirements Document

## Introduction

This document defines the requirements for a Privacy Protection System designed to safeguard Turkish high school students' personally identifiable information (PII) when using API-based Large Language Models (LLMs) for educational purposes. The system must comply with KVKK (Turkish Data Protection Law) and GDPR regulations while maintaining scientific rigor for academic research publication.

## Glossary

- **PII (Personally Identifiable Information)**: Any data that can identify an individual (TC Kimlik, phone, email, name, etc.)
- **System**: The Privacy Protection System including PII detection, content safety, and logging components
- **Student**: Turkish high school (lise) student using the RAG chatbot
- **LLM**: Large Language Model accessed via API (OpenAI, Claude, etc.)
- **Masking**: Replacing detected PII with placeholder tokens (e.g., [TC_KIMLIK])
- **Risk Score**: Numerical value (0.0-1.0) indicating severity of detected PII
- **KVKK**: Kişisel Verilerin Korunması Kanunu (Turkish Data Protection Law)

## Requirements

### Requirement 1: Turkish PII Detection

**User Story:** As a system administrator, I want to automatically detect Turkish PII in student messages, so that sensitive information is not sent to external LLM APIs.

#### Acceptance Criteria

1. WHEN a student submits text containing a valid TC Kimlik number, THE System SHALL detect it using the official validation algorithm (10th and 11th digit checksums)
2. WHEN a student submits text containing a Turkish mobile phone number (05XX XXX XX XX format), THE System SHALL detect it with at least 95% confidence
3. WHEN a student submits text containing an email address, THE System SHALL detect it using RFC-compliant regex patterns
4. WHEN a student submits text containing Turkish first and last names, THE System SHALL detect them using a database of 1000+ common Turkish names
5. WHEN a student submits text containing an IBAN (TR format), THE System SHALL detect it with at least 99% confidence
6. WHEN a student submits text containing a credit card number, THE System SHALL validate it using the Luhn algorithm
7. WHEN a student submits text containing a date of birth (DD/MM/YYYY or DD.MM.YYYY), THE System SHALL detect dates within reasonable range (1900-2020)
8. WHEN multiple PII types are present in the same text, THE System SHALL detect all instances independently

### Requirement 2: PII Masking

**User Story:** As a student, I want my personal information to be automatically masked, so that my privacy is protected when my questions are sent to the LLM.

#### Acceptance Criteria

1. WHEN PII is detected, THE System SHALL replace it with type-specific tokens ([TC_KIMLIK], [TELEFON], [EMAIL], [ISIM], etc.)
2. WHEN masking multiple PII instances, THE System SHALL process them from end to start to prevent position shift errors
3. WHEN masking is applied, THE System SHALL preserve the original text structure and readability
4. WHEN PII is masked, THE System SHALL maintain the semantic meaning of the question for the LLM
5. THE System SHALL store both original and masked text for audit purposes

### Requirement 3: Risk Assessment

**User Story:** As a system administrator, I want to assess the risk level of detected PII, so that I can take appropriate action based on severity.

#### Acceptance Criteria

1. WHEN TC Kimlik or credit card numbers are detected, THE System SHALL assign maximum risk weight (1.0)
2. WHEN IBAN is detected, THE System SHALL assign high risk weight (0.9)
3. WHEN phone numbers or emails are detected, THE System SHALL assign medium risk weight (0.6-0.7)
4. WHEN names or dates of birth are detected, THE System SHALL assign low risk weight (0.4-0.5)
5. WHEN calculating overall risk score, THE System SHALL normalize the result to 0.0-1.0 range
6. WHEN risk score exceeds 0.8, THE System SHALL block the request and return an error message
7. WHEN risk score is between 0.3 and 0.8, THE System SHALL mask PII and allow the request with warnings
8. WHEN risk score is below 0.3, THE System SHALL allow the request with minimal warnings

### Requirement 4: Content Safety Filtering

**User Story:** As a teacher, I want inappropriate content to be filtered, so that students cannot submit harmful or offensive messages.

#### Acceptance Criteria

1. WHEN text contains profanity from the Turkish profanity list, THE System SHALL flag it as high severity
2. WHEN text contains violence-related keywords (öldür, vur, döv, kan, silah), THE System SHALL flag it as high severity
3. WHEN text contains self-harm keywords (intihar, kendimi öldür, canıma kıy), THE System SHALL flag it as high severity and block the request
4. WHEN text contains drug-related keywords (uyuşturucu, esrar, kokain), THE System SHALL flag it as medium severity
5. WHEN text contains sexual content keywords, THE System SHALL flag it as medium severity
6. WHEN text exhibits spam characteristics (>70% uppercase or <30% unique words), THE System SHALL flag it as low severity
7. WHEN high severity content is detected, THE System SHALL block the request and log the incident
8. WHEN medium severity content is detected, THE System SHALL warn but allow the request

### Requirement 5: Logging and Audit Trail

**User Story:** As a compliance officer, I want all PII detections and content safety incidents to be logged, so that we can demonstrate KVKK/GDPR compliance.

#### Acceptance Criteria

1. WHEN PII is detected, THE System SHALL log the PII types, risk score, action taken, and timestamp
2. WHEN PII is detected, THE System SHALL store only a preview (first 100 characters) of the original text, not the full content
3. WHEN content safety issues are detected, THE System SHALL log the issue type, severity, action taken, and timestamp
4. WHEN logging, THE System SHALL associate logs with user_id if available
5. THE System SHALL store logs in a secure database with appropriate access controls
6. THE System SHALL retain logs for at least 6 months for compliance purposes
7. THE System SHALL provide API endpoints to query logs for administrative purposes

### Requirement 6: Performance Requirements

**User Story:** As a student, I want PII detection to be fast, so that my chatbot experience is not degraded.

#### Acceptance Criteria

1. WHEN processing a typical student question (50-200 words), THE System SHALL complete PII detection in less than 100ms on average
2. WHEN processing multiple requests concurrently, THE System SHALL maintain throughput of at least 10 requests per second
3. WHEN loading Turkish name database, THE System SHALL cache it in memory to avoid repeated file I/O
4. WHEN compiling regex patterns, THE System SHALL pre-compile them at initialization
5. THE System SHALL use efficient algorithms (e.g., back-to-front masking) to minimize processing overhead

### Requirement 7: Integration with RAG System

**User Story:** As a developer, I want the privacy system to integrate seamlessly with the existing RAG chatbot, so that all student interactions are protected.

#### Acceptance Criteria

1. WHEN a student submits a question via `/api/chat`, THE System SHALL automatically check for PII before forwarding to the LLM
2. WHEN a student uses RAGAS quick test via `/api/ragas/quick-test`, THE System SHALL apply the same privacy checks
3. WHEN PII is detected and masked, THE System SHALL include warnings in the response to inform the student
4. WHEN high-risk PII is detected, THE System SHALL return HTTP 400 with a clear error message
5. THE System SHALL implement as FastAPI middleware to ensure all protected endpoints are covered
6. THE System SHALL not interfere with non-protected endpoints (e.g., `/api/health`, `/api/settings`)

### Requirement 8: Scientific Evaluation and Testing

**User Story:** As a researcher, I want to scientifically evaluate the privacy system's effectiveness, so that I can include rigorous metrics in my academic paper.

#### Acceptance Criteria

1. WHEN evaluating the system, THE System SHALL provide Precision metric (TP / (TP + FP)) for each PII type
2. WHEN evaluating the system, THE System SHALL provide Recall metric (TP / (TP + FN)) for each PII type
3. WHEN evaluating the system, THE System SHALL provide F1 Score (harmonic mean of Precision and Recall) for each PII type
4. WHEN evaluating the system, THE System SHALL provide overall Accuracy metric ((TP + TN) / (TP + TN + FP + FN))
5. THE System SHALL include a test dataset with at least 100 test cases (50 positive, 30 negative, 20 edge cases)
6. THE System SHALL generate confusion matrices for each PII type
7. THE System SHALL generate visualizations (bar charts, heatmaps) for scientific publication
8. THE System SHALL export evaluation results in JSON format for further analysis
9. THE System SHALL achieve minimum performance targets: Precision ≥ 0.85, Recall ≥ 0.80, F1 Score ≥ 0.82

### Requirement 9: Test Interface

**User Story:** As a researcher, I want a web interface to test the privacy system, so that I can interactively evaluate its performance and demonstrate it to stakeholders.

#### Acceptance Criteria

1. WHEN accessing the test page, THE System SHALL provide a text input area for entering test messages
2. WHEN clicking "Detect", THE System SHALL display all detected PII with type, position, and confidence
3. WHEN PII is detected, THE System SHALL display the masked version of the text
4. WHEN PII is detected, THE System SHALL display the calculated risk score
5. THE System SHALL provide quick test buttons with pre-filled examples (TC Kimlik, phone, email, name, multiple PII)
6. THE System SHALL provide a batch test interface for evaluating multiple test cases at once
7. WHEN running batch tests, THE System SHALL display progress and results in a table format
8. THE System SHALL provide an admin dashboard showing PII detection logs and statistics

### Requirement 10: Turkish Language Optimization

**User Story:** As a Turkish student, I want the system to correctly handle Turkish characters and names, so that my privacy is protected accurately.

#### Acceptance Criteria

1. THE System SHALL correctly handle Turkish characters (ç, ğ, ı, ö, ş, ü) in all PII detection algorithms
2. THE System SHALL use UTF-8 encoding for all text processing and storage
3. THE System SHALL include a comprehensive list of Turkish first names (at least 1000 names)
4. WHEN detecting names, THE System SHALL recognize Turkish surname patterns (e.g., -oğlu, -can, -er suffixes)
5. WHEN detecting names, THE System SHALL use case-sensitive matching (Turkish names start with uppercase)
6. THE System SHALL correctly parse Turkish date formats (DD/MM/YYYY and DD.MM.YYYY)

## Success Criteria

The Privacy Protection System will be considered successful when:

1. **Technical Performance**: Precision ≥ 0.85, Recall ≥ 0.80, F1 Score ≥ 0.82, Latency < 100ms
2. **Functional Completeness**: All 7 PII types detected, masking works correctly, risk assessment accurate
3. **Compliance**: KVKK/GDPR compliant logging, audit trail maintained, data retention policies enforced
4. **Scientific Rigor**: 100+ test cases, comprehensive evaluation metrics, publication-ready visualizations
5. **User Experience**: Test interface functional, warnings clear, no false positives on educational content
6. **Integration**: Seamlessly integrated with RAG chatbot, all protected endpoints covered

## Out of Scope

The following are explicitly out of scope for this version:

1. Detection of PII in images or audio
2. Real-time monitoring dashboard with live updates
3. Machine learning-based PII detection (rule-based only)
4. Multi-language support beyond Turkish
5. Automatic anonymization of historical data
6. Integration with external compliance tools
7. User consent management interface
