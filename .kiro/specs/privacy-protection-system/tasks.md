# Implementation Plan: Privacy Protection System

## Overview

This implementation plan breaks down the Privacy Protection System into discrete, testable coding tasks. Each task builds incrementally on previous work, with regular checkpoints to ensure correctness. The plan follows a bottom-up approach: core detection logic → integration → testing infrastructure → UI components.

**Total Estimated Time**: 4-5 days (32-40 hours)

**Implementation Order**:
1. Core PII detection engine (Day 1)
2. Content safety and middleware integration (Day 2)
3. Database models and API endpoints (Day 2-3)
4. Test infrastructure and evaluation system (Day 3-4)
5. Frontend test interface (Day 4)

## Tasks

### 1. Set up project structure and dependencies

Create the foundational directory structure and install required dependencies for the privacy protection system.

- Create `backend/app/services/pii_detection.py`
- Create `backend/app/services/content_safety.py`
- Create `backend/app/middleware/privacy_middleware.py`
- Create `backend/app/data/` directory
- Create `backend/tests/test_pii_detection.py`
- Create `backend/tests/test_content_safety.py`
- Create `backend/tests/data/` directory for test datasets
- Add dependencies to `backend/requirements.txt`: `hypothesis` for property-based testing

_Requirements: All requirements (infrastructure)_

### 2. Implement Turkish name database

Create and populate the Turkish name database that will be used for name detection.

- [ ] 2.1 Create `backend/app/data/turkish_names.txt`
  - Populate with 1000+ common Turkish first names
  - Include names with Turkish characters (ç, ğ, ı, ö, ş, ü)
  - Use UTF-8 encoding
  - One name per line, lowercase
  - Sources: TDK, popular baby name lists
  - _Requirements: 10.3_

- [ ] 2.2 Implement name database loader in `TurkishPIIDetector`
  - `_load_turkish_names()` method
  - Load from file with UTF-8 encoding
  - Return as set for O(1) lookup
  - Cache in memory
  - Handle file not found gracefully
  - _Requirements: 10.1, 10.2, 10.3_

### 3. Implement core PII detection classes

Build the foundational data structures and main detector class.

- [ ] 3.1 Define PIIType enum and data classes
  - `PIIType` enum with 7 types (TC_KIMLIK, TELEFON, EMAIL, ISIM, IBAN, KREDI_KARTI, DOGUM_TARIHI)
  - `PIIMatch` dataclass with all required fields
  - `PIIDetectionResult` dataclass with all required fields
  - _Requirements: 1.1-1.7_

- [ ] 3.2 Implement `TurkishPIIDetector` class skeleton
  - `__init__()` method
  - `_load_resources()` method
  - Define regex patterns for all PII types
  - Initialize Turkish name database
  - _Requirements: 1.1-1.7, 10.1-10.6_

- [ ] 3.3 Implement main `detect()` method
  - Call all detection methods
  - Collect matches
  - Generate warnings
  - Calculate risk score
  - Apply masking
  - Measure processing time
  - Return `PIIDetectionResult`
  - _Requirements: 1.1-1.8, 2.1-2.5, 3.1-3.8_

### 4. Implement TC Kimlik detection and validation

Implement Turkish national ID detection with official validation algorithm.

- [ ] 4.1 Implement `_validate_tc_kimlik()` method
  - Check 11 digits
  - Validate first digit ≠ 0
  - Implement 10th digit checksum: (sum_odd * 7 - sum_even) mod 10
  - Implement 11th digit checksum: sum(first_10) mod 10
  - Return boolean
  - _Requirements: 1.1_

- [ ] 4.2 Implement `_detect_tc_kimlik()` method
  - Use regex pattern `\b[1-9]\d{10}\b`
  - Validate each match with `_validate_tc_kimlik()`
  - Create `PIIMatch` objects for valid matches
  - Set confidence to 1.0 (algorithm validated)
  - _Requirements: 1.1_

- [ ]* 4.3 Write property test for TC Kimlik detection
  - **Property 1: TC Kimlik Detection Completeness**
  - Generate random valid TC Kimlik numbers
  - Embed in random text
  - Verify 100% detection rate
  - **Validates: Requirements 1.1**

### 5. Implement phone number detection

Implement Turkish mobile phone number detection with format validation.

- [ ] 5.1 Implement `_detect_phone()` method
  - Use regex pattern `\b0?5\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b`
  - Normalize format (remove spaces/dashes)
  - Validate starts with 5XX (mobile)
  - Validate length is 10 digits
  - Set confidence to 0.95
  - _Requirements: 1.2_

- [ ]* 5.2 Write property test for phone detection
  - **Property 2: Phone Number Detection Accuracy**
  - Generate random Turkish mobile numbers in various formats
  - Verify ≥95% detection rate
  - **Validates: Requirements 1.2**

### 6. Implement email, IBAN, and date detection

Implement detection for email addresses, IBANs, and dates of birth.

- [ ] 6.1 Implement `_detect_email()` method
  - Use RFC-compliant regex pattern
  - Set confidence to 0.98
  - _Requirements: 1.3_

- [ ] 6.2 Implement `_detect_iban()` method
  - Use pattern for TR-format IBAN
  - Set confidence to 0.99
  - _Requirements: 1.5_

- [ ] 6.3 Implement `_detect_date_of_birth()` method
  - Use pattern for DD/MM/YYYY and DD.MM.YYYY
  - Validate date range (1900-2020)
  - Validate day (1-31), month (1-12)
  - Set confidence to 0.85
  - _Requirements: 1.7_

- [ ]* 6.4 Write property tests for email, IBAN, date detection
  - **Property 3: Email Detection Completeness**
  - **Property 5: Multiple PII Independence**
  - Generate random valid emails, IBANs, dates
  - Verify detection rates
  - **Validates: Requirements 1.3, 1.5, 1.7, 1.8**

### 7. Implement credit card detection with Luhn validation

Implement credit card number detection with Luhn algorithm validation.

- [ ] 7.1 Implement `_validate_luhn()` method
  - Double every second digit from right
  - If doubled > 9, subtract 9
  - Sum all digits
  - Valid if sum mod 10 = 0
  - _Requirements: 1.6_

- [ ] 7.2 Implement `_detect_credit_card()` method
  - Use regex pattern for 16-digit numbers
  - Validate with Luhn algorithm
  - Set confidence to 1.0 (algorithm validated)
  - _Requirements: 1.6_

- [ ]* 7.3 Write property test for credit card detection
  - Generate random valid credit card numbers
  - Verify 100% detection rate
  - **Validates: Requirements 1.6**

### 8. Implement Turkish name detection

Implement name detection using Turkish name database and capitalization rules.

- [ ] 8.1 Implement `_detect_names()` method
  - Split text into words
  - Check for capitalized words
  - Look up in Turkish name database
  - Check for adjacent capitalized word (surname)
  - Create `PIIMatch` for full name
  - Set confidence: 0.8 for first name only, 0.9 for full name
  - _Requirements: 1.4, 10.3, 10.4, 10.5_

- [ ]* 8.2 Write property test for name detection
  - **Property 4: Turkish Name Detection**
  - Generate random names from database
  - Embed in text with surnames
  - Verify ≥80% detection rate
  - **Validates: Requirements 1.4, 10.3**

### 9. Implement PII masking

Implement the masking algorithm that replaces detected PII with tokens.

- [ ] 9.1 Implement `_apply_masking()` method
  - Sort matches by position (reverse order)
  - Replace each PII with its masked_text token
  - Process from end to start to avoid position shifts
  - Return masked text
  - _Requirements: 2.1, 2.2, 2.3_

- [ ]* 9.2 Write property tests for masking
  - **Property 6: Masking Preserves Structure**
  - **Property 7: Masking Order Independence**
  - Generate random text with multiple PII
  - Verify structure preservation
  - Verify order independence
  - **Validates: Requirements 2.2, 2.3**

### 10. Implement risk score calculation

Implement the risk assessment algorithm that calculates overall risk score.

- [ ] 10.1 Implement `_calculate_risk_score()` method
  - Define risk weights for each PII type:
    - TC_KIMLIK: 1.0
    - KREDI_KARTI: 1.0
    - IBAN: 0.9
    - TELEFON: 0.7
    - EMAIL: 0.6
    - DOGUM_TARIHI: 0.5
    - ISIM: 0.4
  - Calculate weighted sum: sum(weight * confidence)
  - Normalize to 0.0-1.0 range
  - Return risk score
  - _Requirements: 3.1-3.5_

- [ ]* 10.2 Write property tests for risk assessment
  - **Property 9: Risk Score Blocking Threshold**
  - **Property 10: Risk Score Masking Threshold**
  - Generate text with various PII combinations
  - Verify correct risk score ranges
  - Verify correct actions (block/mask/allow)
  - **Validates: Requirements 3.6, 3.7, 3.8**

### 11. Checkpoint - Core PII detection complete

Ensure all core PII detection functionality is working correctly before moving to integration.

- Run all unit tests: `pytest backend/tests/test_pii_detection.py -v`
- Run all property tests with 100 iterations
- Verify all 7 PII types are detected correctly
- Verify masking works correctly
- Verify risk scores are calculated correctly
- Ask user if any issues or questions arise

### 12. Implement ContentSafetyFilter

Build the content safety filtering system for inappropriate content detection.

- [ ] 12.1 Define ContentIssueType enum and data classes
  - `ContentIssueType` enum (PROFANITY, VIOLENCE, SELF_HARM, DRUGS, SEXUAL, SPAM)
  - `ContentIssue` dataclass
  - `ContentSafetyResult` dataclass
  - _Requirements: 4.1-4.8_

- [ ] 12.2 Implement `ContentSafetyFilter` class
  - `__init__()` method
  - `_load_resources()` method
  - Define sensitive topic keywords:
    - Violence: öldür, vur, döv, kan, silah
    - Self-harm: intihar, kendimi öldür, canıma kıy, ölmek istiyorum
    - Drugs: uyuşturucu, esrar, kokain, eroin
    - Sexual: inappropriate keywords
  - Load profanity list (empty for now, can be populated later)
  - _Requirements: 4.1-4.5_

- [ ] 12.3 Implement content checking methods
  - `_contains_profanity()`: Check against profanity list
  - `_detect_sensitive_topics()`: Check for keyword matches
  - `_is_spam()`: Check for >70% uppercase OR <30% unique words
  - `_calculate_risk_level()`: Determine 'low', 'medium', 'high'
  - `_filter_content()`: Mask profanity with [FİLTRELENDİ]
  - _Requirements: 4.1-4.6_

- [ ] 12.4 Implement main `check()` method
  - Run all detection methods
  - Collect issues
  - Calculate risk level
  - Filter content if needed
  - Return `ContentSafetyResult`
  - _Requirements: 4.1-4.8_

- [ ]* 12.5 Write property tests for content safety
  - **Property 11: Profanity Detection**
  - **Property 12: Self-Harm Content Blocking**
  - **Property 13: Spam Pattern Detection**
  - **Property 14: High Severity Content Blocking**
  - Generate text with various content issues
  - Verify correct detection and severity levels
  - **Validates: Requirements 4.1, 4.3, 4.6, 4.7**

### 13. Implement database models

Create SQLAlchemy models for logging PII detections and content safety events.

- [ ] 13.1 Add models to `backend/app/models/db_models.py`
  - `PIIDetectionLog` model with all required fields
  - `ContentSafetyLog` model with all required fields
  - Add indexes on user_id and detected_at
  - _Requirements: 5.1-5.7_

- [ ] 13.2 Create Alembic migration
  - Generate migration: `alembic revision --autogenerate -m "Add privacy logging tables"`
  - Review migration file
  - Run migration: `alembic upgrade head`
  - _Requirements: 5.1-5.7_

### 14. Implement PrivacyMiddleware

Build the FastAPI middleware that automatically protects endpoints.

- [ ] 14.1 Implement `PrivacyMiddleware` class
  - `__init__()`: Initialize PII detector and content filter
  - Define `PROTECTED_ENDPOINTS` list
  - Implement `__call__()` async method
  - _Requirements: 7.1-7.6_

- [ ] 14.2 Implement request processing logic
  - Check if endpoint is protected
  - Extract text field from request body
  - Run PII detection
  - Run content safety check
  - Calculate overall risk
  - Make decision (block/mask/allow)
  - Modify request body if needed
  - _Requirements: 7.1-7.6_

- [ ] 14.3 Implement logging methods
  - `_log_pii_detection()`: Create PIIDetectionLog entry
  - `_log_content_safety()`: Create ContentSafetyLog entry
  - Handle database errors gracefully
  - _Requirements: 5.1-5.7_

- [ ] 14.4 Implement error responses
  - High-risk PII: Return HTTP 400 with details
  - High-severity content: Return HTTP 400 with details
  - Include masked_text in error response
  - _Requirements: 3.6, 4.7_

- [ ]* 14.5 Write integration tests for middleware
  - Test protected endpoints are intercepted
  - Test non-protected endpoints are not affected
  - Test high-risk blocking
  - Test medium-risk masking
  - Test low-risk allowing
  - Test logging is created
  - **Validates: Requirements 7.1-7.6**

### 15. Implement API endpoints

Create REST API endpoints for privacy detection and statistics.

- [ ] 15.1 Create `backend/app/routers/privacy.py`
  - Define Pydantic request/response models
  - `DetectRequest`, `PIIMatchResponse`, `DetectResponse`
  - _Requirements: 9.1-9.8_

- [ ] 15.2 Implement POST /api/privacy/detect endpoint
  - Accept text in request body
  - Run PII detection
  - Return detection results with all fields
  - _Requirements: 9.1-9.8_

- [ ] 15.3 Implement GET /api/privacy/stats endpoint
  - Query PIIDetectionLog and ContentSafetyLog
  - Calculate statistics:
    - Total detections
    - PII type breakdown
    - Risk distribution
    - Actions taken
  - Return statistics
  - _Requirements: 5.7, 9.8_

- [ ] 15.4 Register router in `backend/app/main.py`
  - Add privacy router
  - Add middleware to app
  - _Requirements: 7.1-7.6_

- [ ]* 15.5 Write API endpoint tests
  - Test /api/privacy/detect with various inputs
  - Test /api/privacy/stats returns correct data
  - Test error handling
  - **Validates: Requirements 9.1-9.8**

### 16. Checkpoint - Backend integration complete

Ensure all backend components are integrated and working together.

- Run all backend tests: `pytest backend/tests/ -v`
- Test middleware with actual HTTP requests
- Verify database logging works
- Verify API endpoints return correct data
- Test end-to-end flow: request → middleware → detection → logging → response
- Ask user if any issues or questions arise

### 17. Create test dataset

Build a comprehensive test dataset for scientific evaluation.

- [ ] 17.1 Create `backend/tests/data/pii_test_dataset.json`
  - Define JSON structure with metadata
  - _Requirements: 8.5_

- [ ] 17.2 Add 50 positive test cases
  - Various PII types (TC Kimlik, phone, email, name, IBAN, credit card, date)
  - Single PII and multiple PII combinations
  - Different text lengths and contexts
  - Mark expected PII with type, text, position, confidence
  - _Requirements: 8.5_

- [ ] 17.3 Add 30 negative test cases
  - Educational content with no PII
  - Questions about science, history, math
  - Ensure no false positives
  - _Requirements: 8.5_

- [ ] 17.4 Add 20 edge cases
  - Invalid formats (wrong TC Kimlik checksum, invalid phone)
  - Ambiguous cases (is "Ahmet Bey" a name or title?)
  - Boundary conditions (exactly 11 digits but not TC Kimlik)
  - Multiple formats of same PII type
  - _Requirements: 8.5_

### 18. Implement evaluation system

Build the scientific evaluation system that calculates metrics and generates reports.

- [ ] 18.1 Create `backend/tests/evaluate_pii_detection.py`
  - Define `EvaluationResult` dataclass
  - Define `PIIDetectionEvaluator` class
  - _Requirements: 8.1-8.9_

- [ ] 18.2 Implement evaluation logic
  - `_load_test_dataset()`: Load JSON test cases
  - `evaluate()`: Run detection on all test cases
  - `_evaluate_single_case()`: Compare expected vs detected
  - Calculate TP, FP, TN, FN for each case
  - Measure latency for each detection
  - _Requirements: 8.1-8.4, 6.1_

- [ ] 18.3 Implement metrics calculation
  - `_calculate_metrics()`: Calculate overall metrics
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
  - `_calculate_per_type_metrics()`: Calculate per-PII-type metrics
  - Calculate average latency
  - _Requirements: 8.1-8.4, 6.1_

- [ ] 18.4 Implement report generation
  - `generate_report()`: Create comprehensive report
  - Print metrics to console
  - `_create_visualizations()`: Generate charts
    - F1 Score by PII type (bar chart)
    - Precision vs Recall (grouped bar chart)
    - Confusion matrix (heatmap)
  - `_save_json_report()`: Export to JSON
  - _Requirements: 8.6-8.8_

- [ ]* 18.5 Write property test for evaluation metrics
  - **Property 18: Evaluation Precision Calculation**
  - **Property 19: Evaluation Recall Calculation**
  - **Property 20: Evaluation F1 Score Calculation**
  - Verify metric formulas are correct
  - **Validates: Requirements 8.1-8.3**

### 19. Run full evaluation and verify success criteria

Execute the complete evaluation and verify the system meets all performance targets.

- [ ] 19.1 Run evaluation script
  - Execute: `python backend/tests/evaluate_pii_detection.py`
  - Review console output
  - Check generated visualizations
  - Review JSON report
  - _Requirements: 8.1-8.9_

- [ ] 19.2 Verify success criteria
  - Precision ≥ 0.85 ✓
  - Recall ≥ 0.80 ✓
  - F1 Score ≥ 0.82 ✓
  - Latency < 100ms (average) ✓
  - False Positive Rate < 0.15 ✓
  - _Requirements: 8.9, 6.1_

- [ ] 19.3 Analyze results and iterate if needed
  - If metrics don't meet targets, identify weak areas
  - Adjust detection algorithms or thresholds
  - Re-run evaluation
  - Repeat until targets are met
  - _Requirements: 8.9_

### 20. Checkpoint - Evaluation system complete

Ensure evaluation system is working and producing publication-ready results.

- Verify all metrics are calculated correctly
- Verify visualizations are generated
- Verify JSON report is exported
- Verify success criteria are met
- Ask user if any issues or questions arise

### 21. Implement frontend types

Create TypeScript type definitions for the privacy system.

- [ ] 21.1 Create `frontend/src/types/privacy.ts`
  - `PIIType` enum
  - `PIIMatch` interface
  - `PIIDetectionResult` interface
  - `ContentIssue` interface
  - `ContentSafetyResult` interface
  - _Requirements: 9.1-9.8_

### 22. Implement privacy test page

Build the interactive web interface for testing the privacy system.

- [ ] 22.1 Create `frontend/src/app/dashboard/privacy-test/page.tsx`
  - Text input area (textarea)
  - "Detect PII" button
  - Loading state
  - Results display section
  - _Requirements: 9.1-9.5_

- [ ] 22.2 Implement detection logic
  - Call POST /api/privacy/detect
  - Handle loading state
  - Handle errors
  - Display results
  - _Requirements: 9.1-9.3_

- [ ] 22.3 Implement results display
  - Show detected PII list with type, text, confidence
  - Show masked text
  - Show risk score with color coding (red >0.8, yellow 0.3-0.8, green <0.3)
  - Show warnings
  - _Requirements: 9.2-9.4_

- [ ] 22.4 Add quick test buttons
  - Pre-filled examples for each PII type
  - "TC Kimlik Example" button
  - "Phone Example" button
  - "Email Example" button
  - "Name Example" button
  - "Multiple PII Example" button
  - _Requirements: 9.5_

### 23. Implement privacy warning component

Create a reusable component for displaying privacy warnings.

- [ ] 23.1 Create `frontend/src/components/privacy-warning.tsx`
  - Accept PIIDetectionResult as prop
  - Display alert with appropriate severity
  - List detected PII types
  - Show risk score
  - Show masked text
  - _Requirements: 7.3, 9.2-9.4_

### 24. Implement batch test interface

Build an interface for running batch evaluations from the frontend.

- [ ] 24.1 Create `frontend/src/app/dashboard/privacy-test/batch/page.tsx`
  - File upload for test dataset JSON
  - "Run Batch Test" button
  - Progress bar
  - Results table
  - _Requirements: 9.6-9.7_

- [ ] 24.2 Implement batch testing logic
  - Load test dataset
  - Run detection on each test case
  - Track progress
  - Calculate metrics (Precision, Recall, F1)
  - Display results in table
  - _Requirements: 9.6-9.7_

- [ ] 24.3 Add results visualization
  - Show metrics summary
  - Show per-type breakdown
  - Export results as CSV
  - _Requirements: 9.7_

### 25. Implement admin dashboard

Build an administrative dashboard for viewing logs and statistics.

- [ ] 25.1 Create `frontend/src/app/dashboard/privacy-admin/page.tsx`
  - Statistics cards (total detections, blocks, masks)
  - PII detection logs table
  - Content safety logs table
  - _Requirements: 9.8_

- [ ] 25.2 Implement logs display
  - Fetch from GET /api/privacy/stats
  - Display in paginated tables
  - Show timestamp, user_id, PII types, risk score, action
  - Add filters (date range, PII type, action)
  - _Requirements: 9.8_

- [ ] 25.3 Add charts
  - PII type distribution (pie chart)
  - Detections over time (line chart)
  - Risk score distribution (histogram)
  - Use Chart.js or Recharts
  - _Requirements: 9.8_

### 26. Final checkpoint - Complete system test

Perform end-to-end testing of the entire system.

- [ ] 26.1 Test complete flow
  - Student submits question via chat
  - Middleware intercepts and detects PII
  - PII is masked
  - Request forwarded to LLM
  - Response returned with warnings
  - Logs created in database
  - _Requirements: 7.1-7.6_

- [ ] 26.2 Test all UI components
  - Privacy test page works
  - Batch test interface works
  - Admin dashboard displays data
  - Warnings are shown correctly
  - _Requirements: 9.1-9.8_

- [ ] 26.3 Performance testing
  - Test with 100 concurrent requests
  - Verify latency < 100ms
  - Verify throughput ≥ 10 req/sec
  - Monitor memory usage
  - _Requirements: 6.1, 6.2_

- [ ] 26.4 Verify all success criteria
  - Technical: Precision ≥ 0.85, Recall ≥ 0.80, F1 ≥ 0.82, Latency < 100ms
  - Functional: All 7 PII types detected, masking works, risk assessment accurate
  - Compliance: Logging works, audit trail maintained
  - Scientific: 100+ test cases, metrics calculated, visualizations generated
  - User Experience: Test interface works, warnings clear
  - Integration: Seamlessly integrated with RAG chatbot

- Ask user for final review and approval

## Notes

- Tasks marked with `*` are optional property-based tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation and allow for course correction
- Property tests validate universal correctness properties with 100+ iterations
- Unit tests validate specific examples and edge cases
- The implementation follows a bottom-up approach: core logic → integration → testing → UI
- Total estimated time: 32-40 hours (4-5 days)

## Success Criteria

The implementation will be considered complete when:

1. ✅ All 7 PII types are detected correctly
2. ✅ Masking works without position errors
3. ✅ Risk assessment correctly blocks/masks/allows based on thresholds
4. ✅ Content safety filtering detects inappropriate content
5. ✅ Middleware automatically protects all specified endpoints
6. ✅ Database logging works for all detections
7. ✅ API endpoints return correct data
8. ✅ Evaluation system produces scientific metrics
9. ✅ Success criteria met: Precision ≥ 0.85, Recall ≥ 0.80, F1 ≥ 0.82, Latency < 100ms
10. ✅ Frontend test interface is functional
11. ✅ Admin dashboard displays logs and statistics
12. ✅ All tests pass (unit tests and property tests)
13. ✅ End-to-end flow works correctly
