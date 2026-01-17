# Requirements Document: RAGAS Test Set Export & Multiple Test Runs

## Introduction

Bu doküman RAGAS sistemindeki iki önemli sorunu çözmek için gereksinimleri tanımlar:
1. Test seti export işleminde alternative ground truth'ların eksik olması
2. Aynı test seti ile birden fazla test yapabilme ve sonuçları karşılaştırabilme

## Glossary

- **Test Set**: Soru-cevap çiftlerinden oluşan değerlendirme veri seti
- **Evaluation Run**: Bir test setinin belirli parametrelerle çalıştırılması ve sonuçları
- **Ground Truth**: Bir soru için beklenen doğru cevap
- **Alternative Ground Truth**: Bir soru için kabul edilebilir alternatif doğru cevaplar
- **Export**: Test setini JSON formatında dışa aktarma işlemi

## Requirements

### Requirement 1: Complete Test Set Export

**User Story:** As a teacher, I want to export test sets with all their data including alternative ground truths, so that I can backup, share, or migrate my test data.

#### Acceptance Criteria

1. WHEN a user exports a test set, THE System SHALL include the alternative_ground_truths field for each question
2. WHEN a question has alternative ground truths, THE System SHALL export them as an array of strings
3. WHEN a question has no alternative ground truths, THE System SHALL export an empty array or null
4. WHEN the exported JSON is imported back, THE System SHALL preserve all alternative ground truths

### Requirement 2: Multiple Test Run Management

**User Story:** As a teacher, I want to run the same test set multiple times with different parameters and keep all results, so that I can compare different configurations and track improvements over time.

#### Acceptance Criteria

1. WHEN a user creates a new evaluation run, THE System SHALL always create a new run record (never update existing)
2. WHEN a user views a test set, THE System SHALL display all evaluation runs for that test set
3. WHEN displaying evaluation runs, THE System SHALL show the run name, parameters, date, and status
4. WHEN a user wants to run a test again, THE System SHALL suggest a unique name (e.g., "Test Run #2", "Test Run - 2024-01-13")
5. WHEN a user views evaluation results, THE System SHALL clearly indicate which run the results belong to

### Requirement 3: Test Run Naming and Organization

**User Story:** As a teacher, I want to give meaningful names to my test runs and organize them, so that I can easily identify and compare different experiments.

#### Acceptance Criteria

1. WHEN creating a new evaluation run, THE System SHALL allow the user to provide a custom name
2. WHEN no name is provided, THE System SHALL auto-generate a name with timestamp (e.g., "Run - 2024-01-13 14:30")
3. WHEN viewing test runs, THE System SHALL sort them by creation date (newest first)
4. WHEN a user wants to copy a test run configuration, THE System SHALL provide a "Run Again" button that pre-fills parameters

### Requirement 4: Test Run Comparison

**User Story:** As a teacher, I want to compare results from different test runs side-by-side, so that I can see which parameters produce better results.

#### Acceptance Criteria

1. WHEN viewing a test set, THE System SHALL provide a "Compare Runs" feature
2. WHEN comparing runs, THE System SHALL display metrics side-by-side in a table
3. WHEN comparing runs, THE System SHALL highlight the best score for each metric
4. WHEN comparing runs, THE System SHALL show the configuration differences between runs

### Requirement 5: Test Set Duplication (Optional)

**User Story:** As a teacher, I want to duplicate a test set, so that I can create variations for different courses or experiments.

#### Acceptance Criteria

1. WHEN a user clicks "Duplicate Test Set", THE System SHALL create a copy with all questions
2. WHEN duplicating, THE System SHALL append " (Copy)" to the test set name
3. WHEN duplicating, THE System SHALL preserve all question data including alternative ground truths
4. WHEN duplicating, THE System SHALL NOT copy evaluation runs (only the test set and questions)
