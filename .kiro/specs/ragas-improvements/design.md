# Design Document: RAGAS Test Set Export & Multiple Test Runs

## Overview

Bu tasarım, RAGAS sisteminde test seti export işlemini düzeltir ve birden fazla test çalıştırma yönetimini iyileştirir. Sistem zaten birden fazla evaluation run'ı destekliyor, ancak kullanıcı arayüzü ve export fonksiyonları eksik.

## Architecture

### Current State
- Backend zaten her test için yeni `EvaluationRun` kaydı oluşturuyor ✅
- Export endpoint'i `alternative_ground_truths` alanını eksik bırakıyor ❌
- Frontend'de test run'ları yeterince görünür değil ❌

### Proposed Changes
1. Export endpoint'ini güncelle (alternative_ground_truths ekle)
2. Frontend'de test run yönetimini iyileştir
3. Test run karşılaştırma özelliği ekle (opsiyonel)

## Components and Interfaces

### 1. Backend - Export Endpoint Update

**File**: `backend/app/routers/ragas.py`

**Current Implementation**:
```python
return TestSetExport(
    name=test_set.name,
    description=test_set.description,
    questions=[{
        "question": q.question,
        "ground_truth": q.ground_truth,
        "expected_contexts": q.expected_contexts,
        "question_metadata": q.question_metadata,
    } for q in questions],
)
```

**Updated Implementation**:
```python
return TestSetExport(
    name=test_set.name,
    description=test_set.description,
    questions=[{
        "question": q.question,
        "ground_truth": q.ground_truth,
        "alternative_ground_truths": q.alternative_ground_truths,  # ADD THIS
        "expected_contexts": q.expected_contexts,
        "question_metadata": q.question_metadata,
    } for q in questions],
)
```

### 2. Frontend - Test Run Display

**File**: `frontend/src/app/dashboard/ragas/test-sets/[id]/page.tsx`

**Current State**: Test runs görünmüyor veya yeterince belirgin değil

**Proposed UI**:
```
┌─────────────────────────────────────────────────────┐
│ Test Set: Python Basics                            │
│ ─────────────────────────────────────────────────── │
│                                                     │
│ [Questions Tab] [Evaluation Runs Tab]              │
│                                                     │
│ Evaluation Runs (3)                                 │
│ ┌─────────────────────────────────────────────┐   │
│ │ ✓ Run #1 - Baseline                         │   │
│ │   2024-01-13 10:30 | 10 questions           │   │
│ │   Avg Correctness: 85%                      │   │
│ │   [View Results] [Run Again]                │   │
│ └─────────────────────────────────────────────┘   │
│ ┌─────────────────────────────────────────────┐   │
│ │ ✓ Run #2 - Higher Temperature               │   │
│ │   2024-01-13 14:15 | 10 questions           │   │
│ │   Avg Correctness: 82%                      │   │
│ │   [View Results] [Run Again]                │   │
│ └─────────────────────────────────────────────┘   │
│                                                     │
│ [+ New Evaluation Run]                              │
└─────────────────────────────────────────────────────┘
```

### 3. Frontend - New Evaluation Dialog Enhancement

**Current**: Dialog sadece parametreleri alıyor
**Proposed**: Run name alanı ekle

```typescript
interface EvaluationDialogState {
  name: string;  // ADD THIS - with auto-generated default
  config: EvaluationConfig;
  evaluation_model?: string;
}
```

**Auto-generated name format**: `"Run - {date} {time}"` veya `"Run #{count}"`

### 4. Frontend - Run Comparison (Optional)

**New Component**: `RunComparisonDialog`

```
┌─────────────────────────────────────────────────────┐
│ Compare Evaluation Runs                             │
│ ─────────────────────────────────────────────────── │
│                                                     │
│ Select runs to compare:                             │
│ ☑ Run #1 - Baseline                                │
│ ☑ Run #2 - Higher Temperature                      │
│ ☐ Run #3 - Different Model                         │
│                                                     │
│ [Compare Selected]                                  │
│                                                     │
│ Comparison Results:                                 │
│ ┌─────────────────────────────────────────────┐   │
│ │ Metric          │ Run #1  │ Run #2  │ Best  │   │
│ │─────────────────│─────────│─────────│───────│   │
│ │ Faithfulness    │ 0.85    │ 0.82    │ Run#1 │   │
│ │ Answer Correct. │ 0.88    │ 0.90    │ Run#2 │   │
│ │ Context Recall  │ 0.75    │ 0.78    │ Run#2 │   │
│ └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Data Models

### TestSetExport Schema (Updated)

```python
class TestSetExport(BaseModel):
    name: str
    description: Optional[str] = None
    questions: List[TestQuestionBase]  # TestQuestionBase already has alternative_ground_truths
```

**Note**: `TestQuestionBase` zaten `alternative_ground_truths` alanına sahip, sadece export endpoint'inde kullanılmıyor.

## Error Handling

### Export Errors
- **Missing alternative_ground_truths**: Null veya empty array olarak handle et
- **Large export files**: Limit yok, ancak browser download timeout'u olabilir

### Run Creation Errors
- **Duplicate run names**: İzin ver (timestamp farklı olacak)
- **Invalid parameters**: Validation error döndür

## Testing Strategy

### Unit Tests
1. Export endpoint'inin alternative_ground_truths döndürdüğünü test et
2. Auto-generated run name formatını test et
3. Run comparison logic'ini test et

### Integration Tests
1. Export → Import → Verify alternative_ground_truths preserved
2. Create multiple runs → Verify all stored separately
3. Compare runs → Verify correct metrics displayed

### Manual Testing
1. Export bir test set → JSON'da alternative_ground_truths var mı kontrol et
2. Aynı test seti ile 3 farklı run oluştur → Hepsi görünüyor mu?
3. Run'ları karşılaştır → Metrikler doğru mu?
