# Bloom Taxonomy Update - Test Checklist

## Pre-Testing Setup
- [ ] Backend is running (Docker or local)
- [ ] Frontend is running on localhost:3000
- [ ] Database is accessible
- [ ] At least one course with documents and LLM settings configured

## Backend API Tests

### 1. Bloom Levels Endpoint
```bash
curl -X GET "http://localhost:8000/api/test-generation/bloom-levels" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Expected Response:**
```json
{
  "levels": [
    {
      "id": "remembering",
      "name": "Hatırlama",
      "description": "Temel tanım ve bilgi soruları (simple evolution)",
      "default_ratio": 0.30
    },
    {
      "id": "understanding_applying",
      "name": "Anlama/Uygulama",
      "description": "Senaryo ve problem çözme soruları (reasoning/conditional evolution)",
      "default_ratio": 0.40
    },
    {
      "id": "analyzing_evaluating",
      "name": "Analiz/Değerlendirme",
      "description": "Karşılaştırma ve değerlendirme soruları (multi_context/comparative evolution)",
      "default_ratio": 0.30
    }
  ],
  "total_ratio": 1.0
}
```

- [ ] Endpoint returns 200 OK
- [ ] All three levels are present with correct IDs
- [ ] Default ratios sum to 1.0 (30% + 40% + 30%)
- [ ] Descriptions include RAGAS evolution types

### 2. Question Generation Test
```bash
# Create a test set first, then generate questions
curl -X POST "http://localhost:8000/api/test-generation/generate-from-course" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "test_set_id=YOUR_TEST_SET_ID" \
  -F "total_questions=30" \
  -F "remembering_ratio=0.30" \
  -F "understanding_applying_ratio=0.40" \
  -F "analyzing_evaluating_ratio=0.30"
```

**Expected Behavior:**
- [ ] Generates 30 questions total
- [ ] ~9 questions with bloom_level="remembering"
- [ ] ~12 questions with bloom_level="understanding_applying"
- [ ] ~9 questions with bloom_level="analyzing_evaluating"
- [ ] Each question has question_metadata with bloom_level field

## Frontend UI Tests

### 1. Test Generation Page (`/dashboard/ragas/test-sets/generate`)

**Bloom Level Sliders:**
- [ ] Three sliders are visible
- [ ] Slider 1: "Hatırlama" - default 30%
- [ ] Slider 2: "Anlama/Uygulama" - default 40%
- [ ] Slider 3: "Analiz/Değerlendirme" - default 30%
- [ ] Total percentage shows 100% in green
- [ ] Descriptions mention RAGAS evolution types

**Generation Process:**
- [ ] Select a course
- [ ] Select or create a test set
- [ ] Adjust Bloom level ratios (ensure they sum to 100%)
- [ ] Click "Ders İçeriğinden Üret"
- [ ] Progress bar shows generation status
- [ ] Success message appears with question count
- [ ] Redirects to test set detail page

### 2. Test Set Detail Page (`/dashboard/ragas/test-sets/[id]`)

**Question Display:**
- [ ] Each question shows a colored Bloom level badge
- [ ] New questions show updated level names:
  - 🧠 Hatırlama (Blue)
  - 🔧 Anlama/Uygulama (Purple)
  - ⭐ Analiz/Değerlendirme (Orange)
- [ ] Legacy questions (if any) still display correctly:
  - 🔧 Uygulama/Analiz (Purple)
  - ⭐ Değerlendirme/Sentez (Orange)

**Question Content Verification:**
- [ ] "Hatırlama" questions ask for definitions, lists, or direct recall
- [ ] "Anlama/Uygulama" questions present scenarios or "how" questions
- [ ] "Analiz/Değerlendirme" questions require comparison or evaluation

## Database Verification

### Check Question Metadata
```sql
SELECT 
  id,
  question,
  question_metadata->>'bloom_level' as bloom_level,
  question_metadata->>'llm_provider' as llm_provider,
  question_metadata->>'llm_model' as llm_model
FROM test_questions
WHERE test_set_id = YOUR_TEST_SET_ID
ORDER BY id DESC
LIMIT 10;
```

**Expected:**
- [ ] New questions have bloom_level values: "remembering", "understanding_applying", or "analyzing_evaluating"
- [ ] Old questions (if any) may have: "applying_analyzing" or "evaluating_creating"
- [ ] All questions have llm_provider and llm_model in metadata

## Integration Tests

### 1. Full Workflow Test
1. [ ] Navigate to Test Generation page
2. [ ] Create a new test set named "Bloom Taxonomy Test"
3. [ ] Set total questions to 30
4. [ ] Keep default ratios (30/40/30)
5. [ ] Generate questions
6. [ ] Wait for completion
7. [ ] View generated questions
8. [ ] Verify distribution matches expectations
9. [ ] Check question quality for each Bloom level

### 2. Custom Ratio Test
1. [ ] Create another test set
2. [ ] Set custom ratios: 20% / 50% / 30%
3. [ ] Generate 20 questions
4. [ ] Verify distribution: ~4 / ~10 / ~6

### 3. Backward Compatibility Test
1. [ ] Find an old test set (created before this update)
2. [ ] View questions
3. [ ] Verify old Bloom level names still display correctly
4. [ ] Add new questions to the same test set
5. [ ] Verify both old and new questions display properly

## Performance Tests

- [ ] Generation of 50 questions completes in < 5 minutes
- [ ] UI remains responsive during generation
- [ ] Progress updates appear smoothly
- [ ] No memory leaks or connection issues

## Error Handling Tests

### Invalid Ratios
- [ ] Try ratios that don't sum to 100% - should show error
- [ ] Try negative ratios - should be prevented by UI
- [ ] Try ratios > 100% - should be prevented by UI

### Missing Configuration
- [ ] Try generating without selecting test set - should show error
- [ ] Try generating for course without LLM settings - should show error
- [ ] Try generating for course without documents - should show error

## Regression Tests

- [ ] Existing test sets still load correctly
- [ ] Existing evaluation runs still work
- [ ] Export/import functionality still works
- [ ] Question editing still works
- [ ] Question deletion still works

## Sign-Off

- [ ] All backend tests pass
- [ ] All frontend tests pass
- [ ] Database verification complete
- [ ] Integration tests pass
- [ ] Performance acceptable
- [ ] Error handling works correctly
- [ ] No regressions detected

**Tested By:** _______________
**Date:** _______________
**Notes:** _______________
