# RAGAS Evaluation System - Technical Design

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Test Set    │  │ Run Tests   │  │ Results Dashboard       │ │
│  │ Management  │  │ Panel       │  │ (Charts, Tables)        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend API                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ /api/ragas/ │  │ Test Runner │  │ Results Storage         │ │
│  │ endpoints   │  │ Service     │  │ (PostgreSQL)            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAGAS Service (Docker)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ RAGAS       │  │ Evaluation  │  │ Metrics                 │ │
│  │ Library     │  │ Engine      │  │ Calculator              │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Database Schema

### TestSet
```sql
CREATE TABLE test_sets (
    id SERIAL PRIMARY KEY,
    course_id INTEGER REFERENCES courses(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### TestQuestion
```sql
CREATE TABLE test_questions (
    id SERIAL PRIMARY KEY,
    test_set_id INTEGER REFERENCES test_sets(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    expected_contexts TEXT[], -- Array of expected source texts
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### EvaluationRun
```sql
CREATE TABLE evaluation_runs (
    id SERIAL PRIMARY KEY,
    test_set_id INTEGER REFERENCES test_sets(id),
    course_id INTEGER REFERENCES courses(id),
    status VARCHAR(50) DEFAULT 'pending', -- pending, running, completed, failed
    config JSONB, -- chunk_size, overlap, embedding_model, etc.
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    total_questions INTEGER,
    processed_questions INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### EvaluationResult
```sql
CREATE TABLE evaluation_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    question_id INTEGER REFERENCES test_questions(id),
    question TEXT,
    ground_truth TEXT,
    generated_answer TEXT,
    retrieved_contexts TEXT[],
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    context_precision FLOAT,
    context_recall FLOAT,
    answer_correctness FLOAT,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### RunSummary (Aggregated metrics per run)
```sql
CREATE TABLE run_summaries (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    avg_faithfulness FLOAT,
    avg_answer_relevancy FLOAT,
    avg_context_precision FLOAT,
    avg_context_recall FLOAT,
    avg_answer_correctness FLOAT,
    avg_latency_ms FLOAT,
    total_questions INTEGER,
    successful_questions INTEGER,
    failed_questions INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## API Endpoints

### Test Set Management
- `POST /api/ragas/test-sets` - Create test set
- `GET /api/ragas/test-sets` - List test sets for course
- `GET /api/ragas/test-sets/{id}` - Get test set details
- `PUT /api/ragas/test-sets/{id}` - Update test set
- `DELETE /api/ragas/test-sets/{id}` - Delete test set
- `POST /api/ragas/test-sets/{id}/import` - Import questions from JSON
- `GET /api/ragas/test-sets/{id}/export` - Export to JSON

### Questions
- `POST /api/ragas/test-sets/{id}/questions` - Add question
- `PUT /api/ragas/questions/{id}` - Update question
- `DELETE /api/ragas/questions/{id}` - Delete question

### Evaluation
- `POST /api/ragas/evaluate` - Start evaluation run
- `GET /api/ragas/runs` - List evaluation runs
- `GET /api/ragas/runs/{id}` - Get run details with results
- `GET /api/ragas/runs/{id}/status` - Get run status (for polling)
- `DELETE /api/ragas/runs/{id}` - Delete run

### Results & Reports
- `GET /api/ragas/runs/{id}/summary` - Get aggregated metrics
- `GET /api/ragas/runs/{id}/export` - Export results (PDF/Excel)
- `GET /api/ragas/compare` - Compare multiple runs

## RAGAS Docker Service

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install ragas langchain openai datasets pandas

COPY ragas_service/ .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Service API
- `POST /evaluate` - Evaluate single question
- `POST /evaluate-batch` - Evaluate batch of questions
- `GET /health` - Health check

## Frontend Components

### Pages
1. **RAGAS Dashboard** (`/dashboard/ragas`)
   - Overview of all test sets and recent runs
   - Quick stats and trends

2. **Test Set Editor** (`/dashboard/ragas/test-sets/[id]`)
   - Add/edit/delete questions
   - Import/export functionality

3. **Run Evaluation** (`/dashboard/ragas/evaluate`)
   - Select test set and configuration
   - Real-time progress tracking

4. **Results Viewer** (`/dashboard/ragas/runs/[id]`)
   - Detailed metrics per question
   - Charts and visualizations
   - Export options

5. **Compare Runs** (`/dashboard/ragas/compare`)
   - Side-by-side comparison
   - Trend analysis

## Evaluation Flow

```
1. User creates test set with questions
2. User starts evaluation run with config
3. Backend creates EvaluationRun record
4. For each question:
   a. Call RAG chat endpoint to get answer + sources
   b. Send to RAGAS service for metric calculation
   c. Store result in EvaluationResult
   d. Update progress
5. Calculate and store RunSummary
6. Mark run as completed
```

## Configuration Options for Runs

```json
{
  "test_set_id": 1,
  "course_id": 1,
  "config": {
    "search_type": "hybrid",
    "search_alpha": 0.5,
    "top_k": 5,
    "llm_model": "openai/gpt-4o-mini",
    "llm_temperature": 0.7
  }
}
```
