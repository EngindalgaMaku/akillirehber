# Design Document

## Overview

This design document describes the implementation of an enhanced PDF report for RAGAS evaluation results. The report will include comprehensive statistical analysis, visualizations, and detailed configuration information to provide users with deep insights into their RAG system performance.

## Architecture

### Component Structure

```
frontend/src/app/dashboard/ragas/runs/[id]/
├── page.tsx (main component)
├── components/
│   ├── PDFReportGenerator.tsx (new)
│   └── StatisticsCalculator.ts (new utility)
└── utils/
    └── chartGenerator.ts (new utility)
```

### Data Flow

1. User clicks "PDF" export button
2. Component calculates statistics from evaluation results
3. Component generates charts using Chart.js
4. Component builds HTML with embedded charts
5. Browser opens print dialog with formatted report

## Components and Interfaces

### 1. StatisticsCalculator Utility

**Purpose**: Calculate statistical measures for metric arrays

**Interface**:
```typescript
interface MetricStatistics {
  mean: number;
  median: number;
  stdDev: number;
  variance: number;
  min: number;
  max: number;
  q1: number;
  q3: number;
}

function calculateStatistics(values: number[]): MetricStatistics;
function calculateOverallAverage(metrics: {
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number;
  answer_correctness: number;
}): number;
```

### 2. ChartGenerator Utility

**Purpose**: Generate Chart.js charts as base64 images

**Interface**:
```typescript
interface ChartConfig {
  type: 'bar' | 'line' | 'boxplot';
  data: any;
  options: any;
}

async function generateChartImage(config: ChartConfig): Promise<string>;
async function generateMetricsBarChart(metrics: Record<string, number>): Promise<string>;
async function generateMetricsLineChart(results: EvaluationResult[]): Promise<string>;
async function generateBoxPlot(metricName: string, values: number[]): Promise<string>;
```

### 3. Enhanced PDF Report Generator

**Purpose**: Generate comprehensive PDF report with statistics and charts

**Key Functions**:
- `buildConfigurationSection()`: Extract and format model/settings info
- `buildStatisticsSection()`: Calculate and format statistical measures
- `buildVisualizationsSection()`: Generate and embed charts
- `buildQuestionAnalysisTable()`: Create sortable results table
- `generateEnhancedPDF()`: Orchestrate all sections into final HTML

## Data Models

### Enhanced Evaluation Result

```typescript
interface EnhancedEvaluationResult extends EvaluationResult {
  overall_score: number; // Average of all 5 metrics
  rank: number; // Ranking by overall score
}
```

### Report Statistics

```typescript
interface ReportStatistics {
  overall_average: number;
  metrics: {
    faithfulness: MetricStatistics;
    answer_relevancy: MetricStatistics;
    context_precision: MetricStatistics;
    context_recall: MetricStatistics;
    answer_correctness: MetricStatistics;
  };
  configuration: {
    llm_provider: string;
    llm_model: string;
    embedding_model: string;
    evaluation_model: string;
    search_alpha: number;
    search_top_k: number;
  };
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do.*

### Property 1: Statistical Calculations Accuracy

*For any* array of metric values, the calculated mean should equal the sum of values divided by the count, and standard deviation should be the square root of variance.

**Validates: Requirements 3.1, 3.2**

### Property 2: Overall Average Calculation

*For any* set of five metric averages, the overall average should equal their sum divided by five.

**Validates: Requirements 1.1**

### Property 3: Configuration Completeness

*For any* evaluation run with results, all configuration fields (LLM model, embedding model, evaluation model, search parameters) should be present in the PDF report.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

### Property 4: Chart Generation

*For any* valid metric data, chart generation should either produce a valid base64 image string or return a fallback error message.

**Validates: Requirements 8.4, 8.5**

### Property 5: Question Ranking Consistency

*For any* set of evaluation results, questions sorted by overall score should maintain descending order.

**Validates: Requirements 5.4**

## Error Handling

1. **Missing Data**: If configuration fields are missing, display "N/A"
2. **Chart Rendering Failure**: Show text-based fallback with data values
3. **Invalid Metric Values**: Filter out null/undefined before calculations
4. **Empty Results**: Display message "No results to display"
5. **Browser Print API Failure**: Show error toast with retry option

## Testing Strategy

### Unit Tests

- Test statistical calculation functions with known inputs
- Test overall average calculation
- Test chart configuration generation
- Test HTML template generation

### Property-Based Tests

- Test statistical calculations with random metric arrays (Property 1)
- Test overall average with random metric sets (Property 2)
- Test configuration extraction with various run data (Property 3)
- Test chart generation with edge cases (Property 4)
- Test question ranking with random scores (Property 5)

### Integration Tests

- Test full PDF generation with sample evaluation data
- Test print dialog opening
- Test PDF rendering in different browsers

## Implementation Notes

### Chart.js Configuration

- Use `chart.js` v4.x for modern features
- Use `chartjs-node-canvas` for server-side rendering (if needed)
- Configure charts with `responsive: false` and fixed dimensions for print
- Export charts as PNG with 2x resolution for print quality

### Statistical Calculations

- Use standard formulas for mean, median, std dev, variance
- Handle edge cases: empty arrays, single values, all same values
- Round display values to 1 decimal place for readability

### PDF Styling

- Use CSS Grid for layout
- Use CSS `@media print` rules for print-specific styling
- Use `page-break-inside: avoid` for sections
- Use consistent color palette matching the app theme

### Performance Considerations

- Generate charts asynchronously to avoid blocking UI
- Show loading indicator during PDF generation
- Cache chart images if generating multiple times
- Limit chart data points for large result sets (>100 questions)
