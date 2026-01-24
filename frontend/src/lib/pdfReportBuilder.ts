// PDF Report Builder

/**
 * PDF Report Builder Utility
 * 
 * This module provides functions to build HTML sections for the enhanced RAGAS PDF report.
 * Each function generates a specific section of the report with proper formatting and styling.
 */

import { EvaluationRunDetail, EvaluationResult } from './api';
import { MetricStatistics } from './statisticsCalculator';
import { ConfigurationInfo } from './configurationExtractor';

/**
 * Build the report header section
 * Includes report title, evaluation run name, and generation date/time
 * 
 * Requirements: 7.3, 6.5
 */
export function buildReportHeader(run: EvaluationRunDetail): string {
  const generationDate = new Date().toLocaleString('tr-TR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
  
  const evaluationDate = new Date(run.created_at).toLocaleString('tr-TR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });

  return `
    <div class="report-header">
      <h1 class="report-title">RAGAS Evaluation Report</h1>
      <div class="header-info">
        <div class="info-row">
          <span class="info-label">Evaluation Name:</span>
          <span class="info-value">${run.name || `Evaluation #${run.id}`}</span>
        </div>
        <div class="info-row">
          <span class="info-label">Evaluation Date:</span>
          <span class="info-value">${evaluationDate}</span>
        </div>
        <div class="info-row">
          <span class="info-label">Report Generated:</span>
          <span class="info-value">${generationDate}</span>
        </div>
        <div class="info-row">
          <span class="info-label">Total Questions:</span>
          <span class="info-value">${run.total_questions}</span>
        </div>
      </div>
    </div>
  `;
}

/**
 * Build the RAGAS metrics explanation section
 * Explains what each of the 5 RAGAS metrics measures
 * 
 * Requirements: User Experience, Documentation
 */
export function buildMetricsExplanation(): string {
  return `
    <div class="metrics-explanation-section" id="metrics-explanation">
      <h2>About RAGAS Metrics</h2>
      <p class="section-intro">
        RAGAS (Retrieval Augmented Generation Assessment) provides five key metrics to evaluate 
        the quality of RAG (Retrieval-Augmented Generation) systems. Each metric measures a different 
        aspect of system performance:
      </p>

      <div class="metric-explanation-grid">
        <div class="metric-explanation-card">
          <div class="metric-icon">📊</div>
          <h3>Faithfulness</h3>
          <p class="metric-description">
            Measures how factually accurate the generated answer is based on the retrieved context. 
            It evaluates whether the answer contains only information that can be verified from the 
            provided context, without hallucinations or unsupported claims.
          </p>
          <p class="metric-formula"><strong>Focus:</strong> Factual accuracy and groundedness</p>
        </div>

        <div class="metric-explanation-card">
          <div class="metric-icon">🎯</div>
          <h3>Answer Relevancy</h3>
          <p class="metric-description">
            Evaluates how relevant and appropriate the generated answer is to the original question. 
            A high score indicates the answer directly addresses the question without unnecessary 
            information or tangential content.
          </p>
          <p class="metric-formula"><strong>Focus:</strong> Question-answer alignment</p>
        </div>

        <div class="metric-explanation-card">
          <div class="metric-icon">🔍</div>
          <h3>Context Precision</h3>
          <p class="metric-description">
            Measures the quality of the retrieval system by evaluating whether the most relevant 
            context chunks are ranked higher. It assesses if the retrieval system successfully 
            prioritizes the most useful information.
          </p>
          <p class="metric-formula"><strong>Focus:</strong> Retrieval ranking quality</p>
        </div>

        <div class="metric-explanation-card">
          <div class="metric-icon">📚</div>
          <h3>Context Recall</h3>
          <p class="metric-description">
            Evaluates whether all necessary information from the ground truth is present in the 
            retrieved context. A high score indicates the retrieval system successfully found all 
            relevant information needed to answer the question.
          </p>
          <p class="metric-formula"><strong>Focus:</strong> Retrieval completeness</p>
        </div>

        <div class="metric-explanation-card">
          <div class="metric-icon">✅</div>
          <h3>Answer Correctness</h3>
          <p class="metric-description">
            Measures the overall correctness of the generated answer by comparing it with the ground 
            truth answer. It considers both semantic similarity and factual overlap to provide a 
            comprehensive correctness score.
          </p>
          <p class="metric-formula"><strong>Focus:</strong> Answer accuracy vs. ground truth</p>
        </div>
      </div>

      <div class="metrics-note">
        <p><strong>Note:</strong> All metrics are scored from 0% to 100%, where higher scores indicate better performance. 
        The overall average combines all five metrics to provide a single quality indicator.</p>
      </div>
    </div>
  `;
}

/**
 * Build the summary statistics section
 * Displays overall average, total/successful/failed counts, and average latency
 * 
 * Requirements: 1.2, 6.1, 6.2, 6.3, 6.4
 */
export function buildSummaryStatistics(
  run: EvaluationRunDetail,
  overallAverage: number
): string {
  if (!run.summary) {
    return `
      <div class="summary-section">
        <h2>Summary Statistics</h2>
        <p class="no-data">No summary data available</p>
      </div>
    `;
  }

  const getScoreClass = (score: number): string => {
    if (score >= 80) return 'score-good';
    if (score >= 60) return 'score-medium';
    return 'score-bad';
  };

  return `
    <div class="summary-section" id="summary">
      <h2>Summary Statistics</h2>
      
      <div class="overall-average ${getScoreClass(overallAverage)}">
        <div class="overall-label">Overall Average Score</div>
        <div class="overall-value">${overallAverage.toFixed(1)}%</div>
      </div>

      <div class="summary-grid">
        <div class="summary-card">
          <div class="summary-label">Total Questions</div>
          <div class="summary-value">${run.summary.total_questions}</div>
        </div>
        <div class="summary-card success">
          <div class="summary-label">Successful</div>
          <div class="summary-value">${run.summary.successful_questions}</div>
        </div>
        <div class="summary-card failed">
          <div class="summary-label">Failed</div>
          <div class="summary-value">${run.summary.failed_questions}</div>
        </div>
        <div class="summary-card">
          <div class="summary-label">Average Latency</div>
          <div class="summary-value">${run.summary.avg_latency_ms ? run.summary.avg_latency_ms.toFixed(0) + 'ms' : 'N/A'}</div>
        </div>
      </div>
    </div>
  `;
}

/**
 * Build the configuration information section
 * Displays all model and parameter information in a table format
 * 
 * Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
 */
export function buildConfigurationSection(config: ConfigurationInfo | null): string {
  if (!config) {
    return `
      <div class="configuration-section">
        <h2>Configuration Information</h2>
        <p class="no-data">No configuration data available</p>
      </div>
    `;
  }

  return `
    <div class="configuration-section" id="configuration">
      <h2>Configuration Information</h2>
      <table class="config-table">
        <tbody>
          <tr>
            <td class="config-label">LLM Provider</td>
            <td class="config-value">${config.llm_provider}</td>
          </tr>
          <tr>
            <td class="config-label">LLM Model</td>
            <td class="config-value">${config.llm_model}</td>
          </tr>
          <tr>
            <td class="config-label">Embedding Model</td>
            <td class="config-value">${config.embedding_model}</td>
          </tr>
          <tr>
            <td class="config-label">Evaluation Model</td>
            <td class="config-value">${config.evaluation_model}</td>
          </tr>
          <tr>
            <td class="config-label">Search Alpha</td>
            <td class="config-value">${config.search_alpha >= 0 ? config.search_alpha.toFixed(2) : 'N/A'}</td>
          </tr>
          <tr>
            <td class="config-label">Search Top K</td>
            <td class="config-value">${config.search_top_k >= 0 ? config.search_top_k : 'N/A'}</td>
          </tr>
        </tbody>
      </table>
    </div>
  `;
}


/**
 * Build the statistical analysis section
 * Displays statistics table for each metric
 * 
 * Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
 */
export function buildStatisticalAnalysisSection(statistics: {
  faithfulness: MetricStatistics;
  answer_relevancy: MetricStatistics;
  context_precision: MetricStatistics;
  context_recall: MetricStatistics;
  answer_correctness: MetricStatistics;
}): string {
  const formatPercent = (value: number): string => `${value.toFixed(1)}%`;

  const buildMetricTable = (metricName: string, stats: MetricStatistics): string => `
    <div class="metric-stats">
      <h3>${metricName}</h3>
      <table class="stats-table">
        <thead>
          <tr>
            <th>Mean</th>
            <th>Median</th>
            <th>Std Dev</th>
            <th>Variance</th>
            <th>Min</th>
            <th>Max</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>${formatPercent(stats.mean)}</td>
            <td>${formatPercent(stats.median)}</td>
            <td>${formatPercent(stats.stdDev)}</td>
            <td>${formatPercent(stats.variance)}</td>
            <td>${formatPercent(stats.min)}</td>
            <td>${formatPercent(stats.max)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  `;

  return `
    <div class="statistical-analysis-section" id="statistics">
      <h2>Statistical Analysis</h2>
      ${buildMetricTable('Faithfulness', statistics.faithfulness)}
      ${buildMetricTable('Answer Relevancy', statistics.answer_relevancy)}
      ${buildMetricTable('Context Precision', statistics.context_precision)}
      ${buildMetricTable('Context Recall', statistics.context_recall)}
      ${buildMetricTable('Answer Correctness', statistics.answer_correctness)}
    </div>
  `;
}

/**
 * Build the visualizations section
 * Embeds bar chart, line chart, and box plots
 * 
 * Requirements: 4.1, 4.2, 4.3, 4.4
 */
export function buildVisualizationsSection(charts: {
  barChart: string;
  lineChart: string;
  boxPlots: {
    faithfulness: string;
    answer_relevancy: string;
    context_precision: string;
    context_recall: string;
    answer_correctness: string;
  };
}): string {
  const renderChart = (title: string, chartData: string): string => {
    // Check if it's a base64 image or fallback text
    if (chartData.startsWith('data:image/png;base64,')) {
      return `
        <div class="chart-container">
          <img src="${chartData}" alt="${title}" class="chart-image" />
        </div>
      `;
    } else {
      // Fallback text
      return `
        <div class="chart-fallback">
          <p class="fallback-title">${title}</p>
          <pre class="fallback-text">${chartData}</pre>
        </div>
      `;
    }
  };

  return `
    <div class="visualizations-section" id="visualizations">
      <h2>Visualizations</h2>
      
      <div class="chart-section">
        <h3>Average Metric Scores</h3>
        ${renderChart('Bar Chart - Average Scores', charts.barChart)}
      </div>

      <div class="chart-section page-break-before">
        <h3>Metric Trends Across Questions</h3>
        ${renderChart('Line Chart - Metric Trends', charts.lineChart)}
      </div>

      <div class="chart-section page-break-before">
        <h3>Metric Distributions</h3>
        <div class="boxplot-grid">
          ${renderChart('Faithfulness Distribution', charts.boxPlots.faithfulness)}
          ${renderChart('Answer Relevancy Distribution', charts.boxPlots.answer_relevancy)}
          ${renderChart('Context Precision Distribution', charts.boxPlots.context_precision)}
          ${renderChart('Context Recall Distribution', charts.boxPlots.context_recall)}
          ${renderChart('Answer Correctness Distribution', charts.boxPlots.answer_correctness)}
        </div>
      </div>
    </div>
  `;
}

/**
 * Interface for enhanced evaluation result with overall score
 */
interface EnhancedResult extends EvaluationResult {
  overall_score: number;
  rank: number;
}

/**
 * Build the question analysis table
 * Creates table with all question results, sorted by overall score
 * 
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
 */
export function buildQuestionAnalysisTable(results: EvaluationResult[]): string {
  // Calculate overall score for each result
  const enhancedResults: EnhancedResult[] = results.map((result) => {
    const scores = [
      result.faithfulness ?? 0,
      result.answer_relevancy ?? 0,
      result.context_precision ?? 0,
      result.context_recall ?? 0,
      result.answer_correctness ?? 0,
    ];
    const validScores = scores.filter(s => s > 0);
    const overall_score = validScores.length > 0 
      ? validScores.reduce((sum, s) => sum + s, 0) / validScores.length 
      : 0;
    
    return {
      ...result,
      overall_score,
      rank: 0, // Will be set after sorting
    };
  });

  // Sort by overall score descending
  enhancedResults.sort((a, b) => b.overall_score - a.overall_score);

  // Assign ranks
  enhancedResults.forEach((result, index) => {
    result.rank = index + 1;
  });

  // Find best and worst
  const bestScore = enhancedResults.at(0)?.overall_score ?? 0;
  const worstScore = enhancedResults.at(-1)?.overall_score ?? 0;

  const formatScore = (score: number | null | undefined): string => {
    if (score === null || score === undefined) return '-';
    return `${score.toFixed(1)}%`;
  };

  const getRowClass = (result: EnhancedResult): string => {
    if (result.overall_score === bestScore && bestScore > 0) return 'best-question';
    if (result.overall_score === worstScore && worstScore > 0) return 'worst-question';
    return '';
  };

  const buildTableRows = (): string => {
    return enhancedResults.map((result, index) => `
      <tr class="${getRowClass(result)} ${index % 2 === 0 ? 'even-row' : 'odd-row'}">
        <td>${result.rank}</td>
        <td class="question-cell">${result.question_text}</td>
        <td>${formatScore(result.faithfulness)}</td>
        <td>${formatScore(result.answer_relevancy)}</td>
        <td>${formatScore(result.context_precision)}</td>
        <td>${formatScore(result.context_recall)}</td>
        <td>${formatScore(result.answer_correctness)}</td>
        <td class="overall-cell">${formatScore(result.overall_score)}</td>
      </tr>
    `).join('');
  };

  return `
    <div class="question-analysis-section page-break-before" id="question-analysis">
      <h2>Question Analysis</h2>
      <p class="section-description">All questions sorted by overall score (descending)</p>
      <table class="question-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Question</th>
            <th>Faith</th>
            <th>Rel</th>
            <th>Prec</th>
            <th>Rec</th>
            <th>Corr</th>
            <th>Overall</th>
          </tr>
        </thead>
        <tbody>
          ${buildTableRows()}
        </tbody>
      </table>
      <div class="legend">
        <span class="legend-item best">■ Best performing question</span>
        <span class="legend-item worst">■ Worst performing question</span>
      </div>
    </div>
  `;
}

/**
 * Build the detailed results section
 * Keeps existing detailed results format with page breaks
 * 
 * Requirements: 7.4
 */
export function buildDetailedResultsSection(results: EvaluationResult[]): string {
  const formatScore = (score: number | null | undefined): string => {
    if (score === null || score === undefined) return '-';
    const percentage = score * 100;
    return `${percentage.toFixed(1)}%`;
  };

  const renderRetrievedContexts = (result: EvaluationResult): string => {
    if (!result.retrieved_contexts || result.retrieved_contexts.length === 0) {
      return '';
    }
    
    return `
      <div class="content-section">
        <strong>Retrieved Contexts (${result.retrieved_contexts.length}):</strong>
        ${result.retrieved_contexts.map((ctx, i) => `
          <div class="context-item">
            <span class="context-number">${i + 1}.</span>
            <span class="context-text">${ctx}</span>
          </div>
        `).join('')}
      </div>
    `;
  };

  const renderLatency = (result: EvaluationResult): string => {
    if (!result.latency_ms) {
      return '';
    }
    
    return `
      <div class="result-meta">
        Response time: ${result.latency_ms}ms
      </div>
    `;
  };

  const buildResultCard = (result: EvaluationResult, index: number): string => `
    <div class="result-card page-break-before">
      <div class="result-header">
        <h3>Question #${index + 1}</h3>
        ${result.error_message ? '<span class="error-badge">Error</span>' : '<span class="success-badge">Success</span>'}
      </div>
      
      <div class="result-question">
        <strong>Question:</strong> ${result.question_text}
      </div>

      ${result.error_message ? `
        <div class="result-error">
          <strong>Error:</strong> ${result.error_message}
        </div>
      ` : `
        <div class="result-scores">
          <div class="score-item">
            <span class="score-label">Faithfulness:</span>
            <span class="score-value">${formatScore(result.faithfulness)}</span>
          </div>
          <div class="score-item">
            <span class="score-label">Answer Relevancy:</span>
            <span class="score-value">${formatScore(result.answer_relevancy)}</span>
          </div>
          <div class="score-item">
            <span class="score-label">Context Precision:</span>
            <span class="score-value">${formatScore(result.context_precision)}</span>
          </div>
          <div class="score-item">
            <span class="score-label">Context Recall:</span>
            <span class="score-value">${formatScore(result.context_recall)}</span>
          </div>
          <div class="score-item">
            <span class="score-label">Answer Correctness:</span>
            <span class="score-value">${formatScore(result.answer_correctness)}</span>
          </div>
        </div>

        <div class="result-content">
          <div class="content-section">
            <strong>Ground Truth:</strong>
            <p>${result.ground_truth_text}</p>
          </div>
          
          <div class="content-section">
            <strong>Generated Answer:</strong>
            <p>${result.generated_answer || 'N/A'}</p>
          </div>

          ${renderRetrievedContexts(result)}

          ${renderLatency(result)}
        </div>
      `}
    </div>
  `;

  return `
    <div class="detailed-results-section" id="detailed-results">
      <h2 class="page-break-before">Detailed Results</h2>
      ${results.map((result, index) => buildResultCard(result, index)).join('')}
    </div>
  `;
}


/**
 * Build the table of contents
 * Lists all major sections with anchor links
 * 
 * Requirements: 7.5
 */
export function buildTableOfContents(): string {
  const sections = [
    { id: 'metrics-explanation', title: 'About RAGAS Metrics' },
    { id: 'summary', title: 'Summary Statistics' },
    { id: 'configuration', title: 'Configuration Information' },
    { id: 'statistics', title: 'Statistical Analysis' },
    { id: 'visualizations', title: 'Visualizations' },
    { id: 'question-analysis', title: 'Question Analysis' },
    { id: 'detailed-results', title: 'Detailed Results' },
  ];

  const tocItems = sections.map(section => `
    <div class="toc-item">
      <a href="#${section.id}" class="toc-link">
        <span class="toc-title">${section.title}</span>
      </a>
    </div>
  `).join('');

  return `
    <div class="table-of-contents page-break-after">
      <h2 class="toc-title">Table of Contents</h2>
      <div class="toc-list">
        ${tocItems}
      </div>
    </div>
  `;
}
