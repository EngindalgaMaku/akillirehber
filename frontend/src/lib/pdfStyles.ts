/**
 * PDF Styles for Enhanced RAGAS Report
 * 
 * This module exports CSS styles for the PDF report with:
 * - A4 page size configuration
 * - Print-optimized layout
 * - Page numbers and headers
 * - Color coding for metrics
 * - Page break controls
 * 
 * Requirements: 7.1, 7.2, 7.3, 7.4, 7.6, 1.4, 4.4
 */

export function getPdfStyles(): string {
  return `
    <style>
      /* ===== BASE STYLES & PAGE SETUP (Req 7.1, 7.6) ===== */
      @page {
        size: A4;
        margin: 20mm 15mm 25mm 15mm;
      }

      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }

      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 10pt;
        line-height: 1.4;
        color: #333;
        background: white;
      }

      h1 {
        font-size: 24pt;
        font-weight: 700;
        margin-bottom: 12pt;
        color: #1a1a1a;
      }

      h2 {
        font-size: 16pt;
        font-weight: 600;
        margin-top: 16pt;
        margin-bottom: 10pt;
        color: #2c3e50;
        border-bottom: 2pt solid #3498db;
        padding-bottom: 4pt;
      }

      h3 {
        font-size: 12pt;
        font-weight: 600;
        margin-top: 12pt;
        margin-bottom: 8pt;
        color: #34495e;
      }

      p {
        margin-bottom: 8pt;
      }

      /* ===== PAGE NUMBERS & HEADERS (Req 7.2, 7.3) ===== */
      @page {
        @top-center {
          content: "RAGAS Evaluation Report";
          font-size: 9pt;
          color: #666;
          font-weight: 500;
        }
        
        @bottom-right {
          content: "Page " counter(page) " of " counter(pages);
          font-size: 9pt;
          color: #666;
        }
      }

      /* Page counter for browsers that support it */
      body {
        counter-reset: page;
      }

      .page-number {
        position: fixed;
        bottom: 10mm;
        right: 15mm;
        font-size: 9pt;
        color: #666;
      }

      .page-number::after {
        content: counter(page);
        counter-increment: page;
      }

      /* ===== PAGE BREAKS (Req 7.4) ===== */
      .page-break-before {
        page-break-before: always;
        break-before: page;
      }

      .page-break-after {
        page-break-after: always;
        break-after: page;
      }

      .page-break-avoid {
        page-break-inside: avoid;
        break-inside: avoid;
      }

      /* Prevent section splitting */
      .report-header,
      .summary-section,
      .configuration-section,
      .statistical-analysis-section,
      .visualizations-section,
      .question-analysis-section,
      .result-card,
      .metric-stats,
      .chart-section {
        page-break-inside: avoid;
        break-inside: avoid;
      }

      /* ===== COLOR CODING (Req 1.4, 4.4) ===== */
      .score-good {
        color: #27ae60;
        background-color: #e8f8f5;
      }

      .score-medium {
        color: #f39c12;
        background-color: #fef5e7;
      }

      .score-bad {
        color: #e74c3c;
        background-color: #fadbd8;
      }

      /* ===== REPORT HEADER ===== */
      .report-header {
        text-align: center;
        margin-bottom: 20pt;
        padding: 15pt;
        border: 2pt solid #3498db;
        background: linear-gradient(to bottom, #f8f9fa, #ffffff);
      }

      .report-title {
        color: #2c3e50;
        margin-bottom: 10pt;
      }

      .header-info {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8pt;
        margin-top: 10pt;
        text-align: left;
      }

      .info-row {
        display: flex;
        justify-content: space-between;
        padding: 4pt 8pt;
        background: white;
        border-radius: 4pt;
      }

      .info-label {
        font-weight: 600;
        color: #555;
      }

      .info-value {
        color: #2c3e50;
      }

      /* ===== SUMMARY SECTION ===== */
      .summary-section {
        margin-bottom: 20pt;
      }

      /* ===== METRICS EXPLANATION SECTION ===== */
      .metrics-explanation-section {
        margin-bottom: 20pt;
        padding: 15pt;
        background: #f8f9fa;
        border-radius: 8pt;
      }

      .section-intro {
        margin-bottom: 15pt;
        font-size: 10pt;
        line-height: 1.6;
        color: #555;
      }

      .metric-explanation-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 12pt;
        margin-bottom: 15pt;
      }

      .metric-explanation-card {
        padding: 12pt;
        background: white;
        border-left: 4pt solid #3498db;
        border-radius: 6pt;
        box-shadow: 0 1pt 3pt rgba(0,0,0,0.1);
      }

      .metric-icon {
        font-size: 24pt;
        margin-bottom: 8pt;
      }

      .metric-explanation-card h3 {
        margin: 0 0 8pt 0;
        color: #2c3e50;
        font-size: 12pt;
      }

      .metric-description {
        margin-bottom: 8pt;
        font-size: 9pt;
        line-height: 1.5;
        color: #555;
      }

      .metric-formula {
        font-size: 9pt;
        color: #3498db;
        font-style: italic;
      }

      .metrics-note {
        padding: 10pt;
        background: #e8f4f8;
        border-left: 3pt solid #3498db;
        border-radius: 4pt;
        font-size: 9pt;
        color: #555;
      }

      .metrics-note strong {
        color: #2c3e50;
      }

      /* ===== SUMMARY STATISTICS ===== */

      .overall-average {
        text-align: center;
        padding: 15pt;
        margin: 15pt 0;
        border-radius: 8pt;
        border: 2pt solid;
      }

      .overall-label {
        font-size: 12pt;
        font-weight: 600;
        margin-bottom: 8pt;
      }

      .overall-value {
        font-size: 32pt;
        font-weight: 700;
      }

      .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10pt;
        margin-top: 15pt;
      }

      .summary-card {
        padding: 12pt;
        border: 1pt solid #ddd;
        border-radius: 6pt;
        text-align: center;
        background: #f8f9fa;
      }

      .summary-card.success {
        border-color: #27ae60;
        background: #e8f8f5;
      }

      .summary-card.failed {
        border-color: #e74c3c;
        background: #fadbd8;
      }

      .summary-label {
        font-size: 9pt;
        color: #666;
        margin-bottom: 6pt;
      }

      .summary-value {
        font-size: 18pt;
        font-weight: 700;
        color: #2c3e50;
      }

      /* ===== CONFIGURATION SECTION ===== */
      .configuration-section {
        margin-bottom: 20pt;
      }

      .config-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10pt;
      }

      .config-table td {
        padding: 8pt;
        border: 1pt solid #ddd;
      }

      .config-label {
        font-weight: 600;
        background: #f8f9fa;
        width: 40%;
        color: #555;
      }

      .config-value {
        color: #2c3e50;
      }

      /* ===== STATISTICAL ANALYSIS SECTION ===== */
      .statistical-analysis-section {
        margin-bottom: 20pt;
      }

      .metric-stats {
        margin-bottom: 15pt;
      }

      .stats-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 8pt;
      }

      .stats-table th,
      .stats-table td {
        padding: 8pt;
        border: 1pt solid #ddd;
        text-align: center;
      }

      .stats-table th {
        background: #3498db;
        color: white;
        font-weight: 600;
      }

      .stats-table td {
        background: #f8f9fa;
      }

      /* ===== VISUALIZATIONS SECTION ===== */
      .visualizations-section {
        margin-bottom: 20pt;
      }

      .chart-section {
        margin-bottom: 20pt;
      }

      .chart-container {
        text-align: center;
        margin: 15pt 0;
      }

      .chart-image {
        max-width: 100%;
        height: auto;
        border: 1pt solid #ddd;
        border-radius: 4pt;
      }

      .boxplot-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15pt;
        margin-top: 15pt;
      }

      .chart-fallback {
        padding: 15pt;
        background: #f8f9fa;
        border: 1pt solid #ddd;
        border-radius: 4pt;
        margin: 10pt 0;
      }

      .fallback-title {
        font-weight: 600;
        margin-bottom: 8pt;
        color: #555;
      }

      .fallback-text {
        font-family: 'Courier New', monospace;
        font-size: 9pt;
        white-space: pre-wrap;
        color: #666;
      }

      /* ===== QUESTION ANALYSIS TABLE ===== */
      .question-analysis-section {
        margin-bottom: 20pt;
      }

      .section-description {
        font-style: italic;
        color: #666;
        margin-bottom: 10pt;
      }

      .question-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10pt;
        font-size: 9pt;
      }

      .question-table th,
      .question-table td {
        padding: 6pt;
        border: 1pt solid #ddd;
        text-align: center;
      }

      .question-table th {
        background: #3498db;
        color: white;
        font-weight: 600;
        position: sticky;
        top: 0;
      }

      .question-cell {
        text-align: left;
        max-width: 300pt;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .overall-cell {
        font-weight: 700;
      }

      /* Score color coding in tables */
      .question-table .score-good {
        color: #27ae60;
        font-weight: 600;
      }

      .question-table .score-medium {
        color: #f39c12;
        font-weight: 600;
      }

      .question-table .score-bad {
        color: #e74c3c;
        font-weight: 600;
      }

      .question-table .score-na {
        color: #999;
      }

      /* Zebra striping */
      .even-row {
        background: #f8f9fa;
      }

      .odd-row {
        background: white;
      }

      /* Highlight best/worst */
      .best-question {
        background: #d5f4e6 !important;
        border-left: 3pt solid #27ae60;
      }

      .worst-question {
        background: #fadbd8 !important;
        border-left: 3pt solid #e74c3c;
      }

      .legend {
        margin-top: 10pt;
        display: flex;
        gap: 20pt;
        font-size: 9pt;
      }

      .legend-item {
        display: flex;
        align-items: center;
        gap: 6pt;
      }

      .legend-item.best::before {
        content: "■";
        color: #27ae60;
        font-size: 14pt;
      }

      .legend-item.worst::before {
        content: "■";
        color: #e74c3c;
        font-size: 14pt;
      }

      /* ===== DETAILED RESULTS SECTION ===== */
      .detailed-results-section {
        margin-bottom: 20pt;
      }

      .result-card {
        margin-bottom: 15pt;
        padding: 12pt;
        border: 1pt solid #ddd;
        border-radius: 6pt;
        background: #f8f9fa;
      }

      .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10pt;
        padding-bottom: 8pt;
        border-bottom: 1pt solid #ddd;
      }

      .result-header h3 {
        margin: 0;
        color: #2c3e50;
      }

      .success-badge {
        background: #27ae60;
        color: white;
        padding: 4pt 10pt;
        border-radius: 4pt;
        font-size: 9pt;
        font-weight: 600;
      }

      .error-badge {
        background: #e74c3c;
        color: white;
        padding: 4pt 10pt;
        border-radius: 4pt;
        font-size: 9pt;
        font-weight: 600;
      }

      .result-question {
        margin-bottom: 10pt;
        padding: 8pt;
        background: white;
        border-radius: 4pt;
      }

      .result-error {
        padding: 10pt;
        background: #fadbd8;
        border-left: 3pt solid #e74c3c;
        border-radius: 4pt;
        color: #c0392b;
      }

      .result-scores {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8pt;
        margin-bottom: 12pt;
      }

      .score-item {
        padding: 8pt;
        background: white;
        border-radius: 4pt;
        text-align: center;
        border: 1pt solid #ddd;
      }

      .score-label {
        display: block;
        font-size: 8pt;
        color: #666;
        margin-bottom: 4pt;
      }

      .score-value {
        display: block;
        font-size: 12pt;
        font-weight: 700;
        color: #2c3e50;
      }

      /* Color coding for score values */
      .score-item .score-good {
        color: #27ae60;
      }

      .score-item .score-medium {
        color: #f39c12;
      }

      .score-item .score-bad {
        color: #e74c3c;
      }

      .result-content {
        margin-top: 12pt;
      }

      .content-section {
        margin-bottom: 10pt;
        padding: 8pt;
        background: white;
        border-radius: 4pt;
      }

      .content-section strong {
        display: block;
        margin-bottom: 6pt;
        color: #555;
      }

      .context-item {
        display: flex;
        gap: 6pt;
        margin-bottom: 6pt;
        padding: 6pt;
        background: #f8f9fa;
        border-left: 2pt solid #3498db;
        border-radius: 2pt;
      }

      .context-number {
        font-weight: 600;
        color: #3498db;
        min-width: 20pt;
      }

      .context-text {
        flex: 1;
        color: #555;
      }

      .result-meta {
        margin-top: 8pt;
        padding: 6pt;
        background: #e8f4f8;
        border-radius: 4pt;
        font-size: 9pt;
        color: #555;
        text-align: right;
      }

      /* ===== TABLE OF CONTENTS ===== */
      .table-of-contents {
        margin-bottom: 20pt;
        padding: 15pt;
        background: #f8f9fa;
        border: 1pt solid #ddd;
        border-radius: 6pt;
      }

      .toc-title {
        font-size: 16pt;
        font-weight: 600;
        margin-bottom: 12pt;
        color: #2c3e50;
      }

      .toc-list {
        list-style: none;
      }

      .toc-item {
        padding: 6pt 0;
        border-bottom: 1pt dotted #ddd;
      }

      .toc-item:last-child {
        border-bottom: none;
      }

      .toc-link {
        text-decoration: none;
        color: #3498db;
        display: flex;
        justify-content: space-between;
      }

      .toc-link:hover {
        color: #2980b9;
      }

      .toc-page {
        color: #666;
        font-size: 9pt;
      }

      /* ===== UTILITY CLASSES ===== */
      .no-data {
        padding: 15pt;
        text-align: center;
        color: #999;
        font-style: italic;
        background: #f8f9fa;
        border-radius: 4pt;
      }

      /* ===== PRINT-SPECIFIC STYLES ===== */
      @media print {
        body {
          print-color-adjust: exact;
          -webkit-print-color-adjust: exact;
        }

        .no-print {
          display: none !important;
        }

        a {
          text-decoration: none;
          color: inherit;
        }

        /* Ensure colors print correctly */
        .score-good,
        .score-medium,
        .score-bad,
        .best-question,
        .worst-question,
        .summary-card.success,
        .summary-card.failed {
          print-color-adjust: exact;
          -webkit-print-color-adjust: exact;
        }
      }
    </style>
  `;
}


/**
 * Generate page header HTML
 * Adds a header with report title on each page
 * 
 * Requirements: 7.2, 7.3
 */
export function getPageHeader(reportTitle: string = "RAGAS Evaluation Report"): string {
  return `
    <div class="page-header no-print">
      <div class="header-title">${reportTitle}</div>
    </div>
  `;
}

/**
 * Generate page number HTML
 * Adds page numbers using CSS counters
 * 
 * Requirements: 7.2
 */
export function getPageNumber(): string {
  return `
    <div class="page-number"></div>
  `;
}


/**
 * Utility function to wrap content with page break control
 * 
 * Requirements: 7.4
 */
export function wrapWithPageBreak(content: string, breakBefore: boolean = false, breakAfter: boolean = false): string {
  const classes = [];
  if (breakBefore) classes.push('page-break-before');
  if (breakAfter) classes.push('page-break-after');
  classes.push('page-break-avoid');
  
  return `
    <div class="${classes.join(' ')}">
      ${content}
    </div>
  `;
}


/**
 * Get color class based on score threshold
 * Green >= 80%, Yellow >= 60%, Red < 60%
 * 
 * Requirements: 1.4, 4.4
 */
export function getScoreColorClass(score: number): string {
  if (score >= 80) return 'score-good';
  if (score >= 60) return 'score-medium';
  return 'score-bad';
}

/**
 * Format score with color coding
 * 
 * Requirements: 1.4, 4.4
 */
export function formatScoreWithColor(score: number | null | undefined): string {
  if (score === null || score === undefined) {
    return '<span class="score-na">-</span>';
  }
  
  const colorClass = getScoreColorClass(score);
  return `<span class="${colorClass}">${score.toFixed(1)}%</span>`;
}

/**
 * Get metric color for charts
 * Returns hex color code for consistent visualization
 * 
 * Requirements: 4.4
 */
export function getMetricColor(metricName: string): string {
  const colors: Record<string, string> = {
    faithfulness: '#3498db',      // Blue
    answer_relevancy: '#2ecc71',  // Green
    context_precision: '#f39c12', // Orange
    context_recall: '#9b59b6',    // Purple
    answer_correctness: '#e74c3c' // Red
  };
  
  return colors[metricName] || '#95a5a6'; // Default gray
}
