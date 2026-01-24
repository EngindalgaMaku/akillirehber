/**
 * Chart Generator Utility for PDF Reports
 * 
 * This module generates Chart.js charts as base64 PNG images for embedding
 * in PDF reports. It handles bar charts, line charts, and box plots with
 * proper error handling and fallback text descriptions.
 */

import {
  Chart as ChartJS,
  ChartConfiguration,
  CategoryScale,
  LinearScale,
  BarElement,
  BarController,
  LineElement,
  LineController,
  PointElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import { getCommonChartOptions, METRIC_COLORS, getScoreColor, getScoreBackgroundColor } from './chartConfig';

// Register Chart.js components (ensure they're registered before use)
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  BarController,
  LineElement,
  LineController,
  PointElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartDataLabels
);

/**
 * Generate a chart as a base64 PNG image
 * 
 * @param config - Chart.js configuration
 * @param width - Canvas width in pixels
 * @param height - Canvas height in pixels
 * @returns Promise resolving to base64 image string or error message
 */
async function generateChartImage(
  config: ChartConfiguration,
  width: number = 800,
  height: number = 400
): Promise<string> {
  console.log('[generateChartImage] START - width:', width, 'height:', height);
  
  try {
    // Check if we're in a browser environment
    console.log('[generateChartImage] Checking environment...');
    console.log('[generateChartImage] typeof document:', typeof document);
    console.log('[generateChartImage] typeof window:', typeof window);
    
    if (typeof document === 'undefined' || typeof window === 'undefined') {
      console.error('[generateChartImage] Not in browser environment!');
      throw new Error('Chart generation requires browser environment');
    }

    console.log('[generateChartImage] Environment OK, creating canvas...');
    
    // Create an offscreen canvas
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    
    console.log('[generateChartImage] Canvas created:', canvas);
    
    // Get 2D context to ensure canvas is working
    const ctx = canvas.getContext('2d');
    console.log('[generateChartImage] Context obtained:', ctx);
    
    if (!ctx) {
      console.error('[generateChartImage] Failed to get 2D context!');
      throw new Error('Failed to get canvas 2D context');
    }
    
    console.log('[generateChartImage] Creating Chart.js instance...');
    console.log('[generateChartImage] ChartJS constructor:', typeof ChartJS);
    console.log('[generateChartImage] Config type:', config.type);
    
    // Create chart instance
    const chart = new ChartJS(ctx, config);
    
    console.log('[generateChartImage] Chart instance created:', chart);
    console.log('[generateChartImage] Waiting for render (500ms)...');
    
    // Wait for chart to render (increased timeout for complex charts)
    await new Promise(resolve => setTimeout(resolve, 500));
    
    console.log('[generateChartImage] Render wait complete, getting base64...');
    
    // Get base64 image
    const base64Image = canvas.toDataURL('image/png', 1.0);
    
    console.log('[generateChartImage] Base64 obtained, length:', base64Image.length);
    console.log('[generateChartImage] Base64 prefix:', base64Image.substring(0, 50));
    
    // Validate the image was generated
    if (!base64Image || !base64Image.startsWith('data:image/png;base64,')) {
      console.error('[generateChartImage] Invalid base64 image!');
      console.error('[generateChartImage] Starts with:', base64Image.substring(0, 30));
      throw new Error('Failed to generate valid base64 image');
    }
    
    console.log('[generateChartImage] Validation passed, cleaning up...');
    
    // Cleanup
    chart.destroy();
    
    console.log('[generateChartImage] SUCCESS - returning base64 image');
    return base64Image;
  } catch (error) {
    console.error('[generateChartImage] EXCEPTION CAUGHT:', error);
    console.error('[generateChartImage] Error type:', error?.constructor?.name);
    console.error('[generateChartImage] Error message:', error instanceof Error ? error.message : String(error));
    console.error('[generateChartImage] Error stack:', error instanceof Error ? error.stack : 'No stack');
    throw error;
  }
}

/**
 * Generate a bar chart showing average values for all five metrics
 * 
 * Creates a bar chart with color coding:
 * - Green for scores >= 80%
 * - Yellow for scores >= 60%
 * - Red for scores < 60%
 * 
 * @param metrics - Object containing average values for each metric
 * @returns Promise resolving to base64 PNG image or fallback text
 */
export async function generateMetricsBarChart(metrics: {
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number;
  answer_correctness: number;
}): Promise<string> {
  console.log('[BAR CHART] Starting generation with metrics:', metrics);
  
  try {
    // Check environment first
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      console.error('[BAR CHART] Not in browser environment');
      throw new Error('Not in browser environment');
    }
    
    console.log('[BAR CHART] Browser environment OK');
    console.log('[BAR CHART] ChartJS available:', typeof ChartJS);
    console.log('[BAR CHART] ChartJS.register available:', typeof ChartJS.register);
    console.log('[BAR CHART] ChartDataLabels available:', typeof ChartDataLabels);
    
    // Re-register components to ensure they're available (needed for dynamic imports)
    console.log('[BAR CHART] Re-registering Chart.js components...');
    ChartJS.register(
      CategoryScale,
      LinearScale,
      BarElement,
      BarController,
      LineElement,
      LineController,
      PointElement,
      Title,
      Tooltip,
      Legend,
      Filler,
      ChartDataLabels
    );
    console.log('[BAR CHART] Components registered');
    
    const labels = [
      'Faithfulness',
      'Answer Relevancy',
      'Context Precision',
      'Context Recall',
      'Answer Correctness'
    ];
    
    const data = [
      metrics.faithfulness,
      metrics.answer_relevancy,
      metrics.context_precision,
      metrics.context_recall,
      metrics.answer_correctness
    ];
    
    console.log('[BAR CHART] Data prepared:', data);
    
    // Apply color coding based on score thresholds
    const backgroundColors = data.map(value => getScoreBackgroundColor(value));
    const borderColors = data.map(value => getScoreColor(value));
    
    console.log('[BAR CHART] Colors applied');
    
    const config: ChartConfiguration<'bar'> = {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Average Score (%)',
          data,
          backgroundColor: backgroundColors,
          borderColor: borderColors,
          borderWidth: 2,
        }]
      },
      options: {
        ...getCommonChartOptions(),
        plugins: {
          ...getCommonChartOptions().plugins,
          title: {
            display: true,
            text: 'Average Metric Scores',
            font: {
              size: 16,
              weight: 'bold',
            },
            padding: {
              top: 10,
              bottom: 20,
            },
          },
          legend: {
            display: false, // No legend needed for single dataset
          },
          datalabels: {
            display: true,
            anchor: 'end',
            align: 'top',
            formatter: (value: unknown) => {
              // Handle various value types safely
              if (typeof value === 'number') {
                return `${value.toFixed(1)}%`;
              }
              if (typeof value === 'string') {
                const num = parseFloat(value);
                return !isNaN(num) ? `${num.toFixed(1)}%` : '';
              }
              // For objects or other types, return empty string
              return '';
            },
            color: '#374151',
            font: {
              size: 11,
              weight: 'bold',
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: {
              callback: function(value: string | number) {
                // Handle various value types safely
                if (typeof value === 'number') {
                  return `${value}%`;
                }
                if (typeof value === 'string') {
                  const num = parseFloat(value);
                  return !isNaN(num) ? `${num}%` : '';
                }
                return '';
              },
            },
            grid: {
              color: '#E5E7EB',
            },
            title: {
              display: true,
              text: 'Score (%)',
              font: {
                size: 12,
                weight: 'bold',
              },
            },
          },
          x: {
            grid: {
              display: false,
            },
            ticks: {
              font: {
                size: 11,
              },
            },
          },
        },
      },
    };
    
    console.log('[BAR CHART] Config created, calling generateChartImage...');
    const result = await generateChartImage(config, 800, 500);
    console.log('[BAR CHART] Image generated successfully, length:', result.length);
    
    return result;
  } catch (error) {
    console.error('[BAR CHART] ERROR:', error);
    console.error('[BAR CHART] Error stack:', error instanceof Error ? error.stack : 'No stack');
    // Return fallback text description
    const fallback = `FALLBACK: Bar Chart - Faithfulness: ${metrics.faithfulness.toFixed(1)}%, Answer Relevancy: ${metrics.answer_relevancy.toFixed(1)}%, Context Precision: ${metrics.context_precision.toFixed(1)}%, Context Recall: ${metrics.context_recall.toFixed(1)}%, Answer Correctness: ${metrics.answer_correctness.toFixed(1)}%`;
    console.log('[BAR CHART] Returning fallback:', fallback);
    return fallback;
  }
}

/**
 * Interface for evaluation results used in line chart
 */
interface EvaluationResult {
  question_number: number;
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
  context_recall: number | null;
  answer_correctness: number | null;
}

/**
 * Generate a line chart showing metric values across all questions
 * 
 * Creates a multi-line chart with one line per metric, showing how
 * each metric varies across questions. Includes a legend to identify
 * each metric line.
 * 
 * @param results - Array of evaluation results for all questions
 * @returns Promise resolving to base64 PNG image or fallback text
 */
export async function generateMetricsLineChart(results: EvaluationResult[]): Promise<string> {
  try {
    // Re-register components to ensure they're available (needed for dynamic imports)
    ChartJS.register(
      CategoryScale,
      LinearScale,
      BarElement,
      BarController,
      LineElement,
      LineController,
      PointElement,
      Title,
      Tooltip,
      Legend,
      Filler,
      ChartDataLabels
    );
    
    // Sort results by question number
    const sortedResults = [...results].sort((a, b) => a.question_number - b.question_number);
    
    // Extract question numbers for x-axis labels
    const labels = sortedResults.map(r => `Q${r.question_number}`);
    
    // Extract data for each metric, filtering out null values
    const faithfulnessData = sortedResults.map(r => r.faithfulness ?? null);
    const answerRelevancyData = sortedResults.map(r => r.answer_relevancy ?? null);
    const contextPrecisionData = sortedResults.map(r => r.context_precision ?? null);
    const contextRecallData = sortedResults.map(r => r.context_recall ?? null);
    const answerCorrectnessData = sortedResults.map(r => r.answer_correctness ?? null);
    
    const config: ChartConfiguration<'line'> = {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Faithfulness',
            data: faithfulnessData,
            borderColor: METRIC_COLORS.faithfulness.border,
            backgroundColor: METRIC_COLORS.faithfulness.background,
            borderWidth: 2,
            pointRadius: 3,
            pointHoverRadius: 5,
            tension: 0.1,
            spanGaps: true, // Connect lines even with null values
          },
          {
            label: 'Answer Relevancy',
            data: answerRelevancyData,
            borderColor: METRIC_COLORS.answer_relevancy.border,
            backgroundColor: METRIC_COLORS.answer_relevancy.background,
            borderWidth: 2,
            pointRadius: 3,
            pointHoverRadius: 5,
            tension: 0.1,
            spanGaps: true,
          },
          {
            label: 'Context Precision',
            data: contextPrecisionData,
            borderColor: METRIC_COLORS.context_precision.border,
            backgroundColor: METRIC_COLORS.context_precision.background,
            borderWidth: 2,
            pointRadius: 3,
            pointHoverRadius: 5,
            tension: 0.1,
            spanGaps: true,
          },
          {
            label: 'Context Recall',
            data: contextRecallData,
            borderColor: METRIC_COLORS.context_recall.border,
            backgroundColor: METRIC_COLORS.context_recall.background,
            borderWidth: 2,
            pointRadius: 3,
            pointHoverRadius: 5,
            tension: 0.1,
            spanGaps: true,
          },
          {
            label: 'Answer Correctness',
            data: answerCorrectnessData,
            borderColor: METRIC_COLORS.answer_correctness.border,
            backgroundColor: METRIC_COLORS.answer_correctness.background,
            borderWidth: 2,
            pointRadius: 3,
            pointHoverRadius: 5,
            tension: 0.1,
            spanGaps: true,
          },
        ]
      },
      options: {
        ...getCommonChartOptions(),
        plugins: {
          ...getCommonChartOptions().plugins,
          title: {
            display: true,
            text: 'Metric Scores Across Questions',
            font: {
              size: 16,
              weight: 'bold',
            },
            padding: {
              top: 10,
              bottom: 20,
            },
          },
          legend: {
            display: true,
            position: 'bottom',
            labels: {
              padding: 15,
              usePointStyle: true,
              font: {
                size: 11,
              },
            },
          },
          datalabels: {
            display: false, // Too cluttered for line chart
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: {
              callback: function(value: string | number) {
                // Handle various value types safely
                if (typeof value === 'number') {
                  return `${value}%`;
                }
                if (typeof value === 'string') {
                  const num = parseFloat(value);
                  return !isNaN(num) ? `${num}%` : '';
                }
                return '';
              },
            },
            grid: {
              color: '#E5E7EB',
            },
            title: {
              display: true,
              text: 'Score (%)',
              font: {
                size: 12,
                weight: 'bold',
              },
            },
          },
          x: {
            grid: {
              display: false,
            },
            ticks: {
              font: {
                size: 10,
              },
              maxRotation: 45,
              minRotation: 45,
            },
            title: {
              display: true,
              text: 'Question Number',
              font: {
                size: 12,
                weight: 'bold',
              },
            },
          },
        },
      },
    };
    
    return await generateChartImage(config, 1000, 600);
  } catch (error) {
    console.error('Error generating metrics line chart:', error);
    // Return fallback text description
    return `FALLBACK: Line Chart - Showing metric trends across ${results.length} questions. Each metric is represented by a different colored line.`;
  }
}

/**
 * Interface for box plot statistics
 */
interface BoxPlotStats {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
}

/**
 * Generate a box plot showing distribution of a single metric
 * 
 * Creates a visual representation of the metric's distribution showing:
 * - Minimum value
 * - Q1 (25th percentile)
 * - Median (50th percentile)
 * - Q3 (75th percentile)
 * - Maximum value
 * 
 * Uses a custom bar chart implementation since Chart.js doesn't have
 * native box plot support.
 * 
 * @param metricName - Name of the metric (e.g., "Faithfulness")
 * @param values - Array of metric values
 * @param stats - Pre-calculated box plot statistics
 * @returns Promise resolving to base64 PNG image or fallback text
 */
export async function generateBoxPlot(
  metricName: string,
  values: number[],
  stats: BoxPlotStats
): Promise<string> {
  try {
    // Re-register components to ensure they're available (needed for dynamic imports)
    ChartJS.register(
      CategoryScale,
      LinearScale,
      BarElement,
      BarController,
      LineElement,
      LineController,
      PointElement,
      Title,
      Tooltip,
      Legend,
      Filler,
      ChartDataLabels
    );
    
    // Filter out null/undefined values
    const validValues = values.filter(v => v != null && !Number.isNaN(v));
    
    if (validValues.length === 0) {
      return `FALLBACK: Box Plot for ${metricName} - No valid data available`;
    }
    
    // Create a visual representation using a horizontal bar chart
    // We'll show the box (Q1 to Q3) and whiskers (min to max)
    const labels = ['Distribution'];
    
    const config: ChartConfiguration<'bar'> = {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Min to Q1',
            data: [stats.q1 - stats.min],
            backgroundColor: 'rgba(156, 163, 175, 0.3)', // Light gray
            borderColor: 'rgb(107, 114, 128)',
            borderWidth: 1,
            stack: 'stack0',
          },
          {
            label: 'Q1 to Median',
            data: [stats.median - stats.q1],
            backgroundColor: 'rgba(59, 130, 246, 0.6)', // Blue
            borderColor: 'rgb(59, 130, 246)',
            borderWidth: 2,
            stack: 'stack0',
          },
          {
            label: 'Median to Q3',
            data: [stats.q3 - stats.median],
            backgroundColor: 'rgba(59, 130, 246, 0.6)', // Blue
            borderColor: 'rgb(59, 130, 246)',
            borderWidth: 2,
            stack: 'stack0',
          },
          {
            label: 'Q3 to Max',
            data: [stats.max - stats.q3],
            backgroundColor: 'rgba(156, 163, 175, 0.3)', // Light gray
            borderColor: 'rgb(107, 114, 128)',
            borderWidth: 1,
            stack: 'stack0',
          },
        ]
      },
      options: {
        indexAxis: 'y', // Horizontal bar chart
        responsive: false,
        maintainAspectRatio: true,
        devicePixelRatio: 2,
        animation: false,
        plugins: {
          title: {
            display: true,
            text: `${metricName} Distribution`,
            font: {
              size: 16,
              weight: 'bold',
            },
            padding: {
              top: 10,
              bottom: 20,
            },
          },
          legend: {
            display: false,
          },
          tooltip: {
            enabled: false,
          },
          datalabels: {
            display: false,
          },
        },
        scales: {
          x: {
            stacked: true,
            beginAtZero: true,
            max: 100,
            ticks: {
              callback: function(value: string | number) {
                // Handle various value types safely
                if (typeof value === 'number') {
                  return `${value}%`;
                }
                if (typeof value === 'string') {
                  const num = parseFloat(value);
                  return !isNaN(num) ? `${num}%` : '';
                }
                return '';
              },
            },
            grid: {
              color: '#E5E7EB',
            },
            title: {
              display: true,
              text: 'Score (%)',
              font: {
                size: 12,
                weight: 'bold',
              },
            },
          },
          y: {
            stacked: true,
            grid: {
              display: false,
            },
          },
        },
      },
    };
    
    const chartImage = await generateChartImage(config, 800, 300);
    
    // Return just the chart image - stats will be shown in HTML separately
    return chartImage;
  } catch (error) {
    console.error(`Error generating box plot for ${metricName}:`, error);
    // Return fallback text description
    return `FALLBACK: Box Plot for ${metricName} - Min: ${stats.min.toFixed(1)}%, Q1: ${stats.q1.toFixed(1)}%, Median: ${stats.median.toFixed(1)}%, Q3: ${stats.q3.toFixed(1)}%, Max: ${stats.max.toFixed(1)}%`;
  }
}

/**
 * Export BoxPlotStats interface for external use
 */
export type { BoxPlotStats };

/**
 * Validate metric data before chart generation
 * 
 * @param data - Array of metric values
 * @returns Filtered array with only valid numbers
 */
export function validateMetricData(data: (number | null | undefined)[]): number[] {
  return data.filter((value): value is number => 
    value != null && 
    !Number.isNaN(value) && 
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 100
  );
}

/**
 * Check if chart generation is supported in current environment
 * 
 * @returns true if canvas and required APIs are available
 */
export function isChartGenerationSupported(): boolean {
  try {
    if (typeof document === 'undefined') {
      return false; // Server-side rendering
    }
    
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    return context !== null && typeof canvas.toDataURL === 'function';
  } catch (error) {
    console.error('Chart generation support check failed:', error);
    return false;
  }
}

/**
 * Generate fallback text for a chart when rendering fails
 * 
 * @param chartType - Type of chart (bar, line, boxplot)
 * @param metricName - Name of the metric
 * @param data - Data that was supposed to be charted
 * @returns Formatted fallback text
 */
export function generateFallbackText(
  chartType: 'bar' | 'line' | 'boxplot',
  metricName: string,
  data: Record<string, unknown> | unknown[]
): string {
  const timestamp = new Date().toISOString();
  
  switch (chartType) {
    case 'bar':
      return `[Chart Generation Failed - ${timestamp}]\nBar Chart: ${metricName}\nData: ${JSON.stringify(data, null, 2)}`;
    
    case 'line':
      return `[Chart Generation Failed - ${timestamp}]\nLine Chart: ${metricName}\nShowing trends across multiple data points.`;
    
    case 'boxplot':
      return `[Chart Generation Failed - ${timestamp}]\nBox Plot: ${metricName}\nDistribution statistics: ${JSON.stringify(data, null, 2)}`;
    
    default:
      return `[Chart Generation Failed - ${timestamp}]\nChart Type: ${chartType}\nMetric: ${metricName}`;
  }
}

/**
 * Safely generate a chart with comprehensive error handling
 * 
 * Wraps chart generation functions with additional error handling,
 * validation, and fallback mechanisms.
 * 
 * @param generator - Chart generation function
 * @param fallbackMessage - Message to return if generation fails
 * @param args - Arguments to pass to the generator function
 * @returns Promise resolving to chart image or fallback text
 */
export async function safeGenerateChart<T extends unknown[]>(
  generator: (...args: T) => Promise<string>,
  fallbackMessage: string,
  ...args: T
): Promise<string> {
  try {
    // Check if chart generation is supported
    if (!isChartGenerationSupported()) {
      console.warn('Chart generation not supported in this environment');
      return fallbackMessage;
    }
    
    // Attempt to generate chart
    const result = await generator(...args);
    
    // Validate result
    if (!result || typeof result !== 'string') {
      console.error('Chart generation returned invalid result');
      return fallbackMessage;
    }
    
    // Check if result is a base64 image or fallback text
    if (result.startsWith('data:image/png;base64,') || result.startsWith('FALLBACK:')) {
      return result;
    }
    
    // Unexpected result format
    console.warn('Chart generation returned unexpected format');
    return fallbackMessage;
    
  } catch (error) {
    console.error('Error in safe chart generation:', error);
    return fallbackMessage;
  }
}
