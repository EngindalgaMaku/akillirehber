/**
 * Chart.js Configuration for PDF Reports
 * 
 * This module configures Chart.js defaults optimized for print quality
 * in PDF reports. It sets up proper resolution, fonts, and styling
 * for professional-looking charts.
 */

import {
  Chart as ChartJS,
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
  ChartOptions,
} from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';

// Register Chart.js components
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
 * Configure Chart.js defaults for print quality
 * 
 * Sets up:
 * - High resolution for print (2x device pixel ratio)
 * - Professional fonts
 * - Consistent styling
 * - Responsive disabled for fixed sizing
 */
export function configureChartDefaults(): void {
  // Set default font family
  ChartJS.defaults.font.family = "'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif";
  ChartJS.defaults.font.size = 12;
  
  // Set default colors for better print quality
  ChartJS.defaults.color = '#374151'; // Gray-700
  ChartJS.defaults.borderColor = '#E5E7EB'; // Gray-200
  
  // Configure for print quality
  ChartJS.defaults.devicePixelRatio = 2; // 2x resolution for crisp prints
  ChartJS.defaults.responsive = false; // Fixed size for consistent PDF layout
  ChartJS.defaults.maintainAspectRatio = true;
  
  // Configure animations (disabled for PDF generation)
  ChartJS.defaults.animation = false;
  
  // Configure plugins
  ChartJS.defaults.plugins.legend = {
    ...ChartJS.defaults.plugins.legend,
    display: true,
    position: 'bottom',
    labels: {
      ...ChartJS.defaults.plugins.legend?.labels,
      padding: 10,
      usePointStyle: true,
      font: {
        size: 11,
      },
    },
  };
  
  ChartJS.defaults.plugins.tooltip = {
    ...ChartJS.defaults.plugins.tooltip,
    enabled: true,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    padding: 12,
    cornerRadius: 4,
    titleFont: {
      size: 13,
      weight: 'bold',
    },
    bodyFont: {
      size: 12,
    },
  };
  
  // Configure datalabels plugin defaults
  ChartJS.defaults.plugins.datalabels = {
    display: false, // Disabled by default, enable per chart as needed
    color: '#374151',
    font: {
      size: 11,
      weight: 'bold',
    },
  };
}

/**
 * Get common chart options for PDF reports
 */
export function getCommonChartOptions(): Partial<ChartOptions<'bar' | 'line'>> {
  return {
    responsive: false,
    maintainAspectRatio: true,
    devicePixelRatio: 2,
    animation: false,
    plugins: {
      legend: {
        display: true,
        position: 'bottom',
      },
      tooltip: {
        enabled: false, // Disabled for PDF
      },
      datalabels: {
        display: false, // Enable per chart as needed
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value: string | number) {
            return value + '%';
          },
        },
        grid: {
          color: '#E5E7EB',
        },
      },
      x: {
        grid: {
          display: false,
        },
      },
    },
  };
}

/**
 * Color palette for metrics (consistent across all charts)
 */
export const METRIC_COLORS = {
  faithfulness: {
    background: 'rgba(59, 130, 246, 0.8)', // Blue
    border: 'rgb(59, 130, 246)',
  },
  answer_relevancy: {
    background: 'rgba(16, 185, 129, 0.8)', // Green
    border: 'rgb(16, 185, 129)',
  },
  context_precision: {
    background: 'rgba(245, 158, 11, 0.8)', // Amber
    border: 'rgb(245, 158, 11)',
  },
  context_recall: {
    background: 'rgba(139, 92, 246, 0.8)', // Purple
    border: 'rgb(139, 92, 246)',
  },
  answer_correctness: {
    background: 'rgba(236, 72, 153, 0.8)', // Pink
    border: 'rgb(236, 72, 153)',
  },
};

/**
 * Get color based on score threshold
 * Green >= 80%, Yellow >= 60%, Red < 60%
 */
export function getScoreColor(score: number): string {
  if (score >= 80) return 'rgb(16, 185, 129)'; // Green
  if (score >= 60) return 'rgb(245, 158, 11)'; // Yellow/Amber
  return 'rgb(239, 68, 68)'; // Red
}

/**
 * Get background color based on score threshold (with opacity)
 */
export function getScoreBackgroundColor(score: number): string {
  if (score >= 80) return 'rgba(16, 185, 129, 0.8)'; // Green
  if (score >= 60) return 'rgba(245, 158, 11, 0.8)'; // Yellow/Amber
  return 'rgba(239, 68, 68, 0.8)'; // Red
}

// Initialize configuration on module load
configureChartDefaults();
