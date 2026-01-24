/**
 * StatisticsCalculator Utility
 * 
 * Provides statistical calculation functions for RAGAS evaluation metrics.
 */

/**
 * Interface for metric statistics
 */
export interface MetricStatistics {
  mean: number;
  median: number;
  stdDev: number;
  variance: number;
  min: number;
  max: number;
  q1: number;
  q3: number;
}

/**
 * Calculate the arithmetic mean of a number array
 * Handles empty arrays and null values
 * 
 * @param values - Array of numbers
 * @returns Mean value, or 0 if array is empty
 */
export function calculateMean(values: number[]): number {
  // Filter out null, undefined, and NaN values
  const validValues = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v));
  
  if (validValues.length === 0) {
    return 0;
  }
  
  const sum = validValues.reduce((acc, val) => acc + val, 0);
  return sum / validValues.length;
}

/**
 * Calculate the median of a number array
 * Sorts array and finds middle value
 * Handles even-length arrays by averaging two middle values
 * 
 * @param values - Array of numbers
 * @returns Median value, or 0 if array is empty
 */
export function calculateMedian(values: number[]): number {
  // Filter out null, undefined, and NaN values
  const validValues = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v));
  
  if (validValues.length === 0) {
    return 0;
  }
  
  // Sort the array
  const sorted = [...validValues].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  
  // If even length, average the two middle values
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  
  // If odd length, return the middle value
  return sorted[mid];
}

/**
 * Calculate the variance of a number array
 * Calculates sum of squared differences from mean
 * Divides by count for population variance
 * 
 * @param values - Array of numbers
 * @returns Variance value, or 0 if array is empty
 */
export function calculateVariance(values: number[]): number {
  // Filter out null, undefined, and NaN values
  const validValues = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v));
  
  if (validValues.length === 0) {
    return 0;
  }
  
  const mean = calculateMean(validValues);
  const squaredDiffs = validValues.map(v => Math.pow(v - mean, 2));
  const sumSquaredDiffs = squaredDiffs.reduce((acc, val) => acc + val, 0);
  
  return sumSquaredDiffs / validValues.length;
}

/**
 * Calculate the standard deviation of a number array
 * Uses Math.sqrt(variance) formula
 * 
 * @param values - Array of numbers
 * @returns Standard deviation value, or 0 if array is empty
 */
export function calculateStandardDeviation(values: number[]): number {
  const variance = calculateVariance(values);
  return Math.sqrt(variance);
}

/**
 * Find minimum and maximum values in an array
 * 
 * @param values - Array of numbers
 * @returns Object with min and max values, or { min: 0, max: 0 } if array is empty
 */
export function calculateMinMax(values: number[]): { min: number; max: number } {
  // Filter out null, undefined, and NaN values
  const validValues = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v));
  
  if (validValues.length === 0) {
    return { min: 0, max: 0 };
  }
  
  return {
    min: Math.min(...validValues),
    max: Math.max(...validValues)
  };
}

/**
 * Calculate quartiles (Q1 and Q3) for box plot generation
 * Q1 is the 25th percentile, Q3 is the 75th percentile
 * 
 * @param values - Array of numbers
 * @returns Object with q1 and q3 values, or { q1: 0, q3: 0 } if array is empty
 */
export function calculateQuartiles(values: number[]): { q1: number; q3: number } {
  // Filter out null, undefined, and NaN values
  const validValues = values.filter(v => v !== null && v !== undefined && !Number.isNaN(v));
  
  if (validValues.length === 0) {
    return { q1: 0, q3: 0 };
  }
  
  // Sort the array
  const sorted = [...validValues].sort((a, b) => a - b);
  
  // Calculate Q1 (25th percentile)
  const q1Index = Math.floor(sorted.length * 0.25);
  const q1 = sorted[q1Index];
  
  // Calculate Q3 (75th percentile)
  const q3Index = Math.floor(sorted.length * 0.75);
  const q3 = sorted[q3Index];
  
  return { q1, q3 };
}

/**
 * Calculate overall average from five metric averages
 * Sums all five metric averages and divides by 5
 * 
 * @param metrics - Object containing five metric averages
 * @returns Overall average value
 */
export function calculateOverallAverage(metrics: {
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number;
  answer_correctness: number;
}): number {
  const sum = 
    metrics.faithfulness +
    metrics.answer_relevancy +
    metrics.context_precision +
    metrics.context_recall +
    metrics.answer_correctness;
  
  return sum / 5;
}

/**
 * Calculate all statistics for a metric array
 * Combines all statistical calculations into a single function
 * 
 * @param values - Array of numbers
 * @returns MetricStatistics object with all calculated statistics
 */
export function calculateStatistics(values: number[]): MetricStatistics {
  const mean = calculateMean(values);
  const median = calculateMedian(values);
  const variance = calculateVariance(values);
  const stdDev = calculateStandardDeviation(values);
  const { min, max } = calculateMinMax(values);
  const { q1, q3 } = calculateQuartiles(values);
  
  return {
    mean,
    median,
    stdDev,
    variance,
    min,
    max,
    q1,
    q3
  };
}
