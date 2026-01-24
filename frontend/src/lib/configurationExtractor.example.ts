/**
 * Example usage of configurationExtractor utility
 * This file demonstrates how to use the configuration extraction functions
 * in the PDF report generation.
 */

import { extractConfiguration, formatConfiguration } from "./configurationExtractor";
import { EvaluationResult } from "./api";

// Example: Extract configuration from evaluation results
export function exampleUsage(results: EvaluationResult[]) {
  // Extract configuration from results
  const config = extractConfiguration(results);
  
  if (!config) {
    console.log("No configuration found");
    return;
  }

  // Format for display
  const formatted = formatConfiguration(config);
  
  console.log("Configuration Information:");
  console.log("-------------------------");
  Object.entries(formatted).forEach(([key, value]) => {
    console.log(`${key}: ${value}`);
  });
}

// Example: Generate HTML table for PDF report
export function generateConfigurationTable(results: EvaluationResult[]): string {
  const config = extractConfiguration(results);
  const formatted = formatConfiguration(config);
  
  let html = '<table class="config-table">\n';
  html += '  <thead>\n';
  html += '    <tr>\n';
  html += '      <th>Configuration</th>\n';
  html += '      <th>Value</th>\n';
  html += '    </tr>\n';
  html += '  </thead>\n';
  html += '  <tbody>\n';
  
  Object.entries(formatted).forEach(([key, value]) => {
    html += `    <tr>\n`;
    html += `      <td>${key}</td>\n`;
    html += `      <td>${value}</td>\n`;
    html += `    </tr>\n`;
  });
  
  html += '  </tbody>\n';
  html += '</table>\n';
  
  return html;
}

// Example: Generate configuration section for PDF
export function generateConfigurationSection(results: EvaluationResult[]): string {
  const config = extractConfiguration(results);
  const formatted = formatConfiguration(config);
  
  let html = '<div class="configuration-section">\n';
  html += '  <h2>Configuration Information</h2>\n';
  html += '  <div class="config-grid">\n';
  
  Object.entries(formatted).forEach(([key, value]) => {
    html += `    <div class="config-item">\n`;
    html += `      <div class="config-label">${key}</div>\n`;
    html += `      <div class="config-value">${value}</div>\n`;
    html += `    </div>\n`;
  });
  
  html += '  </div>\n';
  html += '</div>\n';
  
  return html;
}
