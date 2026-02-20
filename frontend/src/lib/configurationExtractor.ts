import { EvaluationResult } from "./api";

/**
 * Configuration information extracted from evaluation results
 */
export interface ConfigurationInfo {
  llm_provider: string;
  llm_model: string;
  embedding_model: string;
  evaluation_model: string;
  search_alpha: number;
  search_top_k: number;
}

/**
 * Extract configuration information from evaluation results.
 * Takes the first result with complete configuration data.
 * 
 * @param results - Array of evaluation results
 * @returns Configuration information or null if no valid configuration found
 */
export function extractConfiguration(results: EvaluationResult[]): ConfigurationInfo | null {
  if (!results || results.length === 0) {
    console.warn("No evaluation results provided for configuration extraction");
    return null;
  }

  // Find the first result with configuration data
  const resultWithConfig = results.find(
    (result) =>
      result.llm_provider ||
      result.llm_model ||
      result.embedding_model ||
      result.evaluation_model ||
      result.search_alpha !== undefined ||
      result.search_top_k !== undefined
  );

  if (!resultWithConfig) {
    console.warn("No configuration data found in evaluation results");
    return null;
  }

  return {
    llm_provider: resultWithConfig.llm_provider || "N/A",
    llm_model: resultWithConfig.llm_model || "N/A",
    embedding_model: resultWithConfig.embedding_model || "N/A",
    evaluation_model: resultWithConfig.evaluation_model || resultWithConfig.llm_model || "N/A",
    search_alpha: resultWithConfig.search_alpha ?? -1,
    search_top_k: resultWithConfig.search_top_k ?? -1,
  };
}

/**
 * Format configuration information for display
 * 
 * @param config - Configuration information
 * @returns Formatted configuration object with display-friendly values
 */
export function formatConfiguration(config: ConfigurationInfo | null): Record<string, string> {
  if (!config) {
    return {
      "LLM Provider": "N/A",
      "LLM Model": "N/A",
      "Embedding Model": "N/A",
      "Evaluation Model": "N/A",
      "Search Alpha": "N/A",
      "Search Top K": "N/A",
    };
  }

  return {
    "LLM Provider": config.llm_provider,
    "LLM Model": config.llm_model,
    "Embedding Model": config.embedding_model,
    "Evaluation Model": config.evaluation_model,
    "Search Alpha": config.search_alpha >= 0 ? config.search_alpha.toFixed(2) : "N/A",
    "Search Top K": config.search_top_k >= 0 ? config.search_top_k.toString() : "N/A",
  };
}
