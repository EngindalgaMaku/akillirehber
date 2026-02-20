"""Embedding benchmark service using MTEB.

This service provides functionality to benchmark embedding models
using the Massive Text Embedding Benchmark (MTEB) library.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from app.services.wb_tracker import WBTracker

logger = logging.getLogger(__name__)


class EmbeddingBenchmarkService:
    """Service for benchmarking embedding models using MTEB."""
    
    def __init__(self):
        """Initialize embedding benchmark service."""
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        self.wb_tracker = WBTracker()
        
        # Available MTEB tasks for different languages
        self.available_tasks = {
            "english": [
                "Banking77Classification.v2",
                "EmotionClassification.v2",
                "ImdbClassification.v2",
                "TweetSentimentExtractionClassification.v2",
                "AmazonCounterfactualClassification.v2",
                "AmazonPolarityClassification.v2",
                "AmazonReviewsClassification.v2",
                "MassiveIntentClassification.v2",
                "MassiveScenarioClassification.v2",
                "MTOPDomainClassification.v2",
                "MTOPIntentClassification.v2",
            ],
            "multilingual": [
                "AmazonCounterfactualClassification.v2",
                "AmazonPolarityClassification.v2",
                "AmazonReviewsClassification.v2",
                "MassiveIntentClassification.v2",
                "MassiveScenarioClassification.v2",
                "MTOPDomainClassification.v2",
                "MTOPIntentClassification.v2",
                "MultiHateClassification.v2",
                "TweetSentimentExtractionClassification.v2",
                "XNLI.v2",
                "XNLIV2.v2",
                "XStoryCloze.v2",
            ],
            "retrieval": [
                "ArguAna",
                "ClimateFEVER",
                "CQADupstackAndroidRetrieval",
                "CQADupstackEnglishRetrieval",
                "CQADupstackGamingRetrieval",
                "CQADupstackGisRetrieval",
                "CQADupstackMathematicaRetrieval",
                "CQADupstackPhysicsRetrieval",
                "CQADupstackProgrammersRetrieval",
                "CQADupstackStatsRetrieval",
                "CQADupstackTexRetrieval",
                "CQADupstackUnixRetrieval",
                "CQADupstackWebmastersRetrieval",
                "CQADupstackWordpressRetrieval",
                "DBPedia",
                "FEVER",
                "FiQA2018",
                "HotpotQA",
                "MSMARCO",
                "NFCorpus",
                "NQ",
                "QuoraRetrieval",
                "SCIDOCS",
                "SciFact",
                "Touche2020",
                "TRECCOVID",
            ],
            "clustering": [
                "ArxivClusteringP2P.v2",
                "ArxivClusteringS2S.v2",
                "BiorxivClusteringP2P.v2",
                "BiorxivClusteringS2S.v2",
                "MedrxivClusteringP2P.v2",
                "MedrxivClusteringS2S.v2",
                "RedditClustering.v2",
                "RedditClusteringP2P.v2",
                "StackExchangeClustering.v2",
                "StackExchangeClusteringP2P.v2",
                "TwentyNewsgroupsClustering.v2",
            ],
            "reranking": [
                "AskUbuntuDupQuestions",
                "MindSmallReranking",
                "SciDocsRR",
                "StackOverflowDupQuestions",
            ]
        }
        
        logger.info("EmbeddingBenchmarkService initialized")
    
    def get_available_tasks(self) -> Dict[str, List[str]]:
        """Get available benchmark tasks by category.
        
        Returns:
            Dictionary mapping categories to task lists
        """
        return self.available_tasks
    
    def get_task_info(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific task.
        
        Args:
            task_name: Name of the MTEB task
            
        Returns:
            Task information dictionary or None if not found
        """
        try:
            import mteb
            tasks = mteb.get_tasks(tasks=[task_name])
            if tasks:
                task = tasks[0]
                return {
                    "name": task.metadata.name,
                    "description": task.metadata.description,
                    "type": task.metadata.type,
                    "category": task.metadata.category,
                    "languages": task.metadata.languages,
                    "main_score": task.metadata.main_score,
                    "domains": getattr(task.metadata, 'domains', []),
                    "reference": getattr(task.metadata, 'reference', ''),
                    "annotated": getattr(task.metadata, 'annotated', False)
                }
            return None
        except Exception as e:
            logger.error(f"Error getting task info for {task_name}: {e}")
            return None
    
    def benchmark_model(
        self,
        model_name: str,
        tasks: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Benchmark an embedding model using MTEB.
        
        Args:
            model_name: Name of the embedding model
            tasks: Specific tasks to run (optional)
            categories: Categories of tasks to run (optional)
            output_name: Custom name for results file (optional)
            
        Returns:
            Benchmark results dictionary
        """
        try:
            import mteb
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Starting benchmark for model: {model_name}")
            
            # Determine tasks to run
            if tasks:
                selected_tasks = tasks
            elif categories:
                selected_tasks = []
                for category in categories:
                    if category in self.available_tasks:
                        selected_tasks.extend(self.available_tasks[category])
            else:
                # Default to a quick subset of tasks
                selected_tasks = [
                    "Banking77Classification.v2",
                    "ImdbClassification.v2",
                    "AmazonReviewsClassification.v2",
                    "MSMARCO",
                    "ArxivClusteringP2P.v2"
                ]
            
            logger.info(f"Running {len(selected_tasks)} tasks: {selected_tasks}")
            
            # Get model
            model = mteb.get_model(model_name)
            
            # Get tasks
            tasks = mteb.get_tasks(tasks=selected_tasks)
            
            # Run evaluation
            start_time = time.time()
            results = mteb.evaluate(model, tasks=tasks)
            end_time = time.time()
            
            # Prepare results
            benchmark_results = {
                "model_name": model_name,
                "tasks_run": selected_tasks,
                "categories": categories or ["custom"],
                "total_time_seconds": end_time - start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
                "summary": self._generate_summary(results)
            }
            
            # Save results
            if output_name:
                results_file = self.results_dir / f"{output_name}_{int(time.time())}.json"
            else:
                results_file = self.results_dir / f"{model_name.replace('/', '_')}_{int(time.time())}.json"
            
            with open(results_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)
            
            logger.info(f"Benchmark completed. Results saved to: {results_file}")
            
            # Log to W&B if enabled
            try:
                wb_run_id = self.wb_tracker.log_benchmark_results(
                    model_name=model_name,
                    benchmark_results=benchmark_results,
                    config={
                        "tasks": selected_tasks,
                        "categories": categories,
                        "output_name": output_name
                    }
                )
                if wb_run_id:
                    logger.info(f"Results logged to W&B with run ID: {wb_run_id}")
                    benchmark_results["wb_run_id"] = wb_run_id
                    benchmark_results["wb_url"] = self.wb_tracker.get_project_url()
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")
            
            return benchmark_results
            
        except ImportError as e:
            logger.error(f"MTEB or sentence_transformers not installed: {e}")
            raise ValueError(
                "MTEB and sentence_transformers are required for benchmarking. "
                "Install them with: pip install mteb sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of benchmark results.
        
        Args:
            results: Raw MTEB results
            
        Returns:
            Summary dictionary
        """
        summary = {
            "total_tasks": len(results),
            "task_scores": {},
            "category_averages": {},
            "overall_average": 0.0
        }
        
        all_scores = []
        
        for task_name, task_result in results.items():
            if hasattr(task_result, 'scores') and task_result.scores:
                main_score = task_result.scores.get(task_result.metadata.main_score, 0.0)
                summary["task_scores"][task_name] = {
                    "main_score": main_score,
                    "main_score_name": task_result.metadata.main_score,
                    "category": task_result.metadata.category,
                    "languages": task_result.metadata.languages
                }
                all_scores.append(main_score)
        
        if all_scores:
            summary["overall_average"] = sum(all_scores) / len(all_scores)
        
        return summary
    
    def get_benchmark_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent benchmark results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of benchmark result summaries
        """
        results = []
        
        try:
            for result_file in sorted(self.results_dir.glob("*.json"), reverse=True)[:limit]:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results.append({
                        "filename": result_file.name,
                        "model_name": data.get("model_name"),
                        "timestamp": data.get("timestamp"),
                        "total_time_seconds": data.get("total_time_seconds"),
                        "overall_average": data.get("summary", {}).get("overall_average", 0.0),
                        "total_tasks": data.get("summary", {}).get("total_tasks", 0)
                    })
        except Exception as e:
            logger.error(f"Error loading benchmark history: {e}")
        
        return results
    
    def compare_models(
        self,
        model_names: List[str],
        tasks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple embedding models on the same tasks.
        
        Args:
            model_names: List of model names to compare
            tasks: Tasks to run comparison on (optional)
            
        Returns:
            Comparison results
        """
        comparison_results = {
            "models": model_names,
            "tasks": tasks or ["default"],
            "model_results": {},
            "comparison_table": [],
            "best_model": ""
        }
        
        for model_name in model_names:
            try:
                logger.info(f"Benchmarking model: {model_name}")
                result = self.benchmark_model(model_name, tasks=tasks)
                comparison_results["model_results"][model_name] = result["summary"]
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
                comparison_results["model_results"][model_name] = {"error": str(e)}
        
        # Generate comparison table
        if tasks:
            for task in tasks:
                row = {"task": task}
                for model_name in model_names:
                    model_result = comparison_results["model_results"].get(model_name, {})
                    task_scores = model_result.get("task_scores", {})
                    task_score = task_scores.get(task, {})
                    if isinstance(task_score, dict):
                        row[model_name] = task_score.get("main_score", "N/A")
                    else:
                        row[model_name] = "N/A"
                comparison_results["comparison_table"].append(row)
        
        # Determine best model based on overall average
        best_model = None
        best_score = -1
        for model_name in model_names:
            model_result = comparison_results["model_results"].get(model_name, {})
            overall_avg = model_result.get("overall_average", 0)
            if overall_avg > best_score:
                best_score = overall_avg
                best_model = model_name
        
        comparison_results["best_model"] = best_model or "N/A"
        
        # Log comparison to W&B if enabled
        try:
            wb_run_id = self.wb_tracker.log_comparison_results(
                comparison_results=comparison_results,
                config={"tasks": tasks}
            )
            if wb_run_id:
                logger.info(f"Comparison logged to W&B with run ID: {wb_run_id}")
                comparison_results["wb_run_id"] = wb_run_id
                comparison_results["wb_url"] = self.wb_tracker.get_project_url()
        except Exception as e:
            logger.warning(f"Failed to log comparison to W&B: {e}")
        
        return comparison_results
