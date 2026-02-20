"""Weights & Biases tracking service for embedding benchmarks."""

import logging
import os
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Install with: pip install wandb")


class WBTracker:
    """Service for tracking embedding benchmarks in Weights & Biases."""
    
    def __init__(self):
        """Initialize W&B tracker."""
        self.enabled = WANDB_AVAILABLE and bool(os.environ.get("WANDB_API_KEY"))
        self.project = os.environ.get("WANDB_PROJECT", "embedding-benchmarks")
        self.entity = os.environ.get("WANDB_ENTITY", None)
        
        if self.enabled:
            try:
                wandb.login(key=os.environ.get("WANDB_API_KEY"))
                logger.info(f"W&B tracker initialized for project: {self.project}")
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")
                self.enabled = False
        else:
            logger.info("W&B tracking disabled (WANDB_API_KEY not set or wandb not installed)")
    
    def log_benchmark_results(
        self,
        model_name: str,
        benchmark_results: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Log benchmark results to W&B.
        
        Args:
            model_name: Name of the benchmarked model
            benchmark_results: Results from MTEB benchmark
            config: Additional configuration metadata
            
        Returns:
            W&B run ID if successful, None otherwise
        """
        if not self.enabled:
            logger.info("W&B tracking disabled, skipping log")
            return None
        
        try:
            # Initialize W&B run
            run_name = f"benchmark_{model_name.replace('/', '_')}_{int(time.time())}"
            run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                config=config or {},
                tags=["embedding", "benchmark", "mteb", model_name.split('/')[0]]
            )
            
            # Log overall metrics
            summary = benchmark_results.get("summary", {})
            run.log({
                "overall_average": summary.get("overall_average", 0.0),
                "total_tasks": summary.get("total_tasks", 0),
                "total_time_seconds": benchmark_results.get("total_time_seconds", 0),
                "model_name": model_name,
                "timestamp": benchmark_results.get("timestamp", "")
            })
            
            # Log task-specific scores
            task_scores = summary.get("task_scores", {})
            for task_name, task_data in task_scores.items():
                if isinstance(task_data, dict):
                    score = task_data.get("main_score", 0.0)
                    category = task_data.get("category", "unknown")
                    languages = task_data.get("languages", [])
                    
                    # Log to W&B
                    run.log({
                        f"task/{task_name}": score,
                        f"category/{category}": score,
                        f"task_metadata/{task_name}_category": category,
                        f"task_metadata/{task_name}_languages": ",".join(languages),
                        f"task_metadata/{task_name}_score_name": task_data.get("main_score_name", "unknown")
                    })
            
            # Create summary table
            self._create_summary_table(run, model_name, task_scores)
            
            # Log model metadata
            self._log_model_metadata(run, model_name, benchmark_results)
            
            # Create and log comparison chart
            self._log_comparison_chart(run, task_scores)
            
            run.finish()
            
            logger.info(f"Benchmark results logged to W&B: {run.id}")
            return run.id
            
        except Exception as e:
            logger.error(f"Failed to log to W&B: {e}")
            return None
    
    def log_comparison_results(
        self,
        comparison_results: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Log model comparison results to W&B.
        
        Args:
            comparison_results: Results from model comparison
            config: Additional configuration metadata
            
        Returns:
            W&B run ID if successful, None otherwise
        """
        if not self.enabled:
            logger.info("W&B tracking disabled, skipping comparison log")
            return None
        
        try:
            # Initialize W&B run for comparison
            run_name = f"comparison_{int(time.time())}"
            run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                config=config or {},
                tags=["embedding", "comparison", "mteb"]
            )
            
            models = comparison_results.get("models", [])
            comparison_table = comparison_results.get("comparison_table", [])
            
            # Log comparison table
            if comparison_table:
                # Create a table for W&B
                table_data = []
                headers = ["Task"] + models
                
                for row in comparison_table:
                    table_row = [row.get("task", "")]
                    for model in models:
                        score = row.get(model, "N/A")
                        table_row.append(score)
                    table_data.append(table_row)
                
                # Log as W&B table
                comparison_table_wandb = wandb.Table(
                    columns=headers,
                    data=table_data
                )
                run.log({"comparison_table": comparison_table_wandb})
                
                # Calculate and log average scores per model
                model_averages = {}
                for model in models:
                    scores = []
                    for row in comparison_table:
                        score = row.get(model, "N/A")
                        if score != "N/A":
                            try:
                                scores.append(float(score))
                            except ValueError:
                                continue
                    
                    if scores:
                        model_averages[model] = sum(scores) / len(scores)
                        run.log({f"model_average/{model}": model_averages[model]})
                
                # Log best performing model
                if model_averages:
                    best_model = max(model_averages.items(), key=lambda x: x[1])
                    run.log({
                        "best_model": best_model[0],
                        "best_model_score": best_model[1]
                    })
            
            run.finish()
            
            logger.info(f"Comparison results logged to W&B: {run.id}")
            return run.id
            
        except Exception as e:
            logger.error(f"Failed to log comparison to W&B: {e}")
            return None
    
    def _create_summary_table(
        self,
        run,
        model_name: str,
        task_scores: Dict[str, Any]
    ):
        """Create a summary table for W&B."""
        try:
            table_data = []
            for task_name, task_data in task_scores.items():
                if isinstance(task_data, dict):
                    table_data.append([
                        task_name,
                        task_data.get("main_score", 0.0),
                        task_data.get("main_score_name", "unknown"),
                        task_data.get("category", "unknown"),
                        ",".join(task_data.get("languages", []))
                    ])
            
            if table_data:
                summary_table = wandb.Table(
                    columns=["Task", "Score", "Score Type", "Category", "Languages"],
                    data=table_data
                )
                run.log({"task_summary": summary_table})
                
        except Exception as e:
            logger.error(f"Failed to create summary table: {e}")
    
    def _log_model_metadata(
        self,
        run,
        model_name: str,
        benchmark_results: Dict[str, Any]
    ):
        """Log model metadata to W&B."""
        try:
            # Extract model information
            model_parts = model_name.split("/")
            model_provider = model_parts[0] if model_parts else "unknown"
            model_id = "/".join(model_parts[1:]) if len(model_parts) > 1 else model_name
            
            metadata = {
                "model_provider": model_provider,
                "model_id": model_id,
                "model_name": model_name,
                "tasks_run": len(benchmark_results.get("tasks_run", [])),
                "categories": benchmark_results.get("categories", []),
                "benchmark_date": benchmark_results.get("timestamp", ""),
                "total_time_minutes": benchmark_results.get("total_time_seconds", 0) / 60
            }
            
            # Log as artifact metadata
            run.metadata.update(metadata)
            
            # Create model info artifact
            model_info = {
                "model_name": model_name,
                "provider": model_provider,
                "id": model_id,
                "benchmark_metadata": metadata
            }
            
            # Save model info as JSON and log as artifact
            model_info_path = Path("model_info.json")
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            artifact = wandb.Artifact(
                name=f"model_info_{model_provider}_{model_id}",
                type="model_metadata"
            )
            artifact.add_file(str(model_info_path))
            run.log_artifact(artifact)
            
            # Clean up temporary file
            model_info_path.unlink()
            
        except Exception as e:
            logger.error(f"Failed to log model metadata: {e}")
    
    def _log_comparison_chart(
        self,
        run,
        task_scores: Dict[str, Any]
    ):
        """Create and log comparison chart to W&B."""
        try:
            # Group scores by category
            category_scores = {}
            for task_name, task_data in task_scores.items():
                if isinstance(task_data, dict):
                    category = task_data.get("category", "unknown")
                    score = task_data.get("main_score", 0.0)
                    
                    if category not in category_scores:
                        category_scores[category] = []
                    category_scores[category].append(score)
            
            # Calculate averages by category
            category_averages = {}
            for category, scores in category_scores.items():
                if scores:
                    category_averages[category] = sum(scores) / len(scores)
            
            # Create bar chart
            if category_averages:
                data = [[category, avg] for category, avg in category_averages.items()]
                table = wandb.Table(
                    columns=["Category", "Average Score"],
                    data=data
                )
                run.log({"category_averages": table})
                
        except Exception as e:
            logger.error(f"Failed to create comparison chart: {e}")
    
    def get_project_url(self) -> Optional[str]:
        """Get W&B project URL.
        
        Returns:
            W&B project URL if available
        """
        if not self.enabled:
            return None
        
        try:
            if self.entity:
                return f"https://wandb.ai/{self.entity}/{self.project}"
            else:
                return f"https://wandb.ai/{self.project}"
        except Exception:
            return None
    
    def create_benchmark_report(
        self,
        model_name: str,
        benchmark_results: Dict[str, Any]
    ) -> Optional[str]:
        """Create a comprehensive benchmark report in W&B.
        
        Args:
            model_name: Name of the benchmarked model
            benchmark_results: Results from MTEB benchmark
            
        Returns:
            W&B run ID if successful, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            run_name = f"report_{model_name.replace('/', '_')}_{int(time.time())}"
            run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                job_type="benchmark",
                tags=["embedding", "benchmark", "mteb", "report", model_name.split('/')[0]]
            )
            
            # Create comprehensive report
            summary = benchmark_results.get("summary", {})
            
            # Executive summary
            run.log({
                "executive_summary/overall_score": summary.get("overall_average", 0.0),
                "executive_summary/total_tasks": summary.get("total_tasks", 0),
                "executive_summary/time_minutes": benchmark_results.get("total_time_seconds", 0) / 60,
                "executive_summary/model": model_name
            })
            
            # Performance by category
            task_scores = summary.get("task_scores", {})
            category_performance = {}
            
            for task_name, task_data in task_scores.items():
                if isinstance(task_data, dict):
                    category = task_data.get("category", "unknown")
                    score = task_data.get("main_score", 0.0)
                    
                    if category not in category_performance:
                        category_performance[category] = []
                    category_performance[category].append(score)
            
            for category, scores in category_performance.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    max_score = max(scores)
                    min_score = min(scores)
                    
                    run.log({
                        f"category/{category}/average": avg_score,
                        f"category/{category}/max": max_score,
                        f"category/{category}/min": min_score,
                        f"category/{category}/count": len(scores)
                    })
            
            # Create HTML report
            html_report = self._generate_html_report(model_name, benchmark_results)
            
            # Save and log HTML report
            report_path = Path(f"benchmark_report_{model_name.replace('/', '_')}.html")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            artifact = wandb.Artifact(
                name=f"benchmark_report_{model_name.replace('/', '_')}",
                type="report"
            )
            artifact.add_file(str(report_path))
            run.log_artifact(artifact)
            
            # Clean up
            report_path.unlink()
            
            run.finish()
            
            logger.info(f"Benchmark report created in W&B: {run.id}")
            return run.id
            
        except Exception as e:
            logger.error(f"Failed to create benchmark report: {e}")
            return None
    
    def _generate_html_report(
        self,
        model_name: str,
        benchmark_results: Dict[str, Any]
    ) -> str:
        """Generate HTML report for benchmark results."""
        try:
            summary = benchmark_results.get("summary", {})
            task_scores = summary.get("task_scores", {})
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Benchmark Report: {model_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .score-high {{ color: green; font-weight: bold; }}
                    .score-medium {{ color: orange; font-weight: bold; }}
                    .score-low {{ color: red; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Benchmark Report: {model_name}</h1>
                    <p>Generated on: {benchmark_results.get('timestamp', '')}</p>
                </div>
                
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Overall Average:</strong> {summary.get('overall_average', 0):.3f}
                </div>
                <div class="metric">
                    <strong>Total Tasks:</strong> {summary.get('total_tasks', 0)}
                </div>
                <div class="metric">
                    <strong>Total Time:</strong> {benchmark_results.get('total_time_seconds', 0) / 60:.1f} minutes
                </div>
                
                <h2>Task Results</h2>
                <table>
                    <tr>
                        <th>Task</th>
                        <th>Score</th>
                        <th>Category</th>
                        <th>Languages</th>
                    </tr>
            """
            
            for task_name, task_data in task_scores.items():
                if isinstance(task_data, dict):
                    score = task_data.get("main_score", 0.0)
                    score_class = "score-high" if score > 0.8 else "score-medium" if score > 0.6 else "score-low"
                    
                    html += f"""
                    <tr>
                        <td>{task_name}</td>
                        <td class="{score_class}">{score:.3f}</td>
                        <td>{task_data.get('category', 'unknown')}</td>
                        <td>{', '.join(task_data.get('languages', []))}</td>
                    </tr>
                    """
            
            html += """
                </table>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return "<html><body><h1>Error generating report</h1></body></html>"

    def log_reranker_benchmark(
        self,
        task_name: str,
        provider: str,
        model: str,
        results: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Log reranker benchmark results to W&B.
        
        Args:
            task_name: Name of the MTEB reranking task
            provider: Reranker provider (cohere, alibaba, jina, bge)
            model: Model name
            results: Results from MTEB reranker benchmark
            config: Additional configuration metadata
            
        Returns:
            W&B run ID if successful, None otherwise
        """
        if not self.enabled:
            logger.info("W&B tracking disabled, skipping reranker benchmark log")
            return None
        
        try:
            # Initialize W&B run for reranker benchmark
            run_name = f"reranker_{task_name}_{provider}_{int(time.time())}"
            run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                config=config or {},
                tags=["reranker", "mteb", task_name, provider]
            )
            
            # Log basic metrics
            run.log({
                "task_name": task_name,
                "provider": provider,
                "model": model,
                "main_score": results.get("main_score", 0.0),
                "ndcg_at_10": results.get("ndcg_at_10", 0.0),
                "map_at_10": results.get("map_at_10", 0.0),
                "recall_at_10": results.get("recall_at_10", 0.0),
                "precision_at_10": results.get("precision_at_10", 0.0),
                "execution_time_seconds": results.get("execution_time_seconds", 0.0),
                "total_queries": results.get("total_queries", 0)
            })
            
            # Create summary table
            table_data = [[
                "Task", "Provider", "Model", "Main Score", "NDCG@10", "MAP@10", "Recall@10", "Precision@10"
            ], [
                task_name, provider, model, 
                f"{results.get('main_score', 0.0):.4f}",
                f"{results.get('ndcg_at_10', 0.0):.4f}",
                f"{results.get('map_at_10', 0.0):.4f}",
                f"{results.get('recall_at_10', 0.0):.4f}",
                f"{results.get('precision_at_10', 0.0):.4f}"
            ]]
            
            summary_table = wandb.Table(
                columns=["Task", "Provider", "Model", "Main Score", "NDCG@10", "MAP@10", "Recall@10", "Precision@10"],
                data=table_data
            )
            run.log({"reranker_summary": summary_table})
            
            run.finish()
            
            logger.info(f"Reranker benchmark results logged to W&B: {run.id}")
            return run.id
            
        except Exception as e:
            logger.error(f"Failed to log reranker benchmark to W&B: {e}")
            return None
