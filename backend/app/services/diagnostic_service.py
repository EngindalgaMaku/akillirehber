"""Diagnostic service for comprehensive monitoring and debugging of PDF processing."""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.models.db_models import (
    Document,
    ProcessingStatus,
    ProcessingStatusEnum,
    DiagnosticReport,
    ChunkQualityMetrics,
    Chunk,
)
from app.models.schemas import (
    DiagnosticReportResponse,
    ChunkQualityMetricsResponse,
    SystemDiagnosticsResponse,
    ProcessingStatusResponse,
    FileInfo,
    ExtractionInfo,
    ChunkingInfo,
    ErrorEntry,
    PerformanceMetrics,
)

logger = logging.getLogger(__name__)


class DiagnosticService:
    """Service for comprehensive monitoring and debugging capabilities."""

    def __init__(self, db: Session):
        self.db = db

    def get_document_diagnostics(self, document_id: int) -> Optional[DiagnosticReportResponse]:
        """Get comprehensive diagnostic report for a document.
        
        Args:
            document_id: ID of the document to analyze
            
        Returns:
            Diagnostic report or None if document not found
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return None

        # Get the latest diagnostic report
        report = (
            self.db.query(DiagnosticReport)
            .filter(DiagnosticReport.document_id == document_id)
            .order_by(desc(DiagnosticReport.created_at))
            .first()
        )

        if not report:
            # Generate a new diagnostic report
            report = self._generate_diagnostic_report(document)

        return self._convert_diagnostic_report(report)

    def get_processing_status(self, document_id: int) -> Optional[ProcessingStatusResponse]:
        """Get current processing status for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Processing status or None if not found
        """
        status = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.document_id == document_id)
            .first()
        )
        
        if not status:
            return None
            
        return ProcessingStatusResponse.model_validate(status)

    def get_chunk_quality_metrics(self, document_id: int) -> Optional[ChunkQualityMetricsResponse]:
        """Get chunk quality metrics for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Chunk quality metrics or None if not found
        """
        metrics = (
            self.db.query(ChunkQualityMetrics)
            .filter(ChunkQualityMetrics.document_id == document_id)
            .first()
        )
        
        if not metrics:
            # Generate metrics if chunks exist
            chunks = self.db.query(Chunk).filter(Chunk.document_id == document_id).all()
            if chunks:
                metrics = self._generate_chunk_quality_metrics(document_id, chunks)
            else:
                return None
                
        return ChunkQualityMetricsResponse.model_validate(metrics)

    def validate_processing_pipeline(self) -> Dict[str, any]:
        """Perform end-to-end pipeline health check.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "database_connection": self._check_database_connection(),
            "document_processing": self._check_document_processing_health(),
            "chunking_service": self._check_chunking_service_health(),
            "embedding_service": self._check_embedding_service_health(),
            "overall_health": "healthy"
        }
        
        # Determine overall health
        if any(not result["healthy"] for result in results.values() if isinstance(result, dict)):
            results["overall_health"] = "error"
        elif any(result.get("warnings", []) for result in results.values() if isinstance(result, dict)):
            results["overall_health"] = "warning"
            
        return results

    def get_performance_metrics(self) -> Dict[str, any]:
        """Get system-wide performance metrics.
        
        Returns:
            Dictionary with performance data
        """
        # Get processing statistics
        processing_stats = self._get_processing_statistics()
        
        # Get system resource usage
        system_metrics = self._get_system_metrics()
        
        # Get error statistics
        error_stats = self._get_error_statistics()
        
        return {
            "processing_statistics": processing_stats,
            "system_metrics": system_metrics,
            "error_statistics": error_stats,
            "timestamp": datetime.utcnow()
        }

    def run_system_diagnostics(self) -> SystemDiagnosticsResponse:
        """Run complete system health check and diagnostics.
        
        Returns:
            Comprehensive system diagnostics
        """
        # Get basic statistics
        total_documents = self.db.query(Document).count()
        
        # Get processing statistics
        processing_stats = self._get_processing_statistics()
        
        # Get error summary
        error_summary = self._get_error_statistics()
        
        # Get performance summary
        performance_summary = self._get_system_metrics()
        
        # Generate recommendations
        recommendations = self._generate_system_recommendations(
            processing_stats, error_summary, performance_summary
        )
        
        # Determine system health
        system_health = self._determine_system_health(processing_stats, error_summary)
        
        return SystemDiagnosticsResponse(
            system_health=system_health,
            total_documents=total_documents,
            processing_stats=processing_stats,
            error_summary=error_summary,
            performance_summary=performance_summary,
            recommendations=recommendations,
            timestamp=datetime.utcnow()
        )

    def _generate_diagnostic_report(self, document: Document) -> DiagnosticReport:
        """Generate a new diagnostic report for a document."""
        # Collect file information
        file_info = {
            "filename": document.original_filename,
            "file_size": document.file_size,
            "file_type": document.file_type,
            "char_count": document.char_count
        }
        
        # Collect extraction information
        extraction_info = {
            "success": document.content is not None,
            "char_count": document.char_count,
            "method_used": "automatic"
        }
        
        # Collect chunking information
        chunks = self.db.query(Chunk).filter(Chunk.document_id == document.id).all()
        chunking_info = {
            "success": len(chunks) > 0,
            "total_chunks": len(chunks),
            "strategy_used": "recursive"  # Default, could be enhanced
        }
        
        # Collect error log
        error_log = []
        processing_status = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.document_id == document.id)
            .first()
        )
        
        if processing_status and processing_status.error_message:
            error_log.append({
                "timestamp": processing_status.updated_at.isoformat(),
                "stage": "processing",
                "error_type": "ProcessingError",
                "error_message": processing_status.error_message,
                "context": processing_status.error_details
            })
        
        # Generate performance metrics
        performance_metrics = {}
        if processing_status and processing_status.processing_duration:
            performance_metrics["total_processing_time"] = processing_status.processing_duration
        
        # Generate recommendations
        recommendations = self._generate_document_recommendations(
            document, chunks, processing_status
        )
        
        # Create and save diagnostic report
        report = DiagnosticReport(
            document_id=document.id,
            report_type="processing",
            file_info=file_info,
            extraction_info=extraction_info,
            chunking_info=chunking_info,
            error_log=error_log,
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )
        
        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)
        
        return report

    def _generate_chunk_quality_metrics(self, document_id: int, chunks: List[Chunk]) -> ChunkQualityMetrics:
        """Generate chunk quality metrics for a document."""
        if not chunks:
            return None
            
        # Calculate basic metrics
        chunk_sizes = [chunk.char_count for chunk in chunks]
        total_chunks = len(chunks)
        avg_chunk_size = sum(chunk_sizes) // total_chunks if total_chunks > 0 else 0
        min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
        max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
        
        # Calculate size distribution
        size_ranges = {
            "0-100": 0,
            "101-300": 0,
            "301-500": 0,
            "501-1000": 0,
            "1000+": 0
        }
        
        for size in chunk_sizes:
            if size <= 100:
                size_ranges["0-100"] += 1
            elif size <= 300:
                size_ranges["101-300"] += 1
            elif size <= 500:
                size_ranges["301-500"] += 1
            elif size <= 1000:
                size_ranges["501-1000"] += 1
            else:
                size_ranges["1000+"] += 1
        
        # Analyze overlaps
        overlap_count = sum(1 for chunk in chunks if chunk.has_overlap)
        overlap_analysis = {
            "total_overlaps": overlap_count,
            "overlap_percentage": (overlap_count / total_chunks * 100) if total_chunks > 0 else 0
        }
        
        # Calculate content quality score (simplified)
        quality_score = self._calculate_content_quality_score(chunks)
        
        # Generate recommendations
        recommendations = self._generate_chunk_recommendations(
            total_chunks, avg_chunk_size, size_ranges, overlap_analysis, quality_score
        )
        
        # Create and save metrics
        metrics = ChunkQualityMetrics(
            document_id=document_id,
            total_chunks=total_chunks,
            avg_chunk_size=avg_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            size_distribution=size_ranges,
            overlap_analysis=overlap_analysis,
            content_quality_score=quality_score,
            recommendations=recommendations,
            chunking_strategy="recursive",  # Default, could be enhanced
            chunk_size_config=500,  # Default, could be enhanced
            overlap_config=50  # Default, could be enhanced
        )
        
        self.db.add(metrics)
        self.db.commit()
        self.db.refresh(metrics)
        
        return metrics

    def _convert_diagnostic_report(self, report: DiagnosticReport) -> DiagnosticReportResponse:
        """Convert database model to response schema."""
        return DiagnosticReportResponse(
            id=report.id,
            document_id=report.document_id,
            report_type=report.report_type,
            file_info=FileInfo(**report.file_info) if report.file_info else None,
            extraction_info=ExtractionInfo(**report.extraction_info) if report.extraction_info else None,
            chunking_info=ChunkingInfo(**report.chunking_info) if report.chunking_info else None,
            error_log=[ErrorEntry(**error) for error in report.error_log] if report.error_log else [],
            performance_metrics=PerformanceMetrics(**report.performance_metrics) if report.performance_metrics else None,
            recommendations=report.recommendations or [],
            created_at=report.created_at
        )

    def _check_database_connection(self) -> Dict[str, any]:
        """Check database connection health."""
        try:
            self.db.execute("SELECT 1")
            return {"healthy": True, "message": "Database connection successful"}
        except Exception as e:
            return {"healthy": False, "message": f"Database connection failed: {str(e)}"}

    def _check_document_processing_health(self) -> Dict[str, any]:
        """Check document processing service health."""
        # Check for recent processing failures
        recent_errors = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.status == ProcessingStatusEnum.ERROR)
            .filter(ProcessingStatus.updated_at >= datetime.utcnow().replace(hour=0, minute=0, second=0))
            .count()
        )
        
        total_today = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0))
            .count()
        )
        
        error_rate = (recent_errors / total_today * 100) if total_today > 0 else 0
        
        if error_rate > 50:
            return {"healthy": False, "message": f"High error rate: {error_rate:.1f}%", "error_rate": error_rate}
        elif error_rate > 20:
            return {"healthy": True, "warnings": [f"Elevated error rate: {error_rate:.1f}%"], "error_rate": error_rate}
        else:
            return {"healthy": True, "message": "Document processing healthy", "error_rate": error_rate}

    def _check_chunking_service_health(self) -> Dict[str, any]:
        """Check chunking service health."""
        # Check for documents with no chunks
        documents_without_chunks = (
            self.db.query(Document)
            .outerjoin(Chunk)
            .filter(Document.is_processed == True)
            .filter(Chunk.id == None)
            .count()
        )
        
        total_processed = self.db.query(Document).filter(Document.is_processed == True).count()
        
        if total_processed == 0:
            return {"healthy": True, "message": "No processed documents to check"}
        
        no_chunks_rate = (documents_without_chunks / total_processed * 100)
        
        if no_chunks_rate > 10:
            return {"healthy": False, "message": f"High rate of documents without chunks: {no_chunks_rate:.1f}%"}
        elif no_chunks_rate > 5:
            return {"healthy": True, "warnings": [f"Some documents without chunks: {no_chunks_rate:.1f}%"]}
        else:
            return {"healthy": True, "message": "Chunking service healthy"}

    def _check_embedding_service_health(self) -> Dict[str, any]:
        """Check embedding service health."""
        # This is a placeholder - would need to check Weaviate connection
        return {"healthy": True, "message": "Embedding service check not implemented"}

    def _get_processing_statistics(self) -> Dict[str, any]:
        """Get processing statistics."""
        # Get status distribution
        status_counts = (
            self.db.query(ProcessingStatus.status, func.count(ProcessingStatus.id))
            .group_by(ProcessingStatus.status)
            .all()
        )
        
        status_distribution = {status.value: 0 for status in ProcessingStatusEnum}
        for status, count in status_counts:
            status_distribution[status.value] = count
        
        # Get average processing time
        avg_processing_time = (
            self.db.query(func.avg(ProcessingStatus.processing_duration))
            .filter(ProcessingStatus.processing_duration.isnot(None))
            .scalar()
        )
        
        return {
            "status_distribution": status_distribution,
            "average_processing_time": float(avg_processing_time) if avg_processing_time else None,
            "total_processed": sum(status_distribution.values())
        }

    def _get_system_metrics(self) -> Dict[str, any]:
        """Get current system resource metrics."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                "memory_usage_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "cpu_usage_percent": cpu_percent,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {"error": "Failed to get system metrics"}

    def _get_error_statistics(self) -> Dict[str, any]:
        """Get error statistics."""
        # Get recent errors (last 24 hours)
        recent_errors = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.status == ProcessingStatusEnum.ERROR)
            .filter(ProcessingStatus.updated_at >= datetime.utcnow().replace(hour=0, minute=0, second=0))
            .count()
        )
        
        # Get total errors
        total_errors = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.status == ProcessingStatusEnum.ERROR)
            .count()
        )
        
        return {
            "recent_errors": recent_errors,
            "total_errors": total_errors,
            "error_rate_24h": recent_errors  # Could calculate as percentage if needed
        }

    def _generate_system_recommendations(self, processing_stats: Dict, error_stats: Dict, performance_stats: Dict) -> List[str]:
        """Generate system-wide recommendations."""
        recommendations = []
        
        # Check error rates
        if error_stats.get("recent_errors", 0) > 10:
            recommendations.append("High error rate detected. Review error logs and consider system maintenance.")
        
        # Check processing performance
        avg_time = processing_stats.get("average_processing_time")
        if avg_time and avg_time > 60:  # More than 1 minute
            recommendations.append("Processing times are elevated. Consider optimizing chunking parameters.")
        
        # Check system resources
        memory_usage = performance_stats.get("memory_usage_percent", 0)
        if memory_usage > 80:
            recommendations.append("High memory usage detected. Consider scaling resources.")
        
        cpu_usage = performance_stats.get("cpu_usage_percent", 0)
        if cpu_usage > 80:
            recommendations.append("High CPU usage detected. Consider optimizing processing algorithms.")
        
        return recommendations

    def _determine_system_health(self, processing_stats: Dict, error_stats: Dict) -> str:
        """Determine overall system health."""
        recent_errors = error_stats.get("recent_errors", 0)
        total_processed = processing_stats.get("total_processed", 0)
        
        if total_processed == 0:
            return "healthy"  # No data to evaluate
        
        error_rate = (recent_errors / total_processed * 100) if total_processed > 0 else 0
        
        if error_rate > 50:
            return "error"
        elif error_rate > 20:
            return "warning"
        else:
            return "healthy"

    def _generate_document_recommendations(self, document: Document, chunks: List[Chunk], status: Optional[ProcessingStatus]) -> List[str]:
        """Generate recommendations for a specific document."""
        recommendations = []
        
        # Check if document was processed successfully
        if not document.is_processed:
            recommendations.append("Document processing incomplete. Consider retrying processing.")
        
        # Check chunk count
        if len(chunks) == 0 and document.is_processed:
            recommendations.append("No chunks generated. Check chunking configuration and retry.")
        elif len(chunks) < 3:
            recommendations.append("Very few chunks generated. Consider reducing chunk size or checking document content.")
        
        # Check for errors
        if status and status.status == ProcessingStatusEnum.ERROR:
            recommendations.append("Processing failed with errors. Review error details and retry with different settings.")
        
        # Check processing time
        if status and status.processing_duration and status.processing_duration > 120:  # 2 minutes
            recommendations.append("Processing took longer than expected. Consider optimizing document or chunking parameters.")
        
        return recommendations

    def _generate_chunk_recommendations(self, total_chunks: int, avg_size: int, size_dist: Dict, overlap_analysis: Dict, quality_score: float) -> List[str]:
        """Generate recommendations for chunk quality."""
        recommendations = []
        
        # Check chunk count
        if total_chunks < 5:
            recommendations.append("Very few chunks generated. Consider reducing chunk size.")
        elif total_chunks > 100:
            recommendations.append("Many chunks generated. Consider increasing chunk size for better performance.")
        
        # Check average size
        if avg_size < 200:
            recommendations.append("Chunks are quite small. Consider increasing chunk size for better context.")
        elif avg_size > 1000:
            recommendations.append("Chunks are quite large. Consider reducing chunk size for better granularity.")
        
        # Check size distribution
        small_chunks_pct = (size_dist.get("0-100", 0) / total_chunks * 100) if total_chunks > 0 else 0
        if small_chunks_pct > 30:
            recommendations.append("Many very small chunks detected. Review chunking strategy.")
        
        # Check quality score
        if quality_score < 0.5:
            recommendations.append("Low content quality score. Review document content and chunking parameters.")
        
        return recommendations

    def _calculate_content_quality_score(self, chunks: List[Chunk]) -> float:
        """Calculate a simple content quality score for chunks."""
        if not chunks:
            return 0.0
        
        # Simple heuristic based on chunk size distribution and content
        total_chars = sum(chunk.char_count for chunk in chunks)
        avg_size = total_chars / len(chunks)
        
        # Penalize very small or very large chunks
        size_score = 1.0
        if avg_size < 100:
            size_score = avg_size / 100
        elif avg_size > 1000:
            size_score = 1000 / avg_size
        
        # Penalize too many or too few chunks
        chunk_count_score = 1.0
        if len(chunks) < 3:
            chunk_count_score = len(chunks) / 3
        elif len(chunks) > 50:
            chunk_count_score = 50 / len(chunks)
        
        # Combine scores
        return min(1.0, (size_score + chunk_count_score) / 2)
    def get_total_document_count(self) -> int:
        """Get total number of documents in the system."""
        return self.db.query(Document).count()

    def get_total_chunk_count(self) -> int:
        """Get total number of chunks in the system."""
        return self.db.query(Chunk).count()

    def get_processing_queue_size(self) -> int:
        """Get number of documents currently being processed."""
        return (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.status.in_([
                ProcessingStatusEnum.PENDING,
                ProcessingStatusEnum.EXTRACTING,
                ProcessingStatusEnum.CHUNKING
            ]))
            .count()
        )