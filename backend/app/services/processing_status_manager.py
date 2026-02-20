"""Processing Status Manager for document processing pipeline."""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.db_models import (
    ProcessingStatus,
    ProcessingStatusEnum,
    Document,
)
from app.models.schemas import ProcessingStatusResponse

logger = logging.getLogger(__name__)


class ProcessingStatusManager:
    """Manages processing status tracking and transitions for documents."""

    def __init__(self, db: Session):
        """Initialize the ProcessingStatusManager.
        
        Args:
            db: Database session
        """
        self.db = db

    def create_status(
        self,
        document_id: int,
        status: ProcessingStatusEnum = ProcessingStatusEnum.PENDING
    ) -> ProcessingStatus:
        """Create initial processing status for a document.
        
        Args:
            document_id: ID of the document
            status: Initial status (default: PENDING)
            
        Returns:
            Created ProcessingStatus instance
            
        Raises:
            ValueError: If document doesn't exist or status already exists
        """
        # Verify document exists
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise ValueError(f"Document with ID {document_id} not found")
            
        # Check if status already exists
        existing_status = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.document_id == document_id)
            .first()
        )
        if existing_status:
            raise ValueError(f"Processing status already exists for document {document_id}")
            
        # Create new status
        processing_status = ProcessingStatus(
            document_id=document_id,
            status=status.value,
            started_at=datetime.utcnow()
        )
        
        self.db.add(processing_status)
        self.db.commit()
        self.db.refresh(processing_status)
        
        logger.info(f"Created processing status for document {document_id} with status {status.value}")
        return processing_status

    def update_status(
        self,
        document_id: int,
        new_status: ProcessingStatusEnum,
        error_message: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> ProcessingStatus:
        """Update processing status for a document.
        
        Args:
            document_id: ID of the document
            new_status: New status to set
            error_message: Optional error message for ERROR status
            error_details: Optional detailed error information
            
        Returns:
            Updated ProcessingStatus instance
            
        Raises:
            ValueError: If document or status not found, or invalid transition
        """
        # Get existing status
        processing_status = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.document_id == document_id)
            .first()
        )
        
        if not processing_status:
            raise ValueError(f"No processing status found for document {document_id}")
            
        # Validate status transition
        current_status = ProcessingStatusEnum(processing_status.status)
        if not self._is_valid_transition(current_status, new_status):
            raise ValueError(
                f"Invalid status transition from {current_status.value} to {new_status.value}"
            )
            
        # Update status
        processing_status.status = new_status.value
        processing_status.updated_at = datetime.utcnow()
        
        # Set completion time for terminal states
        if new_status in [ProcessingStatusEnum.COMPLETED, ProcessingStatusEnum.ERROR]:
            processing_status.completed_at = datetime.utcnow()
            
            # Calculate processing duration
            if processing_status.started_at:
                duration = (processing_status.completed_at - processing_status.started_at).total_seconds()
                processing_status.processing_duration = duration
                
        # Handle error status
        if new_status == ProcessingStatusEnum.ERROR:
            processing_status.error_message = error_message
            processing_status.error_details = error_details
            
        # Handle retry status
        if new_status == ProcessingStatusEnum.RETRYING:
            processing_status.retry_count += 1
            processing_status.last_retry_at = datetime.utcnow()
            # Clear previous error information
            processing_status.error_message = None
            processing_status.error_details = None
            
        self.db.commit()
        self.db.refresh(processing_status)
        
        logger.info(
            f"Updated processing status for document {document_id} "
            f"from {current_status.value} to {new_status.value}"
        )
        
        return processing_status

    def get_status(self, document_id: int) -> Optional[ProcessingStatusResponse]:
        """Get current processing status for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            ProcessingStatusResponse or None if not found
        """
        processing_status = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.document_id == document_id)
            .first()
        )
        
        if not processing_status:
            return None
            
        return ProcessingStatusResponse.model_validate(processing_status)

    def get_status_history(self, document_id: int) -> List[ProcessingStatusResponse]:
        """Get processing status history for a document.
        
        Note: Currently returns single status as history tracking is not implemented.
        This method is prepared for future enhancement with status history table.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of ProcessingStatusResponse objects
        """
        current_status = self.get_status(document_id)
        if current_status:
            return [current_status]
        return []

    def mark_for_retry(
        self,
        document_id: int,
        retry_config: Optional[Dict[str, Any]] = None
    ) -> ProcessingStatus:
        """Mark a document for retry processing.
        
        Args:
            document_id: ID of the document
            retry_config: Optional retry configuration
            
        Returns:
            Updated ProcessingStatus instance
            
        Raises:
            ValueError: If document not in ERROR state
        """
        processing_status = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.document_id == document_id)
            .first()
        )
        
        if not processing_status:
            raise ValueError(f"No processing status found for document {document_id}")
            
        current_status = ProcessingStatusEnum(processing_status.status)
        if current_status != ProcessingStatusEnum.ERROR:
            raise ValueError(
                f"Cannot retry document in {current_status.value} status. "
                "Only ERROR status documents can be retried."
            )
            
        # Update to RETRYING status
        return self.update_status(document_id, ProcessingStatusEnum.RETRYING)

    def get_documents_by_status(
        self,
        status: ProcessingStatusEnum,
        limit: Optional[int] = None
    ) -> List[ProcessingStatusResponse]:
        """Get documents by processing status.
        
        Args:
            status: Status to filter by
            limit: Optional limit on number of results
            
        Returns:
            List of ProcessingStatusResponse objects
        """
        query = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.status == status.value)
            .order_by(desc(ProcessingStatus.updated_at))
        )
        
        if limit:
            query = query.limit(limit)
            
        statuses = query.all()
        return [ProcessingStatusResponse.model_validate(status) for status in statuses]

    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing metrics across all documents.
        
        Returns:
            Dictionary containing processing metrics
        """
        from sqlalchemy import func
        
        # Get status distribution
        status_counts = (
            self.db.query(ProcessingStatus.status, func.count(ProcessingStatus.id))
            .group_by(ProcessingStatus.status)
            .all()
        )
        
        status_distribution = {status.value: 0 for status in ProcessingStatusEnum}
        for status, count in status_counts:
            status_distribution[status] = count
            
        # Get average processing time
        avg_processing_time = (
            self.db.query(func.avg(ProcessingStatus.processing_duration))
            .filter(ProcessingStatus.processing_duration.isnot(None))
            .scalar()
        )
        
        # Get retry statistics
        retry_stats = (
            self.db.query(
                func.count(ProcessingStatus.id).label('total_retries'),
                func.avg(ProcessingStatus.retry_count).label('avg_retry_count'),
                func.max(ProcessingStatus.retry_count).label('max_retry_count')
            )
            .filter(ProcessingStatus.retry_count > 0)
            .first()
        )
        
        return {
            'status_distribution': status_distribution,
            'average_processing_time': avg_processing_time or 0.0,
            'retry_statistics': {
                'total_documents_with_retries': retry_stats.total_retries or 0,
                'average_retry_count': retry_stats.avg_retry_count or 0.0,
                'max_retry_count': retry_stats.max_retry_count or 0
            }
        }

    def _is_valid_transition(
        self,
        current_status: ProcessingStatusEnum,
        new_status: ProcessingStatusEnum
    ) -> bool:
        """Validate if a status transition is allowed.
        
        Args:
            current_status: Current processing status
            new_status: Desired new status
            
        Returns:
            True if transition is valid, False otherwise
        """
        # Define valid transitions
        valid_transitions = {
            ProcessingStatusEnum.PENDING: [
                ProcessingStatusEnum.EXTRACTING,
                ProcessingStatusEnum.ERROR
            ],
            ProcessingStatusEnum.EXTRACTING: [
                ProcessingStatusEnum.CHUNKING,
                ProcessingStatusEnum.COMPLETED,
                ProcessingStatusEnum.ERROR
            ],
            ProcessingStatusEnum.CHUNKING: [
                ProcessingStatusEnum.EMBEDDING,
                ProcessingStatusEnum.COMPLETED,
                ProcessingStatusEnum.ERROR
            ],
            ProcessingStatusEnum.EMBEDDING: [
                ProcessingStatusEnum.COMPLETED,
                ProcessingStatusEnum.ERROR
            ],
            ProcessingStatusEnum.ERROR: [
                ProcessingStatusEnum.RETRYING
            ],
            ProcessingStatusEnum.RETRYING: [
                ProcessingStatusEnum.EXTRACTING,
                ProcessingStatusEnum.CHUNKING,
                ProcessingStatusEnum.EMBEDDING,
                ProcessingStatusEnum.ERROR
            ],
            ProcessingStatusEnum.COMPLETED: []  # Terminal state
        }
        
        allowed_transitions = valid_transitions.get(current_status, [])
        return new_status in allowed_transitions

    def reset_status(self, document_id: int) -> ProcessingStatus:
        """Reset processing status to PENDING for a document.
        
        This is useful for completely restarting the processing pipeline.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Updated ProcessingStatus instance
            
        Raises:
            ValueError: If document or status not found
        """
        processing_status = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.document_id == document_id)
            .first()
        )
        
        if not processing_status:
            raise ValueError(f"No processing status found for document {document_id}")
            
        # Reset all fields
        processing_status.status = ProcessingStatusEnum.PENDING.value
        processing_status.started_at = datetime.utcnow()
        processing_status.completed_at = None
        processing_status.error_message = None
        processing_status.error_details = None
        processing_status.processing_duration = None
        processing_status.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(processing_status)
        
        logger.info(f"Reset processing status for document {document_id} to PENDING")
        return processing_status

    def cleanup_old_statuses(self, days_old: int = 30) -> int:
        """Clean up old completed processing statuses.
        
        Args:
            days_old: Number of days old to consider for cleanup
            
        Returns:
            Number of statuses cleaned up
        """
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Only clean up COMPLETED statuses older than cutoff
        deleted_count = (
            self.db.query(ProcessingStatus)
            .filter(ProcessingStatus.status == ProcessingStatusEnum.COMPLETED.value)
            .filter(ProcessingStatus.completed_at < cutoff_date)
            .delete()
        )
        
        self.db.commit()
        
        logger.info(f"Cleaned up {deleted_count} old processing statuses")
        return deleted_count