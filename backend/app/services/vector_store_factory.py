"""Factory for creating vector store instances based on course settings.

This allows switching between Weaviate and ChromaDB per course.
"""

import logging
from typing import Optional
from sqlalchemy.orm import Session

from app.services.vector_store_interface import VectorStoreInterface
from app.services.weaviate_adapter import WeaviateAdapter

logger = logging.getLogger(__name__)


def get_vector_store_for_course(
    course_id: int,
    db: Optional[Session] = None
) -> VectorStoreInterface:
    """Get the appropriate vector store for a course.
    
    Args:
        course_id: Course ID
        db: Database session (optional, for checking course settings)
        
    Returns:
        VectorStoreInterface implementation (Weaviate or ChromaDB)
    """
    # Default to Weaviate
    vector_store_type = "weaviate"
    
    # Check course settings if db session provided
    if db is not None:
        try:
            from app.models.db_models import CourseSettings
            settings = db.query(CourseSettings).filter(
                CourseSettings.course_id == course_id
            ).first()
            
            if settings and hasattr(settings, 'vector_store'):
                vector_store_type = settings.vector_store or "weaviate"
        except Exception as e:
            logger.warning(
                f"Could not load vector store setting for course {course_id}: {e}. "
                f"Defaulting to Weaviate."
            )
    
    # Create appropriate vector store
    if vector_store_type == "chromadb":
        logger.warning(
            f"ChromaDB requested for course {course_id} but temporarily disabled. "
            f"Using Weaviate instead."
        )
        return WeaviateAdapter()
    else:
        logger.info(f"Using Weaviate for course {course_id}")
        return WeaviateAdapter()


def get_default_vector_store() -> VectorStoreInterface:
    """Get the default vector store (Weaviate).
    
    Returns:
        WeaviateAdapter instance
    """
    return WeaviateAdapter()
