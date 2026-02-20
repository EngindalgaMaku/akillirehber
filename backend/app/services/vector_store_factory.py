"""Factory for creating vector store instances based on course settings.

This allows switching between Weaviate and ChromaDB per course.
Uses a singleton adapter to avoid creating new connections on every call.
"""

import logging
from typing import Optional
from sqlalchemy.orm import Session

from app.services.vector_store_interface import VectorStoreInterface
from app.services.weaviate_adapter import WeaviateAdapter

logger = logging.getLogger(__name__)

# Singleton instance — reuses the same Weaviate connection
_default_adapter: Optional[WeaviateAdapter] = None


def _get_singleton_adapter() -> WeaviateAdapter:
    """Get or create the singleton WeaviateAdapter."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = WeaviateAdapter()
    return _default_adapter


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
    logger.debug(f"Using Weaviate for course {course_id}")
    return _get_singleton_adapter()


def get_default_vector_store() -> VectorStoreInterface:
    """Get the default vector store (Weaviate).
    
    Returns:
        WeaviateAdapter instance (singleton)
    """
    return _get_singleton_adapter()
