"""Abstract interface for vector store implementations.

This allows switching between different vector databases (Weaviate, ChromaDB, etc.)
without changing the application code.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Unified search result format across all vector stores."""
    chunk_id: int
    document_id: int
    content: str
    chunk_index: int
    score: float


@dataclass
class ChunkWithEmbedding:
    """Chunk data with embedding vector."""
    chunk_id: int
    document_id: int
    content: str
    chunk_index: int
    vector: List[float]


class VectorStoreInterface(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def ensure_collection(self, course_id: int) -> str:
        """Create collection for course if not exists.
        
        Args:
            course_id: Course ID
            
        Returns:
            Collection name
        """
        pass

    @abstractmethod
    def store_chunks(
        self,
        course_id: int,
        document_id: int,
        chunks: List[ChunkWithEmbedding]
    ) -> List[str]:
        """Store chunks with their embeddings.
        
        Args:
            course_id: Course ID
            document_id: Document ID
            chunks: List of chunks with embeddings
            
        Returns:
            List of object IDs
        """
        pass

    @abstractmethod
    def vector_search(
        self,
        course_id: int,
        query_vector: List[float],
        limit: int = 5
    ) -> List[SearchResult]:
        """Perform vector similarity search.
        
        Args:
            course_id: Course ID
            query_vector: Query embedding vector
            limit: Maximum results to return
            
        Returns:
            List of search results
        """
        pass

    @abstractmethod
    def keyword_search(
        self,
        course_id: int,
        query: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """Perform keyword search.
        
        Args:
            course_id: Course ID
            query: Search query text
            limit: Maximum results to return
            
        Returns:
            List of search results
        """
        pass

    @abstractmethod
    def hybrid_search(
        self,
        course_id: int,
        query: str,
        query_vector: List[float],
        alpha: float = 0.5,
        limit: int = 5
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword.
        
        Args:
            course_id: Course ID
            query: Search query text
            query_vector: Query embedding vector
            alpha: Balance between vector (1) and keyword (0) search
            limit: Maximum results to return
            
        Returns:
            List of search results
        """
        pass

    @abstractmethod
    def delete_by_chunk_id(self, course_id: int, chunk_id: int) -> int:
        """Delete a single vector by chunk_id.
        
        Args:
            course_id: Course ID
            chunk_id: Chunk ID
            
        Returns:
            Number of deleted objects
        """
        pass

    @abstractmethod
    def delete_by_document(self, course_id: int, document_id: int) -> int:
        """Delete all vectors for a document.
        
        Args:
            course_id: Course ID
            document_id: Document ID
            
        Returns:
            Number of deleted objects
        """
        pass

    @abstractmethod
    def delete_by_course(self, course_id: int) -> int:
        """Delete entire collection for a course.
        
        Args:
            course_id: Course ID
            
        Returns:
            Number of deleted objects
        """
        pass

    @abstractmethod
    def get_document_vector_count(
        self, course_id: int, document_id: int
    ) -> int:
        """Get count of vectors for a document.
        
        Args:
            course_id: Course ID
            document_id: Document ID
            
        Returns:
            Number of vectors
        """
        pass

    @abstractmethod
    def close(self):
        """Close database connection."""
        pass
