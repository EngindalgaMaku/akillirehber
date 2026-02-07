"""Weaviate adapter implementing VectorStoreInterface.

This wraps the existing WeaviateService to conform to the interface.
"""

from typing import List
from app.services.vector_store_interface import (
    VectorStoreInterface,
    SearchResult,
    ChunkWithEmbedding
)
from app.services.weaviate_service import (
    WeaviateService,
    SearchResult as WeaviateSearchResult,
    ChunkWithEmbedding as WeaviateChunk
)


class WeaviateAdapter(VectorStoreInterface):
    """Adapter for WeaviateService to implement VectorStoreInterface."""

    def __init__(self, url: str = None):
        """Initialize Weaviate adapter.
        
        Args:
            url: Weaviate server URL
        """
        self._service = WeaviateService(url=url)

    def ensure_collection(self, course_id: int) -> str:
        """Create collection for course if not exists."""
        return self._service.ensure_collection(course_id)

    def store_chunks(
        self,
        course_id: int,
        document_id: int,
        chunks: List[ChunkWithEmbedding]
    ) -> List[str]:
        """Store chunks with their embeddings."""
        # Convert interface chunks to Weaviate chunks
        weaviate_chunks = [
            WeaviateChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                vector=chunk.vector
            )
            for chunk in chunks
        ]
        return self._service.store_chunks(course_id, document_id, weaviate_chunks)

    def vector_search(
        self,
        course_id: int,
        query_vector: List[float],
        limit: int = 5
    ) -> List[SearchResult]:
        """Perform vector similarity search."""
        results = self._service.vector_search(course_id, query_vector, limit)
        return self._convert_results(results)

    def keyword_search(
        self,
        course_id: int,
        query: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """Perform keyword search."""
        results = self._service.keyword_search(course_id, query, limit)
        return self._convert_results(results)

    def hybrid_search(
        self,
        course_id: int,
        query: str,
        query_vector: List[float],
        alpha: float = 0.5,
        limit: int = 5
    ) -> List[SearchResult]:
        """Perform hybrid search."""
        results = self._service.hybrid_search(
            course_id, query, query_vector, alpha, limit
        )
        return self._convert_results(results)

    def delete_by_chunk_id(self, course_id: int, chunk_id: int) -> int:
        """Delete a single vector by chunk_id."""
        return self._service.delete_by_chunk_id(course_id, chunk_id)

    def delete_by_document(self, course_id: int, document_id: int) -> int:
        """Delete all vectors for a document."""
        return self._service.delete_by_document(course_id, document_id)

    def delete_by_course(self, course_id: int) -> int:
        """Delete entire collection for a course."""
        return self._service.delete_by_course(course_id)

    def get_document_vector_count(
        self, course_id: int, document_id: int
    ) -> int:
        """Get count of vectors for a document."""
        return self._service.get_document_vector_count(course_id, document_id)

    def close(self):
        """Close database connection."""
        self._service.close()

    def _convert_results(
        self, weaviate_results: List[WeaviateSearchResult]
    ) -> List[SearchResult]:
        """Convert Weaviate results to interface results."""
        return [
            SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                chunk_index=r.chunk_index,
                score=r.score
            )
            for r in weaviate_results
        ]
