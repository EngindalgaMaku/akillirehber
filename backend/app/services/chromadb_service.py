"""ChromaDB vector database service for storing and searching embeddings.

This is an EXPERIMENTAL alternative to Weaviate for performance comparison.
"""

import logging
from typing import List, Optional
import chromadb
from chromadb.config import Settings

from app.config import get_settings
from app.services.vector_store_interface import (
    VectorStoreInterface,
    SearchResult,
    ChunkWithEmbedding
)

logger = logging.getLogger(__name__)


class ChromaDBService(VectorStoreInterface):
    """Service for interacting with ChromaDB vector database.
    
    EXPERIMENTAL: This is for benchmarking against Weaviate.
    """

    COLLECTION_PREFIX = "course_"

    def __init__(self, url: str = None):
        """Initialize ChromaDB client connection.
        
        Args:
            url: ChromaDB server URL. Defaults to config value.
        """
        settings = get_settings()
        self._url = url or getattr(settings, 'chromadb_url', 'http://localhost:8081')
        self._client = None
        logger.info(f"ChromaDB service initialized with URL: {self._url}")

    def _get_client(self) -> chromadb.HttpClient:
        """Get or create ChromaDB client connection."""
        if self._client is None:
            # Parse URL to get host and port
            url_parts = self._url.replace("http://", "").replace("https://", "").split(":")
            host = url_parts[0]
            port = int(url_parts[1]) if len(url_parts) > 1 else 8000
            
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            logger.info(f"ChromaDB client connected to {host}:{port}")
        return self._client

    def _get_collection_name(self, course_id: int) -> str:
        """Get collection name for a course."""
        return f"{self.COLLECTION_PREFIX}{course_id}"

    def ensure_collection(self, course_id: int) -> str:
        """Create collection for course if not exists.
        
        Args:
            course_id: Course ID
            
        Returns:
            Collection name
        """
        client = self._get_client()
        collection_name = self._get_collection_name(course_id)

        try:
            # Try to get existing collection
            client.get_collection(name=collection_name)
            logger.info(f"ChromaDB collection '{collection_name}' already exists")
        except Exception:
            # Create new collection
            client.create_collection(
                name=collection_name,
                metadata={"course_id": course_id}
            )
            logger.info(f"ChromaDB collection '{collection_name}' created")

        return collection_name

    def close(self):
        """Close ChromaDB client connection."""
        if self._client is not None:
            self._client = None
            logger.info("ChromaDB client connection closed")

    def store_chunks(
        self,
        course_id: int,
        document_id: int,
        chunks: List[ChunkWithEmbedding]
    ) -> List[str]:
        """Store chunks with their embeddings in ChromaDB.
        
        Args:
            course_id: Course ID
            document_id: Document ID
            chunks: List of chunks with embeddings
            
        Returns:
            List of ChromaDB object IDs
        """
        client = self._get_client()
        collection_name = self.ensure_collection(course_id)
        collection = client.get_collection(name=collection_name)

        # Prepare data for ChromaDB
        ids = [f"chunk_{chunk.chunk_id}" for chunk in chunks]
        embeddings = [chunk.vector for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index
            }
            for chunk in chunks
        ]

        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(
            f"Stored {len(chunks)} chunks for document {document_id} "
            f"in ChromaDB collection '{collection_name}'"
        )

        return ids

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
        client = self._get_client()
        collection_name = self._get_collection_name(course_id)

        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            logger.warning(f"ChromaDB collection '{collection_name}' not found")
            return []

        # Query with embedding only
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=limit
        )

        return self._convert_results(results)

    def keyword_search(
        self,
        course_id: int,
        query: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """Perform keyword search using ChromaDB's where_document filter.
        
        Args:
            course_id: Course ID
            query: Search query text
            limit: Maximum results to return
            
        Returns:
            List of search results
        """
        client = self._get_client()
        collection_name = self._get_collection_name(course_id)

        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            logger.warning(f"ChromaDB collection '{collection_name}' not found")
            return []

        # Query with text only (ChromaDB will use its internal search)
        results = collection.query(
            query_texts=[query],
            n_results=limit
        )

        return self._convert_results(results)

    def hybrid_search(
        self,
        course_id: int,
        query: str,
        query_vector: List[float],
        alpha: float = 0.5,
        limit: int = 5
    ) -> List[SearchResult]:
        """Perform pure vector search (ChromaDB doesn't use hybrid for benchmark).
        
        For benchmark purposes, ChromaDB uses PURE VECTOR SEARCH to compare against
        Weaviate's hybrid search. This allows us to measure the value of hybrid search.
        
        Args:
            course_id: Course ID
            query: Search query text (ignored, only vector used)
            query_vector: Query embedding vector
            alpha: Ignored (ChromaDB uses pure vector)
            limit: Maximum results to return
            
        Returns:
            List of search results (pure vector search)
        """
        # ChromaDB uses pure vector search for benchmark
        logger.info(
            f"ChromaDB using PURE VECTOR search (alpha parameter ignored for benchmark)"
        )
        return self.vector_search(course_id, query_vector, limit)

    def delete_by_document(self, course_id: int, document_id: int) -> int:
        """Delete all vectors for a document.
        
        Args:
            course_id: Course ID
            document_id: Document ID
            
        Returns:
            Number of deleted objects
        """
        client = self._get_client()
        collection_name = self._get_collection_name(course_id)

        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            logger.warning(f"ChromaDB collection '{collection_name}' not found")
            return 0

        # Get all chunks for this document
        results = collection.get(
            where={"document_id": document_id}
        )
        
        if not results['ids']:
            return 0
        
        # Delete them
        collection.delete(ids=results['ids'])
        
        deleted_count = len(results['ids'])
        logger.info(
            f"Deleted {deleted_count} chunks for document {document_id} "
            f"from ChromaDB collection '{collection_name}'"
        )
        
        return deleted_count

    def delete_by_course(self, course_id: int) -> int:
        """Delete entire collection for a course.
        
        Args:
            course_id: Course ID
            
        Returns:
            1 if deleted, 0 if not found
        """
        client = self._get_client()
        collection_name = self._get_collection_name(course_id)

        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted ChromaDB collection '{collection_name}'")
            return 1
        except Exception as e:
            logger.warning(
                f"Failed to delete ChromaDB collection '{collection_name}': {e}"
            )
            return 0

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
        client = self._get_client()
        collection_name = self._get_collection_name(course_id)

        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            return 0

        # Get all chunks for this document
        results = collection.get(
            where={"document_id": document_id}
        )
        
        return len(results['ids']) if results['ids'] else 0

    def _convert_results(self, chroma_results: dict) -> List[SearchResult]:
        """Convert ChromaDB results to SearchResult format.
        
        Args:
            chroma_results: ChromaDB query results
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        # ChromaDB returns results in a specific format
        ids = chroma_results.get('ids', [[]])[0]
        documents = chroma_results.get('documents', [[]])[0]
        metadatas = chroma_results.get('metadatas', [[]])[0]
        distances = chroma_results.get('distances', [[]])[0]
        
        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            distance = distances[i] if i < len(distances) else 1.0
            
            # Convert distance to similarity score (lower distance = higher score)
            # ChromaDB uses L2 distance, convert to similarity
            score = 1.0 / (1.0 + distance)
            
            results.append(SearchResult(
                chunk_id=metadata.get('chunk_id', 0),
                document_id=metadata.get('document_id', 0),
                content=document,
                chunk_index=metadata.get('chunk_index', 0),
                score=score
            ))
        
        return results


# Singleton instance
_chromadb_service: Optional[ChromaDBService] = None


def get_chromadb_service() -> ChromaDBService:
    """Get singleton ChromaDBService instance."""
    global _chromadb_service
    if _chromadb_service is None:
        _chromadb_service = ChromaDBService()
    return _chromadb_service
