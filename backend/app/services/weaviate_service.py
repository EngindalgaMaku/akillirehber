"""Weaviate vector database service for storing and searching embeddings."""

from typing import List, Optional
from dataclasses import dataclass

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery, Filter

from app.config import get_settings


@dataclass
class SearchResult:
    """Search result from Weaviate."""
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


class WeaviateService:
    """Service for interacting with Weaviate vector database.
    
    Handles:
    - Collection management per course
    - Storing chunks with embeddings
    - Vector, keyword (BM25), and hybrid search
    - Deletion by document or course
    """

    COLLECTION_PREFIX = "Course_"

    def __init__(self, url: str = None):
        """Initialize Weaviate client connection.
        
        Args:
            url: Weaviate server URL. Defaults to config value.
        """
        settings = get_settings()
        self._url = url or settings.weaviate_url
        self._client = None

    def _get_client(self) -> weaviate.WeaviateClient:
        """Get or create Weaviate client connection."""
        if self._client is None:
            self._client = weaviate.connect_to_local(
                host=self._url.replace("http://", "").split(":")[0],
                port=int(self._url.split(":")[-1]) if ":" in self._url else 8080
            )
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

        if not client.collections.exists(collection_name):
            client.collections.create(
                name=collection_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(
                        name="chunk_id",
                        data_type=DataType.INT,
                        description="PostgreSQL chunk ID"
                    ),
                    Property(
                        name="document_id",
                        data_type=DataType.INT,
                        description="PostgreSQL document ID"
                    ),
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        description="Chunk text content"
                    ),
                    Property(
                        name="chunk_index",
                        data_type=DataType.INT,
                        description="Chunk index within document"
                    ),
                ]
            )

        return collection_name

    def close(self):
        """Close Weaviate client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None


    # ==================== CRUD Operations ====================

    def store_chunks(
        self,
        course_id: int,
        document_id: int,
        chunks: List[ChunkWithEmbedding]
    ) -> List[str]:
        """Store chunks with their embeddings in Weaviate.
        
        Args:
            course_id: Course ID
            document_id: Document ID
            chunks: List of chunks with embeddings
            
        Returns:
            List of Weaviate object UUIDs
        """
        client = self._get_client()
        collection_name = self.ensure_collection(course_id)
        collection = client.collections.get(collection_name)

        uuids = []
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                uuid = batch.add_object(
                    properties={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                    },
                    vector=chunk.vector
                )
                uuids.append(str(uuid))

        return uuids

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

        if not client.collections.exists(collection_name):
            return 0

        collection = client.collections.get(collection_name)
        
        result = collection.data.delete_many(
            where=Filter.by_property("document_id").equal(document_id)
        )
        
        return result.successful if hasattr(result, 'successful') else 0

    def delete_by_course(self, course_id: int) -> int:
        """Delete entire collection for a course.
        
        Args:
            course_id: Course ID
            
        Returns:
            1 if deleted, 0 if not found
        """
        client = self._get_client()
        collection_name = self._get_collection_name(course_id)

        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
            return 1
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

        if not client.collections.exists(collection_name):
            return 0

        collection = client.collections.get(collection_name)
        
        result = collection.aggregate.over_all(
            filters=Filter.by_property("document_id").equal(document_id),
            total_count=True
        )
        
        return result.total_count if result.total_count else 0

    # ==================== Search Operations ====================

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

        if not client.collections.exists(collection_name):
            return []

        collection = client.collections.get(collection_name)
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )

        results = []
        for obj in response.objects:
            score = 1 - obj.metadata.distance if obj.metadata.distance else 0
            results.append(SearchResult(
                chunk_id=obj.properties.get("chunk_id", 0),
                document_id=obj.properties.get("document_id", 0),
                content=obj.properties.get("content", ""),
                chunk_index=obj.properties.get("chunk_index", 0),
                score=score
            ))

        return results

    def keyword_search(
        self,
        course_id: int,
        query: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """Perform BM25 keyword search.
        
        Args:
            course_id: Course ID
            query: Search query text
            limit: Maximum results to return
            
        Returns:
            List of search results
        """
        client = self._get_client()
        collection_name = self._get_collection_name(course_id)

        if not client.collections.exists(collection_name):
            return []

        collection = client.collections.get(collection_name)
        
        response = collection.query.bm25(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(score=True)
        )

        results = []
        for obj in response.objects:
            score = obj.metadata.score if obj.metadata.score else 0
            results.append(SearchResult(
                chunk_id=obj.properties.get("chunk_id", 0),
                document_id=obj.properties.get("document_id", 0),
                content=obj.properties.get("content", ""),
                chunk_index=obj.properties.get("chunk_index", 0),
                score=score
            ))

        return results

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
        client = self._get_client()
        collection_name = self._get_collection_name(course_id)

        if not client.collections.exists(collection_name):
            return []

        collection = client.collections.get(collection_name)
        
        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=limit,
            return_metadata=MetadataQuery(score=True)
        )

        results = []
        for obj in response.objects:
            score = obj.metadata.score if obj.metadata.score else 0
            results.append(SearchResult(
                chunk_id=obj.properties.get("chunk_id", 0),
                document_id=obj.properties.get("document_id", 0),
                content=obj.properties.get("content", ""),
                chunk_index=obj.properties.get("chunk_index", 0),
                score=score
            ))

        return results


# Singleton instance
_weaviate_service: Optional[WeaviateService] = None


def get_weaviate_service() -> WeaviateService:
    """Get singleton WeaviateService instance."""
    global _weaviate_service
    if _weaviate_service is None:
        _weaviate_service = WeaviateService()
    return _weaviate_service
