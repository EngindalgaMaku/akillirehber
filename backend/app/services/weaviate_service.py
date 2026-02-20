"""Weaviate vector database service for storing and searching embeddings."""

import logging
from typing import List, Optional
from dataclasses import dataclass
from urllib.parse import urlparse

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.collections.classes.grpc import HybridFusion
from weaviate.classes.init import Auth

from app.config import get_settings

logger = logging.getLogger(__name__)


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
        self._api_key = getattr(settings, 'weaviate_api_key', None) or None

    def _get_client(self) -> weaviate.WeaviateClient:
        """Get or create Weaviate client connection.
        
        Automatically detects local vs remote URLs and uses the
        appropriate connection method. Supports HTTPS with API key auth.
        Includes automatic reconnection on stale connections.
        
        IMPORTANT: gRPC port (50051) must be open on the remote server.
        The v4 Python client requires gRPC for all query operations.
        """
        if self._client is not None:
            # Check if existing connection is still alive
            try:
                self._client.is_ready()
            except Exception:
                logger.warning("Weaviate connection is stale, reconnecting...")
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

        if self._client is None:
            parsed = urlparse(self._url)
            is_https = parsed.scheme == "https"
            host = parsed.hostname or "localhost"
            
            if is_https or (host not in ("localhost", "127.0.0.1")):
                # Remote connection — gRPC required for queries
                grpc_port = 50051
                http_port = parsed.port or (443 if is_https else 8080)
                
                auth = Auth.api_key(self._api_key) if self._api_key else None
                
                logger.info(f"Connecting to remote Weaviate at {host} (https={is_https})")
                self._client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=http_port,
                    http_secure=is_https,
                    grpc_host=host,
                    grpc_port=grpc_port,
                    grpc_secure=False,  # gRPC port is plain TCP, TLS is handled by reverse proxy for HTTP only
                    auth_credentials=auth,
                    additional_config=weaviate.config.AdditionalConfig(
                        timeout=(30, 120),
                    ),
                    skip_init_checks=True,
                )
                logger.info("Connected to remote Weaviate (skip_init_checks=True)")
            else:
                # Local connection
                port = parsed.port or 8080
                self._client = weaviate.connect_to_local(
                    host=host,
                    port=port,
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

    def export_collection(self, course_id: int) -> List[dict]:
        """Export all objects from a course collection for backup.
        
        Args:
            course_id: Course ID
            
        Returns:
            List of objects with properties and vectors
        """
        client = self._get_client()
        collection_name = self._get_collection_name(course_id)
        
        if not client.collections.exists(collection_name):
            return []
        
        collection = client.collections.get(collection_name)
        
        # Get all objects with vectors
        objects = []
        for item in collection.iterator(include_vector=True):
            objects.append({
                "uuid": str(item.uuid),
                "properties": item.properties,
                "vector": item.vector.get("default") if item.vector else None
            })
        
        return objects
    
    def import_collection(self, course_id: int, objects: List[dict]) -> int:
        """Import objects into a course collection from backup.
        
        Args:
            course_id: Course ID
            objects: List of objects to import
            
        Returns:
            Number of objects imported
        """
        if not objects:
            return 0
            
        client = self._get_client()
        collection_name = self.ensure_collection(course_id)
        collection = client.collections.get(collection_name)
        
        # Insert objects one by one (REST-compatible, no gRPC needed)
        count = 0
        for obj in objects:
            try:
                collection.data.insert(
                    properties=obj["properties"],
                    vector=obj.get("vector")
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to import object: {e}")
        
        return count

    def close(self):
        """Close Weaviate client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure connection is closed."""
        self.close()
        return False

    def __del__(self):
        """Destructor - ensure connection is closed."""
        self.close()


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
        if not chunks:
            return []

        client = self._get_client()
        collection_name = self.ensure_collection(course_id)
        collection = client.collections.get(collection_name)

        logger.info(
            f"Storing {len(chunks)} chunks for document "
            f"{document_id} in collection {collection_name}"
        )

        uuids = []
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                try:
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
                except Exception as e:
                    logger.warning(f"Failed to insert chunk {chunk.chunk_id}: {e}")

        logger.info(
            f"store_chunks complete: {len(uuids)}/{len(chunks)} UUIDs "
            f"returned for document {document_id}"
        )
        return uuids


    def delete_by_chunk_id(self, course_id: int, chunk_id: int) -> int:
        """Delete a single vector by chunk_id.
        
        Args:
            course_id: Course ID
            chunk_id: Chunk ID (DB primary key)
            
        Returns:
            Number of deleted objects (0 or 1)
        """
        client = self._get_client()
        collection_name = self._get_collection_name(course_id)

        if not client.collections.exists(collection_name):
            return 0

        collection = client.collections.get(collection_name)
        
        result = collection.data.delete_many(
            where=Filter.by_property("chunk_id").equal(chunk_id)
        )
        
        return result.successful if hasattr(result, 'successful') else 0

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
            fusion_type=HybridFusion.RELATIVE_SCORE,
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
