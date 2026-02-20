"""Chat integration validation service for ensuring reliable chat functionality."""

import logging
from datetime import datetime, timezone
from typing import Dict, List
from sqlalchemy.orm import Session

from app.models.db_models import Document, Chunk, Course
from app.services.weaviate_service import get_weaviate_service, SearchResult
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class ChatValidationService:
    """Service for validating chat integration and chunk availability."""

    def __init__(self, db: Session):
        self.db = db
        self.weaviate_service = get_weaviate_service()
        self.embedding_service = get_embedding_service()

    def validate_chunk_availability(self, course_id: int) -> Dict[str, any]:
        """Validate that chunks are available for chat in a course.

        Args:
            course_id: Course ID to validate

        Returns:
            Dictionary with validation results
        """
        # Debug: Check Weaviate collection directly
        collection_name = self.weaviate_service._get_collection_name(course_id)
        client = self.weaviate_service._get_client()
        
        print(f"=== Weaviate Debug for Course {course_id} ===")
        print(f"Collection name: {collection_name}")
        
        if client.collections.exists(collection_name):
            print("Collection exists: YES")
            
            # Get total count in collection
            collection = client.collections.get(collection_name)
            total_count = collection.aggregate.over_all(total_count=True)
            print(f"Total vectors in collection: {total_count.total_count}")
            
            # Try to get a sample vector
            try:
                sample = collection.query.fetch_objects(limit=1)
                if sample.objects:
                    obj = sample.objects[0]
                    print(f"Sample vector found - ID: {obj.uuid}")
                    print(f"Sample vector properties: {list(obj.properties.keys())}")
                    if 'document_id' in obj.properties:
                        print(f"Sample document_id: {obj.properties['document_id']}")
                else:
                    print("No objects found in collection")
            except Exception as e:
                print(f"Error fetching sample: {e}")
        else:
            print("Collection exists: NO")
        
        print("=== End Weaviate Debug ===")
        
        # Check if course exists
        course = self.db.query(Course).filter(Course.id == course_id).first()
        if not course:
            return {
                "valid": False,
                "error": "Course not found",
                "details": {"course_id": course_id}
            }

        # Get all processed documents in the course
        processed_docs = (
            self.db.query(Document)
            .filter(Document.course_id == course_id)
            .filter(Document.is_processed.is_(True))
            .all()
        )

        if not processed_docs:
            return {
                "valid": False,
                "error": "No processed documents found in course",
                "details": {
                    "course_id": course_id,
                    "total_documents": self.db.query(Document)
                    .filter(Document.course_id == course_id).count()
                }
            }

        # Check chunks for each document
        validation_results = []
        total_chunks = 0
        documents_with_chunks = 0
        documents_with_vectors = 0

        for doc in processed_docs:
            doc_validation = self._validate_document_chunks(doc)
            validation_results.append(doc_validation)

            if doc_validation["chunk_count"] > 0:
                documents_with_chunks += 1
                total_chunks += doc_validation["chunk_count"]

            if doc_validation["vector_count"] > 0:
                documents_with_vectors += 1

        # Overall validation
        is_valid = (
            documents_with_chunks > 0 and
            documents_with_vectors > 0 and
            total_chunks > 0
        )

        return {
            "valid": is_valid,
            "summary": {
                "total_documents": len(processed_docs),
                "documents_with_chunks": documents_with_chunks,
                "documents_with_vectors": documents_with_vectors,
                "total_chunks": total_chunks,
                "ready_for_chat": is_valid
            },
            "document_details": validation_results,
            "recommendations": self._generate_availability_recommendations(
                len(processed_docs), documents_with_chunks,
                documents_with_vectors, total_chunks
            )
        }

    def validate_chunk_data_consistency(self, course_id: int) -> Dict[str, any]:
        """Validate consistency between database chunks and vector store.

        Args:
            course_id: Course ID to validate

        Returns:
            Dictionary with consistency validation results
        """
        # Get all chunks in the course
        chunks = (
            self.db.query(Chunk)
            .join(Document)
            .filter(Document.course_id == course_id)
            .all()
        )

        if not chunks:
            return {
                "consistent": False,
                "error": "No chunks found in course",
                "details": {"course_id": course_id}
            }

        # Group chunks by document
        doc_chunks = {}
        for chunk in chunks:
            if chunk.document_id not in doc_chunks:
                doc_chunks[chunk.document_id] = []
            doc_chunks[chunk.document_id].append(chunk)

        consistency_results = []
        total_db_chunks = len(chunks)
        total_vector_chunks = 0
        consistent_documents = 0

        for doc_id, doc_chunks_list in doc_chunks.items():
            # Get vector count from Weaviate
            vector_count = self.weaviate_service.get_document_vector_count(
                course_id, doc_id
            )
            total_vector_chunks += vector_count

            db_chunk_count = len(doc_chunks_list)
            is_consistent = db_chunk_count == vector_count

            if is_consistent:
                consistent_documents += 1

            consistency_results.append({
                "document_id": doc_id,
                "db_chunk_count": db_chunk_count,
                "vector_count": vector_count,
                "consistent": is_consistent,
                "difference": abs(db_chunk_count - vector_count)
            })

        overall_consistent = (
            total_db_chunks == total_vector_chunks and
            consistent_documents == len(doc_chunks)
        )

        return {
            "consistent": overall_consistent,
            "summary": {
                "total_documents": len(doc_chunks),
                "consistent_documents": consistent_documents,
                "total_db_chunks": total_db_chunks,
                "total_vector_chunks": total_vector_chunks,
                "consistency_rate": (consistent_documents / len(doc_chunks) * 100)
                                   if doc_chunks else 0
            },
            "document_details": consistency_results,
            "recommendations": self._generate_consistency_recommendations(
                overall_consistent, total_db_chunks, total_vector_chunks,
                consistent_documents, len(doc_chunks)
            )
        }

    def test_chat_search_functionality(self, course_id: int,
                                       test_query: str = "test") -> Dict[str, any]:
        """Test chat search functionality with a sample query.

        Args:
            course_id: Course ID to test
            test_query: Query to test with

        Returns:
            Dictionary with search test results
        """
        try:
            # Get query embedding
            query_vector = self.embedding_service.get_embedding(test_query, input_type="query")

            if not query_vector:
                return {
                    "functional": False,
                    "error": "Failed to generate query embedding",
                    "details": {"query": test_query}
                }

            # Test different search types
            search_results = {}

            # Vector search
            try:
                vector_results = self.weaviate_service.vector_search(
                    course_id=course_id,
                    query_vector=query_vector,
                    limit=5
                )
                search_results["vector"] = {
                    "success": True,
                    "result_count": len(vector_results),
                    "results": vector_results[:2]  # Sample results
                }
            except Exception as e:
                search_results["vector"] = {
                    "success": False,
                    "error": str(e)
                }

            # Keyword search
            try:
                keyword_results = self.weaviate_service.keyword_search(
                    course_id=course_id,
                    query=test_query,
                    limit=5
                )
                search_results["keyword"] = {
                    "success": True,
                    "result_count": len(keyword_results),
                    "results": keyword_results[:2]  # Sample results
                }
            except Exception as e:
                search_results["keyword"] = {
                    "success": False,
                    "error": str(e)
                }

            # Hybrid search
            try:
                hybrid_results = self.weaviate_service.hybrid_search(
                    course_id=course_id,
                    query=test_query,
                    query_vector=query_vector,
                    alpha=0.5,
                    limit=5
                )
                search_results["hybrid"] = {
                    "success": True,
                    "result_count": len(hybrid_results),
                    "results": hybrid_results[:2]  # Sample results
                }
            except Exception as e:
                search_results["hybrid"] = {
                    "success": False,
                    "error": str(e)
                }

            # Determine overall functionality
            successful_searches = sum(1 for result in search_results.values()
                                      if result.get("success", False))
            is_functional = successful_searches > 0

            return {
                "functional": is_functional,
                "test_query": test_query,
                "search_results": search_results,
                "successful_search_types": successful_searches,
                "recommendations": self._generate_search_recommendations(
                    search_results, successful_searches
                )
            }

        except Exception as e:
            logger.error("Chat search test failed: %s", e)
            return {
                "functional": False,
                "error": f"Search test failed: {str(e)}",
                "details": {"query": test_query, "course_id": course_id}
            }

    def validate_source_attribution(self, course_id: int,
                                    search_results: List[SearchResult]) -> Dict[str, any]:
        """Validate that search results can be properly attributed to sources.

        Args:
            course_id: Course ID
            search_results: List of search results to validate

        Returns:
            Dictionary with attribution validation results
        """
        if not search_results:
            return {
                "valid": True,
                "message": "No results to validate",
                "details": []
            }

        attribution_results = []
        valid_attributions = 0

        for result in search_results:
            # Check if document exists and is accessible
            document = (
                self.db.query(Document)
                .filter(Document.id == result.document_id)
                .filter(Document.course_id == course_id)
                .first()
            )

            if not document:
                attribution_results.append({
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "valid": False,
                    "error": "Document not found or not in course"
                })
                continue

            # Check if chunk exists
            chunk = (
                self.db.query(Chunk)
                .filter(Chunk.id == result.chunk_id)
                .filter(Chunk.document_id == result.document_id)
                .first()
            )

            if not chunk:
                attribution_results.append({
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "valid": False,
                    "error": "Chunk not found in database"
                })
                continue

            # Validate content consistency
            content_matches = chunk.content.strip() == result.content.strip()

            attribution_results.append({
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "document_name": document.original_filename,
                "chunk_index": result.chunk_index,
                "valid": content_matches,
                "content_matches": content_matches,
                "error": None if content_matches else "Content mismatch between DB and vector store"
            })

            if content_matches:
                valid_attributions += 1

        attribution_rate = (valid_attributions / len(search_results) * 100) if search_results else 0

        return {
            "valid": attribution_rate > 90,  # 90% threshold for validity
            "attribution_rate": attribution_rate,
            "valid_attributions": valid_attributions,
            "total_results": len(search_results),
            "details": attribution_results,
            "recommendations": self._generate_attribution_recommendations(
                attribution_rate, valid_attributions, len(search_results)
            )
        }

    def get_chat_integration_diagnostics(self, course_id: int) -> Dict[str, any]:
        """Get comprehensive chat integration diagnostics for a course.

        Args:
            course_id: Course ID to diagnose

        Returns:
            Dictionary with comprehensive diagnostics
        """
        diagnostics = {
            "course_id": course_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "unknown"
        }

        # Validate chunk availability
        chunk_validation = self.validate_chunk_availability(course_id)
        diagnostics["chunk_availability"] = chunk_validation

        # Validate data consistency
        consistency_validation = self.validate_chunk_data_consistency(course_id)
        diagnostics["data_consistency"] = consistency_validation

        # Test search functionality
        search_test = self.test_chat_search_functionality(course_id)
        diagnostics["search_functionality"] = search_test

        # Test source attribution with sample results
        if search_test.get("functional", False):
            # Get sample results for attribution testing
            sample_results = []
            for results in search_test.get("search_results", {}).values():
                if results.get("success", False) and results.get("results"):
                    sample_results.extend(results["results"])

            attribution_validation = self.validate_source_attribution(
                course_id, sample_results[:5]  # Test with up to 5 results
            )
            diagnostics["source_attribution"] = attribution_validation
        else:
            diagnostics["source_attribution"] = {
                "valid": False,
                "error": "Cannot test attribution - search functionality not working"
            }

        # Determine overall status
        all_validations = [
            chunk_validation.get("valid", False),
            consistency_validation.get("consistent", False),
            search_test.get("functional", False),
            diagnostics["source_attribution"].get("valid", False)
        ]

        if all(all_validations):
            diagnostics["overall_status"] = "healthy"
        elif any(all_validations):
            diagnostics["overall_status"] = "warning"
        else:
            diagnostics["overall_status"] = "error"

        # Generate overall recommendations
        diagnostics["recommendations"] = self._generate_overall_recommendations(
            chunk_validation, consistency_validation, search_test,
            diagnostics["source_attribution"]
        )

        return diagnostics

    def _validate_document_chunks(self, document: Document) -> Dict[str, any]:
        """Validate chunks for a single document."""
        # Get database chunks
        chunks = (
            self.db.query(Chunk)
            .filter(Chunk.document_id == document.id)
            .all()
        )

        # Get vector count from Weaviate
        vector_count = self.weaviate_service.get_document_vector_count(
            document.course_id, document.id
        )
        
        # Debug log for troubleshooting
        print(f"Document {document.id} validation:")
        print(f"  - Course ID: {document.course_id}")
        print(f"  - Chunks in DB: {len(chunks)}")
        print(f"  - Vectors in Weaviate: {vector_count}")
        print(f"  - Collection exists: {self.weaviate_service._get_client().collections.exists(self.weaviate_service._get_collection_name(document.course_id))}")

        return {
            "document_id": document.id,
            "document_name": document.original_filename,
            "is_processed": document.is_processed,
            "chunk_count": len(chunks),
            "vector_count": vector_count,
            "chunks_match_vectors": len(chunks) == vector_count,
            "ready_for_chat": len(chunks) > 0 and vector_count > 0
        }

    def _generate_availability_recommendations(self, total_docs: int,
                                               docs_with_chunks: int,
                                               docs_with_vectors: int,
                                               total_chunks: int) -> List[str]:
        """Generate recommendations for chunk availability issues."""
        recommendations = []

        if total_docs == 0:
            recommendations.append("No documents found in course. Upload and process documents first.")
        elif docs_with_chunks == 0:
            recommendations.append("No documents have chunks. Process documents to generate chunks.")
        elif docs_with_vectors == 0:
            recommendations.append("No documents have vector embeddings. Run embedding process.")
        elif docs_with_chunks < total_docs:
            recommendations.append(f"Only {docs_with_chunks}/{total_docs} documents have chunks. Process remaining documents.")
        elif docs_with_vectors < docs_with_chunks:
            recommendations.append(f"Only {docs_with_vectors}/{docs_with_chunks} documents have embeddings. Run embedding process.")

        if total_chunks < 10:
            recommendations.append("Very few chunks available. Consider processing more documents or adjusting chunk size.")

        return recommendations

    def _generate_consistency_recommendations(self, is_consistent: bool,
                                              db_chunks: int, vector_chunks: int,
                                              consistent_docs: int, total_docs: int) -> List[str]:
        """Generate recommendations for consistency issues."""
        recommendations = []

        if not is_consistent:
            if db_chunks > vector_chunks:
                recommendations.append("Database has more chunks than vector store. Re-run embedding process.")
            elif vector_chunks > db_chunks:
                recommendations.append("Vector store has more chunks than database. Clean up vector store or re-process documents.")

        if consistent_docs < total_docs:
            inconsistent_rate = ((total_docs - consistent_docs) / total_docs * 100)
            recommendations.append(f"{inconsistent_rate:.1f}% of documents have inconsistent chunk counts. Review and re-process affected documents.")

        return recommendations

    def _generate_search_recommendations(self, search_results: Dict,
                                         successful_searches: int) -> List[str]:
        """Generate recommendations for search functionality issues."""
        recommendations = []

        if successful_searches == 0:
            recommendations.append("All search types failed. Check Weaviate connection and embedding service.")
        elif successful_searches < 3:
            failed_types = [search_type for search_type, result in search_results.items()
                            if not result.get("success", False)]
            recommendations.append(f"Some search types failed: {', '.join(failed_types)}. Check configuration and services.")

        # Check for empty results
        for search_type, result in search_results.items():
            if result.get("success", False) and result.get("result_count", 0) == 0:
                recommendations.append(f"{search_type.title()} search returns no results. Check if documents are properly embedded.")

        return recommendations

    def _generate_attribution_recommendations(self, attribution_rate: float,
                                              valid_attributions: int,
                                              total_results: int) -> List[str]:
        """Generate recommendations for attribution issues."""
        recommendations = []

        if attribution_rate < 50:
            recommendations.append("Low attribution rate. Check data consistency between database and vector store.")
        elif attribution_rate < 90:
            recommendations.append("Some attribution issues detected. Review chunk synchronization process.")

        if valid_attributions == 0 and total_results > 0:
            recommendations.append("No valid attributions found. Check if search results reference correct documents and chunks.")

        return recommendations

    def _generate_overall_recommendations(self, chunk_validation: Dict,
                                          consistency_validation: Dict,
                                          search_test: Dict,
                                          attribution_validation: Dict) -> List[str]:
        """Generate overall recommendations for chat integration."""
        recommendations = []

        # Priority order: chunks -> consistency -> search -> attribution
        if not chunk_validation.get("valid", False):
            recommendations.append("CRITICAL: No chunks available for chat. Process documents first.")
        elif not consistency_validation.get("consistent", False):
            recommendations.append("HIGH: Data inconsistency detected. Synchronize database and vector store.")
        elif not search_test.get("functional", False):
            recommendations.append("HIGH: Search functionality not working. Check Weaviate and embedding services.")
        elif not attribution_validation.get("valid", False):
            recommendations.append("MEDIUM: Source attribution issues. Review chunk synchronization.")
        else:
            recommendations.append("Chat integration is healthy and ready for use.")

        return recommendations