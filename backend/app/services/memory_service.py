"""
Memory Service for Chat History Management

Implements short-term and long-term memory using Weaviate:
- Short-term memory: Recent conversation context (last N messages)
- Long-term memory: Summarized important information from past conversations
"""

from typing import List, Dict
from datetime import datetime, timezone
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter, Sort

from app.services.embedding_service import get_embedding_service
from app.services.llm_service import LLMService


class MemoryService:
    """Service for managing user conversation memory in Weaviate."""
    
    SHORT_TERM_COLLECTION = "ChatShortTermMemory"
    LONG_TERM_COLLECTION = "ChatLongTermMemory"
    
    def __init__(self, weaviate_client: weaviate.WeaviateClient):
        self.client = weaviate_client
        self.embedding_service = get_embedding_service()
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure memory collections exist in Weaviate."""
        try:
            # Short-term memory collection
            if not self.client.collections.exists(self.SHORT_TERM_COLLECTION):
                self.client.collections.create(
                    name=self.SHORT_TERM_COLLECTION,
                    properties=[
                        Property(name="user_id", data_type=DataType.INT),
                        Property(name="course_id", data_type=DataType.INT),
                        Property(name="role", data_type=DataType.TEXT),
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="timestamp", data_type=DataType.DATE),
                        Property(name="session_id", data_type=DataType.TEXT),
                    ],
                    vectorizer_config=Configure.Vectorizer.none(),
                )
                print(f"[MEMORY] Created collection: {self.SHORT_TERM_COLLECTION}")
            
            # Long-term memory collection
            if not self.client.collections.exists(self.LONG_TERM_COLLECTION):
                self.client.collections.create(
                    name=self.LONG_TERM_COLLECTION,
                    properties=[
                        Property(name="user_id", data_type=DataType.INT),
                        Property(name="course_id", data_type=DataType.INT),
                        Property(name="summary", data_type=DataType.TEXT),
                        Property(name="key_topics", data_type=DataType.TEXT_ARRAY),
                        Property(name="timestamp", data_type=DataType.DATE),
                        Property(name="importance_score", data_type=DataType.NUMBER),
                    ],
                    vectorizer_config=Configure.Vectorizer.none(),
                )
                print(f"[MEMORY] Created collection: {self.LONG_TERM_COLLECTION}")
                
        except Exception as e:
            print(f"[MEMORY] Error ensuring collections: {e}")
    
    def add_message_to_short_term(
        self,
        user_id: int,
        course_id: int,
        role: str,
        content: str,
        session_id: str,
        embedding_model: str = "voyage/voyage-3-lite"
    ):
        """Add a message to short-term memory."""
        try:
            collection = self.client.collections.get(self.SHORT_TERM_COLLECTION)
            
            # Get embedding for the message
            vector = self.embedding_service.get_embedding(content, model=embedding_model)
            
            # Add to Weaviate with RFC3339 timestamp (without microseconds)
            timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
            
            collection.data.insert(
                properties={
                    "user_id": user_id,
                    "course_id": course_id,
                    "role": role,
                    "content": content,
                    "timestamp": timestamp,
                    "session_id": session_id,
                },
                vector=vector
            )
            
            print(f"[MEMORY] Added message to short-term memory: user={user_id}, course={course_id}, timestamp={timestamp}")
            
            # Clean old messages (keep last 50 per user per course)
            self._cleanup_short_term_memory(user_id, course_id, keep_last=50)
            
        except Exception as e:
            print(f"[MEMORY] Error adding to short-term memory: {e}")
    
    def get_short_term_context(
        self,
        user_id: int,
        course_id: int,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent conversation context from short-term memory."""
        try:
            collection = self.client.collections.get(self.SHORT_TERM_COLLECTION)
            
            # Query recent messages with proper Sort object
            response = collection.query.fetch_objects(
                filters=(
                    Filter.by_property("user_id").equal(user_id) &
                    Filter.by_property("course_id").equal(course_id)
                ),
                limit=limit,
                sort=Sort.by_property("timestamp", ascending=False)
            )
            
            # Convert to list and reverse to get chronological order
            messages = []
            for obj in response.objects:
                messages.append({
                    "role": obj.properties.get("role"),
                    "content": obj.properties.get("content"),
                    "timestamp": obj.properties.get("timestamp"),
                })
            
            messages.reverse()  # Chronological order
            return messages
            
        except Exception as e:
            print(f"[MEMORY] Error getting short-term context: {e}")
            return []
    
    def search_relevant_memories(
        self,
        user_id: int,
        course_id: int,
        query: str,
        embedding_model: str = "voyage/voyage-3-lite",
        limit: int = 5
    ) -> List[Dict]:
        """Search for relevant past conversations using semantic search."""
        try:
            collection = self.client.collections.get(self.SHORT_TERM_COLLECTION)
            
            # Get query embedding
            query_vector = self.embedding_service.get_embedding(query, model=embedding_model)
            
            # Semantic search
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                filters=(
                    Filter.by_property("user_id").equal(user_id) &
                    Filter.by_property("course_id").equal(course_id)
                )
            )
            
            memories = []
            for obj in response.objects:
                memories.append({
                    "role": obj.properties.get("role"),
                    "content": obj.properties.get("content"),
                    "timestamp": obj.properties.get("timestamp"),
                    "distance": obj.metadata.distance if hasattr(obj.metadata, 'distance') else None
                })
            
            return memories
            
        except Exception as e:
            print(f"[MEMORY] Error searching memories: {e}")
            return []
    
    def _cleanup_short_term_memory(self, user_id: int, course_id: int, keep_last: int = 50):
        """Remove old messages, keeping only the most recent ones."""
        try:
            collection = self.client.collections.get(self.SHORT_TERM_COLLECTION)
            
            # Get all messages for this user/course with proper Sort object
            response = collection.query.fetch_objects(
                filters=(
                    Filter.by_property("user_id").equal(user_id) &
                    Filter.by_property("course_id").equal(course_id)
                ),
                limit=1000,  # Get all
                sort=Sort.by_property("timestamp", ascending=False)
            )
            
            # Delete old ones
            if len(response.objects) > keep_last:
                to_delete = response.objects[keep_last:]
                for obj in to_delete:
                    collection.data.delete_by_id(obj.uuid)
                
                print(f"[MEMORY] Cleaned up {len(to_delete)} old messages")
                
        except Exception as e:
            print(f"[MEMORY] Error cleaning up short-term memory: {e}")
    
    def summarize_to_long_term(
        self,
        user_id: int,
        course_id: int,
        llm_service: LLMService,
        embedding_model: str = "voyage/voyage-3-lite"
    ):
        """Summarize recent conversations and store in long-term memory."""
        try:
            # Get recent messages
            messages = self.get_short_term_context(user_id, course_id, limit=20)
            
            if len(messages) < 5:
                return  # Not enough to summarize
            
            # Create conversation text
            conversation = "\n".join([
                f"{msg['role']}: {msg['content']}" for msg in messages
            ])
            
            # Generate summary using LLM
            summary_prompt = f"""Aşağıdaki konuşmayı özetle. Önemli konuları, sorulan soruları ve verilen cevapları belirt.

Konuşma:
{conversation}

Özet:"""
            
            summary = llm_service.generate_response(
                prompt=summary_prompt,
                max_tokens=500
            )
            
            # Extract key topics (simple keyword extraction)
            key_topics = self._extract_key_topics(conversation)
            
            # Get embedding for summary
            vector = self.embedding_service.get_embedding(summary, model=embedding_model)
            
            # Store in long-term memory with RFC3339 timestamp (without microseconds)
            timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
            
            collection = self.client.collections.get(self.LONG_TERM_COLLECTION)
            collection.data.insert(
                properties={
                    "user_id": user_id,
                    "course_id": course_id,
                    "summary": summary,
                    "key_topics": key_topics,
                    "timestamp": timestamp,
                    "importance_score": 0.5,  # Can be calculated based on various factors
                },
                vector=vector
            )
            
            print(f"[MEMORY] Created long-term memory summary for user={user_id}, course={course_id}, timestamp={timestamp}")
            
            print(f"[MEMORY] Created long-term memory summary for user={user_id}, course={course_id}")
            
        except Exception as e:
            print(f"[MEMORY] Error creating long-term summary: {e}")
    
    def get_long_term_context(
        self,
        user_id: int,
        course_id: int,
        query: str,
        embedding_model: str = "voyage/voyage-3-lite",
        limit: int = 3
    ) -> List[Dict]:
        """Get relevant long-term memories based on current query."""
        try:
            collection = self.client.collections.get(self.LONG_TERM_COLLECTION)
            
            # Get query embedding
            query_vector = self.embedding_service.get_embedding(query, model=embedding_model)
            
            # Semantic search in long-term memory
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                filters=(
                    Filter.by_property("user_id").equal(user_id) &
                    Filter.by_property("course_id").equal(course_id)
                )
            )
            
            memories = []
            for obj in response.objects:
                memories.append({
                    "summary": obj.properties.get("summary"),
                    "key_topics": obj.properties.get("key_topics", []),
                    "timestamp": obj.properties.get("timestamp"),
                    "importance_score": obj.properties.get("importance_score"),
                })
            
            return memories
            
        except Exception as e:
            print(f"[MEMORY] Error getting long-term context: {e}")
            return []
    
    def _extract_key_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """Simple keyword extraction from text."""
        # This is a simple implementation - can be improved with NLP
        words = text.lower().split()
        
        # Filter common words (basic stopwords)
        stopwords = {'bir', 'bu', 've', 'için', 'ile', 'mi', 'mı', 'ne', 'nedir', 'nasıl', 
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_topics]]
    
    def clear_user_memory(self, user_id: int, course_id: int):
        """Clear all memory for a specific user in a course."""
        try:
            # Clear short-term
            collection = self.client.collections.get(self.SHORT_TERM_COLLECTION)
            collection.data.delete_many(
                where=Filter.by_property("user_id").equal(user_id) &
                      Filter.by_property("course_id").equal(course_id)
            )
            
            # Clear long-term
            collection = self.client.collections.get(self.LONG_TERM_COLLECTION)
            collection.data.delete_many(
                where=Filter.by_property("user_id").equal(user_id) &
                      Filter.by_property("course_id").equal(course_id)
            )
            
            print(f"[MEMORY] Cleared all memory for user={user_id}, course={course_id}")
            
        except Exception as e:
            print(f"[MEMORY] Error clearing memory: {e}")


def get_memory_service(weaviate_client: weaviate.WeaviateClient) -> MemoryService:
    """Get or create memory service instance."""
    return MemoryService(weaviate_client)
