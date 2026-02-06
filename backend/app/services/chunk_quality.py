"""Chunk quality metrics and reporting service.

Feature: semantic-chunker-enhancement, Phase 4: Quality Metrics
This module provides quality metrics calculation for semantic chunks,
including coherence scores, inter-chunk similarity, and quality reports.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChunkQualityMetrics:
    """Quality metrics for a single chunk.
    
    Feature: semantic-chunker-enhancement, Task 7.8
    Validates: Requirements 7.1, 7.2, 7.3, 7.4
    """
    chunk_index: int
    semantic_coherence: float  # 0-1, intra-chunk sentence similarity
    sentence_count: int
    avg_sentence_similarity: float  # 0-1
    topic_consistency: float  # 0-1
    has_questions: bool = False
    has_qa_pairs: bool = False


@dataclass
class QualityReport:
    """Overall quality report for chunking results.
    
    Feature: semantic-chunker-enhancement, Task 7.9
    Validates: Requirements 7.5
    """
    total_chunks: int
    avg_coherence: float
    min_coherence: float
    max_coherence: float
    chunks_below_threshold: List[int] = field(default_factory=list)  # chunk indices
    inter_chunk_similarities: List[float] = field(default_factory=list)
    merge_recommendations: List[Tuple[int, int]] = field(default_factory=list)  # (chunk1, chunk2)
    split_recommendations: List[int] = field(default_factory=list)  # chunk indices
    overall_quality_score: float = 0.0  # 0-1
    recommendations: List[str] = field(default_factory=list)


class ChunkQualityAnalyzer:
    """Analyzer for calculating chunk quality metrics.
    
    Feature: semantic-chunker-enhancement, Phase 4: Quality Metrics
    Provides methods for:
    - Semantic coherence calculation (intra-chunk similarity)
    - Inter-chunk similarity measurement
    - Low coherence detection
    - Topic distribution analysis
    - Quality report generation
    - Merge/split recommendations
    """
    
    # Thresholds - adjusted for realistic text coherence
    # Typical sentence similarity within a paragraph is 0.3-0.6
    # Only flag chunks with very low coherence
    LOW_COHERENCE_THRESHOLD = 0.25  # Chunks below this are flagged
    HIGH_SIMILARITY_THRESHOLD = 0.7  # Consecutive chunks above this should merge
    
    def __init__(
        self,
        embedding_provider=None,
        embedding_model: str = "openai/text-embedding-3-small"
    ):
        """Initialize the quality analyzer.
        
        Args:
            embedding_provider: Optional embedding provider for similarity calculations.
                              If None, will initialize on first use.
            embedding_model: Model to use for embeddings.
        """
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._np = None
    
    def _ensure_numpy(self):
        """Load numpy if not already loaded."""
        if self._np is None:
            try:
                import numpy as np
                self._np = np
            except ImportError as exc:
                raise ImportError(
                    "numpy is required for quality metrics. "
                    "Install it with: pip install numpy"
                ) from exc
    
    def _ensure_embedding_provider(self):
        """Initialize embedding provider if not already done."""
        if self._embedding_provider is None:
            from app.services.embedding_provider import (
                EmbeddingProviderConfig,
                EmbeddingProviderManager,
                OpenRouterProvider,
                OpenAIProvider,
                VoyageProvider,
                OllamaProvider,
                CohereProvider,
                JinaProvider,
                AlibabaProvider,
            )
            
            providers = []
            
            # Try Voyage first if available
            voyage = VoyageProvider()
            if voyage.is_available():
                providers.append(voyage)
            
            # Try Ollama if available (local, fast)
            ollama = OllamaProvider()
            if ollama.is_available():
                providers.append(ollama)
            
            # Try Cohere if available
            cohere = CohereProvider()
            if cohere.is_available():
                providers.append(cohere)
            
            # Try Jina if available
            jina = JinaProvider()
            if jina.is_available():
                providers.append(jina)
            
            # Try Alibaba if available
            alibaba = AlibabaProvider()
            if alibaba.is_available():
                providers.append(alibaba)
            
            openrouter = OpenRouterProvider()
            if openrouter.is_available():
                providers.append(openrouter)
            
            openai = OpenAIProvider()
            if openai.is_available():
                providers.append(openai)
            
            if not providers:
                raise ValueError(
                    "No embedding providers available. "
                    "Set VOYAGE_API_KEY, COHERE_API_KEY, JINA_AI_API_KEY, "
                    "DASHSCOPE_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY or run Ollama locally."
                )
            
            config = EmbeddingProviderConfig(
                batch_size=32,
                max_retries=3,
                retry_delay=1.0
            )
            
            self._embedding_provider = EmbeddingProviderManager(
                providers=providers,
                config=config
            )
    
    def _cosine_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        self._ensure_numpy()
        np = self._np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting on . ! ?
        pattern = r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜa-zçğıöşü0-9])'
        sentences = re.split(pattern, text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def calculate_semantic_coherence(
        self,
        chunk_text: str,
        embeddings: Optional[List[List[float]]] = None
    ) -> float:
        """Calculate semantic coherence for a chunk.
        
        Feature: semantic-chunker-enhancement, Task 7.1
        Validates: Requirements 7.1
        
        Coherence is calculated as the average pairwise cosine similarity
        between all sentences within the chunk.
        
        Args:
            chunk_text: The text content of the chunk.
            embeddings: Optional pre-computed sentence embeddings.
                       If None, will compute embeddings.
        
        Returns:
            Coherence score between 0 and 1.
            - 1.0 means all sentences are identical/very similar
            - 0.0 means sentences are completely unrelated
        """
        sentences = self._split_into_sentences(chunk_text)
        
        # Single sentence or empty chunk has perfect coherence
        if len(sentences) <= 1:
            return 1.0
        
        # Get embeddings if not provided
        if embeddings is None:
            self._ensure_embedding_provider()
            embeddings = self._embedding_provider.get_embeddings(
                sentences, self._embedding_model, input_type="document"
            )
        
        if len(embeddings) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        self._ensure_numpy()
        similarities = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # Return average similarity as coherence score
        coherence = sum(similarities) / len(similarities)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, coherence))

    def calculate_inter_chunk_similarity(
        self,
        chunk1_text: str,
        chunk2_text: str
    ) -> float:
        """Calculate similarity between two consecutive chunks.
        
        Feature: semantic-chunker-enhancement, Task 7.3
        Validates: Requirements 7.2
        
        Uses the last sentence of chunk1 and first sentence of chunk2
        to measure how well they connect.
        
        Args:
            chunk1_text: Text of the first chunk.
            chunk2_text: Text of the second chunk.
        
        Returns:
            Similarity score between 0 and 1.
        """
        sentences1 = self._split_into_sentences(chunk1_text)
        sentences2 = self._split_into_sentences(chunk2_text)
        
        if not sentences1 or not sentences2:
            return 0.0
        
        # Get last sentence of chunk1 and first sentence of chunk2
        last_sentence = sentences1[-1]
        first_sentence = sentences2[0]
        
        # Get embeddings
        self._ensure_embedding_provider()
        embeddings = self._embedding_provider.get_embeddings(
            [last_sentence, first_sentence], self._embedding_model, input_type="document"
        )
        
        if len(embeddings) < 2:
            return 0.0
        
        # Calculate similarity
        similarity = self._cosine_similarity(embeddings[0], embeddings[1])
        
        return max(0.0, min(1.0, similarity))
    
    def calculate_all_inter_chunk_similarities(
        self,
        chunks: List[str]
    ) -> List[float]:
        """Calculate inter-chunk similarities for all consecutive chunk pairs.
        
        Feature: semantic-chunker-enhancement, Task 7.3
        Validates: Requirements 7.2
        
        Args:
            chunks: List of chunk texts.
        
        Returns:
            List of similarity scores for consecutive pairs.
            Length will be len(chunks) - 1.
        """
        if len(chunks) < 2:
            return []
        
        similarities = []
        for i in range(len(chunks) - 1):
            sim = self.calculate_inter_chunk_similarity(chunks[i], chunks[i + 1])
            similarities.append(sim)
        
        return similarities
    
    def detect_low_coherence_chunks(
        self,
        chunk_metrics: List[ChunkQualityMetrics],
        threshold: float = None
    ) -> List[int]:
        """Detect chunks with low semantic coherence.
        
        Feature: semantic-chunker-enhancement, Task 7.5
        Validates: Requirements 7.4
        
        Args:
            chunk_metrics: List of quality metrics for each chunk.
            threshold: Coherence threshold (default: 0.5).
        
        Returns:
            List of chunk indices with coherence below threshold.
        """
        if threshold is None:
            threshold = self.LOW_COHERENCE_THRESHOLD
        
        low_coherence_indices = []
        
        for metrics in chunk_metrics:
            if metrics.semantic_coherence < threshold:
                low_coherence_indices.append(metrics.chunk_index)
        
        return low_coherence_indices
    
    def calculate_topic_consistency(
        self,
        chunk_text: str,
        embeddings: Optional[List[List[float]]] = None
    ) -> float:
        """Calculate topic consistency for a chunk.
        
        Feature: semantic-chunker-enhancement, Task 7.7
        Validates: Requirements 7.3
        
        Uses clustering-like analysis on sentence embeddings to determine
        how focused the chunk is on a single topic.
        
        Args:
            chunk_text: The text content of the chunk.
            embeddings: Optional pre-computed sentence embeddings.
        
        Returns:
            Topic consistency score between 0 and 1.
        """
        sentences = self._split_into_sentences(chunk_text)
        
        if len(sentences) <= 1:
            return 1.0
        
        # Get embeddings if not provided
        if embeddings is None:
            self._ensure_embedding_provider()
            embeddings = self._embedding_provider.get_embeddings(
                sentences, self._embedding_model, input_type="document"
            )
        
        if len(embeddings) < 2:
            return 1.0
        
        self._ensure_numpy()
        np = self._np
        
        # Calculate centroid of all embeddings
        embeddings_array = np.array(embeddings)
        centroid = np.mean(embeddings_array, axis=0)
        
        # Calculate average distance from centroid
        distances = []
        for emb in embeddings:
            sim = self._cosine_similarity(emb, centroid.tolist())
            distances.append(sim)
        
        # Topic consistency is average similarity to centroid
        consistency = sum(distances) / len(distances)
        
        return max(0.0, min(1.0, consistency))
    
    def _detect_questions(self, text: str) -> bool:
        """Check if text contains questions."""
        # Simple question detection
        if '?' in text:
            return True
        
        question_words = [
            'ne ', 'nasıl', 'neden', 'niçin', 'kim', 'nerede',
            'hangi', 'kaç', 'what', 'how', 'why', 'who', 'where',
            'when', 'which'
        ]
        text_lower = text.lower()
        return any(q in text_lower for q in question_words)
    
    def _detect_qa_pairs(self, text: str) -> bool:
        """Check if text contains Q&A pairs."""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) < 2:
            return False
        
        # Check if there's a question followed by a non-question
        for i in range(len(sentences) - 1):
            if self._detect_questions(sentences[i]):
                if not self._detect_questions(sentences[i + 1]):
                    return True
        
        return False
    
    def calculate_chunk_metrics(
        self,
        chunk_text: str,
        chunk_index: int
    ) -> ChunkQualityMetrics:
        """Calculate all quality metrics for a single chunk.
        
        Feature: semantic-chunker-enhancement, Task 7.8
        Validates: Requirements 7.1, 7.2, 7.3, 7.4
        
        Args:
            chunk_text: The text content of the chunk.
            chunk_index: Index of the chunk.
        
        Returns:
            ChunkQualityMetrics with all calculated metrics.
        """
        sentences = self._split_into_sentences(chunk_text)
        sentence_count = len(sentences)
        
        # Get embeddings once for reuse
        embeddings = None
        if sentence_count > 1:
            self._ensure_embedding_provider()
            embeddings = self._embedding_provider.get_embeddings(
                sentences, self._embedding_model, input_type="document"
            )
        
        # Calculate coherence
        coherence = self.calculate_semantic_coherence(chunk_text, embeddings)
        
        # Calculate topic consistency
        topic_consistency = self.calculate_topic_consistency(chunk_text, embeddings)
        
        # Detect questions and Q&A pairs
        has_questions = self._detect_questions(chunk_text)
        has_qa_pairs = self._detect_qa_pairs(chunk_text)
        
        return ChunkQualityMetrics(
            chunk_index=chunk_index,
            semantic_coherence=coherence,
            sentence_count=sentence_count,
            avg_sentence_similarity=coherence,  # Same as coherence for now
            topic_consistency=topic_consistency,
            has_questions=has_questions,
            has_qa_pairs=has_qa_pairs
        )

    def generate_merge_recommendations(
        self,
        inter_chunk_similarities: List[float],
        threshold: float = None
    ) -> List[Tuple[int, int]]:
        """Generate recommendations for merging consecutive chunks.
        
        Feature: semantic-chunker-enhancement, Task 7.10
        Validates: Requirements 7.5
        
        Identifies consecutive chunks with high inter-chunk similarity
        that should be merged.
        
        Args:
            inter_chunk_similarities: List of similarities between consecutive chunks.
            threshold: Similarity threshold for merge recommendation (default: 0.8).
        
        Returns:
            List of (chunk_index, next_chunk_index) tuples to merge.
        """
        if threshold is None:
            threshold = self.HIGH_SIMILARITY_THRESHOLD
        
        recommendations = []
        
        for i, similarity in enumerate(inter_chunk_similarities):
            if similarity > threshold:
                recommendations.append((i, i + 1))
        
        return recommendations
    
    def generate_split_recommendations(
        self,
        chunk_metrics: List[ChunkQualityMetrics],
        threshold: float = None
    ) -> List[int]:
        """Generate recommendations for splitting chunks.
        
        Feature: semantic-chunker-enhancement, Task 7.11
        Validates: Requirements 7.5
        
        Identifies chunks with low coherence that should be split.
        
        Args:
            chunk_metrics: List of quality metrics for each chunk.
            threshold: Coherence threshold below which to recommend split (default: 0.5).
        
        Returns:
            List of chunk indices that should be split.
        """
        if threshold is None:
            threshold = self.LOW_COHERENCE_THRESHOLD
        
        recommendations = []
        
        for metrics in chunk_metrics:
            # Only recommend split if chunk has multiple sentences
            if metrics.semantic_coherence < threshold and metrics.sentence_count > 1:
                recommendations.append(metrics.chunk_index)
        
        return recommendations
    
    def generate_quality_report(
        self,
        chunks: List[str],
        chunk_metrics: Optional[List[ChunkQualityMetrics]] = None
    ) -> QualityReport:
        """Generate a comprehensive quality report for chunking results.
        
        Feature: semantic-chunker-enhancement, Task 7.9
        Validates: Requirements 7.5
        
        Args:
            chunks: List of chunk texts.
            chunk_metrics: Optional pre-computed metrics. If None, will calculate.
        
        Returns:
            QualityReport with all metrics and recommendations.
        """
        if not chunks:
            return QualityReport(
                total_chunks=0,
                avg_coherence=0.0,
                min_coherence=0.0,
                max_coherence=0.0,
                overall_quality_score=0.0
            )
        
        # Calculate metrics if not provided
        if chunk_metrics is None:
            chunk_metrics = []
            for i, chunk_text in enumerate(chunks):
                metrics = self.calculate_chunk_metrics(chunk_text, i)
                chunk_metrics.append(metrics)
        
        # Calculate coherence statistics
        coherences = [m.semantic_coherence for m in chunk_metrics]
        avg_coherence = sum(coherences) / len(coherences)
        min_coherence = min(coherences)
        max_coherence = max(coherences)
        
        # Detect low coherence chunks
        chunks_below_threshold = self.detect_low_coherence_chunks(chunk_metrics)
        
        # Calculate inter-chunk similarities
        inter_chunk_similarities = self.calculate_all_inter_chunk_similarities(chunks)
        
        # Generate merge recommendations
        merge_recommendations = self.generate_merge_recommendations(inter_chunk_similarities)
        
        # Generate split recommendations
        split_recommendations = self.generate_split_recommendations(chunk_metrics)
        
        # Calculate overall quality score
        overall_quality_score = self._calculate_overall_quality_score(
            avg_coherence,
            len(chunks_below_threshold),
            len(chunks),
            len(merge_recommendations),
            len(split_recommendations)
        )
        
        # Generate text recommendations
        recommendations = self._generate_text_recommendations(
            chunks_below_threshold,
            merge_recommendations,
            split_recommendations,
            avg_coherence
        )
        
        return QualityReport(
            total_chunks=len(chunks),
            avg_coherence=avg_coherence,
            min_coherence=min_coherence,
            max_coherence=max_coherence,
            chunks_below_threshold=chunks_below_threshold,
            inter_chunk_similarities=inter_chunk_similarities,
            merge_recommendations=merge_recommendations,
            split_recommendations=split_recommendations,
            overall_quality_score=overall_quality_score,
            recommendations=recommendations
        )
    
    def _calculate_overall_quality_score(
        self,
        avg_coherence: float,
        low_coherence_count: int,
        total_chunks: int,
        merge_count: int,
        split_count: int
    ) -> float:
        """Calculate overall quality score.
        
        Score is based on:
        - Normalized coherence (weight: 0.6) - scaled to realistic range
        - Percentage of chunks with acceptable coherence (weight: 0.3)
        - Penalty for merge/split recommendations (weight: 0.1)
        
        Note: Typical sentence similarity within paragraphs is 0.3-0.6.
        We normalize coherence to this realistic range for better scoring.
        """
        if total_chunks == 0:
            return 0.0
        
        # Normalize coherence to realistic range (0.2-0.7 maps to 0-1)
        # This accounts for the fact that even good chunks have ~0.3-0.5 coherence
        min_expected = 0.2
        max_expected = 0.7
        normalized_coherence = (avg_coherence - min_expected) / (max_expected - min_expected)
        normalized_coherence = max(0.0, min(1.0, normalized_coherence))
        coherence_score = normalized_coherence * 0.6
        
        # Score from percentage of acceptable chunks (threshold is now 0.25)
        good_chunk_ratio = 1 - (low_coherence_count / total_chunks)
        good_chunk_score = good_chunk_ratio * 0.3
        
        # Smaller penalty for recommendations
        recommendation_count = merge_count + split_count
        max_recommendations = total_chunks
        recommendation_ratio = 1 - min(recommendation_count / max(max_recommendations, 1), 1)
        recommendation_score = recommendation_ratio * 0.1
        
        overall = coherence_score + good_chunk_score + recommendation_score
        
        return max(0.0, min(1.0, overall))
    
    def _generate_text_recommendations(
        self,
        low_coherence_chunks: List[int],
        merge_recommendations: List[Tuple[int, int]],
        split_recommendations: List[int],
        avg_coherence: float
    ) -> List[str]:
        """Generate human-readable recommendations."""
        recommendations = []
        
        # Adjusted thresholds for realistic text
        if avg_coherence < 0.25:
            recommendations.append(
                "Overall coherence is low. Consider using a different chunking strategy "
                "or adjusting the similarity threshold."
            )
        elif avg_coherence < 0.35:
            recommendations.append(
                "Coherence is moderate. Some chunks may benefit from refinement."
            )
        
        if low_coherence_chunks:
            recommendations.append(
                f"Chunks {low_coherence_chunks} have low coherence (< 0.25). "
                "Consider splitting them into smaller, more focused chunks."
            )
        
        if merge_recommendations:
            merge_pairs = [f"({a}, {b})" for a, b in merge_recommendations]
            recommendations.append(
                f"Chunk pairs {', '.join(merge_pairs)} have high similarity (> 0.7). "
                "Consider merging them."
            )
        
        if split_recommendations:
            recommendations.append(
                f"Chunks {split_recommendations} should be split due to low coherence."
            )
        
        if not recommendations:
            recommendations.append("Chunk quality is good. No recommendations.")
        
        return recommendations
