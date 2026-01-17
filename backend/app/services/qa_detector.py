"""Question-Answer detection service for semantic chunking.

This module provides comprehensive Q&A detection for Turkish and English texts,
ensuring question-answer pairs stay together during chunking.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from app.services.language_detector import Language

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """Represents a detected question-answer pair."""
    question: str
    question_index: int
    answer: str
    answer_index: int
    confidence: float  # 0-1, based on semantic similarity if available
    merged_text: str


class QADetector:
    """Detect and handle question-answer pairs in text.
    
    Features:
    - Comprehensive Turkish question patterns (50+ patterns)
    - English question word detection
    - Answer detection in following 1-3 sentences
    - Semantic similarity confirmation (optional)
    - Q&A pair merging for chunking
    """
    
    # Comprehensive Turkish question words
    TURKISH_QUESTION_WORDS: Set[str] = {
        # Basic question words
        'ne', 'nasıl', 'neden', 'niçin', 'niye',
        'kim', 'kimi', 'kimin', 'kime', 'kimden', 'kimle',
        'nerede', 'nereye', 'nereden', 'nere', 'neresi',
        'hangi', 'hangisi', 'hangileri',
        'kaç', 'kaçıncı', 'kaçar', 'kaçta',
        'ne zaman', 'ne kadar', 'ne için', 'ne gibi',
        'neyi', 'neyin', 'neye', 'neden', 'neyle',
        
        # Indirect question indicators
        'acaba', 'merak ediyorum', 'bilmiyorum',
        'anlamadım', 'anlayamadım', 'söyler misiniz',
    }
    
    # Turkish question particles (all vowel harmony forms)
    TURKISH_QUESTION_PARTICLES: Set[str] = {
        # Basic particles
        'mi', 'mı', 'mu', 'mü',
        # With -dir suffix
        'midir', 'mıdır', 'mudur', 'müdür',
        # First person singular
        'miyim', 'mıyım', 'muyum', 'müyüm',
        # Second person singular
        'misin', 'mısın', 'musun', 'müsün',
        # Third person singular (same as basic)
        # First person plural
        'miyiz', 'mıyız', 'muyuz', 'müyüz',
        # Second person plural
        'misiniz', 'mısınız', 'musunuz', 'müsünüz',
        # Third person plural
        'midirler', 'mıdırlar', 'mudurlar', 'müdürler',
        'miler', 'mılar', 'mular', 'müler',
        # Past tense forms
        'miydim', 'mıydım', 'muydum', 'müydüm',
        'miydin', 'mıydın', 'muydun', 'müydün',
        'miydi', 'mıydı', 'muydu', 'müydü',
        'miydik', 'mıydık', 'muyduk', 'müydük',
        'miydiniz', 'mıydınız', 'muydunuz', 'müydünüz',
        'miydiler', 'mıydılar', 'muydular', 'müydüler',
    }
    
    # English question words
    ENGLISH_QUESTION_WORDS: Set[str] = {
        'what', 'when', 'where', 'who', 'whom', 'whose',
        'why', 'how', 'which', 'can', 'could', 'would',
        'should', 'will', 'shall', 'may', 'might', 'must',
        'do', 'does', 'did', 'is', 'are', 'was', 'were',
        'has', 'have', 'had', 'isn\'t', 'aren\'t', 'wasn\'t',
        'weren\'t', 'don\'t', 'doesn\'t', 'didn\'t',
        'haven\'t', 'hasn\'t', 'hadn\'t', 'won\'t', 'wouldn\'t',
        'can\'t', 'couldn\'t', 'shouldn\'t', 'mustn\'t',
    }
    
    # Default similarity threshold for Q&A confirmation
    DEFAULT_SIMILARITY_THRESHOLD = 0.6
    
    # Maximum sentences to look ahead for answer
    MAX_ANSWER_DISTANCE = 3
    
    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        """Initialize the Q&A detector.
        
        Args:
            similarity_threshold: Minimum similarity for Q&A confirmation (0-1)
        """
        self.similarity_threshold = similarity_threshold
    
    def is_question(self, sentence: str, language: Language = Language.UNKNOWN) -> bool:
        """Check if a sentence is a question.
        
        Args:
            sentence: Sentence to check
            language: Language of the sentence
            
        Returns:
            True if sentence is a question
        """
        if not sentence or not sentence.strip():
            return False
        
        sentence = sentence.strip()
        
        # Check for question mark (universal)
        if sentence.endswith('?'):
            return True
        
        # Language-specific checks
        if language in (Language.TURKISH, Language.MIXED, Language.UNKNOWN):
            if self._is_turkish_question(sentence):
                return True
        
        if language in (Language.ENGLISH, Language.MIXED, Language.UNKNOWN):
            if self._is_english_question(sentence):
                return True
        
        return False
    
    def _is_turkish_question(self, sentence: str) -> bool:
        """Check if sentence is a Turkish question."""
        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        
        # Check for question words at the start
        for qword in self.TURKISH_QUESTION_WORDS:
            if ' ' in qword:
                # Multi-word question phrase
                if sentence_lower.startswith(qword):
                    return True
            elif words and words[0] == qword:
                return True
        
        # Check for question particles anywhere in sentence
        for particle in self.TURKISH_QUESTION_PARTICLES:
            # Check as standalone word
            if particle in words:
                return True
            # Check as suffix (e.g., "geldi mi" -> "geldimi")
            pattern = rf'\b\w+{re.escape(particle)}\b'
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _is_english_question(self, sentence: str) -> bool:
        """Check if sentence is an English question."""
        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        
        if not words:
            return False
        
        # Check if starts with question word
        first_word = words[0].rstrip('.,!?')
        if first_word in self.ENGLISH_QUESTION_WORDS:
            return True
        
        return False

    def find_answer(
        self,
        question: str,
        following_sentences: List[str],
        embeddings: Optional[List[List[float]]] = None,
        question_embedding: Optional[List[float]] = None
    ) -> Optional[Tuple[str, int, float]]:
        """Find the answer to a question in following sentences.
        
        Args:
            question: The question sentence
            following_sentences: List of sentences after the question (max 3)
            embeddings: Optional embeddings for following sentences
            question_embedding: Optional embedding for the question
            
        Returns:
            Tuple of (answer_text, answer_index, confidence) or None
        """
        if not following_sentences:
            return None
        
        # Limit to MAX_ANSWER_DISTANCE sentences
        candidates = following_sentences[:self.MAX_ANSWER_DISTANCE]
        
        # If embeddings available, use semantic similarity
        if embeddings and question_embedding and len(embeddings) >= len(candidates):
            return self._find_answer_semantic(
                question, candidates, embeddings[:len(candidates)], question_embedding
            )
        
        # Otherwise, use heuristic-based detection
        return self._find_answer_heuristic(question, candidates)
    
    def _find_answer_semantic(
        self,
        question: str,
        candidates: List[str],
        embeddings: List[List[float]],
        question_embedding: List[float]
    ) -> Optional[Tuple[str, int, float]]:
        """Find answer using semantic similarity."""
        import numpy as np
        
        best_idx = -1
        best_similarity = 0.0
        
        q_vec = np.array(question_embedding)
        q_norm = np.linalg.norm(q_vec)
        
        if q_norm == 0:
            return self._find_answer_heuristic(question, candidates)
        
        for i, emb in enumerate(embeddings):
            c_vec = np.array(emb)
            c_norm = np.linalg.norm(c_vec)
            
            if c_norm == 0:
                continue
            
            similarity = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = i
        
        if best_idx >= 0 and best_similarity >= self.similarity_threshold:
            return (candidates[best_idx], best_idx, best_similarity)
        
        # Fall back to heuristic if no good semantic match
        return self._find_answer_heuristic(question, candidates)
    
    def _find_answer_heuristic(
        self,
        question: str,
        candidates: List[str]
    ) -> Optional[Tuple[str, int, float]]:
        """Find answer using heuristics."""
        if not candidates:
            return None
        
        # Simple heuristic: first non-question sentence is likely the answer
        for i, candidate in enumerate(candidates):
            candidate = candidate.strip()
            if not candidate:
                continue
            
            # Skip if it's another question
            if candidate.endswith('?'):
                continue
            
            # Skip very short responses (likely not an answer)
            if len(candidate) < 10:
                continue
            
            # Found a potential answer
            # Confidence based on position (closer = higher confidence)
            confidence = 0.8 - (i * 0.1)  # 0.8, 0.7, 0.6 for positions 0, 1, 2
            return (candidate, i, confidence)
        
        return None
    
    def detect_qa_pairs(
        self,
        sentences: List[str],
        language: Language = Language.UNKNOWN,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[QAPair]:
        """Detect all Q&A pairs in a list of sentences.
        
        Args:
            sentences: List of sentences
            language: Language of the text
            embeddings: Optional embeddings for all sentences
            
        Returns:
            List of detected QAPair objects
        """
        qa_pairs = []
        used_indices = set()
        
        for i, sentence in enumerate(sentences):
            if i in used_indices:
                continue
            
            if not self.is_question(sentence, language):
                continue
            
            # Get following sentences
            following = sentences[i + 1:i + 1 + self.MAX_ANSWER_DISTANCE]
            
            # Get embeddings if available
            q_emb = embeddings[i] if embeddings and i < len(embeddings) else None
            f_embs = None
            if embeddings and i + 1 < len(embeddings):
                end_idx = min(i + 1 + self.MAX_ANSWER_DISTANCE, len(embeddings))
                f_embs = embeddings[i + 1:end_idx]
            
            # Find answer
            result = self.find_answer(sentence, following, f_embs, q_emb)
            
            if result:
                answer_text, answer_offset, confidence = result
                answer_idx = i + 1 + answer_offset
                
                # Create merged text
                merged = f"{sentence} {answer_text}"
                
                qa_pairs.append(QAPair(
                    question=sentence,
                    question_index=i,
                    answer=answer_text,
                    answer_index=answer_idx,
                    confidence=confidence,
                    merged_text=merged
                ))
                
                # Mark indices as used
                used_indices.add(i)
                used_indices.add(answer_idx)
        
        return qa_pairs
    
    def merge_qa_pairs(
        self,
        sentences: List[str],
        language: Language = Language.UNKNOWN,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """Merge Q&A pairs into single sentences.
        
        Args:
            sentences: List of sentences
            language: Language of the text
            embeddings: Optional embeddings for semantic matching
            
        Returns:
            List of sentences with Q&A pairs merged
        """
        if not sentences:
            return sentences
        
        # Detect Q&A pairs
        qa_pairs = self.detect_qa_pairs(sentences, language, embeddings)
        
        if not qa_pairs:
            return sentences
        
        # Build set of indices to skip (they're merged into Q&A pairs)
        skip_indices = set()
        qa_at_index = {}
        
        for qa in qa_pairs:
            skip_indices.add(qa.answer_index)
            qa_at_index[qa.question_index] = qa
        
        # Build result
        result = []
        for i, sentence in enumerate(sentences):
            if i in skip_indices:
                continue
            
            if i in qa_at_index:
                # Use merged Q&A text
                result.append(qa_at_index[i].merged_text)
            else:
                result.append(sentence)
        
        return result
    
    def get_question_patterns_count(self, language: Language) -> int:
        """Get count of question patterns for a language.
        
        Args:
            language: Language to check
            
        Returns:
            Number of question patterns
        """
        if language == Language.TURKISH:
            return len(self.TURKISH_QUESTION_WORDS) + len(self.TURKISH_QUESTION_PARTICLES)
        elif language == Language.ENGLISH:
            return len(self.ENGLISH_QUESTION_WORDS)
        elif language == Language.MIXED:
            return (
                len(self.TURKISH_QUESTION_WORDS) +
                len(self.TURKISH_QUESTION_PARTICLES) +
                len(self.ENGLISH_QUESTION_WORDS)
            )
        else:
            return 0
