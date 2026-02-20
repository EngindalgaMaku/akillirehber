"""Adaptive threshold calculation for semantic chunking.

This module provides intelligent threshold calculation based on
text characteristics like vocabulary diversity and sentence length.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TextCharacteristics:
    """Text analysis results."""
    length: int
    sentence_count: int
    word_count: int
    unique_word_count: int
    vocabulary_diversity: float  # unique words / total words (0-1)
    avg_sentence_length: float  # average characters per sentence
    avg_word_length: float
    has_questions: bool
    question_count: int


@dataclass
class ThresholdRecommendation:
    """Threshold recommendation with explanation."""
    recommended_threshold: float
    base_threshold: float
    diversity_factor: float
    length_factor: float
    confidence: float  # 0-1
    reasoning: str
    text_characteristics: TextCharacteristics


class AdaptiveThresholdCalculator:
    """Calculate optimal similarity threshold based on text characteristics.
    
    The adaptive threshold adjusts based on:
    - Vocabulary diversity: High diversity → lower threshold (more splits)
    - Sentence length: Long sentences → lower threshold
    - Question presence: Questions may need different handling
    
    Formula:
        threshold = base_threshold * diversity_factor * length_factor
        
    Where:
        - diversity_factor: 0.8 (high), 1.0 (medium), 1.2 (low)
        - length_factor: 0.9 (long), 1.0 (medium), 1.1 (short)
    """
    
    # Default base threshold (percentile)
    DEFAULT_BASE_THRESHOLD = 0.5
    
    # Diversity thresholds
    HIGH_DIVERSITY_THRESHOLD = 0.7
    LOW_DIVERSITY_THRESHOLD = 0.4
    
    # Sentence length thresholds (characters)
    LONG_SENTENCE_THRESHOLD = 100
    SHORT_SENTENCE_THRESHOLD = 50
    
    # Threshold bounds
    MIN_THRESHOLD = 0.3
    MAX_THRESHOLD = 0.9
    
    def __init__(self, base_threshold: float = DEFAULT_BASE_THRESHOLD):
        """Initialize the calculator.
        
        Args:
            base_threshold: Base threshold value (0-1)
        """
        self.base_threshold = base_threshold
    
    def analyze_text(self, text: str) -> TextCharacteristics:
        """Analyze text characteristics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            TextCharacteristics with analysis results
        """
        if not text or not text.strip():
            return TextCharacteristics(
                length=0,
                sentence_count=0,
                word_count=0,
                unique_word_count=0,
                vocabulary_diversity=0.0,
                avg_sentence_length=0.0,
                avg_word_length=0.0,
                has_questions=False,
                question_count=0
            )
        
        # Basic counts
        length = len(text)
        
        # Split into sentences (simple approach)
        sentences = self._split_sentences(text)
        sentence_count = len(sentences)
        
        # Word analysis
        words = self._extract_words(text)
        word_count = len(words)
        unique_words = set(w.lower() for w in words)
        unique_word_count = len(unique_words)
        
        # Calculate metrics
        vocabulary_diversity = (
            unique_word_count / word_count if word_count > 0 else 0.0
        )
        
        avg_sentence_length = (
            sum(len(s) for s in sentences) / sentence_count
            if sentence_count > 0 else 0.0
        )
        
        avg_word_length = (
            sum(len(w) for w in words) / word_count
            if word_count > 0 else 0.0
        )
        
        # Question detection
        question_count = sum(1 for s in sentences if s.strip().endswith('?'))
        has_questions = question_count > 0
        
        return TextCharacteristics(
            length=length,
            sentence_count=sentence_count,
            word_count=word_count,
            unique_word_count=unique_word_count,
            vocabulary_diversity=vocabulary_diversity,
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            has_questions=has_questions,
            question_count=question_count
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on . ! ?
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text."""
        # Match word characters including Turkish special chars
        pattern = r'[a-zA-ZçğıöşüÇĞİÖŞÜ]+'
        return re.findall(pattern, text)
    
    def calculate_threshold(
        self,
        characteristics: TextCharacteristics
    ) -> float:
        """Calculate optimal threshold based on text characteristics.
        
        Args:
            characteristics: Analyzed text characteristics
            
        Returns:
            Calculated threshold value (0.3-0.9)
        """
        # Calculate diversity factor
        diversity_factor = self._calculate_diversity_factor(
            characteristics.vocabulary_diversity
        )
        
        # Calculate length factor
        length_factor = self._calculate_length_factor(
            characteristics.avg_sentence_length
        )
        
        # Calculate final threshold
        threshold = self.base_threshold * diversity_factor * length_factor
        
        # Clamp to valid range
        return max(self.MIN_THRESHOLD, min(self.MAX_THRESHOLD, threshold))
    
    def _calculate_diversity_factor(self, diversity: float) -> float:
        """Calculate diversity factor.
        
        High diversity (>0.7): 0.8 (lower threshold, more splits)
        Medium diversity (0.4-0.7): 1.0 (keep base)
        Low diversity (<0.4): 1.2 (higher threshold, fewer splits)
        """
        if diversity > self.HIGH_DIVERSITY_THRESHOLD:
            return 0.8
        elif diversity < self.LOW_DIVERSITY_THRESHOLD:
            return 1.2
        else:
            return 1.0
    
    def _calculate_length_factor(self, avg_length: float) -> float:
        """Calculate sentence length factor.
        
        Long sentences (>100 chars): 0.9 (lower threshold)
        Medium sentences (50-100 chars): 1.0 (keep base)
        Short sentences (<50 chars): 1.1 (higher threshold)
        """
        if avg_length > self.LONG_SENTENCE_THRESHOLD:
            return 0.9
        elif avg_length < self.SHORT_SENTENCE_THRESHOLD:
            return 1.1
        else:
            return 1.0
    
    def recommend_threshold(self, text: str) -> ThresholdRecommendation:
        """Provide threshold recommendation with explanation.
        
        Args:
            text: Input text
            
        Returns:
            ThresholdRecommendation with value and reasoning
        """
        # Analyze text
        characteristics = self.analyze_text(text)
        
        # Calculate factors
        diversity_factor = self._calculate_diversity_factor(
            characteristics.vocabulary_diversity
        )
        length_factor = self._calculate_length_factor(
            characteristics.avg_sentence_length
        )
        
        # Calculate threshold
        threshold = self.calculate_threshold(characteristics)
        
        # Calculate confidence based on text size
        # More text = higher confidence in analysis
        if characteristics.sentence_count >= 10:
            confidence = 0.9
        elif characteristics.sentence_count >= 5:
            confidence = 0.7
        elif characteristics.sentence_count >= 2:
            confidence = 0.5
        else:
            confidence = 0.3
        
        # Build reasoning
        reasoning = self._build_reasoning(
            characteristics, diversity_factor, length_factor, threshold
        )
        
        return ThresholdRecommendation(
            recommended_threshold=threshold,
            base_threshold=self.base_threshold,
            diversity_factor=diversity_factor,
            length_factor=length_factor,
            confidence=confidence,
            reasoning=reasoning,
            text_characteristics=characteristics
        )
    
    def _build_reasoning(
        self,
        characteristics: TextCharacteristics,
        diversity_factor: float,
        length_factor: float,
        threshold: float
    ) -> str:
        """Build human-readable reasoning for recommendation."""
        parts = []
        
        # Diversity reasoning
        if diversity_factor < 1.0:
            parts.append(
                f"High vocabulary diversity ({characteristics.vocabulary_diversity:.2f}) "
                f"suggests varied topics - using lower threshold for more splits."
            )
        elif diversity_factor > 1.0:
            parts.append(
                f"Low vocabulary diversity ({characteristics.vocabulary_diversity:.2f}) "
                f"suggests focused content - using higher threshold for fewer splits."
            )
        else:
            parts.append(
                f"Medium vocabulary diversity ({characteristics.vocabulary_diversity:.2f}) "
                f"- using base threshold."
            )
        
        # Length reasoning
        if length_factor < 1.0:
            parts.append(
                f"Long average sentence length ({characteristics.avg_sentence_length:.0f} chars) "
                f"- adjusting for more granular splits."
            )
        elif length_factor > 1.0:
            parts.append(
                f"Short average sentence length ({characteristics.avg_sentence_length:.0f} chars) "
                f"- adjusting for fewer splits."
            )
        
        # Question note
        if characteristics.has_questions:
            parts.append(
                f"Text contains {characteristics.question_count} question(s) - "
                f"Q&A pairs will be kept together."
            )
        
        parts.append(f"Recommended threshold: {threshold:.2f}")
        
        return " ".join(parts)
