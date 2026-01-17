"""Language detection service for multi-language text processing."""

import hashlib
import logging
from enum import Enum
from typing import Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported languages."""
    TURKISH = "tr"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class LanguageConfig:
    """Language-specific configuration."""
    language: Language
    abbreviations: Set[str]
    question_patterns: Set[str]
    sentence_pattern: str
    tokenizer: Optional[object] = None


class LanguageDetector:
    """Detect language of text and provide language-specific configuration.
    
    Uses langdetect library for initial detection with fallback to
    character-based heuristics for Turkish text.
    """
    
    # Turkish-specific characters
    TURKISH_CHARS = set('çğıöşüÇĞİÖŞÜ')
    
    # Cache for detection results (text hash -> language)
    _cache = {}
    
    def __init__(self):
        """Initialize the language detector."""
        self._langdetect = None
    
    def _ensure_langdetect(self):
        """Lazy load langdetect library."""
        if self._langdetect is None:
            try:
                import langdetect
                self._langdetect = langdetect
                # Set seed for reproducibility
                langdetect.DetectorFactory.seed = 0
            except ImportError:
                logger.warning(
                    "langdetect library not available. "
                    "Install with: pip install langdetect"
                )
                self._langdetect = False
    
    def detect_language(self, text: str) -> Language:
        """Detect primary language of text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language enum (TURKISH, ENGLISH, MIXED, UNKNOWN)
        """
        if not text or not text.strip():
            return Language.UNKNOWN
        
        # Check cache first
        text_hash = self._hash_text(text)
        if text_hash in self._cache:
            return self._cache[text_hash]
        
        # Detect language
        language = self._detect_language_internal(text)
        
        # Cache result
        self._cache[text_hash] = language
        
        return language
    
    def _detect_language_internal(self, text: str) -> Language:
        """Internal language detection logic."""
        # Try langdetect first
        self._ensure_langdetect()
        
        if self._langdetect and self._langdetect is not False:
            try:
                detected = self._langdetect.detect(text)
                
                # Map langdetect codes to our Language enum
                if detected == 'tr':
                    return Language.TURKISH
                elif detected == 'en':
                    return Language.ENGLISH
                else:
                    # For other languages, check if it's mixed with Turkish/English
                    return self._check_mixed_language(text)
                    
            except Exception as e:
                logger.debug(f"langdetect failed: {e}, using fallback")
        
        # Fallback to character-based heuristics
        return self._detect_by_characters(text)
    
    def _detect_by_characters(self, text: str) -> Language:
        """Detect language based on character frequency.
        
        Turkish text contains specific characters (ç, ğ, ı, ö, ş, ü)
        that are not present in English.
        """
        # Count Turkish-specific characters
        turkish_char_count = sum(1 for char in text if char in self.TURKISH_CHARS)
        total_chars = len(text)
        
        if total_chars == 0:
            return Language.UNKNOWN
        
        # If ANY Turkish characters present, it's Turkish or Mixed
        # (Turkish characters are never used in pure English)
        if turkish_char_count > 0:
            # Check for common English patterns
            english_words = {'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but'}
            text_lower = text.lower()
            english_word_count = sum(1 for word in english_words if f' {word} ' in text_lower)
            
            # If both Turkish chars and English words, it's mixed
            if english_word_count > 0:
                return Language.MIXED
            
            # Otherwise, it's Turkish
            return Language.TURKISH
        
        # No Turkish characters - check for English
        english_words = {'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but'}
        text_lower = text.lower()
        english_word_count = sum(1 for word in english_words if f' {word} ' in text_lower)
        
        if english_word_count > 0:
            return Language.ENGLISH
        
        return Language.UNKNOWN
    
    def _check_mixed_language(self, text: str) -> Language:
        """Check if text contains mixed Turkish and English."""
        # Count Turkish characters
        turkish_char_count = sum(1 for char in text if char in self.TURKISH_CHARS)
        
        # Count English indicators
        english_words = {'the', 'is', 'are', 'was', 'were', 'and', 'or'}
        text_lower = text.lower()
        english_word_count = sum(1 for word in english_words if f' {word} ' in text_lower)
        
        # If both Turkish and English indicators present, it's mixed
        if turkish_char_count > 0 and english_word_count > 0:
            return Language.MIXED
        
        # Otherwise, use character-based detection
        return self._detect_by_characters(text)
    
    def get_language_config(self, language: Language) -> LanguageConfig:
        """Get language-specific configuration.
        
        Args:
            language: Detected language
            
        Returns:
            LanguageConfig with tokenizer, patterns, etc.
        """
        if language == Language.TURKISH:
            return self._get_turkish_config()
        elif language == Language.ENGLISH:
            return self._get_english_config()
        elif language == Language.MIXED:
            # For mixed language, use combined config
            return self._get_mixed_config()
        else:
            # Unknown language - use universal config
            return self._get_universal_config()
    
    def _get_turkish_config(self) -> LanguageConfig:
        """Get Turkish language configuration."""
        abbreviations = {
            'Dr.', 'Prof.', 'Doç.', 'Yrd.', 'Öğr.', 'Uzm.',
            'örn.', 'vs.', 'vb.', 'vd.', 'bkz.', 'krş.',
            'A.Ş.', 'Ltd.', 'Şti.', 'Inc.', 'Co.', 'Corp.',
            'No.', 'Tel.', 'Fax.', 'Apt.', 'Cad.', 'Sok.',
            'Mah.', 'İl.', 'İlçe.', 'Köy.', 'Bld.', 'Blv.'
        }
        
        question_patterns = {
            # Basic question words
            'ne', 'nasıl', 'neden', 'niçin', 'niye',
            'kim', 'kimi', 'kimin', 'kime', 'kimden',
            'nerede', 'nereye', 'nereden', 'nere',
            'hangi', 'hangisi', 'hangileri',
            'kaç', 'kaçıncı', 'kaçar',
            'ne zaman', 'ne kadar', 'ne için',
            
            # Indirect questions
            'acaba', 'merak ediyorum', 'bilmiyorum',
            'anlamadım', 'anlayamadım',
            
            # Question particles (all forms)
            'mi', 'mı', 'mu', 'mü',
            'midir', 'mıdır', 'mudur', 'müdür',
            'misin', 'mısın', 'musun', 'müsün',
            'miyim', 'mıyım', 'muyum', 'müyüm',
            'miyiz', 'mıyız', 'muyuz', 'müyüz',
            'misiniz', 'mısınız', 'musunuz', 'müsünüz',
            'midirler', 'mıdırlar', 'mudurlar', 'müdürler'
        }
        
        # Turkish sentence pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9"\'([])'
        
        return LanguageConfig(
            language=Language.TURKISH,
            abbreviations=abbreviations,
            question_patterns=question_patterns,
            sentence_pattern=sentence_pattern
        )
    
    def _get_english_config(self) -> LanguageConfig:
        """Get English language configuration."""
        abbreviations = {
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.',
            'Inc.', 'Ltd.', 'Corp.', 'Co.', 'etc.', 'e.g.', 'i.e.',
            'vs.', 'vol.', 'no.', 'p.', 'pp.', 'ed.', 'eds.'
        }
        
        question_patterns = {
            'what', 'when', 'where', 'who', 'whom', 'whose',
            'why', 'how', 'which', 'can', 'could', 'would',
            'should', 'will', 'shall', 'may', 'might', 'must',
            'do', 'does', 'did', 'is', 'are', 'was', 'were',
            'has', 'have', 'had'
        }
        
        # English sentence pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"\'([])'
        
        return LanguageConfig(
            language=Language.ENGLISH,
            abbreviations=abbreviations,
            question_patterns=question_patterns,
            sentence_pattern=sentence_pattern
        )
    
    def _get_mixed_config(self) -> LanguageConfig:
        """Get mixed language configuration."""
        # Combine Turkish and English configs
        turkish_config = self._get_turkish_config()
        english_config = self._get_english_config()
        
        return LanguageConfig(
            language=Language.MIXED,
            abbreviations=turkish_config.abbreviations | english_config.abbreviations,
            question_patterns=turkish_config.question_patterns | english_config.question_patterns,
            sentence_pattern=turkish_config.sentence_pattern  # Use Turkish pattern (more inclusive)
        )
    
    def _get_universal_config(self) -> LanguageConfig:
        """Get universal configuration for unknown languages."""
        # Basic universal abbreviations
        abbreviations = {
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.',
            'Inc.', 'Ltd.', 'Corp.', 'Co.', 'etc.'
        }
        
        # Universal sentence pattern (basic)
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"\'([])'
        
        return LanguageConfig(
            language=Language.UNKNOWN,
            abbreviations=abbreviations,
            question_patterns=set(),
            sentence_pattern=sentence_pattern
        )
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text caching.
        
        Args:
            text: Text to hash
            
        Returns:
            Hash string
        """
        # Use first 500 chars for hash (enough for language detection)
        sample = text[:500]
        return hashlib.sha256(sample.encode()).hexdigest()[:16]
    
    def clear_cache(self):
        """Clear the detection cache."""
        self._cache.clear()
