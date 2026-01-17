"""Enhanced sentence tokenization with multi-language support."""

import re
import logging
from typing import List, Set, Dict
from dataclasses import dataclass

from app.services.language_detector import Language

logger = logging.getLogger(__name__)


@dataclass
class TokenizationResult:
    """Result of sentence tokenization."""
    sentences: List[str]
    original_text: str
    language: Language
    sentence_count: int


class EnhancedSentenceTokenizer:
    """Advanced sentence tokenization with multi-language support.
    
    Features:
    - Turkish and English abbreviation handling
    - Decimal number preservation
    - Quoted text handling
    - URL and email preservation
    - Language-aware sentence splitting patterns
    """
    
    # Turkish abbreviations (30+)
    TURKISH_ABBREVIATIONS = {
        'Dr.', 'Prof.', 'Doç.', 'Yrd.', 'Öğr.', 'Uzm.',
        'örn.', 'vs.', 'vb.', 'vd.', 'bkz.', 'krş.',
        'A.Ş.', 'Ltd.', 'Şti.', 'Inc.', 'Co.', 'Corp.',
        'No.', 'Tel.', 'Fax.', 'Apt.', 'Cad.', 'Sok.',
        'Mah.', 'İl.', 'İlçe.', 'Köy.', 'Bld.', 'Blv.'
    }
    
    # English abbreviations (20+)
    ENGLISH_ABBREVIATIONS = {
        'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.',
        'Inc.', 'Ltd.', 'Corp.', 'Co.', 'etc.', 'e.g.', 'i.e.',
        'vs.', 'vol.', 'no.', 'p.', 'pp.', 'ed.', 'eds.'
    }
    
    # Placeholder patterns (no trailing space - preserve original spacing)
    ABBREV_PLACEHOLDER = "___ABBREV_{}___"
    DECIMAL_PLACEHOLDER = "___DECIMAL_{}___"
    QUOTE_PLACEHOLDER = "___QUOTE_{}___"
    URL_PLACEHOLDER = "___URL_{}___"
    EMAIL_PLACEHOLDER = "___EMAIL_{}___"
    
    def __init__(self):
        """Initialize the sentence tokenizer."""
        self._placeholder_map: Dict[str, str] = {}
        self._placeholder_counter = 0
    
    def tokenize(
        self,
        text: str,
        language: Language,
        sentence_pattern: str = None
    ) -> List[str]:
        """Split text into sentences with language-aware processing.
        
        Args:
            text: Input text to tokenize
            language: Detected language
            sentence_pattern: Optional custom sentence pattern
            
        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []
        
        # Reset placeholder tracking
        self._placeholder_map = {}
        self._placeholder_counter = 0
        
        # Get abbreviations for the language
        abbreviations = self._get_abbreviations(language)
        
        # Step 1: Protect special patterns with placeholders (NOT abbreviations)
        protected_text = text
        protected_text = self._handle_urls_emails(protected_text)
        protected_text = self._handle_quoted_text(protected_text)
        protected_text = self._handle_decimals(protected_text)
        # Note: We DON'T replace abbreviations - we handle them in splitting
        
        # Step 2: Split into sentences (abbreviation-aware)
        if sentence_pattern:
            pattern = sentence_pattern
        else:
            pattern = self._get_sentence_pattern(language)
        
        sentences = self._split_by_pattern_with_abbreviations(
            protected_text, pattern, abbreviations
        )
        
        # Step 3: Restore original patterns
        restored_sentences = []
        for sentence in sentences:
            restored = self._restore_placeholders(sentence)
            if restored.strip():
                restored_sentences.append(restored.strip())
        
        return restored_sentences
    
    def _get_abbreviations(self, language: Language) -> Set[str]:
        """Get abbreviations for the specified language."""
        if language == Language.TURKISH:
            return self.TURKISH_ABBREVIATIONS
        elif language == Language.ENGLISH:
            return self.ENGLISH_ABBREVIATIONS
        elif language == Language.MIXED:
            return self.TURKISH_ABBREVIATIONS | self.ENGLISH_ABBREVIATIONS
        else:
            # Unknown language - use basic set
            return {'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'etc.'}
    
    def _get_sentence_pattern(self, language: Language) -> str:
        """Get sentence splitting pattern for the language."""
        if language == Language.TURKISH or language == Language.MIXED:
            # Turkish pattern - includes Turkish capital letters
            return r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9"\'([])'
        else:
            # English/Universal pattern
            return r'(?<=[.!?])\s+(?=[A-Z0-9"\'([])'
    
    def _handle_abbreviations(
        self,
        text: str,
        abbreviations: Set[str]
    ) -> str:
        """Replace abbreviations with placeholders before splitting.
        
        Args:
            text: Input text
            abbreviations: Set of abbreviations to protect
            
        Returns:
            Text with abbreviations replaced by placeholders
        """
        result = text
        
        for abbrev in abbreviations:
            if abbrev in result:
                # Create placeholder - keep the period but mark it specially
                # Replace "Dr." with "Dr___ABBREV_0___" (period removed, added to placeholder)
                placeholder = self._create_placeholder(
                    self.ABBREV_PLACEHOLDER,
                    abbrev
                )
                # Replace the abbreviation but keep any trailing space
                result = result.replace(abbrev + ' ', placeholder + ' ')
                result = result.replace(abbrev, placeholder)
        
        return result
    
    def _handle_decimals(self, text: str) -> str:
        """Protect decimal numbers from splitting.
        
        Args:
            text: Input text
            
        Returns:
            Text with decimals replaced by placeholders
        """
        # Pattern for decimal numbers: digits.digits
        decimal_pattern = r'\b(\d+\.\d+)\b'
        
        def replace_decimal(match):
            decimal = match.group(1)
            return self._create_placeholder(self.DECIMAL_PLACEHOLDER, decimal)
        
        return re.sub(decimal_pattern, replace_decimal, text)
    
    def _handle_quoted_text(self, text: str) -> str:
        """Protect quoted text from splitting.
        
        Args:
            text: Input text
            
        Returns:
            Text with quoted segments replaced by placeholders
        """
        result = text
        
        # Handle different quote styles
        quote_patterns = [
            (r'"([^"]+)"', '"', '"'),  # Double quotes
            (r"'([^']+)'", "'", "'"),  # Single quotes
            (r'«([^»]+)»', '«', '»'),  # French quotes
        ]
        
        for pattern, open_q, close_q in quote_patterns:
            def replace_quote(match):
                quoted = match.group(0)  # Full match including quotes
                content = match.group(1)  # Content without quotes
                
                # Check if content ends with sentence-ending punctuation
                if content.endswith(('.', '!', '?')):
                    # Keep the punctuation outside the placeholder
                    punct = content[-1]
                    content_without_punct = content[:-1]
                    quoted_without_punct = open_q + content_without_punct + close_q
                    placeholder = self._create_placeholder(
                        self.QUOTE_PLACEHOLDER,
                        quoted_without_punct
                    )
                    return placeholder + punct
                else:
                    # No sentence-ending punctuation, replace as-is
                    return self._create_placeholder(
                        self.QUOTE_PLACEHOLDER,
                        quoted
                    )
            
            result = re.sub(pattern, replace_quote, result)
        
        return result
    
    def _handle_urls_emails(self, text: str) -> str:
        """Protect URLs and email addresses from splitting.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs and emails replaced by placeholders
        """
        result = text
        
        # URL pattern (simplified but effective)
        url_pattern = r'https?://[^\s]+'
        
        def replace_url(match):
            url = match.group(0)
            return self._create_placeholder(self.URL_PLACEHOLDER, url)
        
        result = re.sub(url_pattern, replace_url, result)
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        def replace_email(match):
            email = match.group(0)
            return self._create_placeholder(self.EMAIL_PLACEHOLDER, email)
        
        result = re.sub(email_pattern, replace_email, result)
        
        return result
    
    def _split_by_pattern(self, text: str, pattern: str) -> List[str]:
        """Split text using the specified regex pattern.
        
        Args:
            text: Text to split
            pattern: Regex pattern for sentence boundaries
            
        Returns:
            List of sentence segments
        """
        # Try regex split
        sentences = re.split(pattern, text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no splits found, try splitting on newlines
        if len(sentences) <= 1 and len(text) > 100:
            sentences = text.split('\n')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # If still single chunk, return as-is
        if not sentences:
            return [text] if text.strip() else []
        
        return sentences
    
    def _split_by_pattern_with_abbreviations(
        self,
        text: str,
        pattern: str,
        abbreviations: Set[str]
    ) -> List[str]:
        """Split text by pattern, but don't split after abbreviations.
        
        Args:
            text: Text to split
            pattern: Regex pattern for sentence boundaries
            abbreviations: Set of abbreviations to avoid splitting after
            
        Returns:
            List of sentence segments
        """
        # First, do a normal split
        sentences = self._split_by_pattern(text, pattern)
        
        # Now merge sentences that were incorrectly split after abbreviations
        merged_sentences = []
        i = 0
        while i < len(sentences):
            current = sentences[i]
            
            # Check if current sentence ends with an abbreviation
            ends_with_abbrev = False
            for abbrev in abbreviations:
                if current.rstrip().endswith(abbrev):
                    ends_with_abbrev = True
                    break
            
            # If it ends with abbreviation and there's a next sentence, merge them
            if ends_with_abbrev and i + 1 < len(sentences):
                merged = current + ' ' + sentences[i + 1]
                merged_sentences.append(merged)
                i += 2  # Skip the next sentence since we merged it
            else:
                merged_sentences.append(current)
                i += 1
        
        return merged_sentences
    
    def _create_placeholder(self, template: str, original: str) -> str:
        """Create a unique placeholder for the original text.
        
        Args:
            template: Placeholder template
            original: Original text to preserve
            
        Returns:
            Unique placeholder string
        """
        placeholder = template.format(self._placeholder_counter)
        self._placeholder_map[placeholder] = original
        self._placeholder_counter += 1
        return placeholder
    
    def _restore_placeholders(self, text: str) -> str:
        """Restore original text from placeholders.
        
        Args:
            text: Text with placeholders
            
        Returns:
            Text with original content restored
        """
        result = text
        
        # Replace all placeholders with original text
        for placeholder, original in self._placeholder_map.items():
            result = result.replace(placeholder, original)
        
        return result
    
    def tokenize_with_metadata(
        self,
        text: str,
        language: Language,
        sentence_pattern: str = None
    ) -> TokenizationResult:
        """Tokenize text and return result with metadata.
        
        Args:
            text: Input text
            language: Detected language
            sentence_pattern: Optional custom pattern
            
        Returns:
            TokenizationResult with sentences and metadata
        """
        sentences = self.tokenize(text, language, sentence_pattern)
        
        return TokenizationResult(
            sentences=sentences,
            original_text=text,
            language=language,
            sentence_count=len(sentences)
        )


class SentenceTokenizerFactory:
    """Factory for creating sentence tokenizers with different configurations."""
    
    @staticmethod
    def create_default() -> EnhancedSentenceTokenizer:
        """Create tokenizer with default configuration."""
        return EnhancedSentenceTokenizer()
    
    @staticmethod
    def create_for_language(language: Language) -> EnhancedSentenceTokenizer:
        """Create tokenizer optimized for specific language.
        
        Args:
            language: Target language
            
        Returns:
            Configured tokenizer
        """
        # For now, all languages use the same tokenizer
        # In the future, we could have language-specific implementations
        return EnhancedSentenceTokenizer()
