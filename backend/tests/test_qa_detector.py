"""Property-based tests for Q&A detection and adaptive threshold.

Feature: semantic-chunker-enhancement
Tests for Q&A detection, answer finding, and adaptive threshold calculation.
"""

from hypothesis import given, settings, strategies as st, assume

from app.services.language_detector import Language
from app.services.qa_detector import QADetector
from app.services.adaptive_threshold import AdaptiveThresholdCalculator


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

# Turkish question templates
TURKISH_QUESTION_TEMPLATES = [
    "Ne zaman {}?",
    "Nasıl {}?",
    "Neden {}?",
    "Kim {}?",
    "Nerede {}?",
    "Hangi {}?",
    "Kaç {}?",
    "{} mi?",
    "{} mı?",
    "{} mu?",
    "{} mü?",
    "Acaba {}?",
]

# English question templates
ENGLISH_QUESTION_TEMPLATES = [
    "What is {}?",
    "When did {}?",
    "Where is {}?",
    "Who is {}?",
    "Why did {}?",
    "How does {}?",
    "Can you {}?",
    "Is {} correct?",
    "Are {} available?",
    "Do you {}?",
]


def generate_turkish_question(topic: str) -> str:
    """Generate a Turkish question about a topic."""
    import random
    template = random.choice(TURKISH_QUESTION_TEMPLATES)
    return template.format(topic)


def generate_english_question(topic: str) -> str:
    """Generate an English question about a topic."""
    import random
    template = random.choice(ENGLISH_QUESTION_TEMPLATES)
    return template.format(topic)


# =============================================================================
# Property 4: Comprehensive Turkish Question Detection
# Feature: semantic-chunker-enhancement, Property 4: Comprehensive Turkish Question Detection
# Validates: Requirements 1.2, 9.1
# =============================================================================

class TestTurkishQuestionDetection:
    """Tests for Turkish question detection."""
    
    # Feature: semantic-chunker-enhancement, Property 4
    def test_question_mark_detection(self):
        """Questions ending with ? are detected."""
        detector = QADetector()
        
        questions = [
            "Bu ne?",
            "Nasıl yapılır?",
            "Kim geldi?",
            "Nerede buluşacağız?",
        ]
        
        for q in questions:
            assert detector.is_question(q, Language.TURKISH), f"Failed: {q}"
    
    # Feature: semantic-chunker-enhancement, Property 4
    def test_turkish_question_words_detected(self):
        """All Turkish question words are detected."""
        detector = QADetector()
        
        question_words = [
            'ne', 'nasıl', 'neden', 'niçin', 'niye',
            'kim', 'nerede', 'hangi', 'kaç',
        ]
        
        for word in question_words:
            sentence = f"{word.capitalize()} bu durum?"
            assert detector.is_question(sentence, Language.TURKISH), \
                f"Failed to detect: {sentence}"
    
    # Feature: semantic-chunker-enhancement, Property 4
    def test_turkish_question_particles_detected(self):
        """Turkish question particles (mi/mı/mu/mü) are detected."""
        detector = QADetector()
        
        particles = ['mi', 'mı', 'mu', 'mü']
        
        for particle in particles:
            sentence = f"Gelecek {particle}?"
            assert detector.is_question(sentence, Language.TURKISH), \
                f"Failed to detect particle: {particle}"
    
    # Feature: semantic-chunker-enhancement, Property 4
    @given(st.text(min_size=5, max_size=50))
    @settings(max_examples=100, deadline=None)
    def test_question_mark_always_detected(self, text: str):
        """Property: Any text ending with ? is detected as question."""
        assume(text.strip())
        
        detector = QADetector()
        question = text.strip() + "?"
        
        assert detector.is_question(question, Language.TURKISH)
    
    # Feature: semantic-chunker-enhancement, Property 4
    def test_all_question_particle_forms(self):
        """All vowel harmony forms of question particles are detected."""
        detector = QADetector()
        
        # Test all particle forms
        all_particles = list(detector.TURKISH_QUESTION_PARTICLES)
        
        for particle in all_particles:
            sentence = f"Bu doğru {particle}?"
            result = detector.is_question(sentence, Language.TURKISH)
            assert result, f"Failed to detect particle: {particle}"
    
    # Feature: semantic-chunker-enhancement, Property 4
    def test_question_patterns_count(self):
        """Turkish has 50+ question patterns."""
        detector = QADetector()
        count = detector.get_question_patterns_count(Language.TURKISH)
        
        assert count >= 50, f"Expected 50+ patterns, got {count}"


# =============================================================================
# Property 19: Answer Detection After Question
# Feature: semantic-chunker-enhancement, Property 19: Answer Detection After Question
# Validates: Requirements 9.2
# =============================================================================

class TestAnswerDetection:
    """Tests for answer detection."""
    
    # Feature: semantic-chunker-enhancement, Property 19
    def test_answer_found_in_next_sentence(self):
        """Answer is found in the sentence immediately after question."""
        detector = QADetector()
        
        question = "Bu ne?"
        following = ["Bu bir kitaptır.", "Başka bir şey.", "Son cümle."]
        
        result = detector.find_answer(question, following)
        
        assert result is not None
        answer, idx, confidence = result
        assert idx == 0  # First sentence
        assert "kitap" in answer
    
    # Feature: semantic-chunker-enhancement, Property 19
    def test_skips_questions_as_answers(self):
        """Questions are not selected as answers."""
        detector = QADetector()
        
        question = "Bu ne?"
        following = ["Bu da ne?", "Bu bir kitaptır.", "Son cümle."]
        
        result = detector.find_answer(question, following)
        
        assert result is not None
        answer, idx, confidence = result
        assert idx == 1  # Second sentence (first non-question)
        assert not answer.endswith('?')
    
    # Feature: semantic-chunker-enhancement, Property 19
    def test_max_distance_respected(self):
        """Only looks at first 3 sentences for answer."""
        detector = QADetector()
        
        question = "Bu ne?"
        following = [
            "Soru mu?",  # Question, skip
            "Başka soru?",  # Question, skip
            "Yine soru?",  # Question, skip
            "Bu cevaptır.",  # Beyond max distance
        ]
        
        result = detector.find_answer(question, following)
        
        # Should not find answer (all within range are questions)
        assert result is None
    
    # Feature: semantic-chunker-enhancement, Property 19
    @given(st.lists(st.text(min_size=10, max_size=100), min_size=1, max_size=5))
    @settings(max_examples=100, deadline=None)
    def test_answer_detection_attempts(self, following: list):
        """Property: Answer detection is attempted for all following sentences."""
        assume(all(s.strip() for s in following))
        
        detector = QADetector()
        question = "Bu ne?"
        
        # Clean up following sentences
        clean_following = [s.strip() for s in following if s.strip()]
        
        # Should attempt to find answer (may or may not succeed)
        result = detector.find_answer(question, clean_following)
        
        # If result found, it should be from within MAX_ANSWER_DISTANCE
        if result:
            _, idx, _ = result
            assert idx < detector.MAX_ANSWER_DISTANCE


# =============================================================================
# Property 5: Q&A Pair Preservation
# Feature: semantic-chunker-enhancement, Property 5: Q&A Pair Preservation
# Validates: Requirements 1.5, 9.2, 9.3, 9.4
# =============================================================================

class TestQAPairPreservation:
    """Tests for Q&A pair detection and merging."""
    
    # Feature: semantic-chunker-enhancement, Property 5
    def test_qa_pairs_detected(self):
        """Q&A pairs are correctly detected."""
        detector = QADetector()
        
        sentences = [
            "Bu ne?",
            "Bu bir kitaptır.",
            "Başka bir cümle.",
            "Nasıl yapılır?",
            "Şöyle yapılır.",
        ]
        
        pairs = detector.detect_qa_pairs(sentences, Language.TURKISH)
        
        assert len(pairs) == 2
        assert pairs[0].question == "Bu ne?"
        assert pairs[0].answer == "Bu bir kitaptır."
        assert pairs[1].question == "Nasıl yapılır?"
        assert pairs[1].answer == "Şöyle yapılır."
    
    # Feature: semantic-chunker-enhancement, Property 5
    def test_qa_pairs_merged(self):
        """Q&A pairs are merged into single sentences."""
        detector = QADetector()
        
        sentences = [
            "Giriş cümlesi.",
            "Bu ne?",
            "Bu bir kitaptır.",
            "Son cümle.",
        ]
        
        merged = detector.merge_qa_pairs(sentences, Language.TURKISH)
        
        assert len(merged) == 3  # 4 sentences -> 3 (Q&A merged)
        assert "Bu ne?" in merged[1]
        assert "Bu bir kitaptır." in merged[1]
    
    # Feature: semantic-chunker-enhancement, Property 5
    def test_non_qa_sentences_preserved(self):
        """Non-Q&A sentences are preserved unchanged."""
        detector = QADetector()
        
        sentences = [
            "Bu bir cümle.",
            "Bu başka bir cümle.",
            "Son cümle.",
        ]
        
        merged = detector.merge_qa_pairs(sentences, Language.TURKISH)
        
        assert merged == sentences  # No changes
    
    # Feature: semantic-chunker-enhancement, Property 5
    def test_qa_pair_confidence(self):
        """Q&A pairs have confidence scores."""
        detector = QADetector()
        
        sentences = [
            "Bu ne?",
            "Bu bir kitaptır.",
        ]
        
        pairs = detector.detect_qa_pairs(sentences, Language.TURKISH)
        
        assert len(pairs) == 1
        assert 0 <= pairs[0].confidence <= 1


# =============================================================================
# Property 17: Text Characteristics Analysis
# Feature: semantic-chunker-enhancement, Property 17: Text Characteristics Analysis
# Validates: Requirements 8.1
# =============================================================================

class TestTextCharacteristicsAnalysis:
    """Tests for text characteristics analysis."""
    
    # Feature: semantic-chunker-enhancement, Property 17
    def test_basic_analysis(self):
        """Basic text characteristics are calculated."""
        calculator = AdaptiveThresholdCalculator()
        
        text = "Bu bir cümle. Bu başka bir cümle. Son cümle."
        
        chars = calculator.analyze_text(text)
        
        assert chars.length > 0
        assert chars.sentence_count == 3
        assert chars.word_count > 0
        assert chars.unique_word_count > 0
        assert 0 <= chars.vocabulary_diversity <= 1
        assert chars.avg_sentence_length > 0
    
    # Feature: semantic-chunker-enhancement, Property 17
    def test_vocabulary_diversity_calculation(self):
        """Vocabulary diversity is correctly calculated."""
        calculator = AdaptiveThresholdCalculator()
        
        # Low diversity (repeated words)
        low_div_text = "bu bu bu bu bu bu bu bu"
        low_chars = calculator.analyze_text(low_div_text)
        
        # High diversity (unique words)
        high_div_text = "bir iki üç dört beş altı yedi sekiz"
        high_chars = calculator.analyze_text(high_div_text)
        
        assert low_chars.vocabulary_diversity < high_chars.vocabulary_diversity
    
    # Feature: semantic-chunker-enhancement, Property 17
    def test_question_detection_in_analysis(self):
        """Questions are detected during analysis."""
        calculator = AdaptiveThresholdCalculator()
        
        text_with_questions = "Bu ne? Nasıl yapılır? Normal cümle."
        text_without_questions = "Bu bir cümle. Başka bir cümle."
        
        with_q = calculator.analyze_text(text_with_questions)
        without_q = calculator.analyze_text(text_without_questions)
        
        assert with_q.has_questions is True
        assert with_q.question_count == 2
        assert without_q.has_questions is False
        assert without_q.question_count == 0
    
    # Feature: semantic-chunker-enhancement, Property 17
    @given(st.text(min_size=10, max_size=500))
    @settings(max_examples=100, deadline=None)
    def test_analysis_always_returns_valid_characteristics(self, text: str):
        """Property: Analysis always returns valid characteristics."""
        calculator = AdaptiveThresholdCalculator()
        
        chars = calculator.analyze_text(text)
        
        assert chars.length >= 0
        assert chars.sentence_count >= 0
        assert chars.word_count >= 0
        assert chars.unique_word_count >= 0
        assert 0 <= chars.vocabulary_diversity <= 1
        assert chars.avg_sentence_length >= 0
    
    # Feature: semantic-chunker-enhancement, Property 17
    def test_empty_text_handling(self):
        """Empty text returns zero characteristics."""
        calculator = AdaptiveThresholdCalculator()

        chars = calculator.analyze_text("")

        assert chars.length == 0
        assert chars.sentence_count == 0
        assert chars.word_count == 0
        assert abs(chars.vocabulary_diversity) < 0.001


# =============================================================================
# Property 18: Adaptive Threshold Response to Diversity
# Feature: semantic-chunker-enhancement, Property 18
# Validates: Requirements 8.2, 8.3
# =============================================================================

class TestAdaptiveThreshold:
    """Tests for adaptive threshold calculation."""

    # Feature: semantic-chunker-enhancement, Property 18
    def test_high_diversity_lowers_threshold(self):
        """High vocabulary diversity results in lower threshold."""
        calculator = AdaptiveThresholdCalculator(base_threshold=0.5)

        # High diversity text (many unique words)
        high_div = "bir iki üç dört beş altı yedi sekiz dokuz on"
        chars = calculator.analyze_text(high_div)
        threshold = calculator.calculate_threshold(chars)

        # Should be lower than base (0.5 * 0.8 = 0.4)
        assert threshold < calculator.base_threshold

    # Feature: semantic-chunker-enhancement, Property 18
    def test_low_diversity_raises_threshold(self):
        """Low vocabulary diversity results in higher threshold."""
        calculator = AdaptiveThresholdCalculator(base_threshold=0.5)

        # Low diversity text (repeated words)
        low_div = "bu bu bu bu bu bu bu bu bu bu bu bu bu bu bu"
        chars = calculator.analyze_text(low_div)
        threshold = calculator.calculate_threshold(chars)

        # Should be higher than base (0.5 * 1.2 = 0.6)
        assert threshold > calculator.base_threshold

    # Feature: semantic-chunker-enhancement, Property 18
    def test_threshold_bounds(self):
        """Threshold is always within valid bounds."""
        calculator = AdaptiveThresholdCalculator()

        texts = [
            "",  # Empty
            "a",  # Very short
            "bu " * 1000,  # Very repetitive
            " ".join(f"kelime{i}" for i in range(100)),  # Very diverse
        ]

        for text in texts:
            chars = calculator.analyze_text(text)
            if chars.word_count > 0:
                threshold = calculator.calculate_threshold(chars)
                assert 0.3 <= threshold <= 0.9

    # Feature: semantic-chunker-enhancement, Property 18
    @given(st.text(min_size=20, max_size=500))
    @settings(max_examples=100, deadline=None)
    def test_threshold_always_in_bounds(self, text: str):
        """Property: Threshold is always within bounds for any text."""
        calculator = AdaptiveThresholdCalculator()

        chars = calculator.analyze_text(text)
        if chars.word_count > 0:
            threshold = calculator.calculate_threshold(chars)
            assert 0.3 <= threshold <= 0.9

    # Feature: semantic-chunker-enhancement, Property 18
    def test_recommendation_includes_reasoning(self):
        """Threshold recommendation includes reasoning."""
        calculator = AdaptiveThresholdCalculator()

        text = "Bu bir test cümlesi. Başka bir cümle daha."
        rec = calculator.recommend_threshold(text)

        assert rec.recommended_threshold > 0
        assert rec.reasoning
        assert len(rec.reasoning) > 10

    # Feature: semantic-chunker-enhancement, Property 18
    def test_confidence_increases_with_text_size(self):
        """Confidence increases with more sentences."""
        calculator = AdaptiveThresholdCalculator()

        short = "Kısa metin."
        medium = "Bir cümle. İki cümle. Üç cümle. Dört cümle. Beş cümle."
        long_text = " ".join([f"Cümle {i}." for i in range(15)])

        short_rec = calculator.recommend_threshold(short)
        medium_rec = calculator.recommend_threshold(medium)
        long_rec = calculator.recommend_threshold(long_text)

        assert short_rec.confidence <= medium_rec.confidence
        assert medium_rec.confidence <= long_rec.confidence


# =============================================================================
# Property 20: Semantic Q&A Confirmation
# Feature: semantic-chunker-enhancement, Property 20
# Validates: Requirements 9.3
# =============================================================================

class TestSemanticQAConfirmation:
    """Tests for semantic Q&A confirmation."""

    # Feature: semantic-chunker-enhancement, Property 20
    def test_semantic_matching_with_embeddings(self):
        """Semantic matching works when embeddings provided."""
        detector = QADetector(similarity_threshold=0.6)

        question = "Bu ne?"
        candidates = ["Bu bir kitaptır.", "Hava güzel."]

        # Mock embeddings (similar vectors for Q and first answer)
        q_emb = [1.0, 0.0, 0.0]
        c_embs = [
            [0.9, 0.1, 0.0],  # Similar to question
            [0.0, 1.0, 0.0],  # Different
        ]

        result = detector.find_answer(question, candidates, c_embs, q_emb)

        assert result is not None
        answer, idx, confidence = result
        assert idx == 0  # First candidate (most similar)
        assert confidence > 0.6

    # Feature: semantic-chunker-enhancement, Property 20
    def test_falls_back_to_heuristic_without_embeddings(self):
        """Falls back to heuristic when no embeddings."""
        detector = QADetector()

        question = "Bu ne?"
        candidates = ["Bu bir kitaptır.", "Başka cümle."]

        result = detector.find_answer(question, candidates)

        assert result is not None
        answer, idx, _ = result
        assert idx == 0  # First non-question

    # Feature: semantic-chunker-enhancement, Property 20
    def test_similarity_threshold_respected(self):
        """Low similarity results fall back to heuristic."""
        detector = QADetector(similarity_threshold=0.9)

        question = "Bu ne?"
        candidates = ["Bu bir kitaptır."]

        # Very different embeddings
        q_emb = [1.0, 0.0, 0.0]
        c_embs = [[0.0, 1.0, 0.0]]  # Orthogonal

        result = detector.find_answer(question, candidates, c_embs, q_emb)

        # Should fall back to heuristic since similarity < 0.9
        assert result is not None


# =============================================================================
# Property 5 Extended: Nested Questions
# Feature: semantic-chunker-enhancement, Property 5
# Validates: Requirements 9.5
# =============================================================================

class TestNestedQuestions:
    """Tests for nested question handling."""

    # Feature: semantic-chunker-enhancement, Property 5
    def test_nested_question_in_answer(self):
        """Questions within answers are handled."""
        detector = QADetector()

        sentences = [
            "Bu ne?",
            "Bu bir kitap. Peki sen ne okuyorsun?",
            "Ben roman okuyorum.",
        ]

        pairs = detector.detect_qa_pairs(sentences, Language.TURKISH)

        # First Q&A pair should be detected
        assert len(pairs) >= 1
        assert pairs[0].question == "Bu ne?"

    # Feature: semantic-chunker-enhancement, Property 5
    def test_consecutive_questions(self):
        """Consecutive questions are handled separately."""
        detector = QADetector()

        sentences = [
            "Bu ne?",
            "Nasıl yapılır?",
            "Bu bir kitaptır.",
        ]

        pairs = detector.detect_qa_pairs(sentences, Language.TURKISH)

        # Second question should find the answer
        assert any(p.question == "Nasıl yapılır?" for p in pairs)



# =============================================================================
# Integration Tests for Q&A and Adaptive Threshold with SemanticChunker
# Feature: semantic-chunker-enhancement
# Validates: Requirements 1.5, 8.1, 8.2, 9.1, 9.2, 9.3, 9.4
# =============================================================================

class TestSemanticChunkerQAIntegration:
    """Integration tests for Q&A detection in SemanticChunker."""

    # Feature: semantic-chunker-enhancement, Property 5
    def test_chunker_with_qa_detection_enabled(self):
        """SemanticChunker with Q&A detection enabled."""
        from app.services.chunker import SemanticChunker

        chunker = SemanticChunker(
            use_provider_manager=False,
            enable_cache=False,
            enable_qa_detection=True,
            enable_adaptive_threshold=False
        )

        # Verify Q&A detector is initialized lazily
        assert chunker._qa_detector is None
        chunker._ensure_qa_detector()
        assert chunker._qa_detector is not None

    # Feature: semantic-chunker-enhancement, Property 5
    def test_chunker_with_qa_detection_disabled(self):
        """SemanticChunker with Q&A detection disabled."""
        from app.services.chunker import SemanticChunker

        chunker = SemanticChunker(
            use_provider_manager=False,
            enable_cache=False,
            enable_qa_detection=False,
            enable_adaptive_threshold=False
        )

        # Q&A detector should not be initialized
        assert chunker._qa_detector is None
        assert chunker._enable_qa_detection is False


class TestSemanticChunkerAdaptiveThresholdIntegration:
    """Integration tests for adaptive threshold in SemanticChunker."""

    # Feature: semantic-chunker-enhancement, Property 18
    def test_chunker_with_adaptive_threshold_enabled(self):
        """SemanticChunker with adaptive threshold enabled."""
        from app.services.chunker import SemanticChunker

        chunker = SemanticChunker(
            use_provider_manager=False,
            enable_cache=False,
            enable_qa_detection=False,
            enable_adaptive_threshold=True
        )

        # Verify threshold calculator is initialized lazily
        assert chunker._threshold_calculator is None
        chunker._ensure_threshold_calculator()
        assert chunker._threshold_calculator is not None

    # Feature: semantic-chunker-enhancement, Property 18
    def test_chunker_with_adaptive_threshold_disabled(self):
        """SemanticChunker with adaptive threshold disabled."""
        from app.services.chunker import SemanticChunker

        chunker = SemanticChunker(
            use_provider_manager=False,
            enable_cache=False,
            enable_qa_detection=False,
            enable_adaptive_threshold=False
        )

        # Threshold calculator should not be initialized
        assert chunker._threshold_calculator is None
        assert chunker._enable_adaptive_threshold is False

    # Feature: semantic-chunker-enhancement, Property 18
    def test_chunker_uses_default_threshold_when_disabled(self):
        """Chunker uses 0.5 default when adaptive threshold disabled."""
        from app.services.chunker import SemanticChunker

        chunker = SemanticChunker(
            use_provider_manager=False,
            enable_cache=False,
            enable_qa_detection=False,
            enable_adaptive_threshold=False
        )

        # When adaptive is disabled and no threshold provided,
        # should use 0.5 default
        assert chunker._enable_adaptive_threshold is False


class TestFullIntegration:
    """Full integration tests with all features enabled."""

    # Feature: semantic-chunker-enhancement
    def test_all_features_enabled(self):
        """SemanticChunker with all new features enabled."""
        from app.services.chunker import SemanticChunker

        chunker = SemanticChunker(
            use_provider_manager=False,
            enable_cache=False,
            enable_qa_detection=True,
            enable_adaptive_threshold=True
        )

        assert chunker._enable_qa_detection is True
        assert chunker._enable_adaptive_threshold is True

    # Feature: semantic-chunker-enhancement
    def test_qa_detector_initialization(self):
        """Q&A detector is properly initialized."""
        from app.services.chunker import SemanticChunker
        from app.services.qa_detector import QADetector

        chunker = SemanticChunker(enable_qa_detection=True)
        chunker._ensure_qa_detector()

        assert isinstance(chunker._qa_detector, QADetector)

    # Feature: semantic-chunker-enhancement
    def test_threshold_calculator_initialization(self):
        """Threshold calculator is properly initialized."""
        from app.services.chunker import SemanticChunker
        from app.services.adaptive_threshold import AdaptiveThresholdCalculator

        chunker = SemanticChunker(enable_adaptive_threshold=True)
        chunker._ensure_threshold_calculator()

        assert isinstance(chunker._threshold_calculator, AdaptiveThresholdCalculator)
