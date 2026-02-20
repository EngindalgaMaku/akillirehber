"""
Giskard Service for RAG Testing

This module provides integration with Giskard library for testing RAG
systems, including hallucination detection, relevance testing, and
Turkish language support.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class GiskardTestConfig:
    """Configuration for Giskard RAG tests."""
    model_name: str
    provider: str
    temperature: float = 0.7
    max_tokens: int = 1000
    num_test_questions: int = 20
    num_irrelevant_questions: int = 50
    language: str = "tr"  # "tr" for Turkish, "en" for English


class TurkishPrompts:
    """Turkish prompt templates for Giskard testing."""

    # System prompt for test generation
    SYSTEM_PROMPT = """Sen Türkçe dilinde uzmanlaşmış bir RAG sistemi
    test uzmanısın.

GÖREVİN:
- Sadece TÜRKÇE sorular üret
- Soruların akademik düzeyde ve anlaşılır olmasını sağla
- Halüsinasyon tespiti için alakasız sorular üret
- Her sorunun bağlamını (konu, zorluk seviyesi) belirt

KURALLAR:
1. Tüm sorular TÜRKÇE olmalı
2. İngilizce kelime kullanma (terimler hariç)
3. Sorular net ve spesifik olmalı
4. Alakasız sorular gerçekten alakasız olmalı
5. Her soru için beklenen cevap tipini belirt"""

    # Prompt for relevant question generation
    RELEVANT_QUESTION_PROMPT = """Aşağıdaki ders notlarını analiz et ve bu
    notlara dayalı olarak TAM OLARAK {num_questions} adet test sorusu üret.

Ders Notları:
{document_content}

Soru Üretim Kuralları:
- Sadece notlarda BULUNAN bilgilere dayalı sorular üret
- Notlarda bulunmayan bir bilgiyi sormak yasaktır
- Her sorunun zorluk seviyesini belirt (Kolay/Orta/Zor)
- Her sorunun beklenen cevabını da yaz
- Çıktıyı SADECE JSON formatında ver (başka metin yazma)

JSON Formatı:
{{
  "questions": [
    {{
      "question": "Soru metni",
      "difficulty": "Kolay/Orta/Zor",
      "expected_answer": "Beklenen cevap",
      "topic": "Konu başlığı"
    }}
  ]
}}"""

    # Prompt for irrelevant question generation
    IRRELEVANT_QUESTION_PROMPT = """Aşağıdaki ders notlarından TAMAMEN
    BAĞIMSIZ TAM OLARAK {num_questions} adet soru üret.

Ders Notları:
{document_content}

Soru Üretim Kuralları:
- Sorular notlarda YOK olan konular hakkında olmalı
- Sorular tamamen alakasız ve uydurma konular içermeli
- Öğrenci bu soruları sorduğunda model "Bilmiyorum" demeli
- Her sorunun beklenen cevabı "Bilmiyorum" olmalı
- Çıktıyı SADECE JSON formatında ver (başka metin yazma)

JSON Formatı:
{{
  "questions": [
    {{
      "question": "Soru metni",
      "expected_answer": "Bilmiyorum",
      "reason": "Neden alakasız olduğu açıklaması"
    }}
  ]
}}"""

    # Agent description for Giskard
    AGENT_DESCRIPTION = """AkıllıRehber RAG sistemi, Türkçe ders notlarına
    dayalı olarak öğrencilere akademik destek sağlayan bir yapay zeka
    asistanıdır.

SİSTEM ÖZELLİKLERİ:
- Sadece yüklenmiş ders notlarından bilgi verir
- Notlarda olmayan sorular için "Bilmiyorum" der
- Türkçe dilinde yanıt verir
- Akademik dil kullanır
- Halüsinasyon yapmamaya çalışır

TEST KURALLARI:
1. Modelin sadece notlarda bulunan bilgilere dayalı cevap verip
   vermediğini test et
2. Alakasız sorulara "Bilmiyorum" cevabı verip vermediğini kontrol et
3. Cevapların Türkçe dilinde olup olmadığını doğrula
4. Halüsinasyon tespiti için modelin uydurma bilgi verip
   vermediğini kontrol et"""

    # Evaluation prompts for Turkish
    EVALUATION_PROMPTS = {
        "relevance": """
Cevabın Relevansını Değerlendir (1-5 puan):

Kriterler:
- 5 puan: Cevap soruya tam ve doğru cevap veriyor
- 4 puan: Cevap soruyu cevaplıyor ama küçük eksiklikler var
- 3 puan: Cevap kısmen doğru ama önemli bilgiler eksik
- 2 puan: Cevap soruyla ilgili ama yanlış bilgi içeriyor
- 1 puan: Cevap soruyla tamamen alakasız

Soru: {question}
Cevap: {answer}
Puan: """,

        "hallucination": """
Halüsinasyon Kontrolü (Evet/Hayır):

Bu cevapta uydurma veya yanlış bilgi var mı?

Kriterler:
- Evet: Cevapta gerçek olmayan veya doğrulanamayan bilgi var
- Hayır: Cevap doğru ve doğrulanabilir

Soru: {question}
Cevap: {answer}
Beklenen Cevap: {expected_answer}
Sonuç: """,

        "language": """
Dil Kontrolü (Türkçe/İngilizce/Karışık):

Cevap hangi dilde yazılmış?

Kriterler:
- Türkçe: Cevap tamamen Türkçe
- İngilizce: Cevap tamamen İngilizce
- Karışık: Cevapta her iki dil de kullanılmış

Cevap: {answer}
Sonuç: """
    }


class EnglishPrompts:
    SYSTEM_PROMPT = """You are an expert RAG system testing specialist.

YOUR TASK:
- Generate questions in the requested language
- Keep questions clear and academic
- Generate irrelevant questions for hallucination detection
- Provide context metadata (topic, difficulty)

RULES:
1. All questions MUST be in ENGLISH
2. Questions must be specific
3. Irrelevant questions must be truly unrelated to the provided notes
4. For irrelevant questions the expected answer must be "I don't know"""

    RELEVANT_QUESTION_PROMPT = (
        """Analyze the following course notes and generate EXACTLY """
        """{num_questions} """
        """test questions based ONLY on the notes.

Course Notes:
{document_content}

Rules:
- Only use information that EXISTS in the notes
- It is forbidden to ask about anything not present in the notes
- Provide difficulty (Easy/Medium/Hard)
- Provide an expected answer
- Output MUST be JSON ONLY (no other text)

JSON format:
{{
  "questions": [
    {{
      "question": "Question text",
      "difficulty": "Easy/Medium/Hard",
      "expected_answer": "Expected answer",
      "topic": "Topic"
    }}
  ]
}}"""

    )

    IRRELEVANT_QUESTION_PROMPT = (
        """Generate EXACTLY {num_questions} questions that are """
        """COMPLETELY UNRELATED """
        """to the following course notes.

Course Notes:
{document_content}

Rules:
- Questions must be about topics NOT present in the notes
- Student should not be able to answer from the notes
- Expected answer MUST be "I don't know"
- Output MUST be JSON ONLY (no other text)

JSON format:
{{
  "questions": [
    {{
      "question": "Question text",
      "expected_answer": "I don't know",
      "reason": "Why this is unrelated"
    }}
  ]
}}"""

    )


class GiskardRAGTester:
    """
    Giskard-based RAG system tester with Turkish language support.

    This class provides methods to test RAG systems for:
    - Hallucination detection
    - Relevance testing
    - Language consistency
    - Response quality
    """

    def __init__(
        self,
        config: GiskardTestConfig,
        llm_service: Any = None
    ):
        """
        Initialize Giskard RAG tester.

        Args:
            config: Test configuration
            llm_service: LLM service instance for generating test questions
        """
        self.config = config
        self.llm_service = llm_service
        self.prompts = (
            EnglishPrompts() if config.language == "en" else TurkishPrompts()
        )
        self.test_results = []

    def generate_test_questions(
        self,
        document_content: str,
        num_relevant: Optional[int] = None,
        num_irrelevant: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Generate test questions for RAG testing.

        Args:
            document_content: Document content to base questions on
            num_relevant: Number of relevant questions
            num_irrelevant: Number of irrelevant questions

        Returns:
            Dictionary with 'relevant' and 'irrelevant' question lists
        """
        if num_relevant is None:
            num_relevant = self.config.num_test_questions
        if num_irrelevant is None:
            num_irrelevant = self.config.num_irrelevant_questions

        results = {
            "relevant": [],
            "irrelevant": []
        }

        if self.llm_service:
            # Generate relevant questions
            relevant_prompt = self.prompts.RELEVANT_QUESTION_PROMPT.format(
                num_questions=num_relevant,
                document_content=document_content[:5000]
            )

            try:
                response = self.llm_service.generate_response([
                    {"role": "system", "content": self.prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": relevant_prompt}
                ])

                # Parse JSON response
                try:
                    parsed = self._parse_llm_json(response)
                    questions = (parsed.get("questions", []) or [])
                    results["relevant"] = questions[:num_relevant]
                    logger.info(
                        f"Generated {len(results['relevant'])} "
                        f"relevant questions"
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse relevant questions JSON. "
                        "LLM response (truncated): %s",
                        (response or "")[:800],
                    )
                    raise

            except Exception as e:
                logger.error(f"Error generating relevant questions: {e}")
                raise

            # Generate irrelevant questions
            irrelevant_prompt = (
                self.prompts.IRRELEVANT_QUESTION_PROMPT.format(
                    num_questions=num_irrelevant,
                    document_content=document_content[:2000]
                )
            )

            try:
                response = self.llm_service.generate_response([
                    {"role": "system", "content": self.prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": irrelevant_prompt}
                ])

                try:
                    parsed = self._parse_llm_json(response)
                    questions = (parsed.get("questions", []) or [])
                    results["irrelevant"] = questions[:num_irrelevant]
                    logger.info(
                        f"Generated {len(results['irrelevant'])} "
                        f"irrelevant questions"
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse irrelevant questions JSON"
                    )
                    results["irrelevant"] = (
                        self._create_fallback_irrelevant_questions(
                            num_irrelevant
                        )
                    )

            except Exception as e:
                logger.error(f"Error generating irrelevant questions: {e}")
                results["irrelevant"] = (
                    self._create_fallback_irrelevant_questions(
                        num_irrelevant
                    )
                )
        else:
            # Fallback without LLM service
            results["relevant"] = self._create_fallback_relevant_questions(
                document_content, num_relevant
            )
            results["irrelevant"] = self._create_fallback_irrelevant_questions(
                num_irrelevant
            )

        return results

    def _parse_llm_json(self, response: str) -> Dict[str, Any]:
        text = (response or "").strip()
        if not text:
            raise json.JSONDecodeError("Empty response", text, 0)

        # Remove markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract first JSON object in the response
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        return json.loads(match.group(0))

    def _create_fallback_relevant_questions(
        self,
        document_content: str,
        num_questions: int
    ) -> List[Dict]:
        """Create fallback relevant questions without LLM."""
        questions = []
        topics = [
            "Ana kavramlar", "Önemli tanımlar", "Temel prensipler",
            "Uygulama örnekleri", "Kritik noktalar"
        ]

        for i in range(min(num_questions, len(topics))):
            questions.append({
                "question": f"{topics[i]} hakkında bilgi verir misiniz?",
                "difficulty": "Orta",
                "expected_answer": (
                    "Ders notlarından ilgili konu hakkında bilgi"
                ),
                "topic": topics[i]
            })

        return questions

    def _create_fallback_irrelevant_questions(
        self,
        num_questions: int
    ) -> List[Dict]:
        """Create fallback irrelevant questions without LLM."""
        questions = []
        irrelevant_topics = [
            "Mars'ta yaşam", "Antik Mısır piramitleri", "Kuantum fiziği",
            "Roma imparatorluğu", "Yapay zeka etiği"
        ]

        for i in range(min(num_questions, len(irrelevant_topics))):
            questions.append({
                "question": (
                    f"{irrelevant_topics[i]} hakkında bilgi verir misiniz?"
                ),
                "expected_answer": "Bilmiyorum",
                "reason": "Bu konu ders notlarında yok"
            })

        return questions

    def evaluate_response(
        self,
        question: str,
        answer: str,
        expected_answer: str,
        question_type: str = "relevant"
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG response.

        Args:
            question: The question asked
            answer: The model's response
            expected_answer: Expected answer
            question_type: "relevant" or "irrelevant"

        Returns:
            Dictionary with evaluation metrics
        """
        evaluation = {
            "question": question,
            "answer": answer,
            "expected_answer": expected_answer,
            "question_type": question_type,
            "metrics": {}
        }

        # Check if irrelevant question got "Bilmiyorum" response
        if question_type == "irrelevant":
            # Check for "Bilmiyorum" or similar responses
            negative_responses = [
                "bilmiyorum", "bilgim yok", "bilgim bulunmuyor",
                "bu konuda bilgi veremem", "notlarda bu konu yok"
            ]
            answer_lower = answer.lower()

            has_negative_response = any(
                neg in answer_lower for neg in negative_responses
            )

            evaluation["metrics"]["correct_refusal"] = has_negative_response
            evaluation["metrics"]["hallucinated"] = not has_negative_response

            if has_negative_response:
                evaluation["metrics"]["score"] = 1.0
            else:
                evaluation["metrics"]["score"] = 0.0
        else:
            # For relevant questions, check if answer is provided
            if answer and len(answer) > 20:
                evaluation["metrics"]["provided_answer"] = True
                evaluation["metrics"]["score"] = 1.0
            else:
                evaluation["metrics"]["provided_answer"] = False
                evaluation["metrics"]["score"] = 0.0

        # Check language consistency
        evaluation["metrics"]["language"] = self._check_language(answer)

        # Calculate overall quality score
        evaluation["metrics"]["quality_score"] = self._calculate_quality_score(
            evaluation["metrics"]
        )

        return evaluation

    def _check_language(self, text: str) -> str:
        """Check if text is Turkish, English, or mixed."""
        # Simple heuristic: check for Turkish characters
        turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        text_has_turkish = any(char in text for char in turkish_chars)

        # Check for English-specific patterns
        english_words = ["the", "and", "is", "of", "to", "in", "that"]
        text_has_english = any(
            word.lower() in text.lower() for word in english_words
        )

        if text_has_turkish and not text_has_english:
            return "Türkçe"
        elif text_has_english and not text_has_turkish:
            return "İngilizce"
        else:
            return "Karışık"

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from metrics."""
        score = 0.0

        if "score" in metrics:
            score += metrics["score"] * 0.7  # 70% weight on correctness

        if metrics.get("language") == "Türkçe":
            score += 0.3  # 30% weight on Turkish language

        return min(score, 1.0)

    def run_test_suite(
        self,
        rag_function: callable,
        document_content: str
    ) -> Dict[str, Any]:
        """
        Run complete test suite on RAG system.

        Args:
            rag_function: Function that takes a question and returns an answer
            document_content: Document content to base tests on

        Returns:
            Dictionary with complete test results
        """
        logger.info(
            f"Starting Giskard test suite for {self.config.model_name}"
        )

        # Generate test questions
        test_questions = self.generate_test_questions(document_content)

        all_evaluations = []

        # Test relevant questions
        logger.info(
            f"Testing {len(test_questions['relevant'])} relevant questions"
        )
        for q in test_questions["relevant"]:
            try:
                answer = rag_function(q["question"])
                evaluation = self.evaluate_response(
                    q["question"],
                    answer,
                    q.get("expected_answer", ""),
                    "relevant"
                )
                all_evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Error testing relevant question: {e}")
                all_evaluations.append({
                    "question": q["question"],
                    "answer": "",
                    "error": str(e),
                    "metrics": {"score": 0.0}
                })

        # Test irrelevant questions (hallucination test)
        logger.info(
            f"Testing {len(test_questions['irrelevant'])} "
            f"irrelevant questions"
        )
        for q in test_questions["irrelevant"]:
            try:
                answer = rag_function(q["question"])
                evaluation = self.evaluate_response(
                    q["question"],
                    answer,
                    q.get("expected_answer", "Bilmiyorum"),
                    "irrelevant"
                )
                all_evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Error testing irrelevant question: {e}")
                all_evaluations.append({
                    "question": q["question"],
                    "answer": "",
                    "error": str(e),
                    "metrics": {"score": 0.0}
                })

        # Calculate aggregate metrics
        results = self._calculate_aggregate_metrics(all_evaluations)
        results["evaluations"] = all_evaluations
        results["test_questions"] = test_questions
        results["config"] = {
            "model_name": self.config.model_name,
            "provider": self.config.provider,
            "num_relevant_questions": len(test_questions["relevant"]),
            "num_irrelevant_questions": len(test_questions["irrelevant"])
        }

        logger.info(
            f"Test suite completed. Overall score: "
            f"{results['overall_score']:.2f}"
        )

        return results

    def _calculate_aggregate_metrics(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate aggregate metrics from all evaluations."""
        total = len(evaluations)
        if total == 0:
            return {"overall_score": 0.0, "metrics": {}}

        relevant = [
            e for e in evaluations if e.get("question_type") == "relevant"
        ]
        irrelevant = [
            e for e in evaluations if e.get("question_type") == "irrelevant"
        ]

        # Calculate scores
        relevant_scores = [
            e["metrics"].get("quality_score", 0) for e in relevant
        ]
        irrelevant_scores = [
            e["metrics"].get("quality_score", 0) for e in irrelevant
        ]

        avg_relevant_score = (
            sum(relevant_scores) / len(relevant_scores)
            if relevant_scores else 0
        )
        avg_irrelevant_score = (
            sum(irrelevant_scores) / len(irrelevant_scores)
            if irrelevant_scores else 0
        )

        # Calculate hallucination rate
        hallucinated = [
            e for e in irrelevant
            if e["metrics"].get("hallucinated", False)
        ]
        hallucination_rate = (
            len(hallucinated) / len(irrelevant) if irrelevant else 0
        )

        # Calculate language consistency
        turkish_responses = [
            e for e in evaluations
            if e["metrics"].get("language") == "Türkçe"
        ]
        language_consistency = (
            len(turkish_responses) / total if total > 0 else 0
        )

        # Overall score (weighted)
        overall_score = (
            avg_relevant_score * 0.5 +
            avg_irrelevant_score * 0.4 +
            language_consistency * 0.1
        )

        return {
            "overall_score": round(overall_score, 3),
            "metrics": {
                "relevant_questions": {
                    "count": len(relevant),
                    "avg_score": round(avg_relevant_score, 3),
                    "success_rate": round(
                        sum(1 for e in relevant
                            if e["metrics"].get("score", 0) > 0) /
                        len(relevant),
                        3
                    ) if relevant else 0
                },
                "irrelevant_questions": {
                    "count": len(irrelevant),
                    "avg_score": round(avg_irrelevant_score, 3),
                    "success_rate": round(
                        sum(1 for e in irrelevant
                            if e["metrics"].get("score", 0) > 0) /
                        len(irrelevant),
                        3
                    ) if irrelevant else 0,
                    "hallucination_rate": round(hallucination_rate, 3),
                    "correct_refusal_rate": round(1 - hallucination_rate, 3)
                },
                "language_consistency": round(language_consistency, 3),
                "turkish_response_rate": round(language_consistency, 3)
            },
            "total_evaluations": total
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable test report."""
        report = []
        report.append("=" * 80)
        report.append("GISKARD RAG SİSTEMİ TEST RAPORU")
        report.append("=" * 80)
        report.append("")

        # Config info
        config = results.get("config", {})
        report.append(f"Model: {config.get('model_name', 'N/A')}")
        report.append(f"Provider: {config.get('provider', 'N/A')}")
        report.append("")

        # Overall score
        overall = results.get("overall_score", 0)
        report.append(f"GENEL SKOR: {overall:.2%}")
        report.append("")

        # Metrics
        metrics = results.get("metrics", {})

        # Relevant questions
        relevant = metrics.get("relevant_questions", {})
        report.append("ALAKALI SORULAR TESTİ:")
        report.append(f"  - Soru Sayısı: {relevant.get('count', 0)}")
        report.append(f"  - Ortalama Skor: {relevant.get('avg_score', 0):.2%}")
        report.append(
            f"  - Başarı Oranı: {relevant.get('success_rate', 0):.2%}"
        )
        report.append("")

        # Irrelevant questions (hallucination test)
        irrelevant = metrics.get("irrelevant_questions", {})
        report.append("ALAKASIZ SORULAR TESTİ (HALÜSİNASYON):")
        report.append(f"  - Soru Sayısı: {irrelevant.get('count', 0)}")
        report.append(
            f"  - Ortalama Skor: {irrelevant.get('avg_score', 0):.2%}"
        )
        report.append(
            f"  - Başarı Oranı: {irrelevant.get('success_rate', 0):.2%}"
        )
        report.append(
            f"  - Halüsinasyon Oranı: "
            f"{irrelevant.get('halucination_rate', 0):.2%}"
        )
        report.append(
            f"  - Doğru Reddetme Oranı: "
            f"{irrelevant.get('correct_refusal_rate', 0):.2%}"
        )
        report.append("")

        # Language consistency
        report.append("DİL TUTARLILIĞI:")
        report.append(
            f"  - Türkçe Cevap Oranı: "
            f"{metrics.get('turkish_response_rate', 0):.2%}"
        )
        report.append("")

        # Recommendations
        report.append("ÖNERİLER:")
        if irrelevant.get("halucination_rate", 0) > 0.3:
            report.append(
                "  ⚠️  Halüsinasyon oranı yüksek. Model daha sık "
                "'Bilmiyorum' demeli."
            )
        if metrics.get("turkish_response_rate", 0) < 0.9:
            report.append(
                "  ⚠️  Dil tutarlılığı düşük. Türkçe promptlarını güçlendirin."
            )
        if relevant.get("success_rate", 0) < 0.8:
            report.append(
                "  ⚠️  Alakalı sorulara yanıt oranı düşük. "
                "Retrieval mekanizmasını kontrol edin."
            )
        if overall > 0.8:
            report.append(
                "  ✅ Sistem genel olarak iyi performans gösteriyor."
            )

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def create_giskard_tester(
    model_name: str,
    provider: str,
    llm_service: Any = None,
    **kwargs
) -> GiskardRAGTester:
    """
    Factory function to create a Giskard RAG tester.

    Args:
        model_name: Name of the model to test
        provider: LLM provider name
        llm_service: Optional LLM service for generating test questions
        **kwargs: Additional configuration parameters

    Returns:
        Configured GiskardRAGTester instance
    """
    config = GiskardTestConfig(
        model_name=model_name,
        provider=provider,
        **kwargs
    )

    return GiskardRAGTester(config=config, llm_service=llm_service)
