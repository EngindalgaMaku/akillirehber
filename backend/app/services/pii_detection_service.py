"""PII Detection Service using few-shot embedding classification.

Kullanıcı mesajlarını few-shot embedding classification ile
kişisel bilgi içerip içermediğini tespit eder. Gerçek Türkçe örnek
cümleler kullanarak k-NN tarzı sınıflandırma yapar.
"""

import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple

from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

# --- Few-Shot Örnek Cümleler ---
# Her örnek (text, category) tuple formatındadır.

PII_EXAMPLES: List[Tuple[str, str]] = [
    ("Benim telefon numaram 0532 123 45 67", "telefon_numarasi"),
    ("Numaramı vereyim: 0555 987 65 43 bana ulaşabilirsiniz", "telefon_numarasi"),
    ("TC kimlik numaram 12345678901", "tc_kimlik"),
    ("Kimlik numaramı paylaşıyorum: 98765432109", "tc_kimlik"),
    ("E-posta adresim ornek@gmail.com", "eposta"),
    ("Bana mail atın: ahmet.yilmaz@hotmail.com", "eposta"),
    ("Ev adresim Atatürk Caddesi No:15 Kadıköy İstanbul", "adres"),
    ("Kredi kartı numaram 4532 1234 5678 9012", "kredi_karti"),
    ("Kart bilgilerim: 5412 7534 0000 1234, son kullanma 12/27", "kredi_karti"),
    ("Şifrem abc123456 olarak ayarladım", "sifre"),
    ("IBAN numaram TR33 0006 1005 1978 6457 8413 26", "iban"),
    ("Doğum tarihim 15 Mart 1995", "dogum_tarihi"),
    ("Pasaport numaram U12345678", "pasaport"),
    ("SGK numaram 1234567890 olarak kayıtlı", "sgk"),
]

NON_PII_EXAMPLES: List[Tuple[str, str]] = [
    ("Fotosantez nedir açıklar mısınız?", "akademik_soru"),
    ("Bu konunun sınavda çıkma ihtimali nedir?", "ders_sorusu"),
    ("Newton'un hareket yasalarını anlatır mısınız?", "akademik_soru"),
    ("Mitoz ve mayoz bölünme arasındaki fark nedir?", "akademik_soru"),
    ("Osmanlı İmparatorluğu ne zaman kuruldu?", "akademik_soru"),
    ("Integral nasıl hesaplanır?", "akademik_soru"),
    ("Python'da liste nasıl sıralanır?", "teknik_soru"),
    ("Ödev teslim tarihi ne zaman?", "ders_sorusu"),
    ("Bu dersin final sınavı kaç puan üzerinden?", "ders_sorusu"),
    ("Makine öğrenmesi ile derin öğrenme arasındaki fark nedir?", "teknik_soru"),
    ("Küresel ısınmanın nedenleri nelerdir?", "akademik_soru"),
    ("Ekonomide arz ve talep dengesi nasıl oluşur?", "akademik_soru"),
    ("Veri yapıları dersinde ağaç yapısını anlatır mısınız?", "teknik_soru"),
    ("Türkçede fiil çekimi kuralları nelerdir?", "akademik_soru"),
]

# --- Yapılandırma Sabitleri ---
PII_THRESHOLD: float = 0.35  # Eşik değeri - bu değerin üstündeki PII skoru tespit olarak kabul edilir
KNN_K: int = 5               # En yakın komşu sayısı (k-NN)


class FewShotEmbeddingCache:
    """Örnek cümlelerin embedding'lerini cache'ler.

    Thread-safe, model bazlı invalidation destekler.
    Singleton pattern ile modül seviyesinde tek instance kullanılır.
    """

    def __init__(self):
        self._cache: Optional[Dict] = None
        self._cached_model: Optional[str] = None
        self._lock: threading.Lock = threading.Lock()

    def get_embeddings(
        self, embedding_model: str
    ) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
        """Cache'lenmiş PII ve non-PII embedding'lerini döndürür.

        Returns:
            (pii_embeddings, non_pii_embeddings) veya None (cache boşsa/model değiştiyse)
        """
        with self._lock:
            if self._cache is not None and self._cached_model == embedding_model:
                return self._cache["pii_embeddings"], self._cache["non_pii_embeddings"]
            return None

    def initialize(self, embedding_model: str, embedding_service) -> bool:
        """Örnek cümlelerin embedding'lerini hesaplar ve cache'ler.

        Returns:
            True başarılıysa, False hata durumunda
        """
        with self._lock:
            # Double-check: başka thread zaten initialize etmiş olabilir
            if self._cache is not None and self._cached_model == embedding_model:
                return True

            try:
                pii_texts = [text for text, _ in PII_EXAMPLES]
                non_pii_texts = [text for text, _ in NON_PII_EXAMPLES]

                all_texts = pii_texts + non_pii_texts
                all_embeddings = embedding_service.get_embeddings(
                    all_texts,
                    model=embedding_model,
                    input_type="query",
                )

                if not all_embeddings or len(all_embeddings) < len(all_texts):
                    logger.warning(
                        "[PII] Cache initialization failed: insufficient embeddings returned"
                    )
                    return False

                pii_embeddings = all_embeddings[: len(pii_texts)]
                non_pii_embeddings = all_embeddings[len(pii_texts) :]

                self._cache = {
                    "model": embedding_model,
                    "pii_embeddings": pii_embeddings,
                    "non_pii_embeddings": non_pii_embeddings,
                }
                self._cached_model = embedding_model

                logger.info(
                    "[PII] Cache initialized: %d PII + %d non-PII examples cached for model=%s",
                    len(pii_embeddings),
                    len(non_pii_embeddings),
                    embedding_model,
                )
                return True

            except Exception as e:
                logger.warning("[PII] Cache initialization error: %s", e)
                return False


# Modül seviyesinde singleton instance
_embedding_cache = FewShotEmbeddingCache()


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """İki vektör arasındaki cosine similarity hesapla."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def detect_pii(
    message: str,
    embedding_model: str,
) -> Tuple[bool, float, str]:
    """Mesajda kişisel bilgi olup olmadığını tespit et.

    Args:
        message: Kullanıcı mesajı
        embedding_model: Kullanılacak embedding modeli

    Returns:
        Tuple of (is_pii: bool, confidence: float, matched_label: str)
    """
    if not message or not message.strip():
        return False, 0.0, ""

    embedding_service = get_embedding_service()

    # Örnek cümle metinlerini çıkar (geçici — Task 4'te k-NN'e dönüşecek)
    pii_texts = [text for text, _ in PII_EXAMPLES]
    non_pii_texts = [text for text, _ in NON_PII_EXAMPLES]
    pii_categories = [cat for _, cat in PII_EXAMPLES]
    non_pii_categories = [cat for _, cat in NON_PII_EXAMPLES]

    # Tüm metinleri tek seferde embed et
    all_texts = [message] + pii_texts + non_pii_texts
    all_embeddings = embedding_service.get_embeddings(
        all_texts,
        model=embedding_model,
        input_type="query",
    )

    if not all_embeddings or len(all_embeddings) < len(all_texts):
        logger.warning("[PII] Embedding generation failed, skipping PII check")
        return False, 0.0, ""

    message_embedding = all_embeddings[0]
    pii_embeddings = all_embeddings[1 : 1 + len(pii_texts)]
    non_pii_embeddings = all_embeddings[1 + len(pii_texts) :]

    # PII örnekleriyle similarity hesapla
    pii_scores = []
    for i, emb in enumerate(pii_embeddings):
        score = cosine_similarity(message_embedding, emb)
        pii_scores.append((score, pii_categories[i]))

    # Non-PII örnekleriyle similarity hesapla
    non_pii_scores = []
    for i, emb in enumerate(non_pii_embeddings):
        score = cosine_similarity(message_embedding, emb)
        non_pii_scores.append((score, non_pii_categories[i]))

    # En yüksek PII ve non-PII skorlarını bul
    max_pii_score, max_pii_label = max(pii_scores, key=lambda x: x[0])
    max_non_pii_score, _ = max(non_pii_scores, key=lambda x: x[0])

    # PII skoru non-PII skorundan yüksekse ve eşik değerinin üstündeyse → PII
    is_pii = max_pii_score > max_non_pii_score and max_pii_score >= PII_THRESHOLD

    logger.info(
        "[PII] message=%s... | pii_score=%.3f (%s) | non_pii_score=%.3f | is_pii=%s",
        message[:50],
        max_pii_score,
        max_pii_label,
        max_non_pii_score,
        is_pii,
    )

    return is_pii, max_pii_score, max_pii_label
