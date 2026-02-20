"""PII Detection Service — iki katmanlı kişisel bilgi tespit sistemi.

Katman 1 (Regex): TC kimlik, telefon, e-posta, IBAN, kredi kartı gibi
    yapısal kalıpları düzenli ifadelerle hızlıca tespit eder.
Katman 2 (Few-Shot Embedding): Regex'e takılmayan belirsiz durumları
    k-NN tabanlı embedding sınıflandırması ile değerlendirir.

Bu iki katmanlı yaklaşım hem hız hem kapsam açısından avantaj sağlar.
"""

import logging
import re
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


# --- Katman 1: Regex Tabanlı PII Tespiti ---
# Yapısal kalıpları düzenli ifadelerle tespit eder.

PII_REGEX_PATTERNS: List[Tuple[str, str]] = [
    # TC Kimlik Numarası (11 haneli, başı sıfır olmayan)
    (r"\b[1-9]\d{10}\b", "tc_kimlik"),
    # Türk cep telefonu (05xx ile başlayan, çeşitli formatlar)
    (r"\b0\s?5\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b", "telefon_numarasi"),
    # +90 ile başlayan telefon
    (r"\+90\s?5\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b", "telefon_numarasi"),
    # E-posta adresi
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "eposta"),
    # IBAN (TR ile başlayan, 26 karakter)
    (r"\bTR\s?\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{2}\b", "iban"),
    # Kredi kartı numarası (4 × 4 haneli gruplar)
    (r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "kredi_karti"),
    # Pasaport numarası (Türk: U + 8 hane)
    (r"\b[Uu]\d{8}\b", "pasaport"),
]

# Derlenmiş regex pattern'leri (performans için)
_COMPILED_PII_PATTERNS: List[Tuple["re.Pattern", str]] = [
    (re.compile(pattern, re.IGNORECASE), label)
    for pattern, label in PII_REGEX_PATTERNS
]


def detect_pii_regex(message: str) -> Tuple[bool, float, str]:
    """Katman 1: Regex tabanlı PII tespiti.

    Yapısal kalıpları (TC kimlik, telefon, e-posta, IBAN, kredi kartı, pasaport)
    düzenli ifadelerle tespit eder. Eşleşme bulunursa 1.0 güven skoru döner.

    Args:
        message: Kullanıcı mesajı

    Returns:
        (is_pii, confidence, matched_label)
    """
    if not message or not message.strip():
        return False, 0.0, ""

    for pattern, label in _COMPILED_PII_PATTERNS:
        if pattern.search(message):
            logger.info(
                "[PII-REGEX] Tespit: label=%s, message=%s...",
                label,
                message[:50],
            )
            return True, 1.0, label

    return False, 0.0, ""


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

def _knn_classify(
    message_embedding: List[float],
    pii_embeddings: List[List[float]],
    non_pii_embeddings: List[List[float]],
    k: int = KNN_K,
) -> Tuple[bool, float, str]:
    """k-NN sınıflandırma mantığı.

    Kullanıcı mesajının embedding'ini tüm örnek cümlelerin embedding'leriyle
    cosine similarity ile karşılaştırır, en yakın k komşuya göre sınıflandırır.

    Args:
        message_embedding: Kullanıcı mesajının embedding vektörü
        pii_embeddings: PII örnek cümlelerinin embedding'leri
        non_pii_embeddings: Non-PII örnek cümlelerinin embedding'leri
        k: En yakın komşu sayısı

    Returns:
        (is_pii, score, matched_category)
    """
    # Tüm örneklerle similarity hesapla: (score, is_pii, category)
    scored: List[Tuple[float, bool, str]] = []

    for i, emb in enumerate(pii_embeddings):
        score = cosine_similarity(message_embedding, emb)
        category = PII_EXAMPLES[i][1] if i < len(PII_EXAMPLES) else "pii"
        scored.append((score, True, category))

    for i, emb in enumerate(non_pii_embeddings):
        score = cosine_similarity(message_embedding, emb)
        category = NON_PII_EXAMPLES[i][1] if i < len(NON_PII_EXAMPLES) else "non_pii"
        scored.append((score, False, category))

    # En yüksek skorlu k örneği seç
    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = scored[:k]

    # Çoğunluk oylaması
    pii_count = sum(1 for _, is_pii, _ in top_k if is_pii)
    non_pii_count = k - pii_count

    if pii_count > non_pii_count:
        # PII çoğunlukta — ortalama similarity skoru
        avg_score = sum(s for s, _, _ in top_k) / k
        # En yakın PII komşunun kategorisini döndür
        matched_category = next(cat for _, is_pii, cat in top_k if is_pii)

        # Threshold kontrolü
        if avg_score < PII_THRESHOLD:
            return False, avg_score, ""

        return True, avg_score, matched_category
    else:
        # Non-PII çoğunlukta
        avg_score = sum(s for s, _, _ in top_k) / k
        return False, avg_score, ""



def detect_pii(
    message: str,
    embedding_model: str,
) -> Tuple[bool, float, str]:
    """Mesajda kişisel bilgi olup olmadığını iki katmanlı sistemle tespit et.

    Katman 1 (Regex): Yapısal kalıpları düzenli ifadelerle hızlıca tespit eder.
    Katman 2 (Few-Shot Embedding): Regex'e takılmayan durumları k-NN ile değerlendirir.

    Args:
        message: Kullanıcı mesajı
        embedding_model: Kullanılacak embedding modeli

    Returns:
        Tuple of (is_pii: bool, confidence: float, matched_label: str)
    """
    if not message or not message.strip():
        return False, 0.0, ""

    # --- Katman 1: Regex ---
    is_pii_regex, regex_score, regex_label = detect_pii_regex(message)
    if is_pii_regex:
        return True, regex_score, regex_label

    # --- Katman 2: Few-Shot Embedding Classification ---
    try:
        embedding_service = get_embedding_service()

        # Cache'i kontrol et, gerekirse initialize et
        cached = _embedding_cache.get_embeddings(embedding_model)
        if cached is None:
            success = _embedding_cache.initialize(embedding_model, embedding_service)
            if not success:
                logger.warning("[PII] Cache initialization failed, skipping PII check")
                return False, 0.0, ""
            cached = _embedding_cache.get_embeddings(embedding_model)
            if cached is None:
                logger.warning("[PII] Cache unavailable after initialization, skipping PII check")
                return False, 0.0, ""

        pii_embeddings, non_pii_embeddings = cached

        # Sadece kullanıcı mesajının embedding'ini hesapla
        msg_embeddings = embedding_service.get_embeddings(
            [message],
            model=embedding_model,
            input_type="query",
        )

        if not msg_embeddings or len(msg_embeddings) < 1:
            logger.warning("[PII] Message embedding failed, skipping PII check")
            return False, 0.0, ""

        message_embedding = msg_embeddings[0]

        # k-NN sınıflandırma
        is_pii, score, matched_category = _knn_classify(
            message_embedding, pii_embeddings, non_pii_embeddings
        )

        logger.info(
            "[PII] message=%s... | score=%.3f | matched=%s | is_pii=%s",
            message[:50],
            score,
            matched_category,
            is_pii,
        )

        return is_pii, score, matched_category

    except Exception as e:
        logger.warning("[PII] Detection error: %s", e)
        return False, 0.0, ""


