"""PII Detection Service using embedding-based zero-shot classification.

Kullanıcı mesajlarını embedding tabanlı zero-shot classification ile
kişisel bilgi içerip içermediğini tespit eder.
"""

import logging
import numpy as np
from typing import List, Tuple

from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

# PII label'ları ve non-PII label'ları (Türkçe)
PII_LABELS = [
    "kişisel bilgi paylaşımı",
    "telefon numarası veya adres paylaşma",
    "kimlik bilgisi veya TC kimlik numarası",
    "e-posta adresi veya şifre paylaşma",
    "kredi kartı veya banka hesap bilgisi",
    "özel sağlık bilgisi paylaşma",
]

NON_PII_LABELS = [
    "genel akademik soru",
    "ders konusuyla ilgili soru",
    "ödev veya sınav hakkında soru",
    "kavram veya teori sorusu",
    "teknik veya bilimsel soru",
]

# Eşik değeri - bu değerin üstündeki PII skoru tespit olarak kabul edilir
PII_THRESHOLD = 0.35


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

    # Tüm metinleri tek seferde embed et
    all_texts = [message] + PII_LABELS + NON_PII_LABELS
    all_embeddings = embedding_service.get_embeddings(
        all_texts,
        model=embedding_model,
        input_type="query",
    )

    if not all_embeddings or len(all_embeddings) < len(all_texts):
        logger.warning("[PII] Embedding generation failed, skipping PII check")
        return False, 0.0, ""

    message_embedding = all_embeddings[0]
    pii_embeddings = all_embeddings[1 : 1 + len(PII_LABELS)]
    non_pii_embeddings = all_embeddings[1 + len(PII_LABELS) :]

    # PII label'larıyla similarity hesapla
    pii_scores = []
    for i, emb in enumerate(pii_embeddings):
        score = cosine_similarity(message_embedding, emb)
        pii_scores.append((score, PII_LABELS[i]))

    # Non-PII label'larıyla similarity hesapla
    non_pii_scores = []
    for i, emb in enumerate(non_pii_embeddings):
        score = cosine_similarity(message_embedding, emb)
        non_pii_scores.append((score, NON_PII_LABELS[i]))

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
