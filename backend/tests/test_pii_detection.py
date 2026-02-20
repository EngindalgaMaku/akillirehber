"""
PII Detection Service — Precision/Recall Test Suite

KVKK uyumlu kişisel bilgi filtreleme katmanının performansını ölçer.
Regex katmanı (Katman 1) için precision, recall, F1 ve false positive/negative
analizi yapar. Embedding katmanı API bağımlı olduğundan ayrı entegrasyon
testinde değerlendirilir.

Kullanım:
    cd backend
    python -m pytest tests/test_pii_detection.py -v
"""

import sys
import os

# backend/ dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.pii_detection_service import detect_pii_regex


# ==================== Test Veri Seti ====================
# Her kayıt: (mesaj, beklenen_sonuç, kategori_açıklama)

# True Positive örnekleri — PII içeren mesajlar
TRUE_POSITIVE_CASES = [
    # TC Kimlik
    ("TC kimlik numaram 12345678901", True, "tc_kimlik"),
    ("Kimlik no: 98765432109", True, "tc_kimlik"),
    ("12345678901 numaralı kimliğim var", True, "tc_kimlik"),
    # Telefon
    ("Telefon numaram 0532 123 45 67", True, "telefon_numarasi"),
    ("Beni 0555-987-65-43 numaradan arayın", True, "telefon_numarasi"),
    ("05321234567 numaramdan ulaşabilirsiniz", True, "telefon_numarasi"),
    ("+90 532 123 45 67 numaramı veriyorum", True, "telefon_numarasi"),
    # E-posta
    ("E-posta adresim ornek@gmail.com", True, "eposta"),
    ("Mail: ahmet.yilmaz@hotmail.com", True, "eposta"),
    ("test_user123@university.edu.tr adresime gönderin", True, "eposta"),
    # IBAN
    ("IBAN numaram TR33 0006 1005 1978 6457 8413 26", True, "iban"),
    ("TR330006100519786457841326 hesabıma yatırın", True, "iban"),
    # Kredi Kartı
    ("Kart numaram 4532 1234 5678 9012", True, "kredi_karti"),
    ("5412-7534-0000-1234 kartımla ödeme yapacağım", True, "kredi_karti"),
    ("4532123456789012 numaralı kartım", True, "kredi_karti"),
    # Pasaport
    ("Pasaport numaram U12345678", True, "pasaport"),
    ("Pasaportum: u87654321", True, "pasaport"),
]

# True Negative örnekleri — PII içermeyen mesajlar
TRUE_NEGATIVE_CASES = [
    ("Fotosantez nedir açıklar mısınız?", False, "akademik_soru"),
    ("Newton'un hareket yasalarını anlatır mısınız?", False, "akademik_soru"),
    ("Mitoz ve mayoz bölünme arasındaki fark nedir?", False, "akademik_soru"),
    ("Osmanlı İmparatorluğu ne zaman kuruldu?", False, "akademik_soru"),
    ("Integral nasıl hesaplanır?", False, "akademik_soru"),
    ("Python'da liste nasıl sıralanır?", False, "teknik_soru"),
    ("Bu dersin final sınavı kaç puan üzerinden?", False, "ders_sorusu"),
    ("Ödev teslim tarihi ne zaman?", False, "ders_sorusu"),
    ("Makine öğrenmesi ile derin öğrenme arasındaki fark nedir?", False, "teknik_soru"),
    ("Küresel ısınmanın nedenleri nelerdir?", False, "akademik_soru"),
    ("Veri yapıları dersinde ağaç yapısını anlatır mısınız?", False, "teknik_soru"),
    ("HTTP protokolü nasıl çalışır?", False, "teknik_soru"),
    ("TCP/IP katmanları nelerdir?", False, "teknik_soru"),
    ("Bilgisayar ağlarında IP adresi ne işe yarar?", False, "teknik_soru"),
    ("2024 yılında yapay zeka alanında neler değişti?", False, "genel_soru"),
    ("Sınav sonuçları ne zaman açıklanacak?", False, "ders_sorusu"),
    ("Bu konuyu 3 maddede özetler misin?", False, "ders_sorusu"),
    ("Algoritma karmaşıklığı O(n log n) ne demek?", False, "teknik_soru"),
]

# Sınır durumları — yanlış alarm riski taşıyan mesajlar
EDGE_CASES = [
    # Sayısal içerik ama PII değil
    ("Soru 12345678901 numaralı sayfada", True, "tc_kimlik_gibi_gorunen"),
    ("2024 yılında 5321234567 adet ürün satıldı", True, "telefon_gibi_gorunen"),
    # Kısa mesajlar
    ("Merhaba", False, "selamlama"),
    ("Evet", False, "kisa_cevap"),
    ("", False, "bos_mesaj"),
]


# ==================== Test Fonksiyonları ====================

def test_true_positives():
    """PII içeren mesajların doğru tespit edildiğini kontrol et."""
    for message, expected, desc in TRUE_POSITIVE_CASES:
        is_pii, score, label = detect_pii_regex(message)
        assert is_pii == expected, (
            f"FALSE NEGATIVE: '{message}' ({desc}) — "
            f"beklenen=True, sonuç={is_pii}"
        )


def test_true_negatives():
    """PII içermeyen mesajların yanlış alarm vermediğini kontrol et."""
    for message, expected, desc in TRUE_NEGATIVE_CASES:
        is_pii, score, label = detect_pii_regex(message)
        assert is_pii == expected, (
            f"FALSE POSITIVE: '{message}' ({desc}) — "
            f"beklenen=False, sonuç={is_pii}, label={label}"
        )


def test_empty_and_whitespace():
    """Boş ve whitespace mesajların güvenli şekilde işlendiğini kontrol et."""
    for msg in ["", "   ", None]:
        is_pii, score, label = detect_pii_regex(msg or "")
        assert not is_pii, f"Boş/whitespace mesaj PII olarak algılandı: '{msg}'"


def test_precision_recall_report():
    """Precision, Recall ve F1 skorlarını hesapla ve raporla.

    Bu test her zaman geçer — amacı metrikleri raporlamaktır.
    """
    all_cases = TRUE_POSITIVE_CASES + TRUE_NEGATIVE_CASES

    tp = fp = fn = tn = 0
    false_positives = []
    false_negatives = []

    for message, expected, desc in all_cases:
        is_pii, score, label = detect_pii_regex(message)

        if expected and is_pii:
            tp += 1
        elif expected and not is_pii:
            fn += 1
            false_negatives.append((message, desc))
        elif not expected and is_pii:
            fp += 1
            false_positives.append((message, desc, label))
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 60)
    print("PII Regex Katmanı — Precision/Recall Raporu")
    print("=" * 60)
    print(f"  Toplam test: {len(all_cases)}")
    print(f"  TP: {tp}  |  FP: {fp}  |  FN: {fn}  |  TN: {tn}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    if false_positives:
        print(f"\n  False Positives ({len(false_positives)}):")
        for msg, desc, lbl in false_positives:
            print(f"    - [{lbl}] {msg} ({desc})")

    if false_negatives:
        print(f"\n  False Negatives ({len(false_negatives)}):")
        for msg, desc in false_negatives:
            print(f"    - {msg} ({desc})")

    # Minimum beklentiler
    assert precision >= 0.80, f"Precision çok düşük: {precision:.4f}"
    assert recall >= 0.80, f"Recall çok düşük: {recall:.4f}"
