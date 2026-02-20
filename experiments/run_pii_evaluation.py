"""
KVKK Uyumlu PII Filtreleme Performans Değerlendirme Betiği

İki katmanlı PII tespit sisteminin performansını ölçer:
- Katman 1 (Regex): Yapısal kalıplar (TC kimlik, telefon, e-posta, IBAN, kredi kartı)
- Katman 2 (Few-Shot Embedding): Belirsiz durumlar için k-NN sınıflandırma

Metrikler: Precision, Recall, F1, False Positive/Negative analizi

Kullanım:
    python run_pii_evaluation.py                          # Sadece regex (offline)
    python run_pii_evaluation.py --with-api --course-id 1 # API üzerinden tam test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# ==================== Test Veri Seti ====================

PII_TEST_CASES = [
    # --- TC Kimlik ---
    {"message": "TC kimlik numaram 12345678901", "expected": True, "category": "tc_kimlik"},
    {"message": "Kimlik no: 98765432109", "expected": True, "category": "tc_kimlik"},
    {"message": "12345678901 numaralı kimliğim", "expected": True, "category": "tc_kimlik"},
    # --- Telefon ---
    {"message": "Telefon numaram 0532 123 45 67", "expected": True, "category": "telefon"},
    {"message": "Beni 0555-987-65-43 numaradan arayın", "expected": True, "category": "telefon"},
    {"message": "05321234567 numaramdan ulaşabilirsiniz", "expected": True, "category": "telefon"},
    {"message": "+90 532 123 45 67", "expected": True, "category": "telefon"},
    # --- E-posta ---
    {"message": "E-posta adresim ornek@gmail.com", "expected": True, "category": "eposta"},
    {"message": "Mail: ahmet.yilmaz@hotmail.com", "expected": True, "category": "eposta"},
    {"message": "test@university.edu.tr adresime gönderin", "expected": True, "category": "eposta"},
    # --- IBAN ---
    {"message": "IBAN numaram TR33 0006 1005 1978 6457 8413 26", "expected": True, "category": "iban"},
    {"message": "TR330006100519786457841326 hesabıma yatırın", "expected": True, "category": "iban"},
    # --- Kredi Kartı ---
    {"message": "Kart numaram 4532 1234 5678 9012", "expected": True, "category": "kredi_karti"},
    {"message": "5412-7534-0000-1234 kartımla ödeme", "expected": True, "category": "kredi_karti"},
    # --- Pasaport ---
    {"message": "Pasaport numaram U12345678", "expected": True, "category": "pasaport"},
    # --- Şifre (sadece embedding katmanı yakalayabilir) ---
    {"message": "Şifrem abc123456 olarak ayarladım", "expected": True, "category": "sifre"},
    # --- Adres (sadece embedding katmanı yakalayabilir) ---
    {"message": "Ev adresim Atatürk Caddesi No:15 Kadıköy İstanbul", "expected": True, "category": "adres"},
    # --- Doğum Tarihi (sadece embedding katmanı yakalayabilir) ---
    {"message": "Doğum tarihim 15 Mart 1995", "expected": True, "category": "dogum_tarihi"},
    # --- Non-PII: Akademik Sorular ---
    {"message": "Fotosantez nedir açıklar mısınız?", "expected": False, "category": "akademik"},
    {"message": "Newton'un hareket yasalarını anlatır mısınız?", "expected": False, "category": "akademik"},
    {"message": "Mitoz ve mayoz bölünme arasındaki fark nedir?", "expected": False, "category": "akademik"},
    {"message": "Osmanlı İmparatorluğu ne zaman kuruldu?", "expected": False, "category": "akademik"},
    {"message": "Integral nasıl hesaplanır?", "expected": False, "category": "akademik"},
    {"message": "Python'da liste nasıl sıralanır?", "expected": False, "category": "teknik"},
    {"message": "HTTP protokolü nasıl çalışır?", "expected": False, "category": "teknik"},
    {"message": "TCP/IP katmanları nelerdir?", "expected": False, "category": "teknik"},
    {"message": "Bilgisayar ağlarında IP adresi ne işe yarar?", "expected": False, "category": "teknik"},
    {"message": "Algoritma karmaşıklığı O(n log n) ne demek?", "expected": False, "category": "teknik"},
    # --- Non-PII: Ders Soruları ---
    {"message": "Bu dersin final sınavı kaç puan üzerinden?", "expected": False, "category": "ders"},
    {"message": "Ödev teslim tarihi ne zaman?", "expected": False, "category": "ders"},
    {"message": "Sınav sonuçları ne zaman açıklanacak?", "expected": False, "category": "ders"},
    {"message": "Bu konuyu 3 maddede özetler misin?", "expected": False, "category": "ders"},
    {"message": "Makine öğrenmesi ile derin öğrenme arasındaki fark nedir?", "expected": False, "category": "teknik"},
    {"message": "Küresel ısınmanın nedenleri nelerdir?", "expected": False, "category": "akademik"},
]


def calc_metrics(results):
    """Precision, Recall, F1 hesapla."""
    tp = sum(1 for r in results if r["expected"] and r["detected"])
    fp = sum(1 for r in results if not r["expected"] and r["detected"])
    fn = sum(1 for r in results if r["expected"] and not r["detected"])
    tn = sum(1 for r in results if not r["expected"] and not r["detected"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "total": len(results),
    }


def run_regex_evaluation():
    """Katman 1 (Regex) değerlendirmesi — offline, API gerektirmez."""
    # pii_detection_service'i import et
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    from app.services.pii_detection_service import detect_pii_regex

    results = []
    for case in PII_TEST_CASES:
        is_pii, score, label = detect_pii_regex(case["message"])
        results.append({
            "message": case["message"],
            "expected": case["expected"],
            "detected": is_pii,
            "score": score,
            "label": label,
            "category": case["category"],
            "correct": is_pii == case["expected"],
        })

    return results


def run_api_evaluation(base_url, token, course_id):
    """Katman 1+2 (Regex + Embedding) değerlendirmesi — API üzerinden."""
    import requests

    results = []
    for i, case in enumerate(PII_TEST_CASES):
        try:
            # PII filtresi açık olan bir derse mesaj gönder
            resp = requests.post(
                f"{base_url}/api/courses/{course_id}/chat",
                json={"message": case["message"], "history": [], "search_type": "hybrid"},
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            resp_data = resp.json()

            # PII tespit edilmişse "kişisel bilgi tespit edildi" mesajı döner
            detected = "kişisel bilgi tespit edildi" in resp_data.get("message", "").lower()

            results.append({
                "message": case["message"],
                "expected": case["expected"],
                "detected": detected,
                "response": resp_data.get("message", "")[:100],
                "category": case["category"],
                "correct": detected == case["expected"],
            })
            print(f"  [{i+1}/{len(PII_TEST_CASES)}] {'✓' if detected == case['expected'] else '✗'} {case['message'][:50]}...")
        except Exception as e:
            print(f"  [{i+1}/{len(PII_TEST_CASES)}] HATA: {e}")
            results.append({
                "message": case["message"],
                "expected": case["expected"],
                "detected": False,
                "error": str(e),
                "category": case["category"],
                "correct": False,
            })
        time.sleep(0.5)

    return results


def print_report(title, metrics, false_positives, false_negatives):
    """Raporu yazdır."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Toplam test  : {metrics['total']}")
    print(f"  TP: {metrics['tp']}  |  FP: {metrics['fp']}  |  FN: {metrics['fn']}  |  TN: {metrics['tn']}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  F1 Score     : {metrics['f1']:.4f}")

    if false_positives:
        print(f"\n  False Positives ({len(false_positives)}):")
        for r in false_positives:
            print(f"    - [{r.get('label', '?')}] {r['message'][:60]} ({r['category']})")

    if false_negatives:
        print(f"\n  False Negatives ({len(false_negatives)}):")
        for r in false_negatives:
            print(f"    - {r['message'][:60]} ({r['category']})")


def main():
    parser = argparse.ArgumentParser(description="PII Filtreleme Performans Değerlendirmesi")
    parser.add_argument("--with-api", action="store_true", help="API üzerinden tam test (regex + embedding)")
    parser.add_argument("--course-id", type=int, default=1, help="Ders ID (API testi için)")
    parser.add_argument("--email", default="admin@test.com", help="Kullanıcı e-posta")
    parser.add_argument("--password", default="123456", help="Kullanıcı şifre")
    args = parser.parse_args()

    print("=" * 60)
    print("KVKK Uyumlu PII Filtreleme — Performans Değerlendirmesi")
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test sayısı: {len(PII_TEST_CASES)}")
    print("=" * 60)

    # --- Katman 1: Regex ---
    print("\n[1] Regex Katmanı Değerlendirmesi (offline)...")
    regex_results = run_regex_evaluation()
    regex_metrics = calc_metrics(regex_results)
    regex_fp = [r for r in regex_results if not r["expected"] and r["detected"]]
    regex_fn = [r for r in regex_results if r["expected"] and not r["detected"]]
    print_report("Katman 1: Regex Tabanlı PII Tespiti", regex_metrics, regex_fp, regex_fn)

    output = {
        "timestamp": datetime.now().isoformat(),
        "test_count": len(PII_TEST_CASES),
        "regex_layer": {
            "metrics": regex_metrics,
            "false_positives": [{"message": r["message"], "category": r["category"]} for r in regex_fp],
            "false_negatives": [{"message": r["message"], "category": r["category"]} for r in regex_fn],
        },
    }

    # --- Katman 1+2: API (opsiyonel) ---
    if args.with_api:
        print("\n[2] API Değerlendirmesi (regex + embedding)...")
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
            from utils import load_config, get_auth_token

            config = load_config()
            base_url = config["api_base_url"]
            token = get_auth_token(base_url, args.email, args.password)

            api_results = run_api_evaluation(base_url, token, args.course_id)
            api_metrics = calc_metrics(api_results)
            api_fp = [r for r in api_results if not r["expected"] and r["detected"]]
            api_fn = [r for r in api_results if r["expected"] and not r["detected"]]
            print_report("Katman 1+2: Regex + Embedding (API)", api_metrics, api_fp, api_fn)

            output["api_layer"] = {
                "metrics": api_metrics,
                "false_positives": [{"message": r["message"], "category": r["category"]} for r in api_fp],
                "false_negatives": [{"message": r["message"], "category": r["category"]} for r in api_fn],
            }
        except Exception as e:
            print(f"\n  API testi başarısız: {e}")
            print("  (Sistem çalışıyor mu? PII filtresi açık mı?)")

    # Sonuçları kaydet
    results_dir = "results"
    os.makedirs(os.path.join(os.path.dirname(__file__), results_dir), exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), results_dir, "pii_evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
