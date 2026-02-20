"""
RAGAS Değerlendirme Betiği

Bloom veri setindeki sorular üzerinde RAGAS metriklerini hesaplar:
- Faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness

Kullanım:
    python run_ragas_evaluation.py --course-id 1 --test-set-id 1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import requests

from utils import load_config, get_auth_token, auth_headers


def fetch_test_questions(base_url, token, test_set_id):
    """Test setindeki soruları çek."""
    resp = requests.get(
        f"{base_url}/api/ragas/test-sets/{test_set_id}",
        headers=auth_headers(token),
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("questions", [])


def start_evaluation_run(base_url, token, course_id, test_set_id, config):
    """RAGAS değerlendirme çalıştırması başlat."""
    payload = {
        "test_set_id": test_set_id,
        "course_id": course_id,
        "name": f"experiment-ragas-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "config": {
            "search_type": config.get("search_type", "hybrid"),
            "search_alpha": config.get("search_alpha", 0.5),
            "top_k": config.get("search_top_k", 5),
        },
    }
    resp = requests.post(
        f"{base_url}/api/ragas/runs",
        json=payload,
        headers=auth_headers(token),
    )
    resp.raise_for_status()
    return resp.json()


def poll_run_status(base_url, token, run_id, poll_interval=5):
    """Değerlendirme tamamlanana kadar bekle."""
    while True:
        resp = requests.get(
            f"{base_url}/api/ragas/runs/{run_id}",
            headers=auth_headers(token),
        )
        resp.raise_for_status()
        run = resp.json()
        status = run.get("status", "unknown")
        processed = run.get("processed_questions", 0)
        total = run.get("total_questions", 0)

        print(f"  Durum: {status} — {processed}/{total} soru işlendi", end="\r")

        if status in ("completed", "failed"):
            print()
            return run

        time.sleep(poll_interval)


def fetch_run_results(base_url, token, run_id):
    """Değerlendirme sonuçlarını çek."""
    resp = requests.get(
        f"{base_url}/api/ragas/runs/{run_id}",
        headers=auth_headers(token),
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="RAGAS Değerlendirme Betiği")
    parser.add_argument("--course-id", type=int, help="Ders ID")
    parser.add_argument("--test-set-id", type=int, help="Test seti ID")
    parser.add_argument("--email", default="admin@test.com", help="Kullanıcı e-posta")
    parser.add_argument("--password", default="123456", help="Kullanıcı şifre")
    args = parser.parse_args()

    config = load_config()
    base_url = config["api_base_url"]
    course_id = args.course_id or config["course_id"]
    test_set_id = args.test_set_id or config["test_set_id"]

    print("=" * 60)
    print("RAGAS Değerlendirme Deneyi")
    print("=" * 60)

    # Kimlik doğrulama
    print("\n[1/4] Kimlik doğrulama...")
    token = get_auth_token(base_url, args.email, args.password)
    print("  Başarılı.")

    # Test sorularını kontrol et
    print(f"\n[2/4] Test seti #{test_set_id} kontrol ediliyor...")
    questions = fetch_test_questions(base_url, token, test_set_id)
    print(f"  {len(questions)} soru bulundu.")

    # Değerlendirme başlat
    print(f"\n[3/4] RAGAS değerlendirmesi başlatılıyor (ders={course_id})...")
    run = start_evaluation_run(base_url, token, course_id, test_set_id, config)
    run_id = run["id"]
    print(f"  Çalıştırma ID: {run_id}")

    # Sonuçları bekle
    print("\n[4/4] Sonuçlar bekleniyor...")
    final_run = poll_run_status(base_url, token, run_id)

    if final_run.get("status") == "failed":
        print(f"\n  HATA: {final_run.get('error_message', 'Bilinmeyen hata')}")
        sys.exit(1)

    # Detaylı sonuçları çek
    results = fetch_run_results(base_url, token, run_id)

    # Özet metrikleri yazdır
    print("\n" + "=" * 60)
    print("RAGAS Sonuçları")
    print("=" * 60)
    metrics = [
        ("Faithfulness", results.get("avg_faithfulness")),
        ("Answer Relevancy", results.get("avg_answer_relevancy")),
        ("Context Precision", results.get("avg_context_precision")),
        ("Context Recall", results.get("avg_context_recall")),
        ("Answer Correctness", results.get("avg_answer_correctness")),
    ]
    for name, value in metrics:
        val_str = f"{value:.4f}" if value is not None else "N/A"
        print(f"  {name:25s}: {val_str}")

    # Sonuçları kaydet
    results_dir = config.get("results_dir", "results")
    os.makedirs(os.path.join(os.path.dirname(__file__), results_dir), exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), results_dir, "ragas_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
