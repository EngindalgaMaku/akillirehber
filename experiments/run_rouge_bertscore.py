"""
ROUGE ve BERTScore Değerlendirme Betiği

Bloom veri setindeki sorular üzerinde metin benzerliği metriklerini hesaplar:
- ROUGE-1, ROUGE-2, ROUGE-L
- BERTScore (Precision, Recall, F1)

Kullanım:
    python run_rouge_bertscore.py --course-id 1 --test-set-id 1
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
    return resp.json().get("questions", [])


def run_semantic_similarity_test(base_url, token, course_id, question, config):
    """Tek bir soru için ROUGE ve BERTScore hesapla."""
    payload = {
        "course_id": course_id,
        "question": question["question"],
        "ground_truth": question["ground_truth"],
        "alternative_ground_truths": question.get("alternative_ground_truths"),
        "embedding_provider": "openai",
        "embedding_model": config.get("embedding_model", "openai/text-embedding-3-small"),
        "llm_provider": config.get("llm_provider", "openrouter"),
        "llm_model": config.get("llm_model", "openai/gpt-4o-mini"),
    }
    resp = requests.post(
        f"{base_url}/api/semantic-similarity/quick-test",
        json=payload,
        headers=auth_headers(token),
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="ROUGE ve BERTScore Değerlendirme Betiği")
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
    print("ROUGE ve BERTScore Değerlendirme Deneyi")
    print("=" * 60)

    # Kimlik doğrulama
    print("\n[1/3] Kimlik doğrulama...")
    token = get_auth_token(base_url, args.email, args.password)
    print("  Başarılı.")

    # Test sorularını çek
    print(f"\n[2/3] Test seti #{test_set_id} yükleniyor...")
    questions = fetch_test_questions(base_url, token, test_set_id)
    print(f"  {len(questions)} soru bulundu.")

    # Her soru için değerlendirme yap
    print(f"\n[3/3] Değerlendirme yapılıyor...")
    all_results = []
    totals = {
        "rouge1": 0, "rouge2": 0, "rougel": 0,
        "bertscore_precision": 0, "bertscore_recall": 0, "bertscore_f1": 0,
    }
    success_count = 0

    for i, q in enumerate(questions):
        try:
            result = run_semantic_similarity_test(base_url, token, course_id, q, config)
            all_results.append(result)

            for key in totals:
                val = result.get(key)
                if val is not None:
                    totals[key] += val

            success_count += 1
            print(f"  [{i+1}/{len(questions)}] {q['question'][:50]}... OK")
        except Exception as e:
            print(f"  [{i+1}/{len(questions)}] HATA: {e}")
            all_results.append({"error": str(e), "question": q["question"]})

        # API rate limit koruması
        time.sleep(1)

    # Ortalama metrikleri hesapla
    print("\n" + "=" * 60)
    print("ROUGE ve BERTScore Sonuçları")
    print("=" * 60)

    if success_count > 0:
        averages = {k: v / success_count for k, v in totals.items()}
        metrics = [
            ("ROUGE-1", averages["rouge1"]),
            ("ROUGE-2", averages["rouge2"]),
            ("ROUGE-L", averages["rougel"]),
            ("BERTScore Precision", averages["bertscore_precision"]),
            ("BERTScore Recall", averages["bertscore_recall"]),
            ("BERTScore F1", averages["bertscore_f1"]),
        ]
        for name, value in metrics:
            print(f"  {name:25s}: {value:.4f}")
    else:
        averages = {}
        print("  Başarılı sonuç yok.")

    # Sonuçları kaydet
    results_dir = config.get("results_dir", "results")
    os.makedirs(os.path.join(os.path.dirname(__file__), results_dir), exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), results_dir, "rouge_bertscore_results.json")

    output = {
        "total_questions": len(questions),
        "successful": success_count,
        "averages": averages,
        "details": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
