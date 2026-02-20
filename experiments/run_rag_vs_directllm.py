"""
RAG vs Direct LLM Karşılaştırma Betiği

Aynı sorular üzerinde RAG tabanlı ve yalın LLM yanıtlarını karşılaştırır.
Her iki mod için RAGAS ve ROUGE/BERTScore metriklerini hesaplar.

Kullanım:
    python run_rag_vs_directllm.py --course-id 1 --test-set-id 1
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


def get_course_settings(base_url, token, course_id):
    """Ders ayarlarını çek."""
    resp = requests.get(
        f"{base_url}/api/courses/{course_id}/settings",
        headers=auth_headers(token),
    )
    resp.raise_for_status()
    return resp.json()


def update_course_settings(base_url, token, course_id, updates):
    """Ders ayarlarını güncelle."""
    resp = requests.put(
        f"{base_url}/api/courses/{course_id}/settings",
        json=updates,
        headers=auth_headers(token),
    )
    resp.raise_for_status()
    return resp.json()


def send_chat_message(base_url, token, course_id, message):
    """Chat API'sine mesaj gönder ve yanıt al."""
    payload = {
        "message": message,
        "history": [],
        "search_type": "hybrid",
    }
    resp = requests.post(
        f"{base_url}/api/courses/{course_id}/chat",
        json=payload,
        headers=auth_headers(token),
    )
    resp.raise_for_status()
    return resp.json()


def run_semantic_similarity(base_url, token, course_id, question, answer, ground_truth, config):
    """Yanıt için ROUGE ve BERTScore hesapla."""
    payload = {
        "course_id": course_id,
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": answer,
        "embedding_provider": "openai",
        "embedding_model": config.get("embedding_model", "openai/text-embedding-3-small"),
    }
    resp = requests.post(
        f"{base_url}/api/semantic-similarity/quick-test",
        json=payload,
        headers=auth_headers(token),
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="RAG vs Direct LLM Karşılaştırma Betiği")
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
    print("RAG vs Direct LLM Karşılaştırma Deneyi")
    print("=" * 60)

    # Kimlik doğrulama
    print("\n[1/5] Kimlik doğrulama...")
    token = get_auth_token(base_url, args.email, args.password)
    print("  Başarılı.")

    # Mevcut ayarları kaydet
    print("\n[2/5] Mevcut ders ayarları kaydediliyor...")
    original_settings = get_course_settings(base_url, token, course_id)
    original_direct_llm = original_settings.get("enable_direct_llm", False)
    print(f"  Direct LLM modu: {'Açık' if original_direct_llm else 'Kapalı'}")

    # Test sorularını çek
    print(f"\n[3/5] Test seti #{test_set_id} yükleniyor...")
    questions = fetch_test_questions(base_url, token, test_set_id)
    print(f"  {len(questions)} soru bulundu.")

    results = {"rag": [], "direct_llm": []}

    # --- RAG Modu ---
    print("\n[4/5] RAG modu ile değerlendirme...")
    update_course_settings(base_url, token, course_id, {"enable_direct_llm": False})
    time.sleep(1)

    for i, q in enumerate(questions):
        try:
            # RAG yanıtı al
            chat_resp = send_chat_message(base_url, token, course_id, q["question"])
            answer = chat_resp.get("message", "")
            sources = chat_resp.get("sources", [])

            # Metrikleri hesapla
            sim_resp = run_semantic_similarity(
                base_url, token, course_id,
                q["question"], answer, q["ground_truth"], config
            )

            results["rag"].append({
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "generated_answer": answer,
                "source_count": len(sources),
                "rouge1": sim_resp.get("rouge1"),
                "rouge2": sim_resp.get("rouge2"),
                "rougel": sim_resp.get("rougel"),
                "bertscore_f1": sim_resp.get("bertscore_f1"),
                "cosine_similarity": sim_resp.get("cosine_similarity"),
            })
            print(f"  RAG [{i+1}/{len(questions)}] OK")
        except Exception as e:
            print(f"  RAG [{i+1}/{len(questions)}] HATA: {e}")
            results["rag"].append({"question": q["question"], "error": str(e)})
        time.sleep(1)

    # --- Direct LLM Modu ---
    print("\n[5/5] Direct LLM modu ile değerlendirme...")
    update_course_settings(base_url, token, course_id, {"enable_direct_llm": True})
    time.sleep(1)

    for i, q in enumerate(questions):
        try:
            chat_resp = send_chat_message(base_url, token, course_id, q["question"])
            answer = chat_resp.get("message", "")

            sim_resp = run_semantic_similarity(
                base_url, token, course_id,
                q["question"], answer, q["ground_truth"], config
            )

            results["direct_llm"].append({
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "generated_answer": answer,
                "rouge1": sim_resp.get("rouge1"),
                "rouge2": sim_resp.get("rouge2"),
                "rougel": sim_resp.get("rougel"),
                "bertscore_f1": sim_resp.get("bertscore_f1"),
                "cosine_similarity": sim_resp.get("cosine_similarity"),
            })
            print(f"  Direct LLM [{i+1}/{len(questions)}] OK")
        except Exception as e:
            print(f"  Direct LLM [{i+1}/{len(questions)}] HATA: {e}")
            results["direct_llm"].append({"question": q["question"], "error": str(e)})
        time.sleep(1)

    # Orijinal ayarları geri yükle
    update_course_settings(base_url, token, course_id, {"enable_direct_llm": original_direct_llm})

    # Özet hesapla
    def calc_avg(items, key):
        vals = [r[key] for r in items if key in r and r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    print("\n" + "=" * 60)
    print("RAG vs Direct LLM Karşılaştırma Sonuçları")
    print("=" * 60)

    metric_keys = ["rouge1", "rouge2", "rougel", "bertscore_f1", "cosine_similarity"]
    metric_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1", "Cosine Similarity"]

    print(f"\n  {'Metrik':25s} {'RAG':>10s} {'Direct LLM':>12s} {'Fark':>10s}")
    print("  " + "-" * 60)

    summary = {}
    for name, key in zip(metric_names, metric_keys):
        rag_avg = calc_avg(results["rag"], key)
        llm_avg = calc_avg(results["direct_llm"], key)
        diff = (rag_avg - llm_avg) if rag_avg is not None and llm_avg is not None else None

        rag_str = f"{rag_avg:.4f}" if rag_avg is not None else "N/A"
        llm_str = f"{llm_avg:.4f}" if llm_avg is not None else "N/A"
        diff_str = f"{diff:+.4f}" if diff is not None else "N/A"

        print(f"  {name:25s} {rag_str:>10s} {llm_str:>12s} {diff_str:>10s}")
        summary[key] = {"rag": rag_avg, "direct_llm": llm_avg, "diff": diff}

    # Sonuçları kaydet
    results_dir = config.get("results_dir", "results")
    os.makedirs(os.path.join(os.path.dirname(__file__), results_dir), exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), results_dir, "rag_vs_directllm_results.json")

    output = {
        "summary": summary,
        "rag_results": results["rag"],
        "direct_llm_results": results["direct_llm"],
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Sonuçlar kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
