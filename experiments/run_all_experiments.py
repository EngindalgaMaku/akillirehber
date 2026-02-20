"""
Tüm Deneyleri Çalıştırma Betiği

Makalede sunulan üç deneyi sırasıyla çalıştırır:
1. RAGAS Değerlendirmesi
2. ROUGE ve BERTScore Değerlendirmesi
3. RAG vs Direct LLM Karşılaştırması

Kullanım:
    python run_all_experiments.py --course-id 1 --test-set-id 1
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

from utils import load_config


def run_script(script_name, args_list):
    """Alt betiği çalıştır."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    cmd = [sys.executable, script_path] + args_list
    print(f"\n{'='*60}")
    print(f"Çalıştırılıyor: {script_name}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=script_dir)
    return result.returncode


def generate_summary_csv(results_dir):
    """Tüm sonuçları birleştirip özet CSV oluştur."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_results_dir = os.path.join(script_dir, results_dir)

    rows = []

    # RAGAS sonuçları
    ragas_path = os.path.join(full_results_dir, "ragas_results.json")
    if os.path.exists(ragas_path):
        with open(ragas_path, "r", encoding="utf-8") as f:
            ragas = json.load(f)
        rows.append({
            "Deney": "RAGAS",
            "Metrik": "Faithfulness",
            "Değer": ragas.get("avg_faithfulness"),
        })
        rows.append({
            "Deney": "RAGAS",
            "Metrik": "Answer Relevancy",
            "Değer": ragas.get("avg_answer_relevancy"),
        })
        rows.append({
            "Deney": "RAGAS",
            "Metrik": "Context Precision",
            "Değer": ragas.get("avg_context_precision"),
        })
        rows.append({
            "Deney": "RAGAS",
            "Metrik": "Context Recall",
            "Değer": ragas.get("avg_context_recall"),
        })
        rows.append({
            "Deney": "RAGAS",
            "Metrik": "Answer Correctness",
            "Değer": ragas.get("avg_answer_correctness"),
        })

    # ROUGE/BERTScore sonuçları
    rouge_path = os.path.join(full_results_dir, "rouge_bertscore_results.json")
    if os.path.exists(rouge_path):
        with open(rouge_path, "r", encoding="utf-8") as f:
            rouge = json.load(f)
        avgs = rouge.get("averages", {})
        for key, label in [
            ("rouge1", "ROUGE-1"),
            ("rouge2", "ROUGE-2"),
            ("rougel", "ROUGE-L"),
            ("bertscore_precision", "BERTScore Precision"),
            ("bertscore_recall", "BERTScore Recall"),
            ("bertscore_f1", "BERTScore F1"),
        ]:
            rows.append({
                "Deney": "ROUGE/BERTScore",
                "Metrik": label,
                "Değer": avgs.get(key),
            })

    # RAG vs Direct LLM sonuçları
    compare_path = os.path.join(full_results_dir, "rag_vs_directllm_results.json")
    if os.path.exists(compare_path):
        with open(compare_path, "r", encoding="utf-8") as f:
            compare = json.load(f)
        summary = compare.get("summary", {})
        for key, label in [
            ("rouge1", "ROUGE-1"),
            ("rouge2", "ROUGE-2"),
            ("rougel", "ROUGE-L"),
            ("bertscore_f1", "BERTScore F1"),
            ("cosine_similarity", "Cosine Similarity"),
        ]:
            vals = summary.get(key, {})
            rows.append({
                "Deney": "RAG vs LLM — RAG",
                "Metrik": label,
                "Değer": vals.get("rag"),
            })
            rows.append({
                "Deney": "RAG vs LLM — Direct LLM",
                "Metrik": label,
                "Değer": vals.get("direct_llm"),
            })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(full_results_dir, "summary.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nÖzet tablo kaydedildi: {csv_path}")
        print(f"\n{df.to_string(index=False)}")
    else:
        print("\nSonuç dosyası bulunamadı.")


def main():
    parser = argparse.ArgumentParser(description="Tüm Deneyleri Çalıştır")
    parser.add_argument("--course-id", type=int, help="Ders ID")
    parser.add_argument("--test-set-id", type=int, help="Test seti ID")
    parser.add_argument("--email", default="admin@test.com", help="Kullanıcı e-posta")
    parser.add_argument("--password", default="123456", help="Kullanıcı şifre")
    parser.add_argument("--skip-ragas", action="store_true", help="RAGAS deneyini atla")
    parser.add_argument("--skip-rouge", action="store_true", help="ROUGE/BERTScore deneyini atla")
    parser.add_argument("--skip-compare", action="store_true", help="Karşılaştırma deneyini atla")
    parser.add_argument("--skip-pii", action="store_true", help="PII filtreleme deneyini atla")
    args = parser.parse_args()

    config = load_config()

    common_args = []
    if args.course_id:
        common_args += ["--course-id", str(args.course_id)]
    if args.test_set_id:
        common_args += ["--test-set-id", str(args.test_set_id)]
    common_args += ["--email", args.email, "--password", args.password]

    print("=" * 60)
    print("AkıllıRehber — Tüm Deneyler")
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. RAGAS
    if not args.skip_ragas:
        rc = run_script("run_ragas_evaluation.py", common_args)
        if rc != 0:
            print(f"\nRAGAS deneyi başarısız (kod: {rc})")

    # 2. ROUGE/BERTScore
    if not args.skip_rouge:
        rc = run_script("run_rouge_bertscore.py", common_args)
        if rc != 0:
            print(f"\nROUGE/BERTScore deneyi başarısız (kod: {rc})")

    # 3. RAG vs Direct LLM
    if not args.skip_compare:
        rc = run_script("run_rag_vs_directllm.py", common_args)
        if rc != 0:
            print(f"\nKarşılaştırma deneyi başarısız (kod: {rc})")

    # 4. PII Filtreleme
    if not args.skip_pii:
        rc = run_script("run_pii_evaluation.py", [])
        if rc != 0:
            print(f"\nPII filtreleme deneyi başarısız (kod: {rc})")

    # Özet tablo oluştur
    print("\n" + "=" * 60)
    print("Özet Tablo Oluşturuluyor...")
    print("=" * 60)
    generate_summary_csv(config.get("results_dir", "results"))

    print("\n" + "=" * 60)
    print("Tüm deneyler tamamlandı.")
    print("=" * 60)


if __name__ == "__main__":
    main()
