"""
evaluate.py
-----------
Evaluation script for the Sanskrit RAG System.

Metrics computed:
  • Answer relevance  — keyword hit rate (simple proxy)
  • Retrieval quality — MRR (Mean Reciprocal Rank) using ground-truth chunk keywords
  • Latency          — mean retrieval and generation times
  • Resource usage   — peak RAM (via psutil)

Run:
    python evaluate.py
    python evaluate.py --output results.json
"""

import sys
import os
import json
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import SanskritRAGPipeline

logger = logging.getLogger(__name__)

# ── Ground-truth test set ──────────────────────────────────────────────────────
# Format: (question, [expected_keywords_in_answer], relevant_source_hint)
TEST_SET = [
    (
        "Who was Shankhanaad and what was his problem?",
        ["servant", "foolish", "govardhan", "shankhanaad"],
        "मूर्खभृत्यस्य",
    ),
    (
        "How did Kalidasa help the new poet get one lakh rupees?",
        ["lakh", "poem", "scholar", "kalidasa", "bhoj"],
        "चतुरस्य",
    ),
    (
        "What trick did the old woman use to get the bell?",
        ["monkey", "fruit", "bell", "old woman"],
        "वृद्धायाः",
    ),
    (
        "Why did the devoted person drown?",
        ["effort", "help", "devotee", "god"],
        "भक्त",
    ),
    (
        "What grammar mistake did the foreign scholar make?",
        ["atmanepadi", "badhate", "grammar", "cold"],
        "शीतं",
    ),
    (
        "What did King Bhoj promise to poets?",
        ["lakh", "rupee", "poem", "court"],
        "भोज",
    ),
    (
        "शंखनादः शर्कराम् कथम् आनीतवान्?",
        ["शर्करा", "वस्त्र", "जीर्ण"],
        "शर्करा",
    ),
    (
        "कालीदासः पण्डितं कुत्र अगच्छत्?",
        ["पालखी", "शीत", "पण्डित"],
        "पालखी",
    ),
]


def keyword_hit(answer: str, keywords: list) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords) if keywords else 0.0


def mrr_at_k(retrieved_chunks: list, relevant_hint: str, k: int = 4) -> float:
    """
    Mean Reciprocal Rank proxy:
    Check if any of the top-k retrieved chunks contains the relevant_hint string.
    Returns 1/rank if found, else 0.
    """
    for rank, chunk in enumerate(retrieved_chunks[:k], start=1):
        if relevant_hint.lower() in chunk["text"].lower():
            return 1.0 / rank
    return 0.0


def measure_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


def run_evaluation(output_path: str = None):
    print("\n" + "═" * 72)
    print("  Sanskrit RAG System — Evaluation")
    print("═" * 72 + "\n")

    pipeline = SanskritRAGPipeline()
    pipeline.ingest()

    results = []
    retrieval_times = []
    generation_times = []
    keyword_scores = []
    mrr_scores = []

    for i, (question, keywords, hint) in enumerate(TEST_SET, 1):
        print(f"  [{i}/{len(TEST_SET)}] {question[:60]}")
        try:
            mem_before = measure_memory_mb()
            resp = pipeline.query(question, return_context=True)
            mem_after = measure_memory_mb()

            k_score = keyword_hit(resp["answer"], keywords)
            m_score = mrr_at_k(resp.get("retrieved", []), hint)

            retrieval_times.append(resp["retrieval_time"])
            generation_times.append(resp["generation_time"])
            keyword_scores.append(k_score)
            mrr_scores.append(m_score)

            print(f"       Keyword score : {k_score:.2f}  |  MRR : {m_score:.2f}")
            print(f"       Latency       : ret={resp['retrieval_time']}s  gen={resp['generation_time']}s")
            print(f"       RAM delta     : {mem_after - mem_before:.1f} MB")
            print(f"       Answer snippet: {resp['answer'][:150]}\n")

            results.append({
                "question":       question,
                "answer":         resp["answer"],
                "keyword_score":  k_score,
                "mrr":            m_score,
                "retrieval_time": resp["retrieval_time"],
                "generation_time":resp["generation_time"],
            })

        except Exception as e:
            print(f"       ✗ ERROR: {e}\n")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("─" * 72)
    avg_keyword = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0
    avg_mrr     = sum(mrr_scores)     / len(mrr_scores)     if mrr_scores     else 0
    avg_ret     = sum(retrieval_times)/ len(retrieval_times) if retrieval_times else 0
    avg_gen     = sum(generation_times)/len(generation_times) if generation_times else 0

    print(f"\n  {'Metric':<30} {'Value':>10}")
    print(f"  {'─'*40}")
    print(f"  {'Avg Keyword Hit Rate':<30} {avg_keyword:>10.3f}")
    print(f"  {'Avg MRR@4':<30} {avg_mrr:>10.3f}")
    print(f"  {'Avg Retrieval Time (s)':<30} {avg_ret:>10.3f}")
    print(f"  {'Avg Generation Time (s)':<30} {avg_gen:>10.3f}")
    print(f"  {'Total Queries':<30} {len(results):>10}")
    print(f"\n{'═'*72}\n")

    summary = {
        "avg_keyword_hit_rate":   round(avg_keyword, 4),
        "avg_mrr_at_4":           round(avg_mrr, 4),
        "avg_retrieval_time_s":   round(avg_ret, 4),
        "avg_generation_time_s":  round(avg_gen, 4),
        "total_queries":          len(results),
        "per_query_results":      results,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  Results saved to: {output_path}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    run_evaluation(output_path=args.output)
