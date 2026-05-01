"""
main.py
-------
Entry point for the Sanskrit RAG System.

Demonstrates the full pipeline:
  1. Ingest documents from data/
  2. Build (or load) the FAISS vector index
  3. Run a set of sample queries
  4. Print results with timing metrics

Run:
    python main.py
    python main.py --demo          # run demo queries
    python main.py --eval          # run evaluation on sample Q&A pairs
"""

import sys
import os
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import SanskritRAGPipeline
from config import LOG_LEVEL

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Sample queries ─────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    # Sanskrit questions (Devanagari)
    "शंखनादः कः आसीत्?",
    "कालिदासः किमर्थं चतुरः आसीत्?",
    "वृद्धा राक्षसं कथम् अजयत्?",
    "घण्टाकर्णः कः आसीत्?",
    "भक्तः किमर्थं मृतः?",

    # English questions
    "Who was Shankhanaad and what mistakes did he make?",
    "How did Kalidasa trick the foreign scholar?",
    "What is the moral of the story about the devotee?",
    "What reward did the old woman receive?",
    "What did King Bhoj promise to poets?",

    # Transliterated Sanskrit
    "kAlidAsa kasya darbAre AsIt?",
    "GhaNTAkarNaH kaH AsIt?",
]

# ── Evaluation pairs ───────────────────────────────────────────────────────────
# (question, key_phrase_expected_in_answer)
EVAL_PAIRS = [
    ("Who is Shankhanaad?", "servant"),
    ("What did the old woman do to the monkeys?", "fruits"),
    ("What did King Bhoj promise?", "lakh"),
    ("Why did the devotee drown?", "effort"),
    ("What language error did the foreign scholar make?", "Atmanepadi"),
]


def run_demo(pipeline: SanskritRAGPipeline):
    print("\n" + "═" * 72)
    print("  Sanskrit RAG System  —  Demo Mode")
    print("═" * 72 + "\n")

    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"  [{i}/{len(DEMO_QUERIES)}]  Query: {q}")
        try:
            resp = pipeline.query(q)
            print(f"  Answer: {resp['answer'][:300]}")
            print(
                f"  ⏱  {resp['retrieval_time']}s retrieval + "
                f"{resp['generation_time']}s generation\n"
            )
        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    print("═" * 72)


def run_eval(pipeline: SanskritRAGPipeline):
    """Simple keyword-based evaluation."""
    print("\n" + "═" * 72)
    print("  Sanskrit RAG System  —  Evaluation Mode")
    print("═" * 72 + "\n")

    correct = 0
    total   = len(EVAL_PAIRS)

    for i, (question, expected_keyword) in enumerate(EVAL_PAIRS, 1):
        try:
            resp = pipeline.query(question)
            answer = resp["answer"].lower()
            hit = expected_keyword.lower() in answer
            correct += hit
            status = "✓ PASS" if hit else "✗ FAIL"
            print(f"  [{i}] {status}")
            print(f"       Q: {question}")
            print(f"       A: {resp['answer'][:200]}")
            print(f"       Expected keyword: '{expected_keyword}'  Found: {hit}\n")
        except Exception as e:
            print(f"  [{i}] ✗ ERROR: {e}\n")

    accuracy = correct / total * 100
    print(f"  Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print("═" * 72)
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Sanskrit RAG System")
    parser.add_argument("--demo",    action="store_true", help="Run demo queries")
    parser.add_argument("--eval",    action="store_true", help="Run evaluation")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    args = parser.parse_args()

    # ── Initialise pipeline ────────────────────────────────────────────────────
    logger.info("Initialising Sanskrit RAG Pipeline…")
    pipeline = SanskritRAGPipeline()

    t0 = time.time()
    n_chunks = pipeline.ingest(force_rebuild=args.rebuild)
    elapsed  = time.time() - t0

    print(f"\n  ✓ Pipeline ready — {n_chunks} chunks indexed in {elapsed:.1f}s")
    print(f"  Backend : {__import__('config').LLM_BACKEND}")
    print(f"  Embed   : {__import__('config').EMBEDDING_MODEL}\n")

    if args.demo:
        run_demo(pipeline)
    elif args.eval:
        run_eval(pipeline)
    else:
        # Default: run a few quick sample queries
        sample_qs = [
            "Who is Kalidasa and why is he clever?",
            "शंखनादः कः आसीत्?",
            "What is the moral of the devotee story?",
        ]
        print("  Running sample queries (use --demo for all, --eval for scoring):\n")
        for q in sample_qs:
            resp = pipeline.query(q)
            print(f"  Q: {q}")
            print(f"  A: {resp['answer'][:300]}\n")
            print(f"  ⏱  {resp['retrieval_time']}s + {resp['generation_time']}s\n")
            print("  " + "─" * 68 + "\n")


if __name__ == "__main__":
    main()
