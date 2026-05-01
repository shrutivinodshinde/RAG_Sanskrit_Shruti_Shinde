"""
query_interface.py
------------------
Interactive command-line interface for the Sanskrit RAG system.

Run:
    python query_interface.py

Supports both single-shot mode (--query flag) and interactive REPL.
"""

import sys
import os
import argparse
import logging
import json

# Add code/ directory to path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import SanskritRAGPipeline
from config import DATA_DIR, TOP_K

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,   # suppress verbose library logs in CLI mode
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Pretty printing ────────────────────────────────────────────────────────────

DIVIDER = "─" * 70

def print_header():
    print(f"\n{'═'*70}")
    print("  🕉   Sanskrit RAG System  —  CPU-only Inference")
    print(f"{'═'*70}")
    print("  Type a question in Sanskrit (Devanagari), English, or")
    print("  IAST transliteration.  Commands: 'help', 'status', 'quit'\n")

def print_response(response: dict, show_context: bool = False):
    print(f"\n{DIVIDER}")
    print(f"  Question : {response['question']}")
    print(f"{DIVIDER}")
    print(f"\n  Answer:\n")
    # Indent each line of the answer
    for line in response["answer"].splitlines():
        print(f"    {line}")
    print()

    if show_context and "retrieved" in response:
        print(f"{DIVIDER}")
        print(f"  Retrieved context ({len(response['retrieved'])} chunks):\n")
        for i, chunk in enumerate(response["retrieved"], 1):
            print(f"  [{i}] Source: {chunk['source']}  Score: {chunk['score']:.3f}")
            preview = chunk["text"][:200].replace("\n", " ")
            print(f"      {preview}…\n")

    print(
        f"  ⏱  retrieval: {response['retrieval_time']}s  |  "
        f"generation: {response['generation_time']}s"
    )
    print(f"{DIVIDER}\n")

def print_help():
    print(f"""
{DIVIDER}
  Available commands:
    help          — show this message
    status        — show index statistics
    context on    — show retrieved chunks with each answer
    context off   — hide retrieved chunks
    top <n>       — set number of retrieved chunks (e.g. 'top 3')
    rebuild       — force-rebuild the index from data/
    quit / exit   — exit the program
{DIVIDER}
""")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_interactive(pipeline: SanskritRAGPipeline, top_k: int, show_context: bool):
    """Interactive REPL loop."""
    print_header()
    print("  Initialising… (may take a minute on first run)\n")

    n_chunks = pipeline.ingest()
    print(f"  ✓ Index ready — {n_chunks} chunks loaded.\n")

    while True:
        try:
            raw = input("  Query › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye! 🙏\n")
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("quit", "exit", "q"):
            print("\n  Goodbye! 🙏\n")
            break
        elif cmd == "help":
            print_help()
        elif cmd == "status":
            s = pipeline.status()
            print(f"\n  Status: {json.dumps(s, indent=4)}\n")
        elif cmd == "context on":
            show_context = True
            print("  Context display: ON\n")
        elif cmd == "context off":
            show_context = False
            print("  Context display: OFF\n")
        elif cmd.startswith("top "):
            try:
                top_k = int(cmd.split()[1])
                print(f"  top_k set to {top_k}\n")
            except (IndexError, ValueError):
                print("  Usage: top <integer>  e.g. 'top 3'\n")
        elif cmd == "rebuild":
            print("  Rebuilding index…")
            n = pipeline.ingest(force_rebuild=True)
            print(f"  ✓ Rebuilt — {n} chunks\n")
        else:
            try:
                response = pipeline.query(raw, top_k=top_k, return_context=show_context)
                print_response(response, show_context=show_context)
            except Exception as e:
                print(f"\n  ✗ Error: {e}\n")


def run_single_query(pipeline: SanskritRAGPipeline, question: str,
                     top_k: int, json_output: bool):
    """Single-shot query mode (non-interactive)."""
    pipeline.ingest()
    response = pipeline.query(question, top_k=top_k, return_context=True)

    if json_output:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print_response(response, show_context=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanskrit RAG System — CPU-only Retrieval-Augmented Generation"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Run a single query and exit (non-interactive mode)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K})",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        default=False,
        help="Show retrieved context chunks in interactive mode",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output response as JSON (only in single-query mode)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help=f"Directory containing Sanskrit documents (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Force rebuild of the FAISS index",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    pipeline = SanskritRAGPipeline()

    if args.query:
        run_single_query(pipeline, args.query, args.top_k, args.json)
    else:
        run_interactive(pipeline, args.top_k, args.context)


if __name__ == "__main__":
    main()
