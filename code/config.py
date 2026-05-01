import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_store")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ── Embedding model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_DIM = 384

# ── Chunking ───────────────────────────────────────────────────────────────────
# Larger chunks keep Sanskrit + English pairs together (critical for this corpus)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 40

# ── Retrieval ──────────────────────────────────────────────────────────────────
# TOP_K=5 fetches more candidate chunks so specific Sanskrit queries (e.g.
# "श्वानशावकस्य मृत्युः") find the correct story even when the embedding
# similarity spreads across multiple stories.
# SIMILARITY_THRESHOLD=0.15 catches transliterated / partial Sanskrit queries.
TOP_K = 6
SIMILARITY_THRESHOLD = 0.15   # lowered so transliterated queries still match

# ── LLM Backend ───────────────────────────────────────────────────────────────
# Choose one of: "ollama" | "transformers" | "llama_cpp" | "simple"
#
#   "ollama"       → BEST quality. Requires Ollama installed + model pulled.
#                    Install: https://ollama.com/download
#                    Then run: ollama pull llama3.2:1b
#
#   "transformers" → No extra install. Auto-downloads flan-t5-large (~800 MB).
#                    Weaker answers but zero setup beyond pip install.
#
#   "llama_cpp"    → Good quality. Needs: pip install llama-cpp-python
#                    + manual GGUF model download (see README).
#
#   "simple"       → No LLM. Returns raw retrieved context (debug/testing).
#
LLM_BACKEND = "ollama"

# ── Ollama model recommendations (all CPU-friendly) ───────────────────────────
#
#  SPEED vs QUALITY trade-off — pick based on your RAM:
#
#  ┌─────────────────────────┬──────────┬──────────┬───────────────────────────┐
#  │ Model                   │ RAM      │ Speed    │ Notes                     │
#  ├─────────────────────────┼──────────┼──────────┼───────────────────────────┤
#  │ qwen2.5:1.5b ← FASTEST  │ ~1.0 GB  │ ~5-8s    │ Fastest + multilingual    │
#  │ llama3.2:1b  ← DEFAULT  │ ~1.3 GB  │ ~8-15s   │ Fast, follows instructions│
#  │ llama3.2:3b             │ ~2.0 GB  │ ~20-30s  │ Best quality for QA       │
#  │ gemma2:2b               │ ~1.6 GB  │ ~15-25s  │ Good multilingual support │
#  │ mistral:latest          │ ~4.5 GB  │ ~60-90s  │ Very slow on CPU          │
#  │ llama3:latest           │ ~4.7 GB  │ ~60-90s  │ Very slow on CPU          │
#  └─────────────────────────┴──────────┴──────────┴───────────────────────────┘
#
#  ❌ AVOID phi3:mini — hallucinates Sanskrit text instead of reading context
#  ❌ AVOID mistral:latest / llama3:latest on CPU — 60-90s per query
#
#  Setup (one-time):
#    ollama pull llama3.2:1b      # recommended default (~8-15s)
#    ollama pull qwen2.5:1.5b     # fastest option (~5-8s)
#    ollama pull llama3.2:3b      # better quality (~20-30s)
#
OLLAMA_MODEL = "llama3.2:1b"    # fast default — change to llama3.2:3b for better quality
OLLAMA_HOST  = "http://localhost:11434"

# ── Transformers settings ──────────────────────────────────────────────────────
TRANSFORMERS_MODEL = "google/flan-t5-large"
TRANSFORMERS_MAX_NEW_TOKENS = 150  # 150 is enough for 3-5 sentence answers; was 256
TRANSFORMERS_TEMPERATURE = 0.1     # low = more faithful to context

# ── llama.cpp settings ────────────────────────────────────────────────────────
LLAMA_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)
LLAMA_N_CTX = 2048
LLAMA_N_THREADS = 4
LLAMA_MAX_TOKENS = 150   # was 256 — shorter = faster
LLAMA_TEMPERATURE = 0.1  # low = stick to context
LLAMA_TOP_P = 0.9

# ── Vector store ───────────────────────────────────────────────────────────────
FAISS_INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
METADATA_FILE    = os.path.join(VECTOR_STORE_DIR, "metadata.json")

LOG_LEVEL = "INFO"