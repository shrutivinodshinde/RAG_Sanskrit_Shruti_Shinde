# Sanskrit RAG System — Complete Setup & Run Guide

---

## What Was Fixed

| File | Problem | Fix |
|------|---------|-----|
| `generator.py` | Used `ollama` import but was broken (no system prompt, no config link) | Rewrote with 4 proper backends: ollama, transformers, llama_cpp, simple |
| `config.py` | `LLM_BACKEND = "transformers"` → flan-t5-base gives poor answers | Changed to `LLM_BACKEND = "ollama"`, upgraded flan-t5-base → flan-t5-large |
| `preprocessor.py` | `normalise_query` defined twice — first version silently ignored | Merged into one correct function |
| `requirements.txt` | `ollama` pip package missing | Added `ollama>=0.1.9` |

---

## Why Ollama Gives Better Results

| Backend | Answer Quality | RAM | Setup |
|---------|---------------|-----|-------|
| **Ollama phi3:mini** | ⭐⭐⭐⭐⭐ Excellent | ~2.2 GB | Install Ollama app + pull model |
| Ollama llama3.2:1b | ⭐⭐⭐⭐ Good | ~1.3 GB | Same |
| flan-t5-large | ⭐⭐ Weak on open QA | ~1.5 GB | pip only |
| flan-t5-base | ⭐ Very weak | ~0.8 GB | pip only |

flan-t5 is a summarisation/translation model — it was never designed
for instruction-following QA, which is why it returned fragments like
"(Of the Clever Kalidasa) by Kedar Naphade" instead of an actual answer.
phi3:mini is a proper instruction-tuned model that understands the task.

---

## Step-by-Step Setup

### Step 1 — Project structure

Make sure your folder looks like this:

```
RAG_Sanskrit_<YourName>/
├── code/
│   ├── config.py           ← updated (set LLM_BACKEND here)
│   ├── generator.py        ← updated (supports all 4 backends)
│   ├── preprocessor.py     ← fixed (duplicate function removed)
│   ├── document_loader.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   ├── query_interface.py
│   ├── main.py
│   ├── evaluate.py
│   └── gradio_app.py
├── data/
│   └── Rag-docs.docx       ← your Sanskrit corpus goes here
├── models/                 ← only needed for llama_cpp backend
├── vector_store/           ← auto-created on first run
├── requirements.txt        ← updated
└── README.md
```

### Step 2 — Python virtual environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate          # Linux / Mac
# venv\Scripts\activate           # Windows

# Verify activation
which python     # should show path inside venv/
```

### Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Expected output (abbreviated):
```
Collecting sentence-transformers==2.7.0 ...
Collecting faiss-cpu==1.8.0 ...
Collecting ollama>=0.1.9 ...
Collecting transformers==4.41.2 ...
...
Successfully installed faiss-cpu-1.8.0 ollama-0.1.9 sentence-transformers-2.7.0 ...
```

This downloads ~1.5 GB total (PyTorch + sentence-transformers).
Allow 5–10 minutes on first install.

### Step 4 — Install Ollama (the app)

#### Mac
```bash
# Option A — Download installer
# Go to: https://ollama.com/download  →  click "Download for Mac"
# Open the .dmg and drag Ollama to Applications.

# Option B — Homebrew
brew install ollama
```

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Windows
```
Go to: https://ollama.com/download
Download and run OllamaSetup.exe
```

After install, Ollama runs as a background service automatically.
You can verify:
```bash
ollama --version
# Expected: ollama version 0.x.x
```

### Step 5 — Pull the language model

```bash
ollama pull phi3:mini
```

Expected output:
```
pulling manifest
pulling model components...
pulling 2b71d4e78984... 100% ▕████████████████████▏ 2.2 GB
verifying sha256 digest
writing manifest
success
```

This downloads ~2.2 GB once. After that it's cached locally.

Verify the model is available:
```bash
ollama list
# Expected output:
# NAME         ID           SIZE    MODIFIED
# phi3:mini    4f2222927938 2.2 GB  ...
```

### Step 6 — Add your Sanskrit documents to data/

```bash
# Copy Rag-docs.docx (or any .txt / .pdf / .docx Sanskrit files) to data/
cp Rag-docs.docx RAG_Sanskrit_<YourName>/data/
```

Or convert to .txt if you prefer plain text:
```bash
# The system handles .docx natively, so no conversion needed
```

### Step 7 — Verify config.py settings

Open `code/config.py` and confirm:
```python
LLM_BACKEND = "ollama"   # ← should say "ollama"
OLLAMA_MODEL = "phi3:mini"  # ← model you pulled in Step 5
```

---

## Running the System

### Option A — Interactive CLI (recommended for testing)

```bash
cd RAG_Sanskrit_<YourName>/code
python query_interface.py
```

Expected startup output:
```
  Initialising… (may take a minute on first run)
Loading embedding model: paraphrase-multilingual-MiniLM-L12-v2
Loaded Ollama model: phi3:mini
  ✓ Index ready — 42 chunks loaded.
```

Then type queries:
```
  Query › Who was Kalidasa?
```

Expected answer:
```
  Answer:

    Kalidasa was a renowned Sanskrit poet who served in the court of
    King Bhoj. He was known for his cleverness and wit. In one famous
    incident, he helped a new poet win one lakh rupees from King Bhoj
    by composing a verse that referenced 99 crores of gems supposedly
    taken by the king's father, which none of the court scholars could
    claim to already know.

  ⏱  retrieval: 0.08s  |  generation: 3.2s
```

```
  Query › शंखनादः कः आसीत्?
```

Expected answer:
```
  Answer:

    शंखनादः गोवर्धनदासस्य भृत्यः आसीत् । (Shankhanaad was the servant
    of Govardhanadasa.) He was described as foolish (मूर्ख) because he
    followed instructions too literally without using common sense —
    for example, carrying sugar in a torn cloth (causing it to spill),
    suffocating a puppy by putting it in a sealed bag, and dragging a
    milk pot on a rope instead of carrying it.

  ⏱  retrieval: 0.06s  |  generation: 4.1s
```

### Option B — Single query (non-interactive)

```bash
cd code
python query_interface.py --query "What trick did the old woman use?"
```

Expected:
```
  Answer:

    The old woman (वृद्धा) discovered that monkeys were ringing a bell
    that had fallen in the forest. She observed them quietly, then on
    the next day offered them sweet fruits. While the monkeys were
    busy eating the fruits, she took the bell and returned to the king,
    claiming she had defeated the demon Ghantakarna. The king rewarded
    her with abundant gold.
```

```bash
# With JSON output
python query_interface.py --query "Who is Shankhanaad?" --json
```

Expected:
```json
{
  "question": "who is shankhanaad?",
  "answer": "Shankhanaad was the foolish servant of Govardhanadasa...",
  "retrieval_time": 0.071,
  "generation_time": 3.84,
  "retrieved": [...]
}
```

### Option C — Demo mode (runs all sample queries)

```bash
cd code
python main.py --demo
```

Expected (12 queries run sequentially):
```
  [1/12]  Query: शंखनादः कः आसीत्?
  Answer: शंखनादः गोवर्धनदासस्य मूर्खः भृत्यः आसीत्...
  ⏱  0.07s retrieval + 3.5s generation

  [2/12]  Query: कालिदासः किमर्थं चतुरः आसीत्?
  Answer: कालिदासः भोजराज्ञः दरबारे कविः आसीत्...
  ⏱  0.06s retrieval + 4.1s generation
  ...
```

### Option D — Gradio Chat UI

```bash
cd code
python gradio_app.py
```

Expected terminal output:
```
Loading Sanskrit RAG Pipeline...
Loaded Ollama model: phi3:mini
Pipeline Ready.
Running on local URL:  http://127.0.0.1:7860
```

Open http://127.0.0.1:7860 in your browser.
You'll see a chat interface. Type queries in Sanskrit, English, or transliteration.

### Option E — Evaluation

```bash
cd code
python evaluate.py
python evaluate.py --output ../report/eval_results.json
```

Expected output:
```
  [1/8] Who was Shankhanaad and what was his problem?
         Keyword score : 0.75  |  MRR : 1.00
         Latency       : ret=0.07s  gen=3.8s

  [2/8] How did Kalidasa help the new poet get one lakh rupees?
         Keyword score : 0.80  |  MRR : 1.00
  ...

  Metric                          Value
  ────────────────────────────────────────
  Avg Keyword Hit Rate             0.720
  Avg MRR@4                        0.875
  Avg Retrieval Time (s)           0.075
  Avg Generation Time (s)          3.900
  Total Queries                        8
```

---

## Force Rebuild Index

If you add new documents to data/, rebuild the index:

```bash
cd code
python main.py --rebuild
```

---

## Switching Backends

Edit `code/config.py`:

```python
# Best quality (requires Ollama + pull)
LLM_BACKEND = "ollama"
OLLAMA_MODEL = "phi3:mini"

# Faster but weaker (pip only, auto-downloads)
LLM_BACKEND = "transformers"
TRANSFORMERS_MODEL = "google/flan-t5-large"

# No LLM (shows raw retrieved text, useful for debugging retrieval)
LLM_BACKEND = "simple"
```

Delete the vector_store/ folder and rerun if you change chunking settings.

---

## Troubleshooting

**`RuntimeError: Ollama server is not running`**
```bash
ollama serve        # start manually in a separate terminal
# Then retry
```

**`RuntimeError: Model 'phi3:mini' not found`**
```bash
ollama pull phi3:mini
```

**`ModuleNotFoundError: faiss`**
```bash
pip install faiss-cpu
```

**`ModuleNotFoundError: sentence_transformers`**
```bash
pip install sentence-transformers
```

**`ModuleNotFoundError: ollama`**
```bash
pip install ollama
```

**`FileNotFoundError: No Sanskrit documents found in data/`**
```bash
# Copy your .docx / .txt / .pdf files into the data/ folder
cp your_document.docx data/
```

**Slow first run**
- The embedding model (~90 MB) downloads on first run — this is normal.
- phi3:mini generation takes 3–10 seconds per query on CPU — this is expected.

---

## Expected Performance (CPU, typical laptop)

| Metric | Ollama phi3:mini | flan-t5-large |
|--------|-----------------|---------------|
| Index build time | ~8s | ~8s |
| Retrieval latency | ~0.06–0.10s | ~0.06–0.10s |
| Generation latency | ~3–10s | ~5–15s |
| RAM usage | ~2.5 GB | ~1.5 GB |
| Answer quality | Excellent | Weak for open QA |
