"""
gradio_app.py
-------------
Gradio UI for the Sanskrit RAG System with runtime model switching.

The user can choose any generation model from a dropdown — the embedding model
and FAISS index are always fixed (they never change with model selection).

Architecture:
  Embedding model  →  converts text/query to vectors for retrieval  (FIXED)
  Generation model →  reads retrieved chunks and writes the answer  (SWAPPABLE)

Flow on each query:
  1. pipeline.retrieve_only()  — FAISS search, no LLM involved
  2. _get_generator(label)     — load or fetch cached generator
  3. generator.generate()      — called exactly ONCE with the chosen model

Run:
    python gradio_app.py
"""

import gradio as gr
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import SanskritRAGPipeline
from generator import build_generator_by_name

# ─────────────────────────────────────────────────────────────────────────────
# Available models
#
# Format: "Display Label": ("backend", "model_name_or_None")
#
# "backend" must be one of: "ollama" | "transformers" | "llama_cpp" | "simple"
# model_name=None means use the default from config.py for that backend.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_OPTIONS = {
    # ── Ollama (locally running server) ──────────────────────────────────────
    "⚡ Qwen 2.5 1.5B (Ollama) — Fastest (~5-8s)":
        ("ollama", "qwen2.5:1.5b"),

    "🐋 Llama 3.2 1B (Ollama) — Fast (~8-15s)":
        ("ollama", "llama3.2:1b"),

    "🐋 Llama 3.2 3B (Ollama) — Balanced (~20-30s)":
        ("ollama", "llama3.2:3b"),

    "🌬️ Mistral 7B (Ollama) — Slow on CPU (~60-90s)":
        ("ollama", "mistral:latest"),

    "🦙 Llama 3 8B (Ollama) — Slow on CPU (~60-90s)":
        ("ollama", "llama3:latest"),

    # ── HuggingFace Transformers (auto-downloaded, CPU-only) ─────────────────
    "🤗 Flan-T5 Large (HuggingFace) — ~800 MB (~5-15s)":
        ("transformers", "google/flan-t5-large"),

    "🤗 Flan-T5 XL (HuggingFace) — ~3 GB (~20-40s)":
        ("transformers", "google/flan-t5-xl"),

    "🤗 Flan-Alpaca Large (HuggingFace) — ~800 MB (~5-15s)":
        ("transformers", "declare-lab/flan-alpaca-large"),

    # ── Debug mode ───────────────────────────────────────────────────────────
    "🔍 Raw Context (No LLM) — Instant":
        ("simple", None),
}

# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect the best available default model
#
# Queries Ollama for pulled models at startup.  The first label in
# _PREFERRED_ORDER whose model is already pulled becomes the default.
# Falls back to Flan-T5 (HuggingFace) if Ollama is not running, and to
# Raw Context if nothing else works.
# ─────────────────────────────────────────────────────────────────────────────

_PREFERRED_ORDER = [
    "⚡ Qwen 2.5 1.5B (Ollama) — Fastest (~5-8s)",
    "🐋 Llama 3.2 1B (Ollama) — Fast (~8-15s)",
    "🐋 Llama 3.2 3B (Ollama) — Balanced (~20-30s)",
    "🌬️ Mistral 7B (Ollama) — Slow on CPU (~60-90s)",
    "🦙 Llama 3 8B (Ollama) — Slow on CPU (~60-90s)",
    "🤗 Flan-T5 Large (HuggingFace) — ~800 MB (~5-15s)",
    "🔍 Raw Context (No LLM) — Instant",
]


def _detect_default_model() -> str:
    """
    Return the label of the best model that is actually available right now.
    Checks Ollama first; falls back to HuggingFace, then Raw Context.
    """
    try:
        import ollama as _ollama
        client = _ollama.Client(host="http://localhost:11434")
        models_info = client.list()

        raw_list = (
            models_info.get("models", [])
            if isinstance(models_info, dict)
            else getattr(models_info, "models", [])
        )
        pulled = set()
        for m in raw_list:
            name = (
                (m.get("name") or m.get("model", ""))
                if isinstance(m, dict)
                else (getattr(m, "model", None) or getattr(m, "name", None) or "")
            )
            if name:
                pulled.add(str(name))
                pulled.add(str(name).split(":")[0])   # bare name without tag

        print(f"  Ollama models available: {sorted(pulled)}")

        for label in _PREFERRED_ORDER:
            backend, model_name = MODEL_OPTIONS[label]
            if backend != "ollama":
                continue
            short = model_name.split(":")[0] if model_name else ""
            if model_name in pulled or short in pulled:
                print(f"  Auto-selected default: {label}")
                return label

    except Exception as exc:
        print(f"  Ollama not reachable ({exc}). Falling back to HuggingFace.")

    hf_label = "🤗 Flan-T5 Large (HuggingFace) — ~800 MB (~5-15s)"
    print(f"  Auto-selected default: {hf_label}")
    return hf_label


DEFAULT_MODEL = _detect_default_model()

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline initialisation — runs once at startup, shared across all sessions
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Sanskrit RAG System — Loading pipeline…")
print("=" * 60)

pipeline = SanskritRAGPipeline()
pipeline.ingest()

print("  ✓ FAISS index ready")
print("  ✓ Embedding model loaded (fixed — paraphrase-multilingual-MiniLM-L12-v2)")
print("=" * 60 + "\n")

# Generator cache: { label_string → generator_instance }
# Cached so switching back to a previously loaded model is instant.
_generator_cache: dict = {}


def _get_generator(label: str):
    """
    Return a cached generator or build and cache a new one.

    Raises RuntimeError with clear instructions if the model cannot be loaded.
    """
    if label not in _generator_cache:
        backend, model_name = MODEL_OPTIONS[label]
        print(f"\n[UI] Loading model: {label}")
        try:
            gen = build_generator_by_name(backend, model_name)
            _generator_cache[label] = gen
            print(f"[UI] Model loaded and cached: {label}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{label}'.\n"
                f"Error: {e}\n\n"
                f"Tips:\n"
                f"  • Ollama models  → make sure Ollama is running (`ollama serve`) "
                f"and the model is pulled (`ollama pull {model_name}`).\n"
                f"  • HuggingFace    → check your internet connection on first load."
            )
    return _generator_cache[label]


# ─────────────────────────────────────────────────────────────────────────────
# Core respond function
# ─────────────────────────────────────────────────────────────────────────────

def respond(
    message: str,
    history: list,
    model_label: str,
    top_k: int,
    show_context: bool,
) -> str:
    """
    Called by Gradio on every user message.

    Correct flow (no double-generation):
      Step 1 — pipeline.retrieve_only()  : FAISS search only, no LLM
      Step 2 — _get_generator(label)     : load / fetch from cache
      Step 3 — generator.generate()      : called exactly ONCE

    Parameters
    ----------
    message      : the user's question
    history      : list of {"role": ..., "content": ...} dicts (Gradio messages format)
    model_label  : dropdown label — key into MODEL_OPTIONS
    top_k        : number of chunks to retrieve
    show_context : whether to append the retrieved chunks to the answer
    """
    if not message.strip():
        return "Please enter a question."

    try:
        # ── Step 1: Retrieve (no LLM) ─────────────────────────────────────────
        retrieval = pipeline.retrieve_only(question=message, top_k=int(top_k))
        question         = retrieval["question"]        # normalised
        retrieved_chunks = retrieval["retrieved"]
        context          = retrieval["context"]
        retrieval_time   = retrieval["retrieval_time"]

        # ── Step 2: Load the selected generator ───────────────────────────────
        generator = _get_generator(model_label)
        pipeline.generator = generator                 # keep pipeline in sync

        # ── Step 3: Generate exactly once ─────────────────────────────────────
        gen_result = generator.generate(question=question, context=context)
        answer     = gen_result["answer"]
        gen_time   = gen_result["generation_time"]

        # ── Build response string ─────────────────────────────────────────────
        response = answer
        response += (
            f"\n\n---\n"
            f"⏱ Retrieval: **{retrieval_time}s**  |  "
            f"Generation: **{gen_time}s**  |  "
            f"Model: **{model_label.split('—')[0].strip()}**"
        )

        if show_context:
            response += "\n\n**📄 Retrieved Context Chunks:**\n"
            for i, chunk in enumerate(retrieved_chunks, 1):
                preview = chunk["text"][:300].replace("\n", " ")
                response += (
                    f"\n**[{i}]** `{chunk['source']}`  "
                    f"score: `{chunk['score']:.3f}`\n"
                    f"> {preview}…\n"
                )

        return response

    except Exception as e:
        tb = traceback.format_exc()
        return f"❌ **Error:** {e}\n\n```\n{tb}\n```"


# ─────────────────────────────────────────────────────────────────────────────
# Model info panel content
# ─────────────────────────────────────────────────────────────────────────────

MODEL_INFO = {
    "⚡ Qwen 2.5 1.5B (Ollama) — Fastest (~5-8s)": (
        "**Qwen 2.5 1.5B** via Ollama\n\n"
        "- ✅ **Fastest option** — ~5–8s on CPU\n"
        "- Strong multilingual support (good for Sanskrit + English)\n"
        "- RAM: ~1.0 GB\n"
        "- Setup: `ollama pull qwen2.5:1.5b`"
    ),
    "🐋 Llama 3.2 1B (Ollama) — Fast (~8-15s)": (
        "**Llama 3.2 1B** via Ollama\n\n"
        "- ✅ **Recommended default** — ~8–15s on CPU\n"
        "- Good instruction following for its size\n"
        "- RAM: ~1.3 GB\n"
        "- Setup: `ollama pull llama3.2:1b`"
    ),
    "🐋 Llama 3.2 3B (Ollama) — Balanced (~20-30s)": (
        "**Llama 3.2 3B** via Ollama\n\n"
        "- ✅ Best quality/speed balance — ~20–30s on CPU\n"
        "- Noticeably better answers than 1B\n"
        "- RAM: ~2.0 GB\n"
        "- Setup: `ollama pull llama3.2:3b`"
    ),
    "🌬️ Mistral 7B (Ollama) — Slow on CPU (~60-90s)": (
        "**Mistral 7B** via Ollama\n\n"
        "- ⚠️ **Very slow on CPU** — expect 60–90s per query\n"
        "- Good quality but not worth the wait on CPU\n"
        "- RAM: ~4.5 GB\n"
        "- Setup: `ollama pull mistral:latest`"
    ),
    "🦙 Llama 3 8B (Ollama) — Slow on CPU (~60-90s)": (
        "**Llama 3 8B** via Ollama\n\n"
        "- ⚠️ **Very slow on CPU** — expect 60–90s per query\n"
        "- Best answer quality but too slow for interactive use on CPU\n"
        "- RAM: ~4.7 GB\n"
        "- Setup: `ollama pull llama3:latest`"
    ),
    "🤗 Flan-T5 Large (HuggingFace) — ~800 MB (~5-15s)": (
        "**google/flan-t5-large** via HuggingFace\n\n"
        "- ✅ Fast — ~5–15s on CPU, no Ollama needed\n"
        "- Encoder-decoder model fine-tuned for QA tasks\n"
        "- First run: downloads ~800 MB automatically"
    ),
    "🤗 Flan-T5 XL (HuggingFace) — ~3 GB (~20-40s)": (
        "**google/flan-t5-xl** via HuggingFace\n\n"
        "- Better answers than Large, but slower (~20–40s)\n"
        "- CPU-compatible, no Ollama needed\n"
        "- First run: downloads ~3 GB automatically"
    ),
    "🤗 Flan-Alpaca Large (HuggingFace) — ~800 MB (~5-15s)": (
        "**declare-lab/flan-alpaca-large** via HuggingFace\n\n"
        "- ✅ Fast — ~5–15s on CPU, no Ollama needed\n"
        "- Flan-T5-Large fine-tuned on Alpaca instruction data\n"
        "- First run: downloads ~800 MB automatically"
    ),
    "🔍 Raw Context (No LLM) — Instant": (
        "**No generation model** — returns raw retrieved chunks\n\n"
        "- ✅ Instant response\n"
        "- Use this to verify the right chunks are being retrieved\n"
        "- If answers seem wrong, check retrieval here first"
    ),
}


def update_model_info(label: str) -> str:
    return MODEL_INFO.get(label, "No info available.")


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI layout
# ─────────────────────────────────────────────────────────────────────────────

DESCRIPTION = """
## 🕉️ Sanskrit RAG Chatbot

Ask questions about the Sanskrit stories in **Sanskrit (Devanagari)**, **English**, or **transliteration**.

**How it works:**
1. Your question is embedded using the **fixed embedding model** (always `paraphrase-multilingual-MiniLM-L12-v2`)
2. The **FAISS index** finds the most relevant story chunks
3. The **generation model** you select reads those chunks and writes the answer

Switching the model dropdown only changes *how the answer is written* — retrieval never changes.
"""

EXAMPLE_QUESTIONS = [
    ["Who is Shankhanaad and what mistakes did he make?"],
    ["How did Kalidasa help the new poet get one lakh rupees?"],
    ["What trick did the old woman use to get the gold?"],
    ["Why did the devoted person drown?"],
    ["शंखनादः कः आसीत्?"],
    ["कालिदासः किमर्थं चतुरः आसीत्?"],
    ["What is the moral of the story about the devotee?"],
]

with gr.Blocks(title="Sanskrit RAG System") as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():

        # ── Left column: chat ─────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=480,
                # Gradio 6.0: dict messages format {"role","content"} is the default.
                # The `type` argument was removed — do NOT pass it.
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question in Sanskrit, English, or transliteration…",
                    label="Your Question",
                    scale=5,
                    lines=2,
                )
                send_btn = gr.Button("Send ▶", variant="primary", scale=1)

            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=msg_box,
                label="Example Questions",
            )

        # ── Right column: settings ────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### ⚙️ Generation Model")

            model_dropdown = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=DEFAULT_MODEL,
                label="Select Model",
                info=" generation model changes.",
            )

            model_info_box = gr.Markdown(
                value=update_model_info(DEFAULT_MODEL),
            )

            gr.Markdown("### 🔧 Retrieval Settings")

            top_k_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Chunks to Retrieve (top_k)",
                info="More chunks = more context, but slower generation",
            )

            show_context_toggle = gr.Checkbox(
                value=False,
                label="Show Retrieved Chunks",
                info="Append the raw retrieved chunks to each answer",
            )

            gr.Markdown("---")
            gr.Markdown(
                "**Fixed embedding model:**\n"
                "`paraphrase-multilingual-MiniLM-L12-v2`\n\n"
                "This model handles Devanagari, English, and transliterated Sanskrit. "
                "It never changes regardless of which generation model is selected."
            )

    # ── Event wiring ──────────────────────────────────────────────────────────

    model_dropdown.change(
        fn=update_model_info,
        inputs=model_dropdown,
        outputs=model_info_box,
    )

    def chat(message, history, model_label, top_k, show_context):
        """
        Append user message and bot reply to history in the Gradio messages
        dict format: {"role": "user"/"assistant", "content": "..."}.

        This format is required by gr.Chatbot(type="messages") which is the
        standard in Gradio 4.x+. The old (user, bot) tuple format is no longer
        accepted and throws the 'Data incompatible with messages format' error.
        """
        if history is None:
            history = []

        # Get bot response
        bot_message = respond(message, history, model_label, int(top_k), show_context)

        # Append in dict format — NOT the old tuple format
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": bot_message})

        return history, ""

    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot, model_dropdown, top_k_slider, show_context_toggle],
        outputs=[chatbot, msg_box],
    )

    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, model_dropdown, top_k_slider, show_context_toggle],
        outputs=[chatbot, msg_box],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),   # Gradio 6.0: theme moved here from gr.Blocks()
    )