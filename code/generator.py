"""
generator.py
-------------
LLM Generator for Sanskrit RAG System.

Supports four backends (set LLM_BACKEND in config.py or pass at runtime):

  "ollama"       Best quality. Requires Ollama app + model pull. CPU-native.
  "transformers" Auto-downloads HuggingFace model. Zero extra setup.
  "llama_cpp"    GGUF quantised models. Needs llama-cpp-python + model file.
  "simple"       No LLM — returns raw retrieved context (debug / testing).

All generator classes accept an optional `model_name` parameter so the
Gradio UI can switch models at runtime without restarting the server.
"""

import time
import logging

from config import (
    LLM_BACKEND,
    OLLAMA_MODEL, OLLAMA_HOST,
    TRANSFORMERS_MODEL, TRANSFORMERS_MAX_NEW_TOKENS, TRANSFORMERS_TEMPERATURE,
    LLAMA_MODEL_PATH, LLAMA_N_CTX, LLAMA_N_THREADS,
    LLAMA_MAX_TOKENS, LLAMA_TEMPERATURE, LLAMA_TOP_P,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STRICT ANTI-HALLUCINATION PROMPT  +  BILINGUAL OUTPUT FORMAT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a Sanskrit document assistant. Answer questions using ONLY the CONTEXT provided.

STRICT RULES — follow every one:
1. Read the CONTEXT word by word before writing anything.
2. Your answer MUST use only sentences or phrases that appear in the CONTEXT.
3. NEVER use your own knowledge. NEVER guess. NEVER invent details.
4. If the answer is not in the CONTEXT, reply ONLY with:
   "संदर्भे उत्तरं न प्राप्तम् | The answer was not found in the provided documents."
5. Do NOT mention scholars, kings, debates, or any topic not in the CONTEXT.

OUTPUT FORMAT — always use this exact two-part structure:

🔸 Sanskrit Answer (from context):
<copy the relevant Sanskrit sentence(s) directly from the CONTEXT>

🔹 English Explanation:
<paraphrase the English summary/translation from the CONTEXT in 2-3 sentences>

FORBIDDEN:
- Do NOT add information from outside the CONTEXT.
- Do NOT translate Sanskrit yourself — only use translations already in the CONTEXT.
- Do NOT write more than 3 Sanskrit sentences and 3 English sentences.\
"""

PROMPT_TEMPLATE = """\
CONTEXT (use ONLY this — nothing else):
\"\"\"
{context}
\"\"\"

QUESTION: {question}

STEP 1 — Scan the CONTEXT above for sentences that directly answer this question.
STEP 2 — If relevant sentences exist, format your answer as shown below.
STEP 3 — If the CONTEXT does not contain the answer, say so in Sanskrit and English.

REQUIRED FORMAT:
🔸 Sanskrit Answer (from context):
[paste the Sanskrit sentence(s) from the CONTEXT that answer the question]

🔹 English Explanation:
[paraphrase the English translation/summary from the CONTEXT in 2-3 sentences]

YOUR ANSWER:\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: detect the correct HuggingFace task for a model name
# ─────────────────────────────────────────────────────────────────────────────

# Models that use text2text-generation (seq2seq / encoder-decoder)
_TEXT2TEXT_KEYWORDS = ("t5", "bart", "pegasus", "mbart", "mt5", "flan")

def _detect_hf_task(model_name: str) -> str:
    """
    Return the correct HuggingFace pipeline task string for a given model name.

    Rules
    -----
    - Any model whose name contains a seq2seq keyword → "text2text-generation"
    - Everything else (GPT-style decoder-only)        → "text-generation"

    This fixes the original bug where "declare-lab/flan-alpaca-large" was
    assigned "text-generation" because the name doesn't contain "t5".
    """
    name_lower = model_name.lower()
    for kw in _TEXT2TEXT_KEYWORDS:
        if kw in name_lower:
            return "text2text-generation"
    return "text-generation"


# =========================================================
# Backend 1: Ollama
# =========================================================

class OllamaGenerator:
    """
    Sends requests to a locally running Ollama server.

    Parameters
    ----------
    model_name : str, optional
        Override the model from config.py (useful for runtime switching).

    Recommended models (pull with 'ollama pull <name>'):
      llama3:latest      best instruction following
      mistral:latest     fast + good
      llama3.2:1b        fastest, good multilingual
      llama3.2:3b        best quality / speed balance
      qwen2.5:1.5b       very fast multilingual
    """

    def __init__(self, model_name: str = None):
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError:
            raise ImportError(
                "ollama package not installed.\n"
                "  pip install ollama\n"
                "Also install Ollama: https://ollama.com/download"
            )

        self.model = model_name or OLLAMA_MODEL
        self.host  = OLLAMA_HOST   # from config — honours custom host/port

        # Build a client pointing at the configured host so OLLAMA_HOST
        # in config.py is actually respected (the default ollama client
        # always uses localhost:11434 otherwise).
        self._client = self._ollama.Client(host=self.host)

        self._verify_model()
        logger.info(f"OllamaGenerator ready — model: {self.model}  host: {self.host}")
        print(f"[Generator] Using Ollama model: {self.model}  (host: {self.host})")

    def _get_available_models(self, models_info) -> list:
        """Parse the models list from ollama.list() regardless of SDK version."""
        if isinstance(models_info, dict):
            models_list = models_info.get("models", [])
        else:
            models_list = getattr(models_info, "models", [])

        available = []
        for m in models_list:
            if isinstance(m, dict):
                name = m.get("name", "") or m.get("model", "")
            else:
                name = getattr(m, "model", None) or getattr(m, "name", None) or ""
            if name:
                available.append(str(name))
        return available

    def _verify_model(self):
        """Raise a clear error if Ollama is not running or the model is not pulled."""
        try:
            models_info = self._client.list()
            available = self._get_available_models(models_info)
            available_short = [n.split(":")[0] for n in available]
            model_short = self.model.split(":")[0]

            if self.model not in available and model_short not in available_short:
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama.\n"
                    f"Available models: {available}\n"
                    f"Pull it with:  ollama pull {self.model}"
                )
        except RuntimeError:
            raise
        except Exception as e:
            err = str(e).lower()
            if "connection refused" in err or "connect" in err or "refused" in err:
                raise RuntimeError(
                    f"Ollama server is not running at {self.host}.\n"
                    "Start it with:  ollama serve"
                ) from e
            raise

    def generate(self, question: str, context: str) -> dict:
        # Trim context to 3000 chars — enough for 3-4 full story chunks while
        # still keeping latency reasonable.  2000 was too tight and caused the
        # model to miss the relevant Sanskrit sentences.
        context = context[:3000]

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        start = time.time()
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={
                "temperature":    0.05,   # lower = more faithful to context
                "top_p":          0.9,
                "repeat_penalty": 1.1,
                "num_predict":    250,    # enough for Sanskrit + English answer
                "num_ctx":        4096,   # larger ctx so story chunks fit fully
            },
        )
        generation_time = round(time.time() - start, 3)

        # Handle both dict response (older SDK) and object response (newer SDK)
        if isinstance(response, dict):
            answer = response["message"]["content"].strip()
        else:
            answer = response.message.content.strip()

        # Strip any "ANSWER:" prefix the model sometimes adds
        for prefix in ("ANSWER:", "Answer:", "answer:"):
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        return {"answer": answer, "generation_time": generation_time}


# =========================================================
# Backend 2: HuggingFace Transformers
# =========================================================

class TransformersGenerator:
    """
    Uses a HuggingFace model to generate answers.
    Runs entirely on CPU — no GPU required.

    Supported models (auto-downloaded on first use):
      google/flan-t5-large          ~800 MB   Good extractive QA    ← fast default
      google/flan-t5-xl             ~3.0 GB   Better quality, slower
      declare-lab/flan-alpaca-large ~800 MB   Instruction-tuned flan

    Compatibility note:
      Newer transformers versions (4.50+) removed "text2text-generation" as a
      pipeline task. This class detects seq2seq models and loads them directly
      via AutoModelForSeq2SeqLM instead of using the pipeline, which works on
      ALL transformers versions.
    """

    def __init__(self, model_name: str = None):
        try:
            import transformers as _tf
        except ImportError:
            raise ImportError(
                "transformers not installed.\n"
                "  pip install transformers torch sentencepiece"
            )

        self.model_name = model_name or TRANSFORMERS_MODEL
        self.task = _detect_hf_task(self.model_name)

        print(f"[Generator] Loading HuggingFace model: {self.model_name}  (task={self.task})")
        print("            (first run downloads the model — please wait…)")
        logger.info(f"TransformersGenerator: model={self.model_name}  task={self.task}")

        if self.task == "text2text-generation":
            # ── Seq2seq path: T5, Flan-T5, Flan-Alpaca, BART, mT5 … ──────────
            # Load directly — bypasses the pipeline task registry so this works
            # regardless of whether the transformers version still registers
            # "text2text-generation" as a pipeline task.
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.seq2seq   = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.pipe      = None
        else:
            # ── Decoder-only path: GPT-2, LLaMA-HF, Mistral-HF … ────────────
            from transformers import pipeline as hf_pipeline
            self.pipe      = hf_pipeline(
                "text-generation",
                model=self.model_name,
                device=-1,                          # always CPU
                max_new_tokens=TRANSFORMERS_MAX_NEW_TOKENS,
            )
            self.tokenizer = None
            self.seq2seq   = None

        print(f"[Generator] HuggingFace model ready: {self.model_name}")

    def generate(self, question: str, context: str) -> dict:
        # Flan-T5 input window is 512 tokens (~1500 chars). Trim to avoid truncation.
        context = context[:1500]

        if self.task == "text2text-generation":
            # ── Seq2seq generation (T5 / Flan family) ────────────────────────
            prompt = (
                "Answer the question using ONLY the context below. "
                "Copy key phrases from the context. Do NOT add outside knowledge.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            import torch
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
            start = time.time()
            with torch.no_grad():
                output_ids = self.seq2seq.generate(
                    **inputs,
                    max_new_tokens=TRANSFORMERS_MAX_NEW_TOKENS,
                    num_beams=4,
                    early_stopping=True,
                )
            generation_time = round(time.time() - start, 3)
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        else:
            # ── Decoder-only generation (GPT-2, LLaMA-HF, Mistral-HF …) ─────
            prompt = (
                f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
                f"{PROMPT_TEMPLATE.format(context=context, question=question)} [/INST]"
            )
            start = time.time()
            output = self.pipe(
                prompt,
                temperature=TRANSFORMERS_TEMPERATURE,
                do_sample=True,
            )
            generation_time = round(time.time() - start, 3)
            full   = output[0]["generated_text"]
            answer = full[len(prompt):].strip()

        return {"answer": answer, "generation_time": generation_time}


# =========================================================
# Backend 3: llama-cpp-python
# =========================================================

class LlamaCppGenerator:
    """Uses a local GGUF model via llama-cpp-python."""

    def __init__(self, model_name: str = None):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "  pip install llama-cpp-python"
            )

        import os
        model_path = model_name or LLAMA_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GGUF model not found: {model_path}\n"
                "Download a model and place it in the models/ directory.\n"
                "See README for download instructions."
            )

        self.llm = Llama(
            model_path=model_path,
            n_ctx=LLAMA_N_CTX,
            n_threads=LLAMA_N_THREADS,
            verbose=False,
        )
        logger.info(f"LlamaCppGenerator ready — model: {model_path}")
        print(f"[Generator] Using llama.cpp model: {model_path}")

    def generate(self, question: str, context: str) -> dict:
        prompt      = PROMPT_TEMPLATE.format(context=context, question=question)
        full_prompt = f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]"

        start = time.time()
        output = self.llm(
            full_prompt,
            max_tokens=LLAMA_MAX_TOKENS,
            temperature=LLAMA_TEMPERATURE,
            top_p=LLAMA_TOP_P,
            stop=["Question:", "[INST]", "\n\n\n"],
        )
        generation_time = round(time.time() - start, 3)
        answer = output["choices"][0]["text"].strip()
        return {"answer": answer, "generation_time": generation_time}


# =========================================================
# Backend 4: Simple — no LLM
# =========================================================

class SimpleGenerator:
    """
    Returns raw retrieved context without any LLM processing.
    Useful for:
      • Debugging retrieval quality (is the right content being found?)
      • Fast testing without needing Ollama or HuggingFace models
      • Used internally by gradio_app.py during the retrieval-only step
    """

    def generate(self, question: str, context: str) -> dict:
        answer = context.strip() if context.strip() else "No relevant context found."
        return {"answer": answer, "generation_time": 0.0}


# =========================================================
# Factory functions
# =========================================================

def build_generator():
    """
    Instantiate the default generator from the LLM_BACKEND setting in config.py.
    Called by rag_pipeline.py on first query if no generator has been set.
    """
    return build_generator_by_name(LLM_BACKEND)


def build_generator_by_name(backend: str, model_name: str = None):
    """
    Build a generator by backend name with an optional model override.

    Parameters
    ----------
    backend    : "ollama" | "transformers" | "llama_cpp" | "simple"
    model_name : optional model name / path to override config defaults

    Examples
    --------
    build_generator_by_name("ollama", "llama3:latest")
    build_generator_by_name("ollama", "llama3.2:1b")
    build_generator_by_name("ollama", "mistral:latest")
    build_generator_by_name("transformers", "google/flan-t5-large")
    build_generator_by_name("transformers", "google/flan-t5-xl")
    build_generator_by_name("transformers", "declare-lab/flan-alpaca-large")
    build_generator_by_name("llama_cpp")                    # uses LLAMA_MODEL_PATH
    build_generator_by_name("simple")
    """
    backend = backend.strip().lower()
    logger.info(f"Building generator — backend='{backend}'  model='{model_name}'")

    if backend == "ollama":
        return OllamaGenerator(model_name=model_name)
    elif backend == "transformers":
        return TransformersGenerator(model_name=model_name)
    elif backend == "llama_cpp":
        return LlamaCppGenerator(model_name=model_name)
    elif backend == "simple":
        return SimpleGenerator()
    else:
        logger.warning(
            f"Unknown backend '{backend}'. Falling back to SimpleGenerator."
        )
        return SimpleGenerator()