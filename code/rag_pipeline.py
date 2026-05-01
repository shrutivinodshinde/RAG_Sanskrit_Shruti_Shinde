"""
rag_pipeline.py
---------------
End-to-end Sanskrit RAG Pipeline orchestrator.

Flow:
  Documents → Preprocessing → Chunking → Embeddings → FAISS Retrieval
           → Context Creation → LLM Generation → Final Answer

Key design:
  • retrieve_only() — returns chunks + timing WITHOUT calling any LLM.
    Used by gradio_app.py so the UI can swap generators without wasting
    time on a throwaway generation call.
  • query()         — full pipeline: retrieval + generation in one call.
    Used by main.py, query_interface.py, and evaluate.py.
"""

import os
import time
import logging
from typing import Dict, List, Optional

from config import DATA_DIR, TOP_K
from document_loader import load_directory, load_document
from preprocessor import preprocess_documents, normalise_query
from retriever import SanskritRetriever
from generator import build_generator

logger = logging.getLogger(__name__)


class SanskritRAGPipeline:
    """
    End-to-end Sanskrit RAG Pipeline.

    Attributes
    ----------
    retriever      : SanskritRetriever  — embedding model + FAISS index
    generator      : any Generator      — LLM backend (set lazily on first query)
    _index_ready   : bool               — True once ingest() has completed
    """

    def __init__(self):
        self.retriever = SanskritRetriever()
        self.generator = None          # lazy-loaded on first query()
        self._index_ready = False

    # =========================================================
    # INGESTION
    # =========================================================

    def ingest(
        self,
        data_dir: str = DATA_DIR,
        force_rebuild: bool = False,
    ) -> int:
        """
        Load Sanskrit documents and build (or reload) the vector index.

        Parameters
        ----------
        data_dir      : directory containing .txt / .pdf / .docx documents
        force_rebuild : if True, always rebuild even if a saved index exists

        Returns
        -------
        Number of chunks in the index.
        """

        # ── Try loading an existing saved index ───────────────────────────────
        if not force_rebuild:
            loaded = self.retriever.load()
            if loaded:
                logger.info("Loaded existing FAISS index from disk.")
                self._index_ready = True
                return self.retriever.index.ntotal

        # ── Build from scratch ────────────────────────────────────────────────
        logger.info(f"Ingesting documents from: {data_dir}")
        start_time = time.time()

        documents = load_directory(data_dir)
        if not documents:
            raise FileNotFoundError(
                f"No Sanskrit documents found in: {data_dir}\n"
                f"Add .txt / .pdf / .docx files inside data/"
            )

        chunks = preprocess_documents(documents)
        logger.info(f"Generated {len(chunks)} chunks")

        self.retriever.build_index(chunks)
        self.retriever.save()

        elapsed = round(time.time() - start_time, 2)
        self._index_ready = True

        logger.info(
            f"Ingestion complete | Documents: {len(documents)} | "
            f"Chunks: {len(chunks)} | Time: {elapsed}s"
        )
        return len(chunks)

    # =========================================================
    # ADD A SINGLE NEW DOCUMENT
    # =========================================================

    def add_document(self, filepath: str) -> int:
        """
        Incrementally add a new document to the existing index.

        Parameters
        ----------
        filepath : path to the new .txt / .pdf / .docx file

        Returns
        -------
        Number of new chunks added.
        """
        logger.info(f"Adding new document: {filepath}")
        doc        = load_document(filepath)
        new_chunks = preprocess_documents([doc])

        # Combine existing metadata chunks + new chunks and rebuild
        existing_chunks = list(self.retriever.metadata)
        all_chunks      = existing_chunks + new_chunks

        self.retriever.build_index(all_chunks)
        self.retriever.save()

        logger.info(f"Added {len(new_chunks)} chunks from {filepath}")
        return len(new_chunks)

    # =========================================================
    # RETRIEVE ONLY  ← NEW: used by gradio_app.py
    # =========================================================

    def retrieve_only(
        self,
        question: str,
        top_k: int = TOP_K,
    ) -> Dict:
        """
        Run ONLY the retrieval step — no LLM generation.

        This is the correct method to call from gradio_app.py so that
        the UI can choose which generator to use without triggering a
        wasted generation call with whatever generator happens to be
        currently loaded.

        Parameters
        ----------
        question : user query (Sanskrit / English / transliteration)
        top_k    : maximum number of chunks to return

        Returns
        -------
        dict with keys:
            question       (str)        : normalised query
            retrieved      (list[dict]) : retrieved chunks with scores
            retrieval_time (float)      : seconds taken for retrieval
            context        (str)        : joined chunk texts (ready for LLM)
        """
        if not self._index_ready:
            raise RuntimeError("Index not ready. Run ingest() first.")

        question = normalise_query(question)
        logger.info(f"[retrieve_only] Question: {question}")

        t0      = time.time()
        results = self.retriever.retrieve(query=question, top_k=top_k)
        ret_time = round(time.time() - t0, 3)

        logger.info(f"Retrieved {len(results)} chunks in {ret_time}s")

        context = "\n\n".join(chunk["text"] for chunk in results)

        return {
            "question":       question,
            "retrieved":      results,
            "retrieval_time": ret_time,
            "context":        context,
        }

    # =========================================================
    # FULL QUERY  (retrieval + generation)
    # =========================================================

    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        return_context: bool = True,
    ) -> Dict:
        """
        End-to-end RAG query: retrieval + generation.

        Used by main.py, query_interface.py, and evaluate.py.
        For the Gradio UI use retrieve_only() + generator.generate() separately.

        Parameters
        ----------
        question       : user query
        top_k          : chunks to retrieve
        return_context : if True, include retrieved chunks in the response dict

        Returns
        -------
        dict with keys:
            question        (str)
            answer          (str)
            retrieval_time  (float)
            generation_time (float)
            retrieved       (list[dict])  — only if return_context=True
        """
        # ── Retrieval ──────────────────────────────────────────────────────────
        retrieval_result = self.retrieve_only(question=question, top_k=top_k)
        question         = retrieval_result["question"]   # use normalised form
        context          = retrieval_result["context"]
        results          = retrieval_result["retrieved"]
        retrieval_time   = retrieval_result["retrieval_time"]

        logger.info("Context prepared for LLM")

        # ── Lazy-load default generator ────────────────────────────────────────
        if self.generator is None:
            logger.info("Lazy-loading default generator from config…")
            self.generator = build_generator()

        # ── Generation ────────────────────────────────────────────────────────
        t0        = time.time()
        generated = self.generator.generate(question=question, context=context)
        gen_time  = round(time.time() - t0, 3)

        # Handle generators that return dict vs plain string
        if isinstance(generated, dict):
            answer   = generated.get("answer", "")
            gen_time = generated.get("generation_time", gen_time)
        else:
            answer = str(generated)

        logger.info(f"Answer generated in {gen_time}s")

        # ── Build response ────────────────────────────────────────────────────
        response = {
            "question":        question,
            "answer":          answer,
            "retrieval_time":  retrieval_time,
            "generation_time": gen_time,
        }
        if return_context:
            response["retrieved"] = results

        return response

    # =========================================================
    # STATUS
    # =========================================================

    def status(self) -> Dict:
        """Return a summary of the pipeline's current state."""
        total_chunks = 0
        if self._index_ready and self.retriever.index:
            total_chunks = self.retriever.index.ntotal

        return {
            "index_ready":         self._index_ready,
            "total_chunks":        total_chunks,
            "embedding_dimension": (
                self.retriever.model.get_sentence_embedding_dimension()
                if self._index_ready else "N/A"
            ),
            "generator": (
                type(self.generator).__name__ if self.generator else "not loaded"
            ),
        }
