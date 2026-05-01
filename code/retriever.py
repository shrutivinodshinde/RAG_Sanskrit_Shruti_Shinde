"""
retriever.py
------------
Vector-based retriever using:
  • sentence-transformers  — multilingual embeddings (Devanagari-aware)
  • FAISS (CPU)            — approximate nearest-neighbour search

The index is built once and persisted to disk.  Subsequent runs load it
from disk, avoiding redundant computation.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_DIM,
    FAISS_INDEX_FILE, METADATA_FILE, TOP_K, SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)


class SanskritRetriever:
    """
    Manages embedding generation and FAISS-based retrieval for Sanskrit chunks.

    Attributes
    ----------
    model     : SentenceTransformer  — embedding model
    index     : faiss.Index          — FAISS flat inner-product index
    metadata  : list[dict]           — parallel list of chunk metadata
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
        self.index: faiss.IndexFlatIP = None   # inner-product (cosine after normalise)
        self.metadata: List[Dict] = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def build_index(self, chunks: List[Dict[str, str]]) -> None:
        """
        Encode all chunks and build a FAISS inner-product index.

        Vectors are L2-normalised so inner product == cosine similarity.

        Parameters
        ----------
        chunks : output of preprocessor.preprocess_documents()
        """
        if not chunks:
            raise ValueError("Cannot build index: no chunks provided.")

        texts = [c["text"] for c in chunks]

        logger.info(f"Encoding {len(texts)} chunks …")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,   # normalise for cosine similarity
            convert_to_numpy=True,
        ).astype("float32")

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner product ≡ cosine (normalised)
        self.index.add(embeddings)
        self.metadata = chunks

        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        faiss.write_index(self.index, FAISS_INDEX_FILE)
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Index saved → {FAISS_INDEX_FILE}")
        logger.info(f"Metadata saved → {METADATA_FILE}")

    def load(self) -> bool:
        """
        Load a previously saved index from disk.

        Returns
        -------
        True if loaded successfully, False if index files do not exist.
        """
        if not (os.path.exists(FAISS_INDEX_FILE) and os.path.exists(METADATA_FILE)):
            return False

        self.index = faiss.read_index(FAISS_INDEX_FILE)
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info(f"Index loaded: {self.index.ntotal} vectors")
        return True

    # ── Retrieve ───────────────────────────────────────────────────────────────

    # ── Sanskrit keyword expansion map ───────────────────────────────────────
    # Maps Sanskrit query terms to additional English/Sanskrit keywords that
    # improve retrieval when the embedding model under-weights specific words.
    _QUERY_EXPANSION = {
        "मृत्यु":   "death died suffocated killed मृतः",
        "मृत्युः":  "death died suffocated killed मृतः",
        "श्वान":    "dog puppy श्वानशावक sack bag suffocate",
        "श्वानशावक": "puppy dog bag sack suffocated killed shankhanaad",
        "कथम्":    "how reason cause",
        "घण्टा":   "bell घण्टाकर्ण monkey",
        "वृद्धा":   "old woman bell monkey fruit",
        "भक्त":    "devotee drowned god effort",
        "शंखनाद":  "shankhanaad servant foolish govardhan",
        "कालिदास":  "kalidasa poet scholar bhoj lakh",
    }

    def _expand_query(self, query: str) -> str:
        """Append expansion keywords for known Sanskrit terms to boost recall."""
        extra = []
        for term, expansion in self._QUERY_EXPANSION.items():
            if term in query:
                extra.append(expansion)
        if extra:
            expanded = query + " " + " ".join(extra)
            logger.debug(f"Query expanded: {expanded[:120]}")
            return expanded
        return query

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query.

        Parameters
        ----------
        query     : user query (Sanskrit or English or transliterated)
        top_k     : max number of chunks to return
        threshold : minimum cosine similarity (0–1) to include a result

        Returns
        -------
        List of chunk dicts (same shape as metadata) with an added 'score' key.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Call build_index() or load() first.")

        # Expand Sanskrit queries with related keywords for better recall
        query = self._expand_query(query)

        # Encode query
        q_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # Search
        scores, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:          # FAISS returns -1 for empty slots
                continue
            if float(score) < threshold:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)

        logger.debug(f"Query: '{query[:60]}…'  → {len(results)} results")
        return results

    # ── Convenience ────────────────────────────────────────────────────────────

    def format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM.
        """
        if not results:
            return "No relevant context found."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Context {i} | source: {r['source']} | score: {r['score']:.3f}]\n"
                f"{r['text']}"
            )
        return "\n\n---\n\n".join(parts)