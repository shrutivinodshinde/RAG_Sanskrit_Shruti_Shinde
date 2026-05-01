"""
preprocessor.py
---------------
Sanskrit-aware text preprocessing and chunking pipeline.

Key design choices:
  • Story-boundary chunking: respects ======= delimiters in the corpus so that
    each story's Sanskrit text, English translation, and English summary stay
    in the same chunk (prevents the LLM from seeing orphaned Sanskrit without
    its translation).
  • Respects Sanskrit sentence boundaries: danda (।) and double-danda (॥)
  • Also respects standard Latin punctuation for bilingual texts.
  • Overlapping fixed-size chunks as fallback for very long stories.
"""

import re
import logging
from typing import List, Dict

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH

logger = logging.getLogger(__name__)

# ── Sanskrit-specific Unicode ranges ──────────────────────────────────────────
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
DANDA        = "\u0964"   # ।
DOUBLE_DANDA = "\u0965"   # ॥

# Story separator (lines of = characters, at least 10 wide)
_STORY_SEP_RE = re.compile(r"^={10,}\s*$")

# Sentence-splitting pattern (Sanskrit + English)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।॥\.\!\?])\s+|(?<=[।॥])")


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Light cleaning that preserves Devanagari and meaningful whitespace.
    """
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Story-boundary splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_into_stories(text: str) -> List[Dict[str, str]]:
    """
    Split a document into story sections using ====== delimiters.

    Returns a list of dicts:
        {
            "title": str,   # story heading (e.g. "STORY 1: मूर्खभृत्यस्य …")
            "text":  str,   # story body
        }

    Falls back to a single story if no delimiters are found.
    """
    lines = text.splitlines()
    stories: List[Dict[str, str]] = []
    current_title = ""
    current_lines: List[str] = []
    in_sep = False

    for line in lines:
        if _STORY_SEP_RE.match(line):
            if in_sep:
                # Second separator line → flush current story, start next title block
                if current_lines:
                    body = "\n".join(current_lines).strip()
                    if len(body) >= MIN_CHUNK_LENGTH:
                        stories.append({"title": current_title, "text": body})
                current_lines = []
                in_sep = False
            else:
                # First separator line
                in_sep = True
        elif in_sep:
            # This line is the story heading (between the two === lines)
            current_title = line.strip()
            in_sep = False
        else:
            current_lines.append(line)

    # Flush last story
    if current_lines:
        body = "\n".join(current_lines).strip()
        if len(body) >= MIN_CHUNK_LENGTH:
            stories.append({"title": current_title, "text": body})

    if not stories:
        # No delimiters — treat whole document as one story
        stories = [{"title": "", "text": text.strip()}]

    logger.info(f"split_into_stories → {len(stories)} sections found")
    return stories


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences respecting Sanskrit and English boundaries.
    """
    # Insert a newline after every danda / double-danda
    text = re.sub(r"([।॥])", r"\1\n", text)
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Further split on English sentence endings
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            part = part.strip()
            if len(part) >= MIN_CHUNK_LENGTH:
                sentences.append(part)
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# Chunk production
# ─────────────────────────────────────────────────────────────────────────────

def make_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    title_prefix: str = "",
) -> List[str]:
    """
    Produce overlapping character-level chunks from text.

    When a story fits within chunk_size, it is returned as a single chunk so
    that Sanskrit lines and their English translations are never separated.

    Parameters
    ----------
    text         : cleaned story/document text
    chunk_size   : target maximum characters per chunk
    overlap      : characters of overlap between consecutive chunks
    title_prefix : story title prepended to every chunk (aids retrieval)
    """
    text = text.strip()
    if not text:
        return []

    # ── Whole-story chunk (preferred) ────────────────────────────────────────
    prefix = f"[{title_prefix}]\n" if title_prefix else ""
    full = prefix + text
    if len(full) <= chunk_size:
        return [full]

    # ── Long story → sentence-level packing ──────────────────────────────────
    sentences = split_into_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len: int = len(prefix)

    for sentence in sentences:
        s_len = len(sentence) + 1  # +1 for the space separator

        if current_len + s_len > chunk_size and current:
            chunk_text = prefix + " ".join(current).strip()
            if len(chunk_text) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk_text)

            # Seed next chunk with overlap
            prev = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
            current = [prev, sentence]
            current_len = len(prefix) + len(prev) + s_len
        else:
            current.append(sentence)
            current_len += s_len

    # Flush last chunk
    if current:
        chunk_text = prefix + " ".join(current).strip()
        if len(chunk_text) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk_text)

    logger.debug(f"make_chunks → {len(chunks)} chunks for '{title_prefix}'")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_documents(
    documents: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Full preprocessing pipeline for a list of loaded documents.

    Strategy:
      1. Split each document into story sections (====== delimited).
      2. If the whole story fits in chunk_size, keep it as ONE chunk so that
         Sanskrit text and its English translation are always together.
      3. For very long stories, produce overlapping sentence-level chunks.

    Returns a list of chunk dicts:
        chunk_id       (str)  : unique identifier
        source         (str)  : originating filename
        story_title    (str)  : story heading (for debugging)
        text           (str)  : chunk content
        has_devanagari (bool) : whether chunk contains Sanskrit script
    """
    all_chunks: List[Dict[str, str]] = []
    chunk_counter = 0

    for doc in documents:
        source   = doc["source"]
        raw_text = doc["text"]

        cleaned = clean_text(raw_text)
        stories = split_into_stories(cleaned)

        for story in stories:
            title    = story["title"]
            body     = story["text"]
            raw_chunks = make_chunks(body, title_prefix=title)

            for chunk_text in raw_chunks:
                all_chunks.append({
                    "chunk_id":       f"chunk_{chunk_counter:04d}",
                    "source":         source,
                    "story_title":    title,
                    "text":           chunk_text,
                    "has_devanagari": bool(DEVANAGARI_RE.search(chunk_text)),
                })
                chunk_counter += 1

            logger.info(
                f"'{source}' / '{title[:40]}': {len(raw_chunks)} chunk(s)"
            )

    logger.info(f"Total chunks produced: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────────

def contains_devanagari(text: str) -> bool:
    """Return True if text contains any Devanagari characters."""
    return bool(DEVANAGARI_RE.search(text))


def normalise_query(query: str) -> str:
    """
    Normalize user queries for better retrieval.
    - NFC Unicode normalization.
    - Lowercase for Latin-script queries only (preserves Devanagari case).
    - Collapse extra whitespace.
    """
    import unicodedata
    query = unicodedata.normalize("NFC", query)
    query = re.sub(r"\s+", " ", query).strip()
    if not DEVANAGARI_RE.search(query):
        query = query.lower()
    return query
