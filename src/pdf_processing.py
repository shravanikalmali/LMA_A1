"""
Phase 1 — PDF processing.

Responsibilities:
  1. Extract raw text from the PDF, page by page (PyMuPDF).
  2. Lightly normalise whitespace (but keep the wording intact so that
     exceptions / negations survive).
  3. Split the text into overlapping word-based chunks.

Each chunk is returned as a dict so that downstream stages can attach metadata
(embeddings, ranks, scores) without losing the original text or its source page.
"""
from __future__ import annotations

import re
from typing import List, Dict

import pymupdf  # PyMuPDF (the old `import fitz` name is deprecated)

from . import config


def extract_pages(pdf_path=config.PDF_PATH) -> List[str]:
    """Return a list of per-page text strings."""
    doc = pymupdf.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


def _normalise(text: str) -> str:
    """Collapse runs of whitespace/newlines. Content words are untouched."""
    text = text.replace("\xad", "")          # soft hyphens
    text = re.sub(r"[ \t]+", " ", text)       # collapse spaces/tabs
    text = re.sub(r"\n{2,}", "\n", text)      # collapse blank lines
    return text.strip()


def chunk_text(
    pages: List[str],
    chunk_size=config.CHUNK_SIZE_WORDS,
    overlap=config.CHUNK_OVERLAP_WORDS,
) -> List[Dict]:
    """
    Word-based sliding-window chunking with overlap.

    We flatten the whole document into one word stream (remembering which page
    each word came from) and then slide a window of `chunk_size` words forward
    by `chunk_size - overlap` each step. The overlap keeps a rule and its
    trailing exception from being cleanly severed at a chunk boundary.
    """
    # Build a flat list of (word, page_number) so each chunk can report its page.
    words_with_page = []
    for page_no, page_text in enumerate(pages, start=1):
        for w in _normalise(page_text).split():
            words_with_page.append((w, page_no))

    step = max(1, chunk_size - overlap)
    chunks: List[Dict] = []
    for start in range(0, len(words_with_page), step):
        window = words_with_page[start : start + chunk_size]
        if not window:
            break
        text = " ".join(w for w, _ in window)
        pages_in_chunk = sorted({p for _, p in window})
        chunks.append(
            {
                "chunk_id": len(chunks),
                "text": text,
                "pages": pages_in_chunk,
                "n_words": len(window),
            }
        )
        if start + chunk_size >= len(words_with_page):
            break  # last window reached the end
    return chunks


def build_corpus(pdf_path=config.PDF_PATH) -> List[Dict]:
    """Convenience: PDF path -> list of chunk dicts."""
    pages = extract_pages(pdf_path)
    return chunk_text(pages)


if __name__ == "__main__":
    corpus = build_corpus()
    print(f"Extracted {len(corpus)} chunks")
    print("First chunk:\n", corpus[0]["text"][:300])
