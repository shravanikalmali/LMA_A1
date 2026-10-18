"""
Phase 2 — Baseline retrieval (Bi-Encoder + FAISS vector store).

A Bi-Encoder embeds the query and every chunk *independently* into vectors.
Relevance is then just a geometric comparison (cosine similarity) between the
query vector and each chunk vector. Because chunk vectors are precomputed once
and stored in a FAISS index, search over thousands of chunks is milliseconds.

We L2-normalise all vectors and use a FAISS inner-product index, which makes
inner product == cosine similarity.
"""
from __future__ import annotations

from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from . import config


class BiEncoderRetriever:
    def __init__(self, model_name: str = config.BI_ENCODER_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.corpus: List[Dict] = []

    # -- Indexing (done once) ------------------------------------------------
    def build_index(self, corpus: List[Dict]) -> None:
        """Embed every chunk and store the vectors in a FAISS index."""
        self.corpus = corpus
        texts = [c["text"] for c in corpus]
        emb = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        faiss.normalize_L2(emb)                 # cosine == inner product
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    # -- Query (done per question) ------------------------------------------
    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVE) -> List[Dict]:
        """
        Return the top_k chunks for `query`, each annotated with its
        bi-encoder cosine score and its baseline rank (1 = best).
        """
        if self.index is None:
            raise RuntimeError("Call build_index() before retrieve().")
        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)

        results: List[Dict] = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
            chunk = dict(self.corpus[int(idx)])     # copy so we can annotate
            chunk["bi_score"] = float(score)
            chunk["bi_rank"] = rank
            results.append(chunk)
        return results
