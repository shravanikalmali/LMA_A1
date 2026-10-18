"""
Phase 3 — Cross-Encoder re-ranking.

A Cross-Encoder takes the (query, chunk) pair TOGETHER as a single input and
runs full self-attention across both token sequences. Instead of two separate
vectors, it outputs one relevance score per pair. This lets it notice things a
Bi-Encoder blurs — e.g. that "no rent can be collected" directly answers a rent
question, or that a passage shares keywords but does NOT actually answer.

Cost: it must run the transformer once per candidate, so we only apply it to the
10 candidates the Bi-Encoder already shortlisted — never the whole corpus.
"""
from __future__ import annotations

from typing import List, Dict

from sentence_transformers import CrossEncoder

from . import config


class CrossEncoderReranker:
    def __init__(self, model_name: str = config.CROSS_ENCODER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Score every (query, chunk) pair jointly and re-sort descending.

        Returns a NEW list (copies of the candidate dicts) annotated with
        `ce_score` and the new `ce_rank` (1 = best). The original `bi_rank`
        is preserved so we can measure the rank shift.
        """
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)      # higher = more relevant

        annotated = []
        for c, s in zip(candidates, scores):
            c = dict(c)
            c["ce_score"] = float(s)
            annotated.append(c)

        annotated.sort(key=lambda c: c["ce_score"], reverse=True)
        for new_rank, c in enumerate(annotated, start=1):
            c["ce_rank"] = new_rank
        return annotated
