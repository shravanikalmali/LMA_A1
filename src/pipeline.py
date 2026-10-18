"""
Orchestration + latency profiling for the two pipelines.

Pipeline 1 (Baseline):   bi-encoder retrieve 10 -> take top 3           -> LLM
Pipeline 2 (Re-Ranking): bi-encoder retrieve 10 -> cross-encoder sort   -> LLM
                                                    -> take top 3

Both share the SAME retrieved-10 candidates and the SAME prompt, so the
comparison isolates the effect of re-ranking.

Every stage is timed with time.perf_counter() and reported separately
(retrieval / rerank / generation) as well as a total, as the assignment asks.
"""
from __future__ import annotations

import time
from typing import List, Dict, Optional

from . import config
from .retrieval import BiEncoderRetriever
from .reranker import CrossEncoderReranker
from . import generation


def _timed(fn, *args, **kwargs):
    """Run fn, returning (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def run_query(
    query: str,
    retriever: BiEncoderRetriever,
    reranker: CrossEncoderReranker,
    generate: bool = True,
    top_k_context: int = config.TOP_K_CONTEXT,
) -> Dict:
    """
    Run BOTH pipelines for a single query and return a structured result with
    candidates, selected chunks, answers, rank-shift info and timings.
    """
    # ---- Stage shared by both: bi-encoder retrieval of top-10 --------------
    candidates, t_retrieve = _timed(retriever.retrieve, query)

    # ---- Pipeline 1: baseline top-3 ---------------------------------------
    baseline_top3 = candidates[:top_k_context]

    # ---- Pipeline 2: cross-encoder re-rank, then top-3 --------------------
    reranked, t_rerank = _timed(reranker.rerank, query, candidates)
    reranked_top3 = reranked[:top_k_context]

    result: Dict = {
        "query": query,
        "candidates": candidates,          # all 10, with bi_rank/bi_score
        "reranked": reranked,              # all 10, with ce_rank/ce_score
        "baseline_top3": baseline_top3,
        "reranked_top3": reranked_top3,
        "timings": {
            "retrieve_s": t_retrieve,
            "rerank_s": t_rerank,
        },
    }

    # ---- Generation (optional, needs Gemini key) --------------------------
    if generate:
        ans1, t_gen1 = _timed(generation.generate_answer, query, baseline_top3)
        ans2, t_gen2 = _timed(generation.generate_answer, query, reranked_top3)
        result["answer_baseline"] = ans1
        result["answer_reranked"] = ans2
        result["timings"]["generate_baseline_s"] = t_gen1
        result["timings"]["generate_reranked_s"] = t_gen2

    # ---- Convenience totals (retrieval-only, matching the assignment) ------
    result["timings"]["baseline_pipeline_s"] = t_retrieve
    result["timings"]["rerank_pipeline_s"] = t_retrieve + t_rerank
    return result


def locate_gold(result: Dict, gold_phrase: str, top_k_context: int = config.TOP_K_CONTEXT) -> Dict:
    """
    Find the 'gold' (genuinely-correct) chunk(s) for a query by substring match.

    Because chunks overlap, the answer phrase can appear in more than one chunk.
    We therefore report the BEST (smallest) rank among all matching chunks, both
    before re-ranking (bi) and after (ce), plus whether a gold chunk made the
    top-3 context in each pipeline.
    """
    gp = gold_phrase.lower()
    bi_ranks = [c["bi_rank"] for c in result["candidates"] if gp in c["text"].lower()]
    ce_ranks = [c["ce_rank"] for c in result["reranked"] if gp in c["text"].lower()]
    best_bi = min(bi_ranks) if bi_ranks else None    # None => not even in top-10
    best_ce = min(ce_ranks) if ce_ranks else None
    return {
        "gold_phrase": gold_phrase,
        "bi_rank": best_bi,
        "ce_rank": best_ce,
        "in_baseline_top3": best_bi is not None and best_bi <= top_k_context,
        "in_reranked_top3": best_ce is not None and best_ce <= top_k_context,
    }


def rank_shift_table(result: Dict) -> List[Dict]:
    """
    For each of the 10 candidates, show bi_rank vs ce_rank so rank movement is
    visible at a glance (used for the 'Chunk X: #8 -> #1' analysis).
    """
    by_id = {c["chunk_id"]: c for c in result["reranked"]}
    rows = []
    for c in result["candidates"]:
        ce = by_id[c["chunk_id"]]
        rows.append(
            {
                "chunk_id": c["chunk_id"],
                "bi_rank": c["bi_rank"],
                "ce_rank": ce["ce_rank"],
                "shift": c["bi_rank"] - ce["ce_rank"],  # +ve = promoted
                "bi_score": round(c["bi_score"], 4),
                "ce_score": round(ce["ce_score"], 4),
                "preview": c["text"][:90].replace("\n", " ") + "...",
            }
        )
    return rows
