# Stage 3 — Cross-Encoder Re-Ranking (Phase 3)

**Code:** [`src/reranker.py`](../src/reranker.py)

## Goal
Pipeline 2: take the **same 10 candidates** from the Bi-Encoder, re-score them
with a **Cross-Encoder**, re-sort, and hand the **new top-3** to the LLM.

## What a Cross-Encoder is (explain this to the TA)
A Cross-Encoder feeds the query and a candidate chunk **together** as a single
input: `[CLS] query [SEP] chunk [SEP]`. The transformer runs full
**self-attention across all tokens of both**, so every query word can directly
attend to every chunk word. The output is **one relevance score** for that exact
pair.

Contrast with the Bi-Encoder:

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Input | query and chunk **separately** | query + chunk **together** |
| Output | two vectors, compared by cosine | one relevance score per pair |
| Interaction | none until the final cosine | full token-level cross-attention |
| Speed | very fast (vectors precomputed) | slow (one forward pass **per pair**) |
| Strength | coarse topical match | fine-grained "does this actually answer?" |

Because it looks at both texts jointly, it can tell that *"no rent can be
collected"* directly answers a rent question, and that a keyword-overlapping
chunk does **not** — distinctions the Bi-Encoder blurs.

## What the code does
`rerank(query, candidates)`:
1. Build 10 `(query, chunk_text)` pairs.
2. `model.predict(pairs)` → 10 relevance scores (higher = more relevant).
3. Copy each candidate dict, attach `ce_score`, **sort descending**, assign
   `ce_rank` (1 = best). The original `bi_rank` is preserved so we can measure the
   shift.

The re-ranked answer context is `reranked[:3]`.

## Key decisions

- **Re-rank only the 10 Bi-Encoder candidates, never the whole corpus.** This is
  the heart of two-stage retrieval and the answer to the Scaling Question: the
  Cross-Encoder is accurate but costs one transformer pass *per chunk*, so we only
  pay it on a short list. (If a correct chunk never made the top-10, the
  re-ranker can't recover it — a real limitation we note in Stage 4.)
- **Same model as the assignment suggests** (`ms-marco-MiniLM-L-6-v2`), trained
  specifically on the MS MARCO passage-ranking task.
- **Preserve `bi_rank`** on every candidate so the "#8 → #1" rank-shift table is
  trivial to build (see `src/pipeline.py:rank_shift_table`).

## What changes vs the baseline
Only the *ordering* of the same 10 chunks, and therefore *which 3* reach the LLM.
The corpus, the candidate set, and the prompt are all identical — so any
difference in Answer 1 vs Answer 2 is attributable to re-ranking alone.

## Output of this stage
For each query: the same 10 chunks, now with `ce_score` and `ce_rank`. The new
top-3 become **Answer 2**'s context.
