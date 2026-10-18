# Stage 2 — Baseline Retrieval (Phase 2)

**Code:** [`src/retrieval.py`](../src/retrieval.py)

## Goal
Pipeline 1: given a query, use a **Bi-Encoder** to fetch the top-10 chunks, and
hand the **top-3** to the LLM.

## What a Bi-Encoder is (explain this to the TA)
A Bi-Encoder is a transformer that maps a piece of text to a single fixed-length
vector (an *embedding*). Crucially it encodes the query and each chunk
**separately** — it never sees them together. Relevance is then just the
**cosine similarity** between the query vector and each chunk vector.

- Query → vector `q`
- Chunk_i → vector `d_i` (all precomputed once, at indexing time)
- score(query, chunk_i) = cos(`q`, `d_i`)

Because the chunk vectors are precomputed and stored, answering a query is just a
fast nearest-neighbour search — milliseconds even over huge corpora.

## What the code does

1. **`build_index(corpus)`** (run once):
   - Encode all 42 chunks with `all-MiniLM-L6-v2` → a 42×384 matrix.
   - `faiss.normalize_L2(...)` so every vector has length 1.
   - Store in a FAISS `IndexFlatIP` (inner-product) index.
   - *Trick:* for unit-length vectors, **inner product == cosine similarity**, so
     an IP index gives us cosine ranking exactly, with no approximation.
2. **`retrieve(query, top_k=10)`** (run per query):
   - Encode + normalise the query.
   - `index.search` returns the 10 highest-cosine chunks.
   - Annotate each with `bi_score` (cosine) and `bi_rank` (1 = best).

The baseline's answer context is simply `candidates[:3]`.

## Key decisions

- **`IndexFlatIP` (exact search), not an approximate index (IVF/HNSW).** With
  only 42 chunks, approximate search buys nothing and could introduce ranking
  noise that confounds the Bi-Encoder-vs-Cross-Encoder comparison. We want the
  Bi-Encoder's *true* ranking.
- **Normalise + inner product** rather than raw L2 distance, so the score is a
  clean, interpretable cosine in `[-1, 1]`.
- **Top-10 then top-3** exactly as the assignment specifies. The gap between
  "retrieved 10" and "shown 3" is the whole game: a correct chunk sitting at
  rank 4–10 is retrieved but *invisible* to the LLM — until re-ranking.

## Why the Bi-Encoder fails on some queries (preview)
Because it compares independent topic vectors, it is vulnerable to:
- **Keyword traps:** a chunk that shares surface words ranks high even if it
  doesn't answer the question.
- **Negation blindness:** "rent is collected" and "no rent can be collected" have
  almost identical embeddings, so the Bi-Encoder can't tell permit from prohibit.

These are exactly the cases Stage 4 stress-tests.

## Output of this stage
For each query: a list of 10 candidate chunks, each with `bi_rank` and
`bi_score`. The top-3 of these become **Answer 1**'s context.
