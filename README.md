# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two-stage RAG experiment over the official **Monopoly** rulebook, comparing a
Bi-Encoder baseline against a Cross-Encoder re-ranker and measuring *when*
re-ranking actually improves the LLM's answer.

- **Pipeline 1 (Baseline):** Bi-Encoder → top-10 → top-3 → LLM → *Answer 1*
- **Pipeline 2 (Re-Rank):** Bi-Encoder → top-10 → **Cross-Encoder** → new top-3 → LLM → *Answer 2*

## Quick start

```bash
pip install -r requirements.txt            # PyMuPDF, sentence-transformers, faiss-cpu, google-generativeai, ...
cp .env.example .env                       # then paste your GEMINI_API_KEY into .env
jupyter notebook notebooks/RAG_Baseline_vs_Reranking.ipynb
```

Run the notebook top-to-bottom. The Bi-Encoder and Cross-Encoder run locally;
only answer generation calls Gemini (so it needs the key). Every cell except the
generation cell runs without a key.

## Stack

| Component | Choice |
|---|---|
| PDF extraction | PyMuPDF |
| Chunking | custom word-window (90 words, 20 overlap) → 42 chunks |
| Bi-Encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS `IndexFlatIP` (cosine via normalised inner product) |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Gemini `gemini-2.0-flash` |

## Layout

```
data/        monopoly_rules.pdf              # corpus (official Hasbro rulebook)
src/         modular pipeline code (config, pdf_processing, retrieval, reranker, generation, pipeline, queries)
notebooks/   RAG_Baseline_vs_Reranking.ipynb # the single submission notebook (orchestration)
docs/        00..05_*.md                     # per-stage explanations of every decision
results/     rank_shifts.json, latency.json  # saved experiment outputs
```

## Headline results (5 stress-test queries)

| Query | Failure type | Gold rank Bi → CE | Verdict |
|---|---|:---:|---|
| Q1 | Keyword trap (Free Parking) | 5 → 1 | Re-rank fixes |
| Q2 | Negation (mortgaged rent) | 4 → 1 | Re-rank fixes |
| Q3 | Exception (sell with buildings) | 4 → 1 | Re-rank fixes |
| Q4 | Buried chunk (sell order) | 5 → 3 | Re-rank fixes |
| Q5 | Honest counter-example | 2 → 4 | **Re-rank hurts** |

Latency (this machine): retrieval ≈ 3.5 ms, re-rank ≈ 15 ms (~4.3× retrieval);
LLM generation is network-bound and dominates.

See [`docs/`](docs) for the full reasoning behind every decision and
[`docs/05_report_findings.md`](docs/05_report_findings.md) for the report notes
(to be rewritten in your own words — the report must not be AI-generated).
