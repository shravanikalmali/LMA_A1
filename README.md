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
jupyter notebook notebooks/rag.ipynb
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
| LLM | Gemini `gemini-3.5-flash` |

## Layout

```
data/        monopoly_rules.pdf              # corpus (official Hasbro rulebook)
notebooks/   rag.ipynb                       # the single self-contained submission notebook (all code + explanations inline)
results/     rank_shifts.json, latency.json, run_results.json, hallucination_demo.json   # saved experiment outputs
requirements.txt, README.md, .env.example   # deps, this file, API-key template
```

## Headline results (5 stress-test queries)

| Query | Failure type | Gold rank Bi → CE | Verdict |
|---|---|:---:|---|
| Q1 | Keyword trap (Free Parking) | 5 → 1 | Re-rank fixes |
| Q2 | Negation (mortgaged rent) | 4 → 1 | Re-rank fixes |
| Q3 | Exception (sell with buildings) | 4 → 1 | Re-rank fixes |
| Q4 | Buried chunk (sell order) | 5 → 3 | Re-rank fixes |
| Q5 | Honest counter-example | 2 → 4 | **Re-rank hurts** |

Latency (this machine, 30 samples): retrieval ≈ 2.9 ms, re-rank ≈ 14.6 ms (~5× retrieval);
LLM generation is network-bound and dominates.

