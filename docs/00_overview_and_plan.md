# Stage 0 — Overview, Plan & Key Decisions

This document is your map of the whole project. Each later `docs/0X_*.md` file
drills into one stage. Read this first so you can explain the big picture in the
TA evaluation.

## 1. What the assignment asks (in one line)

Build **two RAG pipelines over the same chunked PDF**, and show *when* and *why*
adding a Cross-Encoder re-ranker produces better LLM answers than plain
Bi-Encoder retrieval — plus what it costs in latency.

- **Pipeline 1 (Baseline):** Bi-Encoder → top-10 → take top-3 → LLM → *Answer 1*
- **Pipeline 2 (Re-Rank):** Bi-Encoder → top-10 → **Cross-Encoder re-sorts** →
  new top-3 → LLM → *Answer 2*

Both pipelines must share the same corpus, the same retrieved-10 candidates, and
the same prompt. Only the *selection of the 3 chunks* differs. That is what makes
the comparison fair.

## 2. Decisions made (and why)

| Decision | Choice | Why |
|---|---|---|
| **Dataset** | Official Hasbro **Monopoly** rulebook (8 pages) | Dense with exceptions, penalties, and "cannot / no … can be" negations — perfect raw material for retrieval failures. |
| **PDF extraction** | PyMuPDF (`pymupdf`) | Reliable text extraction, no external service. |
| **Chunking** | Word-based sliding window, **90 words, 20-word overlap** → 42 chunks | Small enough that a single "except when…" clause lands in its own chunk (so it *can* be buried at rank 4–10); overlap stops rules being severed mid-sentence. |
| **Bi-Encoder** | `all-MiniLM-L6-v2` | Standard, fast, 384-dim; the assignment's suggested model. |
| **Vector store** | FAISS `IndexFlatIP` with L2-normalised vectors | Inner product on normalised vectors = cosine similarity. Exact search (no approximation error to muddy the comparison). |
| **Cross-Encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | The model named in the assignment; trained for passage re-ranking. |
| **LLM** | Gemini (`gemini-2.0-flash`) via API | Suggested stack; key read from `.env`, never hardcoded. |
| **Timing** | `time.perf_counter()`, each stage timed separately | Assignment asks for retrieval vs re-ranking latency, broken down. |
| **Code layout** | Modular `src/` package + one orchestration notebook | Keeps logic reusable/testable; notebook tells the story end-to-end. |

## 3. Why a two-stage design at all? (the core idea)

- A **Bi-Encoder** embeds the query and each chunk **independently**, then
  compares vectors. Chunk vectors are precomputed once, so search is extremely
  fast — but the model never looks at the query and chunk *together*, so it only
  captures coarse topical similarity. It can be fooled by shared keywords and it
  blurs negation ("rent is collected" vs "no rent can be collected").
- A **Cross-Encoder** feeds the query and chunk **together** through one
  transformer with full cross-attention, producing a single relevance score. It
  is much more accurate at judging "does this passage actually answer this
  question?" — but it must run once per pair, so it is far too slow to score the
  whole corpus for every query.
- **Two-stage retrieval** gets the best of both: the cheap Bi-Encoder narrows
  thousands of chunks down to 10 candidates; the expensive Cross-Encoder then
  carefully re-orders just those 10. This is exactly the Scaling Question in the
  report (see `docs/05_report_findings.md`).

## 4. Project structure

```
LMA_A1/
├── data/monopoly_rules.pdf        # corpus (downloaded, official)
├── src/                           # modular, reusable code
│   ├── config.py                  # every tunable knob in one place
│   ├── pdf_processing.py          # Phase 1: extract + chunk
│   ├── retrieval.py               # Phase 2: Bi-Encoder + FAISS
│   ├── reranker.py                # Phase 3: Cross-Encoder
│   ├── generation.py              # Gemini wrapper + shared prompt
│   ├── pipeline.py                # orchestration + latency timing
│   └── queries.py                 # the failure-mode stress tests
├── notebooks/RAG_Baseline_vs_Reranking.ipynb   # the single submission notebook
├── docs/00..05_*.md               # this explanation series
├── results/                       # saved tables/answers from a run
├── requirements.txt, .env.example, README.md
```

## 5. Execution phases (and the doc for each)

1. **Phase 1 – PDF processing** → `docs/01_pdf_processing.md`
2. **Phase 2 – Baseline retrieval** → `docs/02_baseline_retrieval.md`
3. **Phase 3 – Re-ranking** → `docs/03_reranking.md`
4. **Phase 4 – Failure-mode analysis** → `docs/04_failure_analysis.md`
5. **Phase 5 – Report findings** → `docs/05_report_findings.md`

## 6. Honesty note for the report

The assignment report must be **in your own words, not AI-generated**. These
`docs/` files explain the reasoning so *you* understand it; use them to write the
final report yourself. In particular, our query set deliberately includes an
**honest counter-example (Q5)** where re-ranking makes things *worse* — the goal
is to show *when* re-ranking helps, not to pretend it always does.
