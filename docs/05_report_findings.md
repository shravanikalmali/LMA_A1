# Stage 5 — Report Findings (Phase 5)

> **These are the facts and figures for the four required report sections, so you
> can WRITE THE REPORT IN YOUR OWN WORDS.** Do not paste this verbatim — the
> assignment requires the report not be AI-generated. Use it as your notes.

The four required sections are: (1) Architecture comparison, (2) Qualitative &
rank-shift analysis, (3) Latency profiling, (4) Scaling question.

---

## 1. Architecture comparison — Bi-Encoder vs Cross-Encoder

| Aspect | Bi-Encoder (`all-MiniLM-L6-v2`) | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
|---|---|---|
| Input | query and chunk encoded **separately** | query + chunk encoded **together** |
| Output | one vector each → compared by cosine | a **single relevance score** per pair |
| Token interaction | none until the final cosine | **full cross-attention** between all tokens |
| Cost per query | ~1 encode + a vector search over the index | **N transformer passes** (one per candidate) |
| Can precompute? | **Yes** — chunk vectors built once, reused | **No** — depends on the query, recomputed every time |
| Strength | fast, coarse topical similarity | accurate "does this passage answer *this* query?" |
| Weakness | fooled by shared keywords; blurs negation | too slow to score a whole corpus |

Core point to say in your own words: the Bi-Encoder is fast *because* it
precomputes independent embeddings, but that independence is also why it misses
fine distinctions (keyword traps, "not allowed" vs "allowed"). The Cross-Encoder
is accurate *because* it lets the query and chunk attend to each other, but that
same joint processing is why it cannot be precomputed and is expensive.

## 2. Qualitative & rank-shift analysis

Rank shifts (gold chunk, best rank over overlapping chunks):

| Query | Failure type | Bi rank | CE rank | Effect on the LLM answer |
|---|---|:---:|:---:|---|
| Q1 | Keyword trap (Free Parking) | 5 | **1** | Baseline lacks the "receives nothing" rule → risk of inventing a payout; re-rank supplies it. |
| Q2 | Negation (mortgaged rent) | 4 | **1** | Baseline may say rent is collected; re-rank gives "no rent can be collected". |
| Q3 | Exception (sell w/ buildings) | 4 | **1** | Baseline misses the prohibition; re-rank supplies "cannot sell while buildings stand". |
| Q4 | Buried chunk (sell order) | 5 | **3** | Re-rank just gets the "sell evenly in reverse" clause into context. |
| Q5 | **Counter-example** | 2 | **4** | Baseline is correct ("Just Visiting, no penalty"); re-rank **drops** it → worse answer. |

**Actual LLM answers from our run** (Gemini `gemini-3.5-flash`, saved in
`results/run_results.json`). Our prompt forces grounding ("if the context does
not contain the answer, say so"), so a baseline that misses the gold chunk
**abstains** ("The provided rules do not specify this") rather than hallucinating —
an honest failure, but still a failure (the user gets no usable answer).

| Q | Answer 1 (baseline top-3) | Answer 2 (re-ranked top-3) | Difference |
|---|---|---|---|
| Q1 | *"The provided rules do not specify this."* | *"There is no jackpot … 'does not receive any money, property or reward of any kind'."* | Baseline **omits** the rule; re-rank answers correctly. |
| Q2 | *"… do not specify this."* | *"No rent can be collected on mortgaged properties … rent can still be collected on unmortgaged properties in the same group."* | Baseline omits; re-rank gives the full negation + corner case. |
| Q3 | *"… do not specify this."* | *"No. No property can be sold … if buildings are standing … buildings must be sold back to the Bank first."* | Baseline omits; re-rank gives the prohibition. |
| Q4 | *"… removed in the reverse of the manner in which they were erected."* | *"… sold one by one, **evenly**, in reverse …"* | Both roughly right; re-rank adds the precise "one by one, evenly". |
| Q5 | *"No. … you are 'Just Visiting' and incur no penalty."* | *"The provided rules do not specify this."* | **Baseline correct; re-rank now abstains** — re-ranking hurt. |

Write-up point: with a grounded prompt the baseline's failure shows up as
*abstention / omission* (Q1–Q3), not confident hallucination. If you drop the
"say you don't know" instruction, the baseline is more likely to **hallucinate** a
plausible-but-wrong rule (e.g. invent a Free Parking payout) — worth noting as the
more dangerous failure mode a weaker prompt would expose.

## 3. Latency profiling

Measured on this machine over **30 samples** (6 measured passes × 5 queries, 2
warm-up passes discarded; see `results/latency.json`, reproduced by the notebook's
latency cell). Re-run for your own numbers — retrieval wobbles ~3.1–3.3 ms
run-to-run, the re-rank figure is stable:

| Stage | Mean time | Notes |
|---|---:|---|
| Bi-Encoder retrieval (encode query + search 42-chunk index, return 10) | **~3.3 ms** | Dominated by query encoding; FAISS search is negligible at this size. |
| Cross-Encoder re-rank (score 10 query–chunk pairs) | **~14.9 ms** | ~**4.5×** the retrieval cost, for just 10 candidates. |
| Two-stage total (retrieve + re-rank) | **~18.2 ms** | ~**5.5×** the baseline retrieval-only pipeline. |
| LLM generation (Gemini, per answer) | **~4–15 s** (network-bound) | Measured live; by far the dominant term — ~hundreds× retrieval+rerank. |

Key observation: re-ranking adds a **fixed, modest** overhead that grows with the
number of candidates re-scored (here 10). Generation latency dwarfs both, so in
practice the re-ranker's cost is cheap insurance for better context — *as long as*
you only re-rank a short list.

## 4. Scaling question — why not Cross-Encode the whole collection?

Because the Cross-Encoder has **no precomputation shortcut**. For every query it
must run a full transformer forward pass on **each (query, chunk) pair**:

- Bi-Encoder: embed each chunk **once, ever**; at query time do 1 encode + a
  vector search. Cost per query ≈ O(1) model passes + fast ANN search. Scales to
  millions of chunks.
- Cross-Encoder: cost per query = **O(N) model passes** for N chunks, every time,
  because the score depends on the specific query and cannot be cached.

With ~15 ms for 10 pairs, scoring a 42-chunk doc would be ~4× that; scoring a
10,000-chunk corpus would be ~1,000× — hundreds of milliseconds to seconds **per
query**, versus the Bi-Encoder's few milliseconds. That is why the standard design
is **retrieve-then-rerank**: the cheap Bi-Encoder narrows millions → ~10, and the
expensive Cross-Encoder refines only those ~10. You pay the accurate model's cost
on a tiny, query-dependent short list instead of the entire collection.

---

### One-paragraph conclusion (reword in your own voice)
Re-ranking is worth it *when the Bi-Encoder's weaknesses bite* — keyword traps and
negations — because joint attention resolves distinctions independent embeddings
cannot. It is not free and not infallible (Q5), but its overhead is small relative
to generation, and it is only affordable because it runs on a pre-filtered
shortlist rather than the whole corpus.
