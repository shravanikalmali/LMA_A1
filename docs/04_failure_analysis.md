# Stage 4 — Failure-Mode Analysis (Phase 4)

**Code:** [`src/queries.py`](../src/queries.py), [`src/pipeline.py`](../src/pipeline.py)
**Data:** [`results/rank_shifts.json`](../results/rank_shifts.json)

## Goal
Find queries where the Bi-Encoder baseline sends the *wrong* top-3 to the LLM,
and see whether the Cross-Encoder rescues the correct chunk. Document the rank
shifts and the resulting answer differences.

## How we find the "gold" chunk automatically
Every query has a `gold_phrase` — a short string that appears **only** in the
genuinely-correct passage (e.g. `"no rent can be collected"`). Because chunks
overlap, that phrase can appear in two neighbouring chunks, so
`pipeline.locate_gold()` reports the **best (smallest) rank** among all matching
chunks — both before re-ranking (`bi_rank`) and after (`ce_rank`) — and whether a
gold chunk reached the **top-3** in each pipeline. This removes any guesswork.

## How the 5 queries were chosen
We drafted ~25 candidate phrasings and kept only those where the gold chunk
*actually* starts outside the baseline top-3 (so there is a real failure to
study), plus one deliberate counter-example where re-ranking makes things worse.
This empirical selection is itself part of the finding: on a small, well-written
rulebook the Bi-Encoder is often already correct — failures have to be
*engineered* by using query vocabulary that differs from the rule's wording.

## Results (measured, deterministic)

| ID | Failure type | Gold rank (Bi → CE) | In baseline top-3? | In re-rank top-3? | Verdict |
|----|--------------|:---:|:---:|:---:|---|
| Q1 | Keyword Trap / Distractor | **5 → 1** | ✗ | ✓ | **Re-rank fixes** |
| Q2 | Negation / Constraint | **4 → 1** | ✗ | ✓ | **Re-rank fixes** |
| Q3 | Exception / Corner Case | **4 → 1** | ✗ | ✓ | **Re-rank fixes** |
| Q4 | Buried Relevant Chunk | **5 → 3** | ✗ | ✓ | **Re-rank fixes** |
| Q5 | Honest counter-example | **2 → 4** | ✓ | ✗ | **Re-rank hurts** |

### Q1 — Keyword Trap (Free Parking)
*Query:* "How large is the **jackpot paid out** to a player who lands on Free
Parking?" The query borrows the popular *house-rule* vocabulary (jackpot, paid
out). None of those words are in the real rule, which says a player
*"does not receive any money, property or reward of any kind."* The Bi-Encoder is
pulled toward money/payment chunks and ranks the true chunk **#5**. The
Cross-Encoder, reading query+chunk together, recognises it as the actual answer
and lifts it to **#1**. This is the textbook keyword trap: high lexical overlap
with the wrong passages, correct answer phrased with different words.

### Q2 — Negation (mortgaged-property rent)
*Query:* "Explain the **rent payment rules** for a property that has been
mortgaged." The phrase "rent payment rules" matches several generic rent / house
chunks. The true answer is a **negation** — *"no rent can be collected"* — which
the Bi-Encoder embeds almost identically to the positive "rent is collected",
so it lands at **#4**. The Cross-Encoder separates permit from prohibit and
promotes it to **#1**.

### Q3 — Exception / Corner case (selling with buildings)
*Query:* "Can I sell a property to another player while it has houses on it?" The
general SELLING chunk and mortgage chunks out-rank the specific prohibition
*"no property can be sold … if buildings are standing"* (**#4**). Re-rank → **#1**.

### Q4 — Buried relevant chunk (sell order)
*Query:* "In what order must houses be removed when I sell buildings back?" The
correct clause (*houses "sold one by one, evenly, in reverse"*) sits at **#5**,
behind general building/mortgage chunks. Re-rank lifts it to **#3** — just inside
the context window. A narrower win, but still a win.

### Q5 — Honest counter-example (re-ranking HURTS)
*Query:* "I rolled a normal number and stopped on the Jail square; do I owe any
fine?" Here the baseline is **already correct**: the *"Just Visiting … you incur
no penalty"* chunk is at rank **2**, inside the top-3. The Cross-Encoder, however,
is distracted by chunks that are strongly *about jail* (going to jail, getting
out) and demotes every answer-bearing chunk to **#4+**, pushing the correct one
**out** of the context. The re-ranked answer can therefore be *worse* than the
baseline. We include this on purpose: **re-ranking is not a guarantee.**

## Take-aways
- Re-ranking helps most on **keyword traps and negations**, where joint
  attention beats independent embeddings (Q1–Q3).
- It can only rescue chunks that are **in the top-10 at all** — if the
  Bi-Encoder never retrieves the gold chunk, the Cross-Encoder cannot invent it.
- It can **overfit to topicality** and demote a correct-but-subtle chunk (Q5).
- Net: re-ranking is a high-value *refinement*, not a cure. Report honestly.
