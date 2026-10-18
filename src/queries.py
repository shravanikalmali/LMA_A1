"""
Failure-mode stress-test queries for the Monopoly rulebook.

Each query targets a KNOWN tricky passage. The Bi-Encoder is lured by a
lexically-similar but wrong/too-general chunk, while the true answer lives in a
chunk that uses DIFFERENT wording. A good Cross-Encoder should promote that true
chunk into the top 3.

`gold_phrase` is a short substring that appears ONLY in the genuinely-correct
chunk. The notebook uses it to locate the gold chunk's rank automatically
(before and after re-ranking) rather than eyeballing it.

These queries were selected empirically (see docs/04_failure_analysis.md): we
kept the ones where the gold chunk actually starts outside the baseline top-3,
plus one honest counter-example where re-ranking HURTS.

`category` maps to the four failure types named in the assignment.
"""

FAILURE_QUERIES = [
    {
        "id": "Q1",
        "category": "Keyword Trap / Distractor",
        "query": "How large is the jackpot paid out to a player who lands on Free Parking?",
        # The query uses "jackpot / paid out" (the popular house rule). None of
        # those words are in the true chunk, which says you get NOTHING. Several
        # money/payment chunks share the vocabulary and out-rank it.
        "gold_phrase": "does not receive any money",
        "expectation": "fixed",
        "note": "Lexical lure toward money chunks; truth uses a negation.",
    },
    {
        "id": "Q2",
        "category": "Negation / Constraint",
        "query": "Explain the rent payment rules for a property that has been mortgaged to the Bank.",
        # 'rent payment rules' pulls generic rent/Title-Deed chunks; the true
        # answer is the negation 'no rent can be collected'.
        "gold_phrase": "no rent can be collected",
        "expectation": "fixed",
        "note": "Bi-encoder blurs 'collect rent' and 'no rent can be collected'.",
    },
    {
        "id": "Q3",
        "category": "Exception / Corner Case",
        "query": "Can I sell a property to another player while it has houses on it?",
        # General SELLING chunk out-ranks the specific prohibition clause.
        "gold_phrase": "no property can be sold",
        "expectation": "fixed",
        "note": "General selling rule vs the specific 'cannot sell with buildings' clause.",
    },
    {
        "id": "Q4",
        "category": "Buried Relevant Chunk",
        "query": "In what order must houses be removed when I sell buildings back to the Bank?",
        "gold_phrase": "sold one by one",
        "expectation": "fixed",
        "note": "Correct 'sell evenly in reverse' clause buried behind general selling text.",
    },
    {
        "id": "Q5",
        "category": "Honest Counter-Example (re-ranking hurts)",
        "query": "I rolled a normal number and stopped on the Jail square; do I owe any fine?",
        # Included deliberately: here the baseline ALREADY has the gold chunk in
        # the top 3 (rank 2), but the cross-encoder demotes every answer-bearing
        # chunk out of the top 3. Shows re-ranking is not a guarantee — a point
        # the report must make honestly.
        "gold_phrase": "incur no penalty",
        "expectation": "hurts",
        "note": "Baseline correct; cross-encoder pushes the gold chunk out of top-3.",
    },
]
