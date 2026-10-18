# Stage 1 — PDF Processing (Phase 1)

**Code:** [`src/pdf_processing.py`](../src/pdf_processing.py)

## Goal
Turn the PDF into a list of small, searchable text **chunks**, each carrying a
little metadata (an id, the source page(s), word count).

## What the code does, step by step

1. **Extract text** with PyMuPDF — one string per page (`extract_pages`).
   The Monopoly PDF is 8 pages, ~2,900 words, ~16,000 characters.
2. **Normalise whitespace** (`_normalise`): collapse multiple spaces/newlines and
   drop soft hyphens. We deliberately do **not** fix spelling. The scanned PDF has
   OCR quirks (e.g. "helshe", "f~ve", "wery"), but rewording could accidentally
   change a rule, and keeping it raw is more honest about real-world RAG inputs.
3. **Chunk** (`chunk_text`): a **sliding window of 90 words with 20 words of
   overlap**. We flatten the document into one word stream, remembering each
   word's page number, then step forward 70 words (`90 - 20`) at a time.

Result: **42 chunks**, ~89 words each.

## Key decisions & the reasoning

### Why word-based chunking instead of LangChain?
A plain word window is transparent and dependency-light — you can explain exactly
where every chunk boundary is. LangChain's splitters would add a dependency for
no benefit at this scale.

### Why 90 words / 20 overlap specifically?
This is the most important chunking decision for the assignment, because it
controls whether failures are even *possible*:

- **Too large** (e.g. a whole page per chunk): the general rule and its exception
  land in the *same* chunk, so the Bi-Encoder can never "miss" the exception —
  there would be nothing for the re-ranker to rescue.
- **Too small** (e.g. 20 words): rules get shredded into fragments with no
  context, hurting both pipelines equally and making answers incoherent.
- **~90 words** is the sweet spot: a specific clause like *"no rent can be
  collected on mortgaged properties"* sits in its own chunk, separate from the
  general rent chunk — so it *can* be ranked #4–#10 by the Bi-Encoder and then
  promoted by the Cross-Encoder. That is exactly the phenomenon we want to study.
- **20-word overlap** keeps a sentence that straddles a boundary from being cut
  in half, so no clause is lost at the seams.

### Why keep page numbers?
So we can cite where an answer came from and sanity-check that the "gold" chunk
really is the passage we think it is.

## How to verify (quick check)
```python
from src import pdf_processing as pp
corpus = pp.build_corpus()
print(len(corpus))            # -> 42
print(corpus[0]["text"][:200])
```

## Output of this stage
`corpus`: a list of 42 dicts, each `{chunk_id, text, pages, n_words}`.
This exact list feeds **both** pipelines — that shared corpus is a fairness
requirement of the assignment.
