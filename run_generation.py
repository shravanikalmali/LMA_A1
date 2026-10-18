"""
Standalone driver to generate Answer 1 / Answer 2 for every failure query and
save them to results/run_results.json.

Paced for the Gemini FREE tier (5 requests/minute): we make ONE call at a time
and sleep SPACING seconds between calls, so we never burst past the limit.
Run:  python -u run_generation.py
"""
import json, time, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src import pdf_processing as pp, retrieval, reranker, pipeline, generation
from src.queries import FAILURE_QUERIES

SPACING = 14  # seconds between individual LLM calls (-> ~4.3 req/min)

corpus = pp.build_corpus()
r = retrieval.BiEncoderRetriever(); r.build_index(corpus)
rr = reranker.CrossEncoderReranker()

out = {}
first_call = True
for q in FAILURE_QUERIES:
    res = pipeline.run_query(q["query"], r, rr, generate=False)
    g = pipeline.locate_gold(res, q["gold_phrase"])
    answers = {}
    for label, ctx in [("baseline", res["baseline_top3"]),
                       ("reranked", res["reranked_top3"])]:
        if not first_call:
            time.sleep(SPACING)
        first_call = False
        answers[label] = generation.generate_answer(q["query"], ctx)
        print(f"{q['id']} {label}: {answers[label][:70]}", flush=True)
    out[q["id"]] = {
        "query": q["query"], "category": q["category"],
        "gold_phrase": q["gold_phrase"],
        "bi_rank": g["bi_rank"], "ce_rank": g["ce_rank"],
        "answer_baseline": answers["baseline"],
        "answer_reranked": answers["reranked"],
        "timings": res["timings"],
    }

json.dump(out, open("results/run_results.json", "w"), indent=2, default=str)
print("SAVED results/run_results.json", flush=True)
