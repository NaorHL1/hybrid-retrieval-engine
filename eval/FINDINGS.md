# Eval findings (running log)

Evidence behind the numbers in `results_v0.json` and `results_v1_no_parent_doc.json`.
Everything here was measured on this repo's 45-query set (15 per bucket); small-sample
caveats apply. "Single query" means exactly that: not a general rate.

## Method
- `python -m eval.run` scores four configs: `dense`, `bm25`, `rrf`, `rrf+ce` (RRF fused with
  limit 1000, cut to 50 candidates, as in `api.py`).
- **strict** counts only `expected_id`; **lenient** also counts `also_relevant` (near-duplicate
  docs that answer the same question). Bootstrap 95% CI, 1000 resamples, seed 0.
- Candidate pools for labeling came from dense and BM25 only (not the cross-encoder).
- Queries were drafted by the assistant from the documents, then verified by hand. This is a
  bias risk (shared vocabulary with the targets); 10 queries written blind by someone else
  would test it.

## Findings
1. **The corpus contains near-duplicates**, so single-id labels punish correct answers.
   Example: `doc_0010` and `doc_0142` both state that a dentist treated the pilot. The old
   benchmark counted the cross-encoder's top pick (`doc_0142`) as a miss for that query.
2. **`doc_0096` was a parent article** containing 20 child chunks (`0097`-`0116`) plus the
   interview question headings (7% of its text). Removed in `f3cf770`; Hit@3 unchanged.
3. **The old ground truth had an error**: `doc_0106` does not contain "Fleischer Peled";
   `doc_0107` does.
4. **Default FTS tokenization splits hyphenated IDs**: `A6-FKF` -> `a6` + `fkf`, matched with
   OR semantics. The exact ID still ranks first, but `A6-FKX` (nonexistent) returns every
   `A6-*` doc. Measured on a nine-document toy table, not the real corpus.
5. **Glued words from scraping**: at least 25 of 155 docs (16%, a lower bound) contain a
   lowercase-to-Capital join such as `staffEyal`. Joins without a capital (`Zamirheld`)
   are invisible to a regex. Consequence: BM25 returns nothing for `Eyal Zamir`, while
   dense ranks the target 1st. Decision: leave the text unchanged and keep it as a
   documented limitation.
6. **Dense is not uniformly strong on identifiers**: of the 15 identifier queries drafted,
   dense ranked the target first in 10; time strings were the weak case (`10:45 a.m. GST`
   at rank 173 of 460, `17:40 local time` at rank 26). BM25 ranked both 1st or 2nd.
7. **Cross-encoder latency is the dominant cost.** `batch_size=8` roughly halves it on CPU
   (identical eval results). Timings on this machine vary a lot between runs (9-14 s for
   the same setting), so quote minimum and range, not a single run.

## Retracted
- Drafting note for `sem-12` said the target was missing from the candidate pool. Wrong: the
  helper script hid the target from its own candidate list. In the v0 run it is rank 1 in
  BM25 and rank 1 after the cross-encoder.
