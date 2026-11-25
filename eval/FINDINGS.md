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

8. **Look-alike identifier distractors** (36 synthetic docs, `doc_1305`-`doc_1340`: near-miss
   registrations, phone numbers, squawk codes, flight numbers, times, counts, law/case and
   person names; verified by code to contain no full target string). Identifier bucket, strict
   Hit@3 / Hit@1, n=15, single run:

   | config | no distractors | short (130 chars) | length-matched (~409) |
   |---|---|---|---|
   | dense | 0.80 / 0.73 | 0.60 / 0.20 | 0.67 / 0.27 |
   | bm25 | 0.93 / 0.87 | 0.73 / 0.40 | 0.87 / 0.87 |
   | rrf | 0.87 / 0.87 | 0.73 / 0.33 | 0.87 / 0.40 |
   | rrf+ce | 1.00 / 1.00 | 0.87 / 0.67 | 0.87 / 0.67 |

   - The first (short) version was a confound: distractors were ~3x shorter than their targets
     (130 vs 381 chars median), and BM25's length normalization favors short documents.
     With length-matched distractors BM25 recovers fully (Hit@1 0.87).
   - Dense stays hurt at equal length (Hit@1 0.73 -> 0.27): it cannot separate one-character
     differences in IDs.
   - Fusing with dense lowers top-1 relative to BM25 alone (RRF Hit@1 0.40 vs BM25 0.87). The
     README's claim that hybrid helps on identifiers is not supported by this set.
   - The cross-encoder does not depend on length, but its identifier Hit@1 fell from 1.00 to
     0.67 once look-alikes existed.
   - Caveats: 15 queries, one distractor set written by the same assistant, identical filler
     text across all padded docs. `ingest.py` defaults to the length-matched file; set
     `LOOKALIKE_FILE=data/distractors_lookalike.json` for the short stress variant.

## Retracted
- Drafting note for `sem-12` said the target was missing from the candidate pool. Wrong: the
  helper script hid the target from its own candidate list. In the v0 run it is rank 1 in
  BM25 and rank 1 after the cross-encoder.
