# Corrections log

Published claims that turned out wrong, newest first. Code is the source of
truth: `tests/test_readme_claims.py` enforces doc-vs-code consistency in CI,
and anything that slipped through before that gate existed is recorded here
permanently.

## 2026-07-06: LoCoMo category labels were transposed; multi-hop claims retracted

widemem's LoCoMo harnesses labeled category 1 "single-hop" and category 4
"multi-hop". The official LoCoMo evaluation maps category 1 to multi-hop and
category 4 to single-hop. Per-category numbers published before this date have
those two labels transposed: scores reported for "multi-hop" were measured on
single-hop questions, and the reverse.

- Retracted: any claim of multi-hop strength based on those reports.
- Unaffected: overall J scores, temporal (category 2), open-domain (category
  3), and adversarial (category 5) numbers.
- Fix: the category maps in the benchmark harnesses are corrected in a
  follow-up change; results published after it use the official mapping.

## 2026-06: the 45.32 "v1.4 baseline" was a stale-store artifact

Early v1.4.x comparisons reused benchmark stores ingested with v1.3.0 code,
which depressed the baseline to 45.32. Re-ingesting with the code under test
measured ~56 overall J on the same questions. Policy since then: every
published number comes from a fresh ingest with the code being measured, plus
an ingestion control run.
