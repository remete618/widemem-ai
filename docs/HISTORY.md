# Corrections log

Published claims that turned out wrong, newest first. Code is the source of
truth: `tests/test_readme_claims.py` enforces doc-vs-code consistency in CI,
and anything that slipped through before that gate existed is recorded here
permanently.

## 2026-09-02: the audit trail did not cover every write path

The README and widemem.ai stated that every add, update and delete was
logged. Four public methods wrote to the vector store without a history
entry: `delete()`, `pin()`, `import_json()` and `backfill_entities()`.
`delete()` is the one that mattered. It left the ADD entry standing with
nothing recording the removal, so the log read as though the memory still
existed, which is worse than an absent entry.

The same sections implied attribution the schema does not carry.
`HistoryEntry` has no field naming a caller, so the log answers what changed
and when, never who.

- Corrected: "full audit trail" now states its scope, which is writes, not
  reads, and content changes, not callers.
- Unaffected: entries written by the extraction pipeline and the hierarchy
  manager, which were complete throughout, and the content on both sides of
  an update, which was always recorded.
- Fix: the four paths log as of this date, and
  `tests/test_audit_log_coverage.py::test_no_unlogged_mutation_site` fails
  when a method mutates the store without logging. Attribution claims stay
  gated in `tests/test_readme_claims.py` until `HistoryEntry` carries an
  actor field.

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
