# Corrections log

Published claims that turned out wrong, newest first. Code is the source of
truth: `tests/test_readme_claims.py` enforces doc-vs-code consistency in CI,
and anything that slipped through before that gate existed is recorded here
permanently.

## 2026-10-04: docs described behavior the code does not have

A docs review against the code found claims that crashed or were false:

- The active-retrieval example in the README and `examples/ymyl_active_retrieval.py`
  read `c.existing_memory`. The field is `existing_content`; the example raised
  `AttributeError` on the first conflict.
- The README showed `MemoryConfig(uncertainty_mode=...)` changing how search
  answers. Nothing reads that field. The modes work only when passed to
  `build_uncertainty_guidance()`.
- YMYL.md said a weak YMYL match nudges importance to 6.0. No such floor
  exists: a weak match alone changes nothing.
- `docs/mcp.md` gave the MCP server's default LLM as `ollama` / `llama3.2`; the
  code defaults to `openai` / `gpt-4o-mini`. It also listed
  `WIDEMEM_EMBEDDING_MODEL`, which nothing reads.
- `docs/configuration.md` listed `OLLAMA_BASE_URL` and `QDRANT_URL`, which
  nothing reads, and said Qdrant uses `VectorStoreConfig.url`. It does not:
  Qdrant runs embedded with `path`, otherwise on `localhost:6333`.
- The README listed five MCP tools; the server has seven.

- Corrected: each claim above now matches the code. The README also states
  that `WideMemory()` without a FAISS `path` keeps vectors in RAM only.
- Gated: `tests/test_readme_claims.py` now checks against the code the
  README tool list, every env var named in the docs, the MCP env defaults,
  the clarification fields the examples read, any `uncertainty_mode=` in docs
  or examples, the YMYL importance floors (plus behavioural tests in
  `tests/test_ymyl_extraction_floors.py`), and the Qdrant
  `url` row. The FAISS-in-RAM note is not gated.
- Also corrected in YMYL.md: an LLM-tagged fact gets the full strong
  treatment, and the regex is not the only classifier.

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

Separately, widemem.ai described `ttl_days` as auto-expiring memories after
N days. It is a search-time filter: rows past the cutoff stay on disk and
`get()`, `count()` and `export_json()` still return them. `docs/configuration.md`
described this correctly; the site did not.

- Corrected: "full audit trail" now states its scope, which is writes, not
  reads, and content changes, not callers. The `ttl_days` description says
  filter rather than expiry everywhere.
- Unaffected: entries written by the extraction pipeline and the hierarchy
  manager, which were complete throughout, and the content on both sides of
  an update, which was always recorded.
- Fix: the four paths log as of this date, `purge_expired()` provides the
  deletion the retention wording promised, and
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
