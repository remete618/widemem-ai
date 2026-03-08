# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-03-09

### Added

- **Retry/backoff on LLM calls** — All LLM providers now retry up to 3 times with exponential backoff on transient errors (network, rate limits). `ProviderError` is not retried.
- **Memory TTL** — `MemoryConfig(ttl_days=30)` auto-expires memories older than N days at search time. No background jobs needed.
- **Score breakdown** — `MemorySearchResult` exposes `similarity_score`, `temporal_score`, `importance_score`, and `final_score` for debugging and transparency.
- **Batch add** — `memory.add_batch(["text1", "text2", ...])` processes multiple texts in one call.
- **Memory count** — `memory.count(user_id="alice")` returns total memory count with optional filters.
- **Export/import JSON** — `memory.export_json()` and `memory.import_json(data)` for backup, restore, and migration. Import skips existing IDs.
- 14 new tests (140 total)

## [1.2.0] - 2026-03-08

### Fixed

- **Invalid default LLM model** — Changed default from non-existent `gpt-4.1-nano` to `gpt-4o-mini`
- **Negative fact_index exploit** — Conflict resolver now rejects negative indices from LLM responses instead of silently wrapping via Python negative indexing
- **Duplicate fact_index processing** — If LLM returns the same fact_index twice with different actions, only the first is processed
- **Missing fact_index double-add** — Facts with missing `fact_index` in LLM response no longer get added twice (once from LLM action, once from fallback)
- **Unbounded top_k** — `search(top_k=...)` now capped at 1000 to prevent memory exhaustion

### Added

- 3 new tests for conflict resolver edge cases (negative index, duplicate index, missing index)

## [1.1.0] - 2026-03-08

### Added

- **YMYL two-tier confidence system** — Strong patterns (multi-word) get full treatment (importance floor 8.0, decay immunity, forced active retrieval). Weak patterns (single keyword) get moderate boost only. Prevents false positives like "bank of the river".
- **YMYL documentation** — `YMYL.md` with full explanation of the two-tier system, examples, flow diagram, and limitations
- **Duplicate content detection** — Content hash checked before insert, prevents identical memories from being stored
- **list_all() on vector stores** — Proper metadata-based listing (FAISS + Qdrant) replaces zero-vector search hack in hierarchy
- **End-to-end test script** — `scripts/e2e_test.py` for real OpenAI integration testing
- **Resource cleanup** — `WideMemory` supports context manager (`with` statement), `close()`, and `__del__` for proper SQLite cleanup
- **Thread safety** — Pipeline operations protected by threading lock for concurrent access
- **Embedding dimension validation** — FAISS rejects vectors with wrong dimensions instead of silently corrupting
- **Conflict resolver fallback** — Bad LLM JSON gracefully falls back to ADD all facts instead of crashing

### Fixed

- **YMYL regex case sensitivity** — Patterns with mixed case (IRS, MRI, W-2) now match correctly against lowercased text via `re.IGNORECASE`
- **Zero-vector search hack** — Hierarchy manager now uses `list_all()` instead of searching with a zero vector

## [1.0.0] - 2026-03-08

### Added

- **Core memory system** — `WideMemory` with add, search, get, delete, and history
- **Batch conflict resolution** — Single LLM call resolves all new facts against existing memories (ADD/UPDATE/DELETE/NONE)
- **Importance scoring** — Facts rated 1-10 at extraction time, normalized into combined scoring
- **Time decay** — Four decay functions: exponential, linear, step, none
- **Combined scoring** — `final_score = similarity * weight + importance * weight + recency * weight`
- **Hierarchical memory** — Three-tier system (facts, summaries, themes) with automatic query routing and fallback chain
- **Active retrieval** — Contradiction and ambiguity detection with clarification callbacks
- **Self-supervised extraction** — SQLite-backed training data collector, small model fallback chain, training script
- **Topic weights** — Configurable boost/suppress multipliers for retrieval, custom extraction hints
- **Temporal search** — Time-range filters (time_after, time_before) on search
- **History audit trail** — SQLite log of all add/update/delete operations
- **Persistent FAISS** — Save/load to disk via `VectorStoreConfig.path`
- **LLM providers** — OpenAI, Anthropic Claude, Ollama
- **Embedding providers** — OpenAI, sentence-transformers (local)
- **Vector store providers** — FAISS (local), Qdrant (local or cloud)
- **UUID-to-integer ID mapping** — Prevents LLM hallucination of invalid memory IDs during conflict resolution
- **MD5 content hashing** — Skips no-op updates when content hasn't changed
- **Open source release** — README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, LICENSE (Apache 2.0), GitHub templates, CI workflow
- 126 tests, all passing
