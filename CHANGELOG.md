# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.1] - 2026-05-13

### Added

- **Semantic YMYL classification** — Two-stage YMYL pipeline. Strong patterns fire from regex; implied or weak patterns get LLM classification during the existing extraction call. Catches "my chest hurts" while rejecting "bank of the river". Zero additional API cost. Full writeup: https://widemem.ai/blog/semantic-ymyl.
- **Prompt-injection sanitizer** — `widemem.security.sanitize()` strips well-known prompt-injection patterns (instruction overrides, system tags, role markers, jailbreak vocabulary, memory-targeted destructive actions) from input before LLM extraction. Conservative by design to avoid false positives on legitimate clinical or operational content. Wired into `LLMExtractor.extract()`.
- **Healthcare quickstart** — `examples/healthcare_quickstart.py` demonstrates the regulated-industry happy path: ingest a clinical encounter, retrieve YMYL facts, abstain gracefully on a memory miss, pin a critical correction.
- **LRU embedding cache** — `BaseEmbedder` now caches embeddings in process (default size 1024). Cache hits skip the provider call entirely. Significant latency win for repeated queries.
- **Configurable confidence thresholds + `created_at` in search response** — Confidence thresholds for HIGH/MODERATE/LOW are now configurable via env vars or `MemoryConfig`. `MemorySearchResult` exposes `created_at` for downstream temporal logic.
- **`glama.json`** — Manifest for Glama MCP server verification.

### Fixed

- **`Memory.get()` metadata loss** — Was discarding `ymyl_category`, `content_hash`, `run_id`, `created_at`, and `updated_at` when reconstructing from vector store metadata. Now copies all persisted fields and parses ISO timestamps back to `datetime`.
- **`import_json` crash on missing IDs** — `Memory().id` placeholder failed because `content` is required. Now uses `str(uuid.uuid4())` for entries without IDs.
- **Qdrant `host` parameter** — `QdrantClient(url='localhost', ...)` is invalid; switched to `host='localhost'` for non-path connections.
- **`datetime.utcnow()` deprecation** — Migrated internal timestamps to `datetime.now(timezone.utc)`. Removes Python 3.12 deprecation warnings; stored timestamps remain ISO-8601 with UTC offset.
- **macOS libomp pytest segfault** — Lazy-import FAISS in the vector store to avoid OpenMP runtime conflict during pytest collection on Apple Silicon. Pytest now runs to completion on stock macOS Homebrew installs.

### Changed

- **README structure** — Compressed provider sections into a single table, moved full configuration reference to `docs/configuration.md`, full API reference to `docs/api.md`, and MCP setup to `docs/mcp.md`. README now ~640 lines, scannable in 90 seconds.
- **Default LLM is now `gpt-4o-mini`** — Calibrated similarity thresholds for `all-MiniLM-L6-v2` embeddings. Old defaults assumed Ollama llama3.2, which produced poor extraction quality. Opt back into local with `MemoryConfig(llm=LLMConfig(provider="ollama", ...))`.
- **`faiss-cpu` is now optional** — Install via `pip install widemem-ai[faiss]` to keep base install lighter. Qdrant remains the alternative.
- **CI install** — Added `[faiss]` extra to test job; without it FAISS-backed tests fail after v1.4 made `faiss-cpu` an optional dependency.
- **Confidence/abstention framing** — Reframed retrieval confidence as graceful memory-miss handling rather than uncertainty quantification. The implementation is a similarity-threshold abstention; clearer naming reflects that.
- **LICENSE** — Replaced short notice with full Apache 2.0 license text for GitHub license detection.

## [1.4.0] - 2026-03-19

### Added

- **Retrieval modes** — `fast`, `balanced` (default), and `deep` presets that control retrieval depth, candidate pool size, and similarity boost strength. Configure at init or override per query: `mem.search("query", mode=RetrievalMode.DEEP)`.
- **Confidence scoring** — Every search now returns a `RetrievalConfidence` level (HIGH, MODERATE, LOW, NONE) based on how relevant the results are. Access via `response.confidence` and `response.has_relevant`.
- **Uncertainty modes** — Three response strategies for uncertain retrieval: `strict` (refuse if unsure), `helpful` (hedge with related context), `creative` (offer to guess). Set via `MemoryConfig(uncertainty_mode="helpful")`.
- **`mem.pin()`** — Store a memory with elevated importance (default 9.0). Use when the user explicitly asks to remember something or corrects a forgotten fact. Pinned memories resist decay.
- **Frustration detection** — Detects when users say things like "I told you this!" or "you forgot" and provides recovery guidance including automatic fact extraction and pinning.
- **Query-adaptive scoring** — Scoring weights now adapt to query type: factual queries boost similarity (0.75), temporal queries boost recency (0.50), multi-hop queries keep balanced weights.
- **Two-pass re-ranking** — For factual queries, top results by pure similarity get an additive boost to prevent important-but-irrelevant memories from burying the best match.
- **Improved extraction prompt** — Better preservation of dates, proper nouns, and specific details during fact extraction.
- **Temporal answer prompt** — Specialized prompt for temporal questions that demands specific dates.

### Changed

- **Default retrieval mode is now `balanced`** — `top_k=25`, hierarchy enabled, moderate similarity boost. Previous default was equivalent to `fast` mode.
- `SearchResult` wrapper returned from `search()` behaves like a list (backward compatible) but also exposes `.confidence` and `.has_relevant`.
- Factual queries now fetch a larger candidate pool (`top_k * 5` instead of `top_k * 3`) for better recall.

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
