# widemem-ai Audit (Phase 1)

Read-only audit of `widemem-ai` v1.4.1. Findings are ranked by severity, then effort. Every row cites a real `file:line`. A ✓ in the **V** column means I personally re-read the code and confirmed the finding; unmarked rows are reviewer-reported with a cited location but not independently re-verified.

Harness for verifying any fix (from Phase 0): `ruff check widemem/ tests/` and `pytest tests/ --cov=widemem`. Both currently green (512 passed, 73% coverage).

Effort: S = under an hour, M = half a day, L = multi-day.

## High

| ID | V | Finding | Location | Category | Effort | Proposed fix |
|----|---|---------|----------|----------|--------|--------------|
| WM-1 | ✓ | FAISS `_save()` writes `index.faiss` then `state.json` with no temp-file + atomic rename. A crash mid-save leaves index and state mismatched and unrecoverable on next `_load()`. | `storage/vector/faiss_store.py:166-178` | durability | M | Write both to temp paths, then `os.replace()` after both succeed. |
| WM-2 | ✓ | FAISS `_save()` runs on every single insert/update/delete, rewriting the entire index + full metadata JSON. Each write is O(total store size); a 100k-memory store rewrites everything per add. | `faiss_store.py:68,132,178` | perf | M | Debounced/explicit flush, or batch writes; do not save per mutation. |
| WM-3 | ✓ | `HistoryStore` opens one sqlite connection without `check_same_thread=False` and no lock. `search_stream` runs producers via `asyncio.to_thread`, so concurrent access raises "SQLite objects created in a thread..." or corrupts under concurrent commits. | `storage/history.py:16` | concurrency | S | Open with `check_same_thread=False` + a `threading.Lock`, or use per-thread connections. |
| WM-4 | ✓ | Confidence assessment is broken when hybrid search is on. `assess_confidence` reads `similarity_score`, but `hybrid.py:94` overwrites that field with a min-max-normalized blend in [0,1] where the top item is ≈1.0. Thresholds (0.45/0.25/0.12) then collapse to almost-always HIGH. | `retrieval/uncertainty.py:42` + `retrieval/hybrid.py:94` | correctness | M | Assess confidence on a preserved raw cosine similarity, or recalibrate thresholds per signal. |
| WM-5 | ✓ | The search query is never sanitized; `sanitize()` is called only on add-path extraction input (`llm_extractor.py:34`). A malicious query flows raw into the ranking/answer LLM path. | `server.py:140`, `mcp_server.py` | security | M | Run `sanitize()` on the query before `_memory.search`. |
| WM-6 | ✓ | The hierarchy summarizer re-feeds stored `Memory.content` into LLM prompts with no sanitization, so injection text persisted in a memory poisons every later summary/theme. Stored-then-retrieved is the live injection channel for a memory layer. | `hierarchy/summarizer.py:25,41,50` | security | M | Sanitize each `content` before formatting into the prompt; also sanitize extracted fact content before storing. |

## Medium

| ID | V | Finding | Location | Category | Effort | Proposed fix |
|----|---|---------|----------|----------|--------|--------------|
| WM-7 | ✓ | `boost_on_repetition` (the advertised "natural reinforcement" feature) is defined but called nowhere in `widemem/` or `tests/`. It is dead code; repetition never boosts importance. | `scoring/persistence.py:10` | correctness | M | Wire it into the add/update path, or remove it and the claim. |
| WM-8 | ✓ | `"last week/month/year"` sets `start = now - timedelta(days=days*2)`, producing a window twice the unit (14 days for "last week"), contradicting its own comment. | `retrieval/temporal_parser.py:207` | correctness | S | Use `days`, not `days*2`. |
| WM-9 | ✓ | Embedder LRU cache (`OrderedDict`) is mutated in `embed`/`embed_batch` with no lock. Under the server threadpool, concurrent embeds race; `move_to_end` + `popitem` sequences are not atomic. (Note: the FAISS store itself *is* lock-protected.) | `providers/embeddings/base.py:28-37` | concurrency | S | Guard cache reads/writes with a lock, or use a thread-safe cache. |
| WM-10 |  | `int(existing_id)` is unguarded inside the per-fact loop; a non-numeric LLM-returned id raises and aborts `detect_conflicts`. The identical call in `batch_resolver.py:78-81` is already guarded. | `retrieval/active.py:74` | correctness | S | Wrap in try/except and skip on failure, matching `batch_resolver`. |
| WM-11 |  | Cross-store write path is non-atomic: `_execute_actions` mutates the vector store, then separately logs history. A crash between leaves the store mutated with no audit row (committed, for pgvector autocommit). | `core/pipeline.py:178-184` | concurrency | M | Wrap vector mutation + history log in one transaction, or log intent first. |
| WM-12 |  | `count()` and `export_json()` pull up to 100k full records via `list_all` just to take `len()` — an O(n) scan on a hot path. pgvector/Qdrant offer cheap count APIs. | `core/memory.py:554,567` | perf | M | Add `count()` to `BaseVectorStore` backed by `SELECT count(*)` / Qdrant count; stream exports. |
| WM-13 |  | `content_hash` dedup keys on exact bytes only (whitespace/case variants treated as new) and ignores user scope (same content under different `user_id` collides). | `utils/hashing.py:4-5`, `core/pipeline.py:98-101` | correctness | S | Normalize content before hashing; namespace the hash by `user_id`/`agent_id`. |
| WM-14 |  | OpenAI embedder always passes `dimensions=`; models that do not support the param (ada-002, some OpenAI-compatible/Ollama-proxy endpoints) error on every embed. | `providers/embeddings/openai.py:22-26` | correctness | S | Pass `dimensions` only for v3 models that support it. |
| WM-15 |  | Embedder never checks the returned vector length against `config.dimensions`; a model/config mismatch is stored silently and corrupts FAISS dimension assumptions. | `providers/embeddings/base.py:33-37` | correctness | S | Assert `len(result) == dimensions`; raise `ProviderError` on mismatch. |
| WM-16 |  | Retry treats every exception alike: 401/400 retried 3x (wasteful, leaks timing), and 429 `Retry-After` ignored. | `providers/embeddings/base.py:60-75`, `providers/llm/base.py:28-41` | correctness | M | Distinguish retryable (429/5xx/timeout) from non-retryable 4xx; honor `Retry-After`. |
| WM-17 |  | pgvector swallows all index-DDL exceptions, silently leaving large stores doing sequential scans on every search with no signal. | `storage/vector/pgvector_store.py:143-152` | perf | S | Log the DDL failure; attempt an IVFFlat fallback. |
| WM-18 |  | Qdrant `get/search/list_all` call `payload.pop("_widemem_id")` on the live payload; with in-process clients this strips the id from the cached object, corrupting later lookups. | `storage/vector/qdrant_store.py:81,106,135` | correctness | S | Copy the payload before popping. |
| WM-19 |  | Qdrant maps non-UUID ids through `uuid5` (lossy many-to-one). Two distinct ids that collide silently overwrite each other; `search` returns the original id, hiding the loss. | `storage/vector/qdrant_store.py:139-144` | correctness | M | Reject non-UUID ids or keep a reverse-lookup index. |
| WM-20 |  | `/search` and `/add` are sync `def` doing blocking LLM+embedding network calls, so FastAPI runs them in the default 40-thread pool; long LLM calls starve it under load. | `server.py:139,154` | concurrency | M | Raise/bound the executor, or move blocking calls off the request thread. |
| WM-21 |  | `ExtractionCollector` shares one same-thread sqlite connection and stores raw pre-sanitization input (PII/secrets) to `~/.widemem/extractions.db` by default, no opt-in. | `extraction/collector.py:19,41-46` | security/concurrency | S | Make collection opt-in; store sanitized text; `check_same_thread=False` + lock. |

## Low

| ID | V | Finding | Location | Category | Effort | Proposed fix |
|----|---|---------|----------|----------|--------|--------------|
| WM-22 |  | FAISS over-fetches only `top_k*3` when filters are present; a selective filter on a large store can return far fewer than `top_k`. | `storage/vector/faiss_store.py:84-99` | correctness | M | Expand `k` until `top_k` filtered results found or index exhausted. |
| WM-23 |  | `IDMapper` is in-memory only with a monotonic counter; the same UUID gets a different int across restarts, so any int id surfaced to an LLM between restarts mismaps. | `utils/id_mapping.py:7-25` | correctness | S | Document as per-call-scoped only, or derive int deterministically. |
| WM-24 |  | Topic multiplier is applied after the additive temporal boost, so the soft temporal nudge gets scaled by topic weight. | `retrieval/temporal.py:83` | correctness | S | Apply topic multiplier before additive boosts. |
| WM-25 |  | `len(scored) > 5` disables the top-similarity protection pass for pools of exactly 5 (off-by-one). | `retrieval/temporal.py:140` | correctness | S | Use `>= 5` or compare to the actual slice size. |
| WM-26 |  | `best_boost` starts at 1.0 with `max()`, so configured topic weights below 1.0 (intended de-prioritization) are silently ignored. | `scoring/topics.py:18` | correctness | S | Handle sub-1.0 weights explicitly, or document boosts-only. |
| WM-27 |  | `old_importance = metadata.get("importance", 5.0)` assumes numeric; a store round-tripping importance as a string raises `TypeError` on the arithmetic. | `scoring/persistence.py:34` | correctness | S | Coerce to float before arithmetic. |
| WM-28 |  | `_MONTH_YEAR_BARE` matches any `<word> <4-digit>`, so "model 2020" / "version 2021" mis-fire temporal filtering. | `retrieval/temporal_parser.py:165` | correctness | S | Anchor the first group to month names. |
| WM-29 |  | A fresh `BM25Retriever` is built and the full candidate pool re-tokenized on every query. | `retrieval/hybrid.py:79` | perf | M | Cache the index/tokenization keyed by candidate set. |
| WM-30 |  | Anthropic `_generate` returns `response.content[0].text`, assuming the first block is text; a leading non-text block raises or returns wrong data. | `providers/llm/anthropic.py:33` | correctness | S | Concatenate blocks where `.type == "text"`. |
| WM-31 |  | MCP tool handlers return raw `str(e)` to the client, leaking internal paths/URLs. | `mcp_server.py:228,256,...` | security | S | Return generic errors; log detail server-side. |
| WM-32 |  | `_require_auth` compares keys with `!=` (non-constant-time) on network-exposed endpoints. | `server.py:107` | security | S | Use `hmac.compare_digest`. |
| WM-33 |  | MCP `_handle_search` clamps only the upper bound of `top_k`; a negative value passes through unchecked (HTTP layer enforces `ge=1`). | `mcp_server.py:234` | api | S | Clamp `max(1, min(top_k, 100))`. |
| WM-34 |  | `import_json` and `add_batch` embed serially with no batching, making large imports O(n) network round-trips. | `core/memory.py:594,161-173` | perf | S/M | Batch via `embed_batch` across the input set. |

## Test gaps (dimension 5)

The FAISS store is well tested (concurrency, restart, write-then-retrieve, TTL, oversized/empty text). The real gaps:

- **Dead-feature coverage** — `boost_on_repetition` (WM-7) has 0% coverage because nothing calls it. A test would have caught that it is unwired.
- **Cross-call dedup** — duplicate detection across separate `add()` calls (via `_find_existing` vector search) is never tested end to end. (`core/pipeline.py:108`)
- **WideMemory-level restart** — only the bare FAISS store reload is tested, not reopening the full `WideMemory` object on the same paths. (new `tests/test_memory_restart.py`)
- **Empty query** — `search("")` is unguarded while `add("")` short-circuits. (`core/memory.py:340`)
- **0%-coverage public surface** — `mcp_server.py` (entire stdio tool API) and `providers/llm/anthropic.py` (shipped provider) are correctness-critical and untested; `qdrant_store.py`, `responses.py`, `ollama.py`, `sentence_transformers.py` are optional/lower risk.
- **pgvector concurrency** — the production network backend has no concurrent insert/read test (FAISS does).
- **Capacity/eviction** — there is no capacity, eviction, or token-budget enforcement anywhere in storage. "Store at capacity" cannot be tested because the behavior does not exist; a test should pin the intended contract (bounded vs unbounded).

## Cross-cutting harness gap

No static type checker is configured (CI runs ruff + pytest only) despite the package shipping `Typing :: Typed`. Adding `mypy widemem/` or `pyright` to the dev env and CI would give every future loop a second objective signal and catch the dimension/type mismatches behind WM-14, WM-15, and WM-27.

## Suggested loop order

1. **Durability/concurrency core (WM-1, WM-2, WM-3):** highest risk of silent data loss. One fix loop each, verified by a new crash/concurrency test.
2. **Security replay path (WM-5, WM-6):** the live prompt-injection channel.
3. **Confidence calibration (WM-4):** user-visible correctness regression under hybrid search.
4. **Type checker (harness gap):** unlocks cheap verification for the medium tier.
5. Work the medium tier by store backend, then low/test-gap cleanup.
