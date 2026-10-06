# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **The servers could not point embeddings at a host Ollama.** `widemem.server` and `widemem.mcp_server` set the LLM base URL from `WIDEMEM_LLM_BASE_URL` but never set `EmbeddingConfig.base_url`, so the Docker image (`WIDEMEM_EMBEDDING_PROVIDER=ollama`) always embedded against `localhost:11434` inside the container. Both servers now read `WIDEMEM_EMBEDDING_BASE_URL`; blank means unset.

- **Qdrant ignored `VectorStoreConfig.url`.** Without `path` it always connected to `localhost:6333`, so a remote Qdrant server could not be used. `url` now connects to that server, and the new `VectorStoreConfig.api_key` (a `SecretStr`, Qdrant only) is passed with it. Precedence: `path` (embedded), then `url`, then `localhost:6333`. The dimension guard applies to remote collections too.

- **pgvector accepted a table built for another vector size.** `CREATE TABLE IF NOT EXISTS` skipped the existing table, and the first insert failed inside Postgres (`expected 384 dimensions, not 1536`). `PgVectorStore` now reads the `embedding` column's declared size on open and raises `StorageError` naming both sizes, as FAISS and Qdrant do. A table with that name but no `embedding` column also raises `StorageError`; a bare `vector` column with no declared size still opens. The connection is closed when opening fails. `tests/test_pgvector_integration.py` covers these against a real database when `PGVECTOR_TEST_URL` is set.

- **`MemoryConfig(collect_extractions=True)` did not turn on collection.** `WideMemory` built the collector without `enabled`, so it fell back to `WIDEMEM_COLLECT_EXTRACTIONS` and stayed off unless the variable was also set; the variable alone did nothing either, because no collector was built while the flag was `False`. Now either one turns collection on.
- **Behaviour change, privacy:** if `WIDEMEM_COLLECT_EXTRACTIONS` is set in your environment, `WideMemory` (including the MCP and REST servers) now stores raw, pre-sanitization input text in `~/.widemem/extractions.db`. Unset it if you did not intend that.
- **macOS: the local stack could still segfault when faiss loaded before torch.** 2.0.0 pinned faiss's OpenMP threads when torch loaded first. In the opposite order (an app that imports faiss before widemem builds its embedder) and with the model on CPU (any Mac where the model runs on CPU), torch's own thread pool crashed the process. The sentence-transformers embedder now pins torch to one thread in exactly that case and logs it at INFO; batch encoding on that path is 15-35% slower on an M4. One order stays outside widemem's reach: if your app builds the embedder and then calls faiss directly with several threads, call `faiss.omp_set_num_threads(1)` yourself. Also stops calling the deprecated `get_sentence_embedding_dimension()` when `get_embedding_dimension()` exists.

### Changed

- **Breaking: topic weights below 1.0 now suppress.** Existing weights below 1.0 had no effect before, because `get_topic_boost()` floored the multiplier at 1.0; they now lower the score of matching memories, as the docs said. When several topics match, the multiplier is the strongest boost times the strongest suppression; boosts still do not stack. A weight of 0, a negative weight or a non-finite weight (NaN, infinity) now raises `ValidationError` at `TopicConfig` construction (and `ValueError` from `get_topic_boost()`): remove it or use a small positive weight.
- **Breaking: config models reject unknown fields.** `MemoryConfig`, `LLMConfig`, `EmbeddingConfig`, `VectorStoreConfig`, `ScoringConfig`, `YMYLConfig` and `TopicConfig` now raise `ValidationError` on a field they do not define. A typo such as `MemoryConfig(embeddings=...)` (the field is `embedding`) used to be accepted and ignored, so widemem ran on defaults without saying so. If you pass extra keys on purpose, remove them.
- **CI tests the default local stack.** A new `local-stack` job (Ubuntu with CPU-only torch, and macOS, Python 3.12) installs `.[dev,local]` and runs `tests/test_local_stack_e2e.py`: the real `all-MiniLM-L6-v2` embedder, FAISS on disk and SQLite history, with a fake LLM in place of Ollama. It covers the default config, add, add, search after torch loads (the sequence that segfaulted on macOS), persistence across a new instance, confidence on a known match, and the 384 vs 1536 dimension guard. The stack runs in a subprocess so a native crash fails the test instead of killing pytest. The live-model check in `tests/test_confidence_thresholds.py`, previously skipped in CI, runs there too. The main matrix still installs no torch.

## [2.0.1] - 2026-10-04

### Changed

- **Frustration recovery reassures only on HIGH confidence.** `build_frustration_response()` returned `reassure` ("I do have some information about this", nothing pinned) on HIGH or MODERATE. MODERATE means a memory about the same person or topic exists, not that the restated fact was stored, and with the lower `all-MiniLM-L6-v2` thresholds below the gap became visible: "I told you I moved to Boston!" scores 0.42 (MODERATE) when only "live in San Francisco" is stored, so the new fact was reassured and dropped instead of pinned. MODERATE now follows LOW: `recover_and_pin` when a fact can be extracted, otherwise `apologize_and_ask`. Once Boston is stored the query scores 0.80 (HIGH) and still reassures.

### Fixed

- **User-facing messages used em dashes.** The uncertainty and frustration response messages and the MCP health tool description now use plain punctuation. Extraction prompts are unchanged.

- **Correct answers came back LOW under the default local embedder.** The confidence thresholds (0.60 / 0.50 / 0.30) were calibrated for `text-embedding-3-small`, but 2.0 defaults to `all-MiniLM-L6-v2`, whose scores run lower: a fact stored without its subject ("moved to Boston") scores 0.32 against "where does alice live". `all-MiniLM-L6-v2` now gets its own defaults, high 0.60, moderate 0.30, low 0.20, selected from the embedder's resolved model (`EmbeddingConfig(model="")` under sentence-transformers counts as MiniLM, which is what it loads); every other model keeps the old values, and each `WIDEMEM_CONFIDENCE_*` variable still overrides its own level. A warning is logged when the resolved thresholds are not ordered high >= moderate >= low. Measured on 62 labeled queries over facts stored by a real `add()` with `llama3.1:8b` (`tests/fixtures/confidence_minilm.json`, split by person): when the answer fact has no subject, recall at MODERATE or above rises from 19% to 77% on the train split and from 17% to 58% on the holdout; on the answer fact itself, from 46% to 85% (train) and 50% to 83% (holdout). (Top-1 recall reads 62% to 100% on train and 92% to 100% on the holdout, but the top hit is often another fact that merely names the person.) The cost: questions about the same person with no stored answer ("is alice married") reach MODERATE or above in 12 of 12 cases instead of 8 of 12, one unrelated query (0.351) moves from LOW to MODERATE so `explain=True` calls it answerable, and precision over the whole fixture drops from 0.771 to 0.745. Under MiniLM, confidence cannot separate "the answer is stored" from "something about this person is stored", because the name in the query dominates the score; that needs follow-up research, not thresholds. `assess_confidence()` takes an optional `embedding_model` argument.
- **Qdrant did not expand `~` in `VectorStoreConfig.path`.** `path="~/qdrant"` created a literal `./~/qdrant` directory in the working directory. The path is now expanded, as FAISS and the history store already did.

## [2.0.0] - 2026-10-04

widemem is local-first: the defaults run on your machine, and cloud providers are opt-in. Three breaking changes, each with a migration step below.

### Changed

- **BREAKING: local-first defaults.** `MemoryConfig()` now runs on Ollama (`llama3.1:8b`) and sentence-transformers (`all-MiniLM-L6-v2`, 384 dimensions); it used to default to OpenAI and fall back to Ollama only when no `OPENAI_API_KEY` was set, so a key in the environment sent memories to OpenAI without being asked. Cloud providers are now used only when configured. Each provider gets its own default model unless `model` is set (`openai`: `gpt-4o-mini` and `text-embedding-3-small`; `anthropic`: `claude-haiku-4-5-20251001`). The MCP server defaults to Ollama too. `pip install "widemem-ai[local]"` installs the whole local stack. Under all-MiniLM-L6-v2 the existing confidence thresholds still keep unrelated memories out of HIGH, but short extracted facts often score LOW even when they answer the question; per-embedder calibration is open work. `llama3.1:8b` rather than the smaller `llama3.2`: in an add, contradict, YMYL and miss scenario run 3 times each, `llama3.2` kept the stale fact and split "I'm allergic to penicillin" into one fact per word every time; `llama3.1:8b` passed every check. `EmbeddingConfig.dimensions` now defaults to 384 and, when unset, follows the model (`text-embedding-3-small` 1536, `text-embedding-3-large` 3072, `nomic-embed-text` 768). **If you relied on the implicit OpenAI default**, set `LLMConfig(provider="openai")` and `EmbeddingConfig(provider="openai")`; for the MCP or REST server, set `WIDEMEM_LLM_PROVIDER=openai` and `WIDEMEM_EMBEDDING_PROVIDER=openai`. A FAISS index or Qdrant collection built with 1536-dimensional vectors now refuses to open under a 384-dimensional embedder instead of failing on the first write. pgvector has no such check yet: its first insert fails in Postgres.

- **BREAKING: OpenAI is now an optional extra.** `openai` moved from the core dependencies to `pip install "widemem-ai[openai]"` (also in `[all]`). A base install pulls no cloud SDK, and widemem imports and runs on a local stack (Ollama, sentence-transformers, FAISS) without it. The OpenAI providers load the SDK on first use and raise `ProviderError` naming the extra when it is missing. If you use OpenAI, add the extra.

- **BREAKING for the `mcp` extra: the MCP server now requires mcp 2.x.** The pin moves from `mcp>=1.0,<2` to `mcp>=2,<3`. mcp 2.0 removed the low-level `@server.list_tools()` and `@server.call_tool()` decorators the server was built on, so `widemem/mcp_server.py` did not import at all under 2.x. The handlers are now passed to `Server(...)` as `on_list_tools` and `on_call_tool`, which is how mcp 2.x registers them. Supporting both generations was considered and rejected: it needs two handler signatures and doubles the test matrix for an opt-in extra. Pin `widemem-ai<2` if you need mcp 1.x.

  The seven tools, their names, their input schemas and their response bodies are unchanged. This is an internal migration, not a protocol change, and a real stdio handshake against the ported server was used to confirm it: initialize, `tools/list` returning all seven, and a `widemem_health` round trip.

### Added

- **A contract suite every LLM provider must pass** - `tests/test_provider_contract.py` runs openai, anthropic and ollama through the same assertions, each against its real SDK with the HTTP transport mocked. A structural check fails when a provider is added to the package without a case, so a fourth one cannot drift in unseen.

- **LangChain retriever adapter** - `widemem.integrations.langchain.WidememRetriever` is a real `BaseRetriever`, so widemem drops into any chain that takes one. Documents carry the memory id, owner, importance, YMYL category, timestamp and both scores. `min_confidence` is all-or-nothing rather than a per-document filter, because widemem reports confidence for the result set: a chain can branch on an empty list, where a thinned list of weak matches would quietly degrade the answer. The async path runs the synchronous search in a worker thread so it does not stall the event loop. Install with the `langchain` extra. Example in `examples/langchain_retriever.py`.

### Fixed

- **The default local stack segfaulted on macOS.** faiss and torch each bundle libomp; once sentence-transformers had loaded, the first multithreaded FAISS search over a non-empty index crashed the process (exit 139), so the second `add()` never returned. The FAISS store now pins OpenMP to one thread on macOS.

- **Ollama returned invalid JSON on ordinary inputs.** JSON was requested only in the prompt, so local models added prose or a second object and extraction failed. `generate_json` now asks Ollama for constrained output with `format="json"`.

- **Docs described behavior the code does not have.** A crashing active-retrieval example, a config field nothing reads, a YMYL importance floor that does not exist, wrong MCP defaults, unread env vars and a Qdrant `url` the store ignores. Listed in `docs/HISTORY.md`. `tests/test_readme_claims.py` now gates all of them except the FAISS-in-RAM note.

- **The MCP server sent OpenAI requests to the Ollama port by default.** `WIDEMEM_LLM_BASE_URL` fell back to `http://localhost:11434` for every provider, so with `OPENAI_API_KEY` set, `provider=openai` (the 1.6 MCP default) talked to a local Ollama daemon, or to nothing, and `OPENAI_BASE_URL` was overridden. Both servers now pass a base URL only when the variable is set and not blank; the Ollama provider already defaults to its local host on its own.

- **The OpenAI provider did not strip code fences from JSON responses.** It relied on `response_format={"type": "json_object"}` to prevent them, which the OpenAI API honours. `base_url` is a supported setting and points at OpenAI-compatible endpoints that may ignore that field, in which case a fenced reply raised `ProviderError` on valid JSON. All three providers now share `strip_json_fences()` instead of two of them carrying their own copy. Found by the contract suite on its first run.

- **`purge_expired()` stopped at a fixed row cap and still reported success.** The sweep listed with `max_results=1_000_000`, and every backend honours that limit: FAISS stops appending, qdrant scrolls with it, pgvector adds a `LIMIT`. A larger store had its tail examined by nothing while the returned count read as a completed purge. The request is now sized from the store's own count for the same scope, so the caller's view cannot be truncated by the sweep's own request. Bounded-memory paging would need an offset on the vector-store interface and is deliberately not part of this fix.

## [1.6.0] - 2026-09-14

### Added

- **Provider tests run against a real SDK** - `tests/test_llm_providers.py` mocks the HTTP transport instead of the client, so the installed SDK validates every call. A `MagicMock` client accepts any keyword, which is how a parameter the SDK had removed passed CI. CI now installs the `anthropic`, `mcp`, `ollama` and `qdrant` extras too; those suites were being skipped, 14 tests in total. `sentence-transformers` stays out because it pulls torch into every matrix job.
- **`purge_expired(older_than_days, ...)`** - permanently removes memories past a cutoff, the deletion counterpart to the `ttl_days` search filter. YMYL rows are skipped unless `include_ymyl=True`, since a retention sweep that quietly dropped an allergy would defeat decay immunity. `dry_run=True` returns the count without removing anything, and every removal writes a `delete` history entry carrying the content it removed. Deliberately an explicit call rather than a config value: retention driven by a setting would delete data on upgrade.

### Changed

- **SDK ranges widened to the current majors** - `anthropic>=0.30,<2` and `openai>=1.0,<4`. Both generations are exercised: the suite passes on anthropic 0.109 with openai 1.x and on anthropic 1.4 with openai 3.11.
- **Audit-trail documentation states its scope** - the README's "History & Audit Trail" section said every add, update and delete was logged and implied attribution the schema does not carry. It now names the write paths covered, says entries are not attributed to a caller, and points retention at `purge_expired()`. `tests/test_readme_claims.py` gates the attribution wording on a `HistoryEntry` actor field existing, checks the documented `HistoryEntry` fields against the model, and checks `SECURITY.md` covers the shipped minor. Recorded in `docs/HISTORY.md`.

### Fixed

- **The Anthropic provider crashed on any non-zero temperature under anthropic >= 1** - the SDK removed `temperature`, `top_p` and `top_k` in 1.0 because the models it targets ignore them. `AnthropicLLM` still sent `temperature` whenever it was above zero, so `messages.create()` raised `TypeError`. `BaseLLM._retry` treats that as transient, so each call burned three attempts and two backoff sleeps before surfacing as a retry failure rather than a signature error. The provider now reads the installed SDK's signature and omits what it will not accept, warning once at construction when a configured temperature is dropped. Default temperature is 0, so only configs that set one were affected.
- **The benchmark harnesses imported an undeclared `httpx`** - `val.py`, `mini_locomo.py` and `run_ws1.py` built an `httpx.Client` purely to give the OpenAI client a timeout. openai >= 2 is built on `httpx2` and rejects an `httpx` client, and a fresh install no longer brings `httpx` at all. They pass a plain timeout now, which every generation accepts, and a test fails if an http library is imported there again.
- **Four write paths bypassed the history log** - `delete()`, `pin()`, `import_json()` and `backfill_entities()` wrote to the vector store without recording anything. A memory removed through `delete()` (the path behind the MCP `widemem_delete` tool) left its ADD entry standing and nothing marking the removal, so the log read as though the memory still existed. All four now write an entry, and `delete()` captures the removed content so the record can be reconstructed. `tests/test_audit_log_coverage.py` adds a structural guard that fails when a method mutates the store without logging, so a new write path cannot land unlogged.
- **`import_json` dropped `ymyl_category` and `run_id`** - both were absent from the metadata literal the importer builds, so an export/import round trip stripped YMYL decay immunity from every restored row. Found by a `purge_expired` test whose YMYL fixture would not stay protected.

## [1.5.1] - 2026-08-31

### Fixed

- **Cross-scope writes** - `add()` without a `user_id` searched every tenant's memories and handed them to the conflict resolver as UPDATE and DELETE targets. An UPDATE rewrote the victim's row with the caller's scope, making it invisible to its owner's filtered search; a DELETE erased it. Candidates are now scoped before the resolver sees them, and any UPDATE or DELETE whose target is owned by another scope is refused and counted in `stats["skipped"]`. Scope is enforced in the pipeline rather than pushed into a store filter, because a `None` user_id cannot be expressed on two of the three backends.
- **UPDATE dropped metadata** - rebuilding a memory from the resolver action alone discarded `ymyl_category`, `event_time`, `run_id`, `tier` and ownership. An export/import round trip therefore stripped YMYL decay immunity. These now carry from the stored row.
- **Local no-key mode was unusable** - the Ollama fallback built a 768-dimension embedder against a 1536-dimension store, so every add and search raised. Stores are now sized from the embedder in use.
- **YMYL rows evicted by TTL** - `ttl_days` filtered YMYL memories out before scoring, defeating the decay immunity they are promised. They are now exempt from the cut.
- **Filtered search under-returned** - the FAISS store over-fetched a fixed `top_k * 3` then post-filtered, so a tenant holding 5 of 205 memories got 1 result while `count()` reported 5. `k` now grows until `top_k` matches are found or the index is exhausted.
- **Container could not serve a request** - the image selected the `ollama` and `sentence-transformers` providers and installed neither. It now installs what its defaults select, defaults embeddings to `ollama` (no torch), fails the build rather than the first request on a missing provider, keeps state on a declared volume instead of `/tmp`, and runs as a non-root user.
- **`pip install widemem-ai[all]` could not start the server** - the `[all]` extra carried no `fastapi` or `uvicorn`, and none of the security floors the `server` extra pins. All four are now included.

### Changed

- **README benchmark table** - showed v1.4.1 numbers (54.81%, ~214 tokens) while shipping v1.5.0. Updated to the v1.5 figures (55.15% under an independent judge, ~213 tokens) and made the reproduction note precise: the harness and question split are committed, the LoCoMo dataset is not vendored, and result files are not committed.

## [1.5.0] - 2026-07-17

### Changed

- **Benchmark harness integrity** - LoCoMo category mapping corrected to the official evaluation (labels published before 2026-07-06 were transposed; see docs/HISTORY.md). Answer prompts now allow complete counts/lists past the 5-6 word cap. `WM_JUDGE_MODEL` separates the judge model from the answerer; result files record `judge_llm` and `self_graded`.
- **Tracked full-run harness** - `benchmark/run_ws1.py` (the runner behind the published numbers) is now in the repo with env-overridable output paths.
- **Published numbers** - v1.5 LoCoMo: 54.81 -> 55.15 overall under an independent GPT-4o judge (56.32 self-graded); temporal 60.02, ahead of every reference system in our comparison set; ~213 tokens per query. Full breakdown at https://widemem.ai/benchmarks.

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
