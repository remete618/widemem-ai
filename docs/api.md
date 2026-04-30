# API Reference

## WideMemory

| Method | Description |
|---|---|
| `add(text, user_id, agent_id, run_id, on_clarification)` | Extract and store memories. Returns `AddResult`. |
| `add_batch(texts, user_id, agent_id, run_id)` | Process multiple texts. Returns `List[AddResult]`. |
| `search(query, user_id, agent_id, top_k, time_after, time_before, tier, mode)` | Search memories. Returns `SearchResult` (list-compatible, with `.confidence`). |
| `pin(text, user_id, agent_id, importance=9.0)` | Store memory with elevated importance. For facts that must not be forgotten. |
| `get(memory_id)` | Get a single memory by ID. Returns `Memory` or `None`. |
| `delete(memory_id)` | Delete a memory by ID. |
| `get_history(memory_id)` | Get audit trail for a memory. Returns `List[HistoryEntry]`. |
| `summarize(user_id, agent_id, force)` | Trigger hierarchical summarization. Returns `List[Memory]`. |
| `count(user_id, agent_id, tier)` | Count memories with optional filters. Returns `int`. |
| `export_json(user_id, agent_id)` | Export memories as JSON string. |
| `import_json(data)` | Import memories from JSON string. Returns count imported. |
| `close()` | Close connections (called automatically when used as context manager). |

## AddResult

| Field | Type | Description |
|---|---|---|
| `memories` | `List[Memory]` | Newly created or updated memories |
| `clarifications` | `List[Clarification]` | Any detected conflicts |
| `has_clarifications` | `bool` | Whether conflicts were detected |

## SearchResult

`SearchResult` behaves like a list (backward compatible) and also exposes confidence metadata.

| Attribute | Type | Description |
|---|---|---|
| `results` | `List[MemorySearchResult]` | Ranked search results |
| `confidence` | `RetrievalConfidence` | `HIGH`, `MODERATE`, `LOW`, or `NONE` |
| `has_relevant` | `bool` | True if confidence is not `NONE` |

Standard list operations (`len()`, `for`, `[idx]`, `bool()`) work as expected.

## MemorySearchResult

| Field | Type | Description |
|---|---|---|
| `memory` | `Memory` | The memory object |
| `similarity_score` | `float` | Vector similarity (0-1) |
| `temporal_score` | `float` | Recency score after decay (0-1) |
| `importance_score` | `float` | Normalized importance (0-1) |
| `final_score` | `float` | Combined weighted score |

## Memory

| Field | Type | Description |
|---|---|---|
| `id` | `str` | UUID4 |
| `content` | `str` | Memory text |
| `user_id` | `Optional[str]` | Owner of the memory |
| `agent_id` | `Optional[str]` | Agent that wrote the memory |
| `run_id` | `Optional[str]` | Optional run/session identifier |
| `tier` | `MemoryTier` | `FACT`, `SUMMARY`, or `THEME` |
| `importance` | `float` | 0.0 to 10.0 |
| `content_hash` | `str` | Used for dedup |
| `ymyl_category` | `Optional[str]` | Set when YMYL classification fires |
| `metadata` | `dict` | Provider-specific extras |
| `created_at` | `datetime` | UTC creation time |
| `updated_at` | `datetime` | UTC last-update time |

## HistoryEntry

| Field | Type | Description |
|---|---|---|
| `id` | `str` | UUID4 |
| `memory_id` | `str` | Memory this entry refers to |
| `action` | `MemoryAction` | `ADD`, `UPDATE`, `DELETE`, `NONE` |
| `old_content` | `Optional[str]` | Previous content (for UPDATE/DELETE) |
| `new_content` | `Optional[str]` | New content (for ADD/UPDATE) |
| `timestamp` | `datetime` | UTC time of the action |

## Enums

| Enum | Values |
|---|---|
| `MemoryTier` | `FACT`, `SUMMARY`, `THEME` |
| `MemoryAction` | `ADD`, `UPDATE`, `DELETE`, `NONE` |
| `RetrievalMode` | `FAST`, `BALANCED`, `DEEP` |
| `RetrievalConfidence` | `HIGH`, `MODERATE`, `LOW`, `NONE` |
| `UncertaintyMode` | `STRICT`, `HELPFUL`, `CREATIVE` |
| `DecayFunction` | `EXPONENTIAL`, `LINEAR`, `STEP`, `NONE` |
