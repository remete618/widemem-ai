# Configuration Reference

`MemoryConfig` is the top-level configuration object for `WideMemory`. This page documents the public configuration fields defined in `widemem/core/types.py`.

```python
from widemem import MemoryConfig, WideMemory

config = MemoryConfig()
memory = WideMemory(config)
```

For nested configuration objects, import them from `widemem.core.types`:

```python
from widemem.core.types import (
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    ScoringConfig,
    TopicConfig,
    VectorStoreConfig,
    YMYLConfig,
)
```

## MemoryConfig

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `llm` | `LLMConfig` | `LLMConfig()` | LLM provider settings used for extraction, summarization, and conflict resolution. |
| `embedding` | `EmbeddingConfig` | `EmbeddingConfig()` | Embedding provider settings used to convert text into vectors. |
| `vector_store` | `VectorStoreConfig` | `VectorStoreConfig()` | Vector database settings for memory storage and search. |
| `scoring` | `ScoringConfig` | `ScoringConfig()` | Ranking weights and time-decay settings for search results. |
| `ymyl` | `YMYLConfig` | `YMYLConfig()` | Your Money or Your Life settings for high-stakes facts. |
| `topics` | `TopicConfig` | `TopicConfig()` | Topic boost and custom extraction hint settings. |
| `history_db_path` | `str` | `"~/.widemem/history.db"` | SQLite path for the memory history and audit trail. |
| `retrieval_mode` | `RetrievalMode` | `RetrievalMode.BALANCED` | Default retrieval preset: `FAST`, `BALANCED`, or `DEEP`. |
| `uncertainty_mode` | `UncertaintyMode` | `UncertaintyMode.HELPFUL` | Stored on the config but not read by `search()`. Pass a mode to `widemem.retrieval.uncertainty.build_uncertainty_guidance()` instead. |
| `enable_hierarchy` | `bool` | `False` | Forces hierarchical memory routing on when set. |
| `enable_active_retrieval` | `bool` | `False` | Enables contradiction checks and clarification callbacks for new memories. |
| `active_retrieval_threshold` | `float` | `0.6` | Similarity threshold used by active retrieval conflict detection. |
| `collect_extractions` | `bool` | `False` | Stores extraction input/output pairs for later self-supervised training. `WIDEMEM_COLLECT_EXTRACTIONS=1` also turns it on. |
| `extractions_db_path` | `str` | `"~/.widemem/extractions.db"` | SQLite path for collected extraction training examples. |
| `enable_fact_consolidation` | `bool` | `False` | Passes linked candidate memories into conflict resolution so each fact can add, update, delete, or noop deterministically. |
| `ttl_days` | `Optional[int]` | `None` | Hides memories older than this many days from `search()`. A filter, not a deletion: hidden rows stay on disk and `get()`, `count()` and `export_json()` still return them. Use `purge_expired()` to remove them. |
| `parse_temporal_hints` | `bool` | `False` | Auto-parses time ranges from temporal search queries when explicit time filters are not provided. |
| `enable_hybrid_search` | `bool` | `False` | Blends BM25 keyword scores into vector similarity before ranking. |
| `hybrid_bm25_weight` | `float` | `0.5` | Fraction of hybrid similarity taken from BM25 when hybrid search is enabled. |

## LLMConfig

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | `str` | `"ollama"` | LLM backend name. Supported by `WideMemory`: `ollama`, `openai`, `anthropic`. |
| `model` | `str` | `"llama3.1:8b"` | Model name. Left unset, each provider gets its own default: `ollama` `llama3.1:8b`, `openai` `gpt-4o-mini`, `anthropic` `claude-haiku-4-5-20251001`. |
| `api_key` | `Optional[SecretStr]` | `None` | API key passed to providers that need one. |
| `base_url` | `Optional[str]` | `None` | Provider base URL override, commonly used for local or compatible endpoints. |
| `temperature` | `float` | `0.0` | Sampling temperature for LLM generation. |
| `max_tokens` | `int` | `2000` | Maximum tokens requested from the LLM response. |

## EmbeddingConfig

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | `str` | `"sentence-transformers"` | Embedding backend name. Supported by `WideMemory`: `sentence-transformers`, `ollama`, `openai`. |
| `model` | `str` | `"all-MiniLM-L6-v2"` | Embedding model name. Left unset, each provider gets its own default model and size: `sentence-transformers` `all-MiniLM-L6-v2` (384), `ollama` `nomic-embed-text` (768), `openai` `text-embedding-3-small` (1536). |
| `api_key` | `Optional[SecretStr]` | `None` | API key passed to embedding providers that need one. |
| `base_url` | `Optional[str]` | `None` | Provider base URL override, commonly used for Ollama or compatible endpoints. |
| `dimensions` | `int` | `384` | Embedding vector size; must match the model. A stored FAISS index, Qdrant collection or pgvector table refuses to open under a different size. |

## VectorStoreConfig

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | `str` | `"faiss"` | Vector store backend name. Supported by `WideMemory`: `faiss`, `qdrant`, `pgvector`. |
| `path` | `Optional[str]` | `None` | Local persistence path. FAISS without a path keeps vectors in RAM only. Qdrant with a path runs embedded. |
| `url` | `Optional[str]` | `None` | Connection URL for pgvector. Qdrant ignores it: without `path` it connects to `localhost:6333`. |
| `table_name` | `str` | `"widemem_vectors"` | Table name used by the pgvector backend. |

## ScoringConfig

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `decay_function` | `DecayFunction` | `DecayFunction.EXPONENTIAL` | Time-decay function: `EXPONENTIAL`, `LINEAR`, `STEP`, or `NONE`. |
| `decay_rate` | `float` | `0.01` | Decay speed; higher values reduce older memories faster. |
| `similarity_weight` | `float` | `0.5` | Weight applied to vector similarity in the final score. |
| `importance_weight` | `float` | `0.3` | Weight applied to memory importance in the final score. |
| `recency_weight` | `float` | `0.2` | Weight applied to time-decay recency in the final score. |

## YMYLConfig

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | `bool` | `False` | Enables YMYL handling for high-stakes facts. |
| `categories` | `list` | `["health", "medical", "financial", "legal", "safety", "insurance", "tax", "pharmaceutical"]` | YMYL category names used for classification and prioritization. |
| `min_importance` | `float` | `8.0` | Minimum importance assigned to strong YMYL facts. |
| `decay_immune` | `bool` | `True` | Prevents YMYL facts from losing score through time decay. |
| `force_active_retrieval` | `bool` | `True` | Runs active retrieval checks for YMYL facts even when global active retrieval is off. |

## TopicConfig

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `weights` | `Dict[str, float]` | `{}` | Topic-to-multiplier map applied to matching memories during scoring. Above 1.0 boosts, below 1.0 suppresses; must be finite and above 0. |
| `custom_topics` | `list` | `[]` | Topic hints passed to extraction so domain-specific facts can be labeled. |

## Retrieval mode presets

`MemoryConfig.get_retrieval_preset()` starts from the selected `retrieval_mode`. If `enable_hierarchy=True`, the returned preset always has `enable_hierarchy` set to `True`.

| Mode | `top_k` | `fetch_k_multiplier` | `similarity_boost` | `enable_hierarchy` |
| --- | --- | --- | --- | --- |
| `RetrievalMode.FAST` | `10` | `3` | `0.10` | `False` |
| `RetrievalMode.BALANCED` | `25` | `4` | `0.15` | `True` |
| `RetrievalMode.DEEP` | `50` | `5` | `0.20` | `True` |

## Common configurations

### Local (default)

`MemoryConfig()` runs on Ollama for the LLM, sentence-transformers for embeddings and FAISS for vectors. Nothing leaves the machine. Install with `pip install "widemem-ai[local]"`, run `ollama pull llama3.1:8b` (4.9 GB), and set a FAISS `path` to keep vectors across restarts.

```python
from widemem import MemoryConfig, WideMemory
from widemem.core.types import VectorStoreConfig

memory = WideMemory(MemoryConfig(
    vector_store=VectorStoreConfig(provider="faiss", path="./widemem_faiss"),
))
```

sentence-transformers downloads `all-MiniLM-L6-v2` (about 90 MB) on first use; after that it runs offline.

### OpenAI

Install `pip install "widemem-ai[openai,faiss]"` and set `OPENAI_API_KEY`. Each provider picks its own default model.

```python
from widemem import MemoryConfig, WideMemory
from widemem.core.types import EmbeddingConfig, LLMConfig

memory = WideMemory(MemoryConfig(
    llm=LLMConfig(provider="openai"),              # gpt-4o-mini
    embedding=EmbeddingConfig(provider="openai"),  # text-embedding-3-small, 1536
))
```

### Mixed: cloud LLM, local embeddings

Facts go to the LLM for extraction; vectors stay local.

```python
from widemem import MemoryConfig, WideMemory
from widemem.core.types import LLMConfig

memory = WideMemory(MemoryConfig(
    llm=LLMConfig(provider="anthropic"),  # claude-haiku-4-5-20251001; needs [anthropic] and ANTHROPIC_API_KEY
))
```

### Ollama for everything

```python
from widemem import MemoryConfig, WideMemory
from widemem.core.types import EmbeddingConfig

memory = WideMemory(MemoryConfig(
    embedding=EmbeddingConfig(provider="ollama"),  # nomic-embed-text, 768
))
```

## Environment variables

| Variable | Used by |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI LLM and embedding providers. |
| `ANTHROPIC_API_KEY` | Anthropic LLM provider. |
| `WIDEMEM_*` | Provider, model and data path for the MCP server ([mcp.md](mcp.md#environment-variables)) and the REST server. The REST server defaults to `ollama` / `llama3.1:8b` and also reads `WIDEMEM_HOST`, `WIDEMEM_PORT` (or `PORT`) and `WIDEMEM_API_KEY`. |
| `WIDEMEM_CONFIDENCE_HIGH`, `WIDEMEM_CONFIDENCE_MODERATE`, `WIDEMEM_CONFIDENCE_LOW` | Override the similarity thresholds behind `RetrievalConfidence`. Defaults depend on the embedding model: `all-MiniLM-L6-v2` uses 0.60 / 0.30 / 0.20; every other model uses 0.60 / 0.50 / 0.30, calibrated for `text-embedding-3-small`. Each variable overrides its own level on top of the per-model defaults, so setting only `WIDEMEM_CONFIDENCE_MODERATE` keeps the model's `high` and `low`. Keep `high >= moderate >= low`; widemem logs a warning when the resolved set is out of order. |
| `WIDEMEM_COLLECT_EXTRACTIONS` | Set to `1` to log extractions for distillation, even when `collect_extractions` is `False`. Privacy: while set, `WideMemory` (including the MCP and REST servers) stores raw, pre-sanitization input text in `~/.widemem/extractions.db`; unset it if you did not intend that. |
