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
| `uncertainty_mode` | `UncertaintyMode` | `UncertaintyMode.HELPFUL` | How responses handle low-confidence retrieval: `STRICT`, `HELPFUL`, or `CREATIVE`. |
| `enable_hierarchy` | `bool` | `False` | Forces hierarchical memory routing on when set. |
| `enable_active_retrieval` | `bool` | `False` | Enables contradiction checks and clarification callbacks for new memories. |
| `active_retrieval_threshold` | `float` | `0.6` | Similarity threshold used by active retrieval conflict detection. |
| `collect_extractions` | `bool` | `False` | Stores extraction input/output pairs for later self-supervised training. |
| `extractions_db_path` | `str` | `"~/.widemem/extractions.db"` | SQLite path for collected extraction training examples. |
| `enable_fact_consolidation` | `bool` | `False` | Passes linked candidate memories into conflict resolution so each fact can add, update, delete, or noop deterministically. |
| `ttl_days` | `Optional[int]` | `None` | Filters memories older than this many days at search time when set. |
| `parse_temporal_hints` | `bool` | `False` | Auto-parses time ranges from temporal search queries when explicit time filters are not provided. |
| `enable_hybrid_search` | `bool` | `False` | Blends BM25 keyword scores into vector similarity before ranking. |
| `hybrid_bm25_weight` | `float` | `0.5` | Fraction of hybrid similarity taken from BM25 when hybrid search is enabled. |

## LLMConfig

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | `str` | `"openai"` | LLM backend name. Supported by `WideMemory`: `openai`, `anthropic`, `ollama`. |
| `model` | `str` | `"gpt-4o-mini"` | Provider-specific model name. |
| `api_key` | `Optional[SecretStr]` | `None` | API key passed to providers that need one. |
| `base_url` | `Optional[str]` | `None` | Provider base URL override, commonly used for local or compatible endpoints. |
| `temperature` | `float` | `0.0` | Sampling temperature for LLM generation. |
| `max_tokens` | `int` | `2000` | Maximum tokens requested from the LLM response. |

## EmbeddingConfig

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | `str` | `"openai"` | Embedding backend name. Supported by `WideMemory`: `openai`, `sentence-transformers`, `ollama`. |
| `model` | `str` | `"text-embedding-3-small"` | Provider-specific embedding model name. |
| `api_key` | `Optional[SecretStr]` | `None` | API key passed to embedding providers that need one. |
| `base_url` | `Optional[str]` | `None` | Provider base URL override, commonly used for Ollama or compatible endpoints. |
| `dimensions` | `int` | `1536` | Expected embedding vector size; must match the selected embedding model. |

## VectorStoreConfig

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | `str` | `"faiss"` | Vector store backend name. Supported by `WideMemory`: `faiss`, `qdrant`, `pgvector`. |
| `path` | `Optional[str]` | `None` | Local persistence path for backends that support file-backed storage. |
| `url` | `Optional[str]` | `None` | Connection URL for network-backed stores such as pgvector and Qdrant Cloud. |
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
| `weights` | `Dict[str, float]` | `{}` | Topic-to-multiplier map used to boost matching memories during scoring. |
| `custom_topics` | `list` | `[]` | Topic hints passed to extraction so domain-specific facts can be labeled. |

## Retrieval mode presets

`MemoryConfig.get_retrieval_preset()` starts from the selected `retrieval_mode`. If `enable_hierarchy=True`, the returned preset always has `enable_hierarchy` set to `True`.

| Mode | `top_k` | `fetch_k_multiplier` | `similarity_boost` | `enable_hierarchy` |
| --- | --- | --- | --- | --- |
| `RetrievalMode.FAST` | `10` | `3` | `0.10` | `False` |
| `RetrievalMode.BALANCED` | `25` | `4` | `0.15` | `True` |
| `RetrievalMode.DEEP` | `50` | `5` | `0.20` | `True` |

## Common configurations

### Local-only

Use Ollama for the LLM, sentence-transformers for embeddings, and FAISS for local vector storage. This setup avoids hosted LLM and embedding API keys.

```python
from widemem import MemoryConfig, WideMemory
from widemem.core.types import EmbeddingConfig, LLMConfig, VectorStoreConfig

config = MemoryConfig(
    llm=LLMConfig(provider="ollama", model="llama3", base_url="http://localhost:11434"),
    embedding=EmbeddingConfig(
        provider="sentence-transformers",
        model="all-MiniLM-L6-v2",
        dimensions=384,
    ),
    vector_store=VectorStoreConfig(provider="faiss", path="./widemem_faiss"),
)

memory = WideMemory(config)
```

### OpenAI default

The config defaults point to OpenAI for the LLM and embeddings, and FAISS for vector storage. At runtime, `WideMemory` falls back to Ollama for OpenAI-configured providers if no OpenAI API key is available.

```python
from widemem import MemoryConfig, WideMemory

config = MemoryConfig()
memory = WideMemory(config)
```

You can also set the fields explicitly:

```python
from widemem import MemoryConfig, WideMemory
from widemem.core.types import EmbeddingConfig, LLMConfig, VectorStoreConfig

config = MemoryConfig(
    llm=LLMConfig(provider="openai", model="gpt-4o-mini"),
    embedding=EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
    ),
    vector_store=VectorStoreConfig(provider="faiss"),
)

memory = WideMemory(config)
```

### Anthropic

Use Anthropic for the LLM while keeping the default OpenAI embeddings and FAISS vector storage.

```python
from widemem import MemoryConfig, WideMemory
from widemem.core.types import LLMConfig

config = MemoryConfig(
    llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
)

memory = WideMemory(config)
```

### Ollama

Use Ollama for the LLM and embeddings. The default OpenAI model names are remapped internally when the provider is `ollama`, but setting local model names explicitly makes the configuration easier to read.

```python
from widemem import MemoryConfig, WideMemory
from widemem.core.types import EmbeddingConfig, LLMConfig

config = MemoryConfig(
    llm=LLMConfig(provider="ollama", model="llama3.2", base_url="http://localhost:11434"),
    embedding=EmbeddingConfig(
        provider="ollama",
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        dimensions=768,
    ),
)

memory = WideMemory(config)
```

## Environment variables

| Variable | Used by |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI LLM and embedding providers. |
| `ANTHROPIC_API_KEY` | Anthropic LLM provider. |
| `OLLAMA_BASE_URL` | Ollama defaults used by the server and MCP server. |
| `QDRANT_URL` | Remote Qdrant configuration used by environment-driven server setup. |
