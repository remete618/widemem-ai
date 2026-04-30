# Configuration Reference

All settings live in `MemoryConfig`. Most defaults are sane; you only touch what you need.

```python
from widemem import WideMemory, MemoryConfig
from widemem.core.types import (
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    ScoringConfig,
    YMYLConfig,
    TopicConfig,
    DecayFunction,
    RetrievalMode,
    UncertaintyMode,
)

config = MemoryConfig(
    llm=LLMConfig(
        provider="openai",          # "openai", "anthropic", or "ollama"
        model="gpt-4o-mini",
        api_key="sk-...",           # Or set OPENAI_API_KEY env var
        base_url=None,
        temperature=0.0,
        max_tokens=2000,
    ),
    embedding=EmbeddingConfig(
        provider="openai",          # "openai" or "sentence-transformers"
        model="text-embedding-3-small",
        api_key=None,
        base_url=None,
        dimensions=1536,
    ),
    vector_store=VectorStoreConfig(
        provider="faiss",           # "faiss" or "qdrant"
        path=None,                  # Optional path for persistent storage
    ),
    scoring=ScoringConfig(
        decay_function=DecayFunction.EXPONENTIAL,
        decay_rate=0.01,            # Higher means faster decay
        similarity_weight=0.5,
        importance_weight=0.3,
        recency_weight=0.2,
    ),
    ymyl=YMYLConfig(
        enabled=False,
        categories=[
            "health", "medical", "financial", "legal",
            "safety", "insurance", "tax", "pharmaceutical",
        ],
        min_importance=8.0,
        decay_immune=True,
        force_active_retrieval=True,
    ),
    topics=TopicConfig(
        weights={},                 # Topic boost multipliers
        custom_topics=[],           # Hints for extraction
    ),
    history_db_path="~/.widemem/history.db",
    retrieval_mode=RetrievalMode.BALANCED,
    uncertainty_mode=UncertaintyMode.HELPFUL,
    enable_hierarchy=False,
    enable_active_retrieval=False,
    active_retrieval_threshold=0.6,
    collect_extractions=False,
    extractions_db_path="~/.widemem/extractions.db",
    ttl_days=None,
)

memory = WideMemory(config)
```

## Field reference

| Field | Type | Default | Notes |
|---|---|---|---|
| `llm.provider` | str | `"openai"` | `"openai"`, `"anthropic"`, `"ollama"` |
| `llm.model` | str | `"gpt-4o-mini"` | Provider-specific model name |
| `llm.api_key` | SecretStr | None | Reads `OPENAI_API_KEY` env var if not set |
| `llm.base_url` | str | None | Override for self-hosted providers |
| `llm.temperature` | float | 0.0 | LLM sampling temperature |
| `llm.max_tokens` | int | 2000 | Max tokens per LLM response |
| `embedding.provider` | str | `"openai"` | `"openai"` or `"sentence-transformers"` |
| `embedding.model` | str | `"text-embedding-3-small"` | |
| `embedding.dimensions` | int | 1536 | Must match the model |
| `vector_store.provider` | str | `"faiss"` | `"faiss"` or `"qdrant"` |
| `vector_store.path` | str | None | If set, FAISS persists to this directory |
| `scoring.decay_function` | DecayFunction | `EXPONENTIAL` | `EXPONENTIAL`, `LINEAR`, `STEP`, `NONE` |
| `scoring.decay_rate` | float | 0.01 | Higher means faster decay |
| `scoring.similarity_weight` | float | 0.5 | Cosine similarity weight in `final_score` |
| `scoring.importance_weight` | float | 0.3 | Importance weight in `final_score` |
| `scoring.recency_weight` | float | 0.2 | Recency weight in `final_score` |
| `ymyl.enabled` | bool | False | Toggle YMYL prioritization |
| `ymyl.categories` | list | 8 categories | Subset of YMYL categories to enable |
| `ymyl.min_importance` | float | 8.0 | Floor for strong YMYL facts |
| `ymyl.decay_immune` | bool | True | Whether YMYL facts skip decay |
| `ymyl.force_active_retrieval` | bool | True | Force conflict checks on YMYL adds |
| `topics.weights` | dict | {} | `{topic: multiplier}` for retrieval boost |
| `topics.custom_topics` | list | [] | Extraction hints for the LLM |
| `history_db_path` | str | `~/.widemem/history.db` | SQLite path for audit trail |
| `retrieval_mode` | RetrievalMode | `BALANCED` | `FAST`, `BALANCED`, `DEEP` |
| `uncertainty_mode` | UncertaintyMode | `HELPFUL` | `STRICT`, `HELPFUL`, `CREATIVE` |
| `enable_hierarchy` | bool | False | Enable summary/theme tiers |
| `enable_active_retrieval` | bool | False | Enable contradiction detection callbacks |
| `active_retrieval_threshold` | float | 0.6 | Similarity threshold for conflict detection |
| `collect_extractions` | bool | False | Log extraction pairs for self-supervised distillation |
| `extractions_db_path` | str | `~/.widemem/extractions.db` | SQLite path for extraction logs |
| `ttl_days` | int | None | If set, memories older than N days are filtered at search time |

## Retrieval mode presets

| Mode | top_k | fetch_k_multiplier | similarity_boost | enable_hierarchy |
|---|---|---|---|---|
| `FAST` | 10 | 3 | 0.10 | False |
| `BALANCED` (default) | 25 | 4 | 0.15 | True |
| `DEEP` | 50 | 5 | 0.20 | True |

Setting `enable_hierarchy=True` at the config level forces hierarchy on regardless of mode.

## Environment variables

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | OpenAI LLM and embedding providers |
| `ANTHROPIC_API_KEY` | Anthropic LLM provider |
| `OLLAMA_BASE_URL` | Ollama provider (default `http://localhost:11434`) |
| `QDRANT_URL` | Remote Qdrant vector store |
