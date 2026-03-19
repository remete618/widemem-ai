"""Basic widemem usage: add memories and search them."""

from widemem import WideMemory, MemoryConfig

# Default config uses OpenAI for LLM + embeddings, FAISS for storage
mem = WideMemory()

# Add memories
result = mem.add("I live in San Francisco and work at Google as a senior engineer.", user_id="alice")
print(f"Added {len(result.memories)} memories")
for m in result.memories:
    print(f"  - {m.content} (importance: {m.importance})")

# Search
results = mem.search("where does alice work", user_id="alice", top_k=3)
for r in results:
    print(f"  [{r.final_score:.3f}] {r.memory.content}")

# --- Using Anthropic ---
# config = MemoryConfig(llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"))
# mem = WideMemory(config=config)

# --- Fully local (Ollama + sentence-transformers) ---
# from widemem.core.types import LLMConfig, EmbeddingConfig
# config = MemoryConfig(
#     llm=LLMConfig(provider="ollama", model="llama3"),
#     embedding=EmbeddingConfig(provider="sentence-transformers", model="all-MiniLM-L6-v2", dimensions=384),
# )
# mem = WideMemory(config=config)
