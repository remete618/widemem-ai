from widemem import WideMemory, MemoryConfig
from widemem.core.types import LLMConfig, VectorStoreConfig

m = WideMemory(MemoryConfig(llm=LLMConfig(provider="ollama", model="llama3.2:3b"), vector_store=VectorStoreConfig(provider="faiss", path="./demo_data")))
m.add("I live in San Francisco and work as a software engineer", user_id="alice")
m.add("I just moved to Boston", user_id="alice")
print("stored. exiting.")
