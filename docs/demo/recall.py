from widemem import WideMemory, MemoryConfig
from widemem.core.types import LLMConfig, VectorStoreConfig

m = WideMemory(MemoryConfig(llm=LLMConfig(provider="ollama", model="llama3.2:3b"), vector_store=VectorStoreConfig(provider="faiss", path="./demo_data")))
r = m.search("where does alice live", user_id="alice")
print(f"{r[0].memory.content}  (score {r[0].final_score:.2f}, {len(r)} memories match)")
