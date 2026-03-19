"""Hierarchical memory: facts -> summaries -> themes with query routing."""

from widemem import WideMemory, MemoryConfig

config = MemoryConfig(enable_hierarchy=True)
mem = WideMemory(config=config)

# Add many facts about a user
facts = [
    "I live in San Francisco, California",
    "I moved from Boston last year",
    "I speak Spanish and French fluently",
    "I work at Google as a senior engineer",
    "I specialize in distributed systems",
    "I enjoy hiking in the Alps on weekends",
    "I run marathons — my best time is 3:45",
    "I'm vegetarian and love cooking Thai food",
    "I have a golden retriever named Max",
    "I graduated from ETH Zurich in 2015",
]

for fact in facts:
    mem.add(fact, user_id="alice")

# Trigger summarization (groups facts, creates summaries and themes)
results = mem.summarize(user_id="alice", force=True)
print(f"Created {len(results)} summaries/themes:")
for m in results:
    print(f"  [{m.tier.value}] {m.content}")

# Specific query -> routed to facts
print("\n--- Specific query ---")
for r in mem.search("what is alice's marathon time", user_id="alice", top_k=3):
    print(f"  [{r.memory.tier.value}] {r.memory.content}")

# Broad query -> routed to themes/summaries
print("\n--- Broad query ---")
for r in mem.search("tell me about alice", user_id="alice", top_k=3):
    print(f"  [{r.memory.tier.value}] {r.memory.content}")
