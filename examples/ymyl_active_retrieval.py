"""YMYL + Active Retrieval: healthcare memory that catches contradictions.

Demonstrates how widemem handles critical health information:
- YMYL flags medical facts with high importance and decay immunity
- Active retrieval detects contradictions (e.g. medication changes)
- Clarification callback lets the app resolve conflicts interactively
"""

from widemem import WideMemory, MemoryConfig
from widemem.core.types import LLMConfig, YMYLConfig, ScoringConfig, DecayFunction


def handle_clarification(clarifications):
    """Resolve conflicts by accepting the newer information."""
    print("\n--- Contradiction detected ---")
    answers = []
    for c in clarifications:
        print(f"  Existing: {c.existing_memory}")
        print(f"  New:      {c.new_fact}")
        print(f"  Question: {c.question}")
        answers.append("Accept the updated information")
    print("--- Auto-resolving with newest info ---\n")
    return answers


config = MemoryConfig(
    ymyl=YMYLConfig(
        enabled=True,
        categories=["health", "medical", "pharmaceutical", "safety"],
        min_importance=8.0,
        decay_immune=True,
        force_active_retrieval=True,
    ),
    scoring=ScoringConfig(
        decay_function=DecayFunction.EXPONENTIAL,
        decay_rate=0.01,
    ),
    enable_active_retrieval=True,
    active_retrieval_threshold=0.6,
)

mem = WideMemory(config)

# Patient intake
print("=== Patient intake ===")
result = mem.add(
    "Patient has type 2 diabetes, diagnosed 2019. Currently on metformin 500mg twice daily. "
    "Allergic to penicillin. Blood type A+. Emergency contact: spouse Maria, 555-0123.",
    user_id="patient_001",
)
for m in result.memories:
    print(f"  [{m.importance}] {m.content}")

# Later visit — medication change (should trigger contradiction detection)
print("\n=== Follow-up visit ===")
result = mem.add(
    "Doctor increased metformin to 1000mg twice daily due to elevated HbA1c.",
    user_id="patient_001",
    on_clarification=handle_clarification,
)
if result.has_clarifications:
    print(f"Resolved {len(result.clarifications)} medication conflict(s)")
for m in result.memories:
    print(f"  [{m.importance}] {m.content}")

# Search — YMYL facts should rank high regardless of age
print("\n=== Critical info search ===")
results = mem.search("what medications and allergies", user_id="patient_001", top_k=5)
for r in results:
    print(f"  [{r.final_score:.3f}] {r.memory.content}")

# Verify decay immunity — YMYL facts keep full recency score
print("\n=== Recency scores (YMYL facts should be 1.0) ===")
for r in results:
    print(f"  temporal={r.temporal_score:.2f}  {r.memory.content[:60]}...")
