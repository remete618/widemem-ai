"""Healthcare quickstart: ingest a clinical encounter, retrieve YMYL facts,
demonstrate graceful memory-miss when a fact wasn't recorded.

Demonstrates the pieces a healthcare AI agent actually needs:
- YMYL prioritization gives medical facts an importance floor and decay immunity
- Two-stage classification catches both explicit ("Type 2 diabetes") and implied
  ("my chest hurts") medical content
- Active retrieval forces contradiction detection on critical facts
- Confidence levels let the agent abstain instead of fabricating

Run: OPENAI_API_KEY=sk-... python examples/healthcare_quickstart.py
"""

from widemem import WideMemory, MemoryConfig
from widemem.core.types import (
    DecayFunction,
    LLMConfig,
    RetrievalConfidence,
    ScoringConfig,
    UncertaintyMode,
    YMYLConfig,
)


config = MemoryConfig(
    llm=LLMConfig(provider="openai", model="gpt-4o-mini"),
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
    uncertainty_mode=UncertaintyMode.HELPFUL,
    enable_active_retrieval=True,
    active_retrieval_threshold=0.6,
)

with WideMemory(config) as mem:
    # 1. Ingest a doctor-patient encounter note
    print("=== Encounter intake ===")
    encounter = (
        "Mrs. Garcia, 67, presents with chest tightness for three days. "
        "History of type 2 diabetes since 2014, on metformin 500mg twice daily. "
        "Allergic to penicillin (anaphylaxis 2018). Blood type A negative. "
        "Husband reports increased fatigue over the past week."
    )
    result = mem.add(encounter, user_id="garcia-patient", agent_id="cardiology-scribe")
    print(f"Extracted {len(result.memories)} memories.\n")

    print("YMYL-flagged facts (importance floor 8.0, decay-immune):")
    for m in result.memories:
        flag = f"YMYL/{m.ymyl_category}" if m.ymyl_category else "regular"
        print(f"  [{flag:14s}] importance={m.importance:.1f}  {m.content}")

    # 2. Retrieve a recorded YMYL fact
    print("\n=== Recorded fact: drug allergies ===")
    response = mem.search(
        "Does the patient have any known drug allergies?",
        user_id="garcia-patient",
    )
    print(f"Confidence: {response.confidence.value}  has_relevant={response.has_relevant}")
    for r in response[:3]:
        print(f"  -> {r.memory.content}  (score={r.final_score:.2f})")

    # 3. Graceful memory-miss for a fact that wasn't recorded
    print("\n=== Unrecorded fact: home address ===")
    response = mem.search(
        "What is the patient's home address?",
        user_id="garcia-patient",
    )
    print(f"Confidence: {response.confidence.value}  has_relevant={response.has_relevant}")

    if response.confidence == RetrievalConfidence.NONE:
        print("  Action: agent should respond")
        print('    "I do not have that information recorded for this patient."')
    elif response.confidence == RetrievalConfidence.LOW:
        print("  Action: agent should hedge or ask for confirmation.")
    else:
        for r in response[:1]:
            print(f"  -> {r.memory.content}  (score={r.final_score:.2f})")

    # 4. Pin a critical clarification the user explicitly states
    print("\n=== Pinning a critical correction ===")
    pinned = mem.pin(
        "Patient blood type was originally documented as A negative, "
        "but lab confirmed AB negative on 2026-04-30.",
        user_id="garcia-patient",
    )
    print(f"Pinned with importance {pinned.importance:.1f}, ymyl={pinned.ymyl_category}.")

    # 5. Re-query to confirm pinned correction outranks the older fact
    print("\n=== Re-query: blood type ===")
    response = mem.search(
        "What is the patient's blood type?",
        user_id="garcia-patient",
    )
    print(f"Confidence: {response.confidence.value}")
    for r in response[:2]:
        print(f"  -> {r.memory.content}  (score={r.final_score:.2f})")
