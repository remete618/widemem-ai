# YMYL — Your Money or Your Life

## What is YMYL?

YMYL is a concept borrowed from Google's Search Quality Guidelines. It stands for "Your Money or Your Life" — content that, if inaccurate or forgotten, could seriously impact a person's health, financial stability, safety, or legal standing.

In widemem, YMYL is a prioritization system that ensures critical facts about health, finances, legal matters, and safety receive special treatment. Because forgetting someone's allergy is not the same as forgetting their favorite color.

## The Problem

Keyword-based classification has a fundamental weakness: **no semantic awareness**. The word "bank" appears in both "opened a bank account" and "sat on the river bank". A naive keyword matcher would flag both as financial content.

We call these **false positives**, and they're the reason widemem uses a two-tier confidence system instead of a simple keyword match.

## How It Works: Confidence Tiers

widemem classifies YMYL content into two confidence levels:

### Strong Confidence

A fact is classified as **strong YMYL** when:
- It matches a **multi-word pattern** that's unambiguous (e.g., "bank account", "blood type", "insurance policy", "diabetes diagnosis")
- OR it matches **two or more weak keywords** from the same category (e.g., "doctor" + "medication" = strong health)

Strong YMYL facts receive the full treatment:
- Importance floor raised to `min_importance` (default 8.0 out of 10)
- Immune to time decay — recency score stays at 1.0 forever
- Forced active retrieval — contradictions trigger clarification even if `enable_active_retrieval` is off

### Weak Confidence

A fact is classified as **weak YMYL** when:
- It matches only a **single keyword** that could be ambiguous (e.g., "doctor" alone, "bank" alone, "fire" alone)

Weak YMYL facts receive a moderate boost:
- Importance nudged to 6.0 (if below) — a gentle push, not a full override
- Still subject to normal time decay
- No forced active retrieval

### No Match

If no YMYL keywords are found, the fact is treated normally. No boost, no special handling.

## Examples

| Input | Confidence | Category | Why |
|---|---|---|---|
| "diagnosed with diabetes by the doctor" | **Strong** | health | Multi-word "diabetes diagnosis" OR two weak hits (diabetes + doctor) |
| "my bank account balance is low" | **Strong** | financial | Multi-word "bank account" |
| "blood type is O+" | **Strong** | safety | Multi-word "blood type" |
| "insurance premium went up" | **Strong** | insurance | Multi-word "insurance premium" |
| "doctor prescribed medication" | **Strong** | health | Two weak hits: "doctor" + "medication" |
| "went to the doctor" | **Weak** | health | Single weak keyword "doctor" |
| "walked by the bank" | **Weak** | financial | Single weak keyword "bank" |
| "the fire was warm" | No match | — | "fire" alone doesn't match any pattern (removed from weak to avoid camping/cooking false positives) |
| "I like pizza" | No match | — | No YMYL keywords at all |
| "watching Doctor Who" | **Weak** | health | Single weak keyword "doctor" — gets a minor boost (6.0), not the full floor (8.0) |

The "Doctor Who" case is intentionally a weak match. It gets a small importance bump (6.0 instead of 5.0) but NOT the full YMYL treatment (8.0 floor, decay immunity). This is the right trade-off: occasionally bumping a TV show reference slightly is far less harmful than missing a real medical fact.

## Categories

widemem recognizes 8 YMYL categories, each with its own strong and weak keyword patterns:

| Category | Strong Patterns (examples) | Weak Patterns (examples) |
|---|---|---|
| `health` | blood pressure, diabetes diagnosis, mental health, cancer treatment | doctor, hospital, surgery, medication, anxiety |
| `medical` | lab results, medical condition, treatment plan, medical emergency | clinic, vaccine, MRI, scan, nurse |
| `financial` | bank account, savings account, credit score, mortgage rate, 401k | bank, savings, loan, debt, salary, budget |
| `legal` | power of attorney, child custody, court order, estate planning | lawyer, lawsuit, contract, divorce, settlement |
| `safety` | emergency contact, next of kin, blood type, epipen, DNR order | evacuation, flood, fire alarm |
| `insurance` | insurance policy, insurance premium, insurance claim | insurance, premium, coverage, claim |
| `tax` | tax return, tax filing, W-2, 1099, IRS audit | deduction, audit, filing, exemption |
| `pharmaceutical` | side effect, drug interaction, contraindication | drug, dosage, pharmacist, prescription |

## Configuration

```python
from widemem import MemoryConfig
from widemem.core.types import YMYLConfig

config = MemoryConfig(
    ymyl=YMYLConfig(
        enabled=True,                    # Turn YMYL on
        categories=[                     # Which categories to check
            "health", "medical", "financial", "legal",
            "safety", "insurance", "tax", "pharmaceutical",
        ],
        min_importance=8.0,              # Importance floor for strong YMYL facts
        decay_immune=True,               # Strong YMYL facts don't decay
        force_active_retrieval=True,     # Force contradiction detection for strong YMYL
    ),
)
```

### Partial Categories

You don't have to enable all 8 categories. If you only care about health and financial:

```python
YMYLConfig(enabled=True, categories=["health", "financial"])
```

### Disabling Specific Behaviors

```python
# YMYL classification but no decay immunity
YMYLConfig(enabled=True, decay_immune=False)

# YMYL classification but no forced active retrieval
YMYLConfig(enabled=True, force_active_retrieval=False)

# Lower importance floor
YMYLConfig(enabled=True, min_importance=7.0)
```

## How YMYL Flows Through the System

```
User adds text
    │
    ▼
LLM extracts facts with importance 1-10
    │
    ▼
For each fact, run YMYL classification
    │
    ├── Strong match? → importance = max(importance, 8.0)
    ├── Weak match?   → importance = max(importance, 6.0)
    └── No match?     → importance unchanged
    │
    ▼
Batch conflict resolution (ADD/UPDATE/DELETE)
    │
    ▼
Store in vector DB + history
    │
    ▼
On search, apply scoring:
    │
    ├── Strong YMYL + decay_immune? → recency = 1.0 (no decay)
    └── Everything else             → normal decay applied
    │
    ▼
Return ranked results
```

## Limitations

1. **Keyword-based, not semantic.** "My grandmother's homemade medicine" would match "medicine" (weak YMYL) even though it's not a medical fact. The two-tier system mitigates this — it gets a 6.0 bump, not the full 8.0 treatment.

2. **English-centric patterns.** The keyword lists are in English. Non-English medical or financial terms won't match. If you need multilingual YMYL, you'd need to extend the pattern dictionaries.

3. **No negation handling.** "I don't have diabetes" matches "diabetes" the same as "I have diabetes". The importance gets boosted either way. In practice, this is acceptable — a fact about NOT having diabetes is still medically relevant.

4. **Category overlap.** Some keywords appear in multiple categories (e.g., "prescription" is in both health and pharmaceutical). The first matching category wins, based on the order in `config.categories`.

5. **Not a compliance tool.** YMYL is a best-effort prioritization heuristic. It is not HIPAA, GDPR, or any regulatory compliance mechanism. Don't use it as one.

## Why Not Use the LLM for YMYL Classification?

We considered it. Three reasons we didn't:

1. **Cost.** Every fact would need an extra LLM call just to classify it. At scale, this doubles your API bill for extraction.
2. **Latency.** An additional round-trip per fact slows down the add() pipeline significantly.
3. **Determinism.** Keyword matching is fast, cheap, and reproducible. The LLM might classify "walked by the bank" as financial one day and not the next. The two-tier keyword system gives you consistent, predictable behavior.

The LLM already participates indirectly: during extraction, if YMYL is enabled, the system prompt instructs the LLM to rate health/financial/legal facts at 8-10 importance. So the LLM does contribute — it just doesn't make the binary "is this YMYL?" decision.
