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
- It matches only a **single keyword** that could be ambiguous (e.g., "doctor" alone, "bank" alone)

A weak match changes nothing on its own:
- No importance floor
- Still subject to normal time decay
- No forced active retrieval

The extraction LLM still gets a vote. If YMYL is enabled and the LLM tags the fact with a YMYL category, the fact gets the full strong treatment (importance floor, decay immunity, forced active retrieval), whatever the regex said.

### No Match

If no YMYL keywords are found, the fact is treated normally unless the extraction LLM tags it with a YMYL category.

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
| "watching Doctor Who" | **Weak** | health | Single weak keyword "doctor", so no floor and no decay immunity |

The "Doctor Who" case is intentionally a weak match. It gets none of the YMYL treatment (8.0 floor, decay immunity) unless the extraction LLM tags it as health, which it should not.

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
    ├── Strong match?            → importance = max(importance, 8.0)
    ├── LLM tagged a category?   → importance = max(importance, 8.0)
    └── Otherwise (weak or none) → importance unchanged
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
    ├── (Strong match or LLM tag) + decay_immune? → recency = 1.0 (no decay)
    └── Everything else             → normal decay applied
    │
    ▼
Return ranked results
```

## Limitations

1. **Keyword-based, not semantic.** "Went to the doctor" matches "doctor" (weak YMYL) even when it's a routine errand. The two-tier system mitigates this: a single weak keyword gets no special treatment unless the LLM also tags it.

2. **English-centric patterns.** The keyword lists are in English. Non-English medical or financial terms won't match. If you need multilingual YMYL, you'd need to extend the pattern dictionaries.

3. **No negation handling.** "I don't have diabetes" and "I have diabetes" classify the same way (weak health). Neither gets a floor from the regex; whatever the LLM tags applies to both. A fact about NOT having diabetes is still medically relevant, so this is acceptable.

4. **Category overlap.** Some keywords appear in multiple categories (e.g., "prescription" is in both health and pharmaceutical). The first matching category wins, based on the order in `config.categories`.

5. **Not a compliance tool.** YMYL is a best-effort prioritization heuristic. It is not HIPAA, GDPR, or any regulatory compliance mechanism. Don't use it as one.

## How the Regex and the LLM Split the Work

Classification runs in two stages inside the extraction call, so it costs no extra LLM round-trip:

1. **Regex first.** A strong pattern (or two weak keywords in one category) makes the fact YMYL. Fast, cheap and reproducible.
2. **LLM tag otherwise.** With YMYL enabled, the extraction prompt asks the LLM for a `ymyl_category` per fact. If it returns a configured category, the fact is YMYL.

The regex is the deterministic floor; the LLM catches what the keyword lists miss. The regex never promotes a single weak keyword. An LLM tag is accepted as-is, so a fact like "walked by the bank" can be YMYL on one run and not the next.
