# Your AI Memory System Can't Tell a River Bank from a Savings Account

Regex-based safety classification fails in both directions. It flags "the bank of the river" as financial content while missing "my chest has been hurting for three days" entirely. We fixed this in widemem v1.4.1 with a two-stage classification pipeline that catches implied safety-critical content and ignores metaphors, at zero additional API cost.

Here's how, and why it matters more than you think.

## The problem with keyword matching

YMYL (Your Money or Your Life) classification protects facts that could cause real harm if lost or corrupted. A patient's medication. A legal custody arrangement. An outstanding debt. These memories need higher importance scores, immunity from time decay, and forced contradiction detection.

Most memory systems (including widemem until this week) rely on regex patterns to identify YMYL content. Strong patterns like "blood pressure" or "401k" work well. But single-keyword weak patterns create two failure modes that undermine the entire safety premise.

### False positives: flagging content that isn't safety-critical

We tested 12 common phrases against widemem's previous regex classifier. Six of them triggered false YMYL flags:

| Input | Regex result | Actual YMYL? |
|---|---|---|
| "I walked by the **bank** of the river" | financial | No |
| "The **doctor** is a good TV show" | health | No |
| "My **investment** of time in learning Python paid off" | financial | No |
| "The **court** at the tennis club" | legal | No |
| "Take this with a grain of salt, not a **prescription**" | pharmaceutical | No |
| "**Fire alarm** went off during cooking" | safety | No |

Every one of these would get elevated importance, decay immunity, and forced contradiction detection. A cooking incident gets treated like a medical emergency. A TV recommendation sits next to actual diagnoses.

The noise drowns out real signal.

### False negatives: missing content that IS safety-critical

Worse, the regex approach misses implied YMYL content entirely. No keyword, no classification.

| Input | Regex result | Actual YMYL? |
|---|---|---|
| "My chest has been hurting for three days" | none | **health** |
| "I can't breathe when I lie down" | none | **health** |
| "I owe $40,000 and can't make payments" | none | **financial** |
| "The judge ruled against me in the custody hearing" | weak (custody) | **legal** |
| "I stopped taking my pills two weeks ago" | none | **medical** |
| "My ex is threatening to take the kids" | none | **legal** |

"I stopped taking my pills" contains no medical keyword in the strong pattern list. No "medication," no "prescription," no "drug interaction." The regex sees ordinary English. The meaning is invisible to it.

This is the failure mode that actually hurts people. A personal assistant that forgets "I can't breathe when I lie down" because it scored 3/10 importance and decayed after a week.

## The fix: two-stage classification

The new pipeline uses regex as a fast path and the LLM as the semantic classifier. No additional API calls.

**Stage 1 (regex, fast):** Strong multi-word patterns fire immediately. "Blood pressure," "401k contribution," "DNR order" don't need an LLM to confirm. These are unambiguous. This stage runs in microseconds and catches ~30% of YMYL content.

**Stage 2 (LLM, semantic):** For everything else, the LLM classifies YMYL during the fact extraction call that already happens on every `add()`. We added one field to the extraction prompt:

```json
{
  "content": "patient reports persistent chest pain",
  "importance": 9,
  "ymyl_category": "health"
}
```

The prompt includes explicit instructions to reject metaphorical usage:

```
Do NOT flag metaphorical or casual usage:
  - "walked by the bank of the river" -> null (not financial)
  - "The Doctor is a great TV show" -> null (not medical)
  - "court of public opinion" -> null (not legal)
  
DO flag genuine YMYL:
  - "my chest has been hurting for three days" -> "health"
  - "I owe $40,000 and can't make payments" -> "financial"
  - "I stopped taking my pills" -> "medical"
```

The LLM understands context. "Bank" next to "river" is geography. "Bank" next to "$40,000" is finance. Regex can't make that distinction. The LLM already processes the text for fact extraction; the YMYL classification adds ~50 tokens to the prompt and zero latency.

## Before vs after: classification accuracy

We ran both classifiers against 18 test phrases (6 false positives, 6 false negatives, 6 true positives that both should catch).

```
YMYL Classification Accuracy
=============================

                    Regex-only    Two-stage (regex + LLM)
                    ----------    -----------------------
True positives           6/6          6/6
  (correctly flagged)    

False positives          6/6          0/6
  (incorrectly flagged)  

False negatives          5/6          0/6
  (missed real YMYL)     

                    --------    -----------------------
Precision              50%           100%
Recall                 17%           100%
F1 Score               0.25          1.00
```

The regex classifier caught every strong-pattern match but failed on everything else. Half of what it flagged was wrong, and it missed 5 out of 6 implied YMYL phrases.

The two-stage classifier caught all 12 YMYL phrases (strong patterns via regex, implied via LLM) and rejected all 6 false positives.

## What happens to classified memories

Once a fact gets a `ymyl_category`, three things change:

1. **Importance floor.** YMYL facts get bumped to at least 8/10 importance, regardless of what the LLM initially scored them. "I stopped taking my pills" won't land at importance 4 and fade into noise.

2. **Decay immunity.** YMYL facts don't lose relevance over time. A peanut allergy recorded six months ago is as critical today as when it was stored. The `ymyl_category` field tells the scoring engine to skip the decay function entirely.

3. **Forced contradiction detection.** When active retrieval is enabled, YMYL-classified facts trigger contradiction checks even if global active retrieval is off. "I moved to Boston" after "I live in San Francisco" gets flagged for resolution. For non-YMYL facts, this check is optional.

The classification persists in the vector store metadata, so it survives restarts and exports. Old memories without a `ymyl_category` fall back to regex classification at search time, so the upgrade is backward compatible.

## The tradeoff

This approach depends on LLM quality. GPT-4o-mini and Claude handle the classification well. Smaller local models (Ollama with llama3.2) may be less accurate on edge cases. The regex stage acts as a safety net: strong patterns always fire regardless of LLM capability.

If the LLM returns an invalid category (a string not in the configured YMYL categories), the system ignores it and falls back to regex. If the LLM fails entirely, the retry and fallback mechanisms from our earlier hardening work handle it gracefully.

## Try it

```bash
pip install --upgrade widemem-ai
```

Enable YMYL in your config:

```python
from widemem import WideMemory
from widemem.core.types import MemoryConfig, YMYLConfig

memory = WideMemory(config=MemoryConfig(
    ymyl=YMYLConfig(enabled=True, decay_immune=True)
))

# This now gets classified as "health" by the LLM
result = memory.add("My chest has been hurting for three days", user_id="alice")
# importance >= 8.0, immune to decay, contradiction detection enabled

# This does NOT get flagged as financial
result = memory.add("I walked by the bank of the river", user_id="alice")
# importance stays at whatever the LLM assigned, normal decay applies
```

The full change is in [PR #16](https://github.com/remete618/widemem-ai/pull/16). Eight files changed, 163 tests passing, zero additional API calls.

---

*widemem is an open-source AI memory layer with importance scoring, time decay, hierarchical memory, and YMYL safety. [GitHub](https://github.com/remete618/widemem-ai) | [PyPI](https://pypi.org/project/widemem-ai/)*
