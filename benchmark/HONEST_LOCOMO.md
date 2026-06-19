# Honest LoCoMo evaluation

A credibility layer for widemem's LoCoMo benchmarking. It exists because the
widely-cited vendor LoCoMo numbers are not trustworthy, and widemem should not
repeat their mistakes when it publishes its own results.

## Why this exists

Published LoCoMo comparisons in the agent-memory space have three recurring
problems (documented in the mem0 assessment, June 2026):

1. **Self-grading.** The same model family answers the questions and judges
   the answers, which inflates scores. mem0's harness defaults the answerer
   and the judge to the same model.
2. **The hardest category is dropped.** LoCoMo's adversarial category (5)
   contains unanswerable questions where the correct behavior is to abstain.
   Excluding it removes the category that most exposes a memory system that
   confidently makes things up.
3. **No full-context baseline.** An independent reproduction (Zep, May 2025)
   showed a plain full-context baseline can beat the memory systems on LoCoMo.
   A result without that ceiling hides whether the memory layer adds anything.

The same vendor's Zep score and Zep's own Zep score differed by ~17 points
purely from these choices, which is the clearest evidence that a LoCoMo number
without these controls is not a real result.

## The three controls (in `benchmark/honest_core.py`)

1. **Judge must differ from the answerer.** `self_grading_check(answerer, judge)`
   returns `ok=False` and a clear message when the two models are equal or no
   judge is set. The honest run must use a distinct judge model (ideally a
   different provider).
2. **Adversarial category is included and scored as abstention.** Category 5 is
   in `CATEGORY_NAMES`. For these questions `score_question` awards credit only
   when the answer abstains (`is_correct_abstention`), so confidently answering
   an unanswerable question is scored wrong, as it should be.
3. **Full-context baseline.** `build_full_context_prompt` answers each question
   from the entire transcript with no retrieval. The honest report shows
   widemem's retrieval result next to this ceiling; if widemem cannot get close
   to it, the gap is the headline, not a footnote.

All three are pure, deterministic functions with unit tests in
`tests/test_honest_core.py`, so the methodology is verified without an API key.

## Reporting standard

When widemem publishes a LoCoMo number, state alongside it: the answerer model,
the judge model (and that it differs), whether the adversarial category was
included, the full-context baseline score, the number of judge runs averaged,
and the retrieval budget (top_k). A number without these is marked
"not apples-to-apples" and is not used in marketing.

## Running a scored pass (next step)

The scored run wires `honest_core` into the existing harness machinery
(`benchmark/mini_locomo.py` for the search/answer/judge plumbing and the
`locomo-data` dataset) and requires `OPENAI_API_KEY` plus a distinct judge
model. That runner is a follow-up; this PR lands and tests the credibility
core and the methodology so the scored run is built on a verified foundation.
