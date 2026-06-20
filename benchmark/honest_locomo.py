#!/usr/bin/env python3
"""Honest LoCoMo scored runner.

Wires the credibility controls in ``honest_core`` into the LoCoMo dataset and
widemem's retrieval, reusing the plumbing from ``mini_locomo`` (data shape,
prompts, API retry, store loading). Unlike the regression gate, this run:

- uses a judge model DISTINCT from the answerer (no self-grading),
- INCLUDES the adversarial category (5), scored as abstention,
- can report a full-context baseline ceiling alongside the retrieval result.

Requires ``OPENAI_API_KEY``. The pure logic (model guard, question building,
transcript flattening, judge parsing) lives in tested functions here and in
``honest_core``; only ``main`` touches the network. See HONEST_LOCOMO.md.

Usage:
    python benchmark/honest_locomo.py --judge-model gpt-4o --full-context
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import honest_core as hc  # noqa: E402
import mini_locomo as ml  # noqa: E402

DEFAULT_ANSWERER = "gpt-4o-mini"
DEFAULT_JUDGE = "gpt-4o"  # distinct from the answerer by default
DEFAULT_PER_CATEGORY = {1: 13, 2: 13, 3: 12, 4: 12, 5: 10}  # includes adversarial
SESSION_RE = re.compile(r"^session_\d+$")


# --------------------------------------------------------------------------
# Pure, network-free helpers (unit-tested in tests/test_honest_runner.py)
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Honest LoCoMo scored runner")
    ap.add_argument("--answerer-model", default=DEFAULT_ANSWERER)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE,
                    help="MUST differ from the answerer; default is distinct")
    ap.add_argument("--no-adversarial", action="store_true",
                    help="exclude category 5 (NOT recommended; matches the vendor benchmarks)")
    ap.add_argument("--full-context", action="store_true",
                    help="also run the full-context baseline ceiling")
    ap.add_argument("--top-k", type=int, default=ml.TOP_K)
    ap.add_argument("--judge-runs", type=int, default=ml.JUDGE_RUNS)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap questions per category (0 = full stratified sample)")
    ap.add_argument("--data", default=ml.DATA_FILE)
    ap.add_argument("--output", default=None)
    return ap.parse_args(argv)


def validate_models(answerer_model: str, judge_model: str) -> bool:
    """Enforce judge != answerer. Prints the verdict; exits(2) on self-grading."""
    ok, msg = hc.self_grading_check(answerer_model, judge_model)
    print(("OK: " if ok else "REFUSING: ") + msg)
    if not ok:
        raise SystemExit(2)
    return True


def conversation_to_text(conv: dict) -> str:
    """Flatten a LoCoMo conversation into a plain transcript for the
    full-context baseline."""
    convo = conv.get("conversation", {})
    lines = []
    for key in sorted(k for k in convo if SESSION_RE.match(k)):
        turns = convo.get(key) or []
        for turn in turns:
            speaker = turn.get("speaker", "?")
            text = turn.get("text", "")
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def build_questions(data: list, include_adversarial: bool = True) -> list:
    """Build the question set from LoCoMo data, INCLUDING category 5 by default
    (the regression gate drops it; an honest run must not)."""
    out = []
    for i, conv in enumerate(data):
        sample_id = conv["sample_id"]
        speaker_a = conv["conversation"]["speaker_a"]
        speaker_b = conv["conversation"]["speaker_b"]
        for q in conv.get("qa", []):
            cat = q.get("category")
            if cat not in hc.CATEGORY_NAMES:
                continue
            if cat == hc.ADVERSARIAL_CATEGORY and not include_adversarial:
                continue
            qc = dict(q)
            qc["_conv_idx"] = i
            qc["_sample_id"] = sample_id
            qc["_speaker_a"] = speaker_a
            qc["_speaker_b"] = speaker_b
            out.append(qc)
    return out


def pick_answer_prompt(category: int) -> str:
    return ml.TEMPORAL_ANSWER_PROMPT if category == 2 else ml.ANSWER_PROMPT


def parse_judge_label(text: str | None) -> bool | None:
    """Parse a judge response to True/False/None. Mirrors mini_locomo.judge_one,
    extracted so it is testable without the network."""
    if text is None:
        return None
    if "{" in text:
        try:
            parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
            return parsed.get("label", "").upper() == "CORRECT"
        except (json.JSONDecodeError, ValueError):
            pass
    upper = text.upper()
    return "CORRECT" in upper and "WRONG" not in upper


def per_category_counts(include_adversarial: bool, limit: int) -> dict[int, int]:
    counts = dict(DEFAULT_PER_CATEGORY)
    if not include_adversarial:
        counts.pop(hc.ADVERSARIAL_CATEGORY, None)
    if limit and limit > 0:
        counts = {c: min(limit, n) for c, n in counts.items()}
    return counts


# --------------------------------------------------------------------------
# Network code (only reached from main; requires OPENAI_API_KEY)
# --------------------------------------------------------------------------

def _answer_from_memories(q, mem_instances, client, answerer_model, top_k):
    conv_idx = q["_conv_idx"]
    mem = mem_instances[conv_idx]
    results_a = mem.search(query=q["question"], user_id=f"{q['_sample_id']}_{q['_speaker_a']}", top_k=top_k)
    results_b = mem.search(query=q["question"], user_id=f"{q['_sample_id']}_{q['_speaker_b']}", top_k=top_k)
    memories_a = "\n".join(f"[importance={r.memory.importance:.1f}] {r.memory.content}" for r in results_a)
    memories_b = "\n".join(f"[importance={r.memory.importance:.1f}] {r.memory.content}" for r in results_b)
    tokens = sum(len(r.memory.content.split()) for r in list(results_a) + list(results_b))
    prompt = pick_answer_prompt(q["category"]).format(
        speaker_a=q["_speaker_a"], speaker_b=q["_speaker_b"],
        memories_a=memories_a or "(no memories)", memories_b=memories_b or "(no memories)",
        question=q["question"],
    )
    answer = ml.api_call_with_retry(client, answerer_model, [{"role": "user", "content": prompt}])
    return (answer or "ERROR: API failed"), tokens


def _answer_full_context(conv_text, question, client, answerer_model):
    prompt = hc.build_full_context_prompt(conv_text, question)
    return ml.api_call_with_retry(client, answerer_model, [{"role": "user", "content": prompt}]) or "ERROR"


def _judge(question, gold, predicted, client, judge_model, runs):
    correct = valid = 0
    for _ in range(runs):
        prompt = ml.JUDGE_PROMPT.format(question=str(question), gold_answer=str(gold), generated_answer=str(predicted))
        verdict = parse_judge_label(
            ml.api_call_with_retry(client, judge_model, [{"role": "user", "content": prompt}],
                                   temperature=0.1, max_tokens=200)
        )
        if verdict is None:
            continue
        valid += 1
        correct += 1 if verdict else 0
    return (correct / valid) if valid else 0.0


def main():
    args = parse_args()
    # Self-grading guard runs FIRST, before the key check, so the refusal is
    # visible even without credentials.
    validate_models(args.answerer_model, args.judge_model)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        raise SystemExit(1)

    import httpx
    from openai import OpenAI

    with open(args.data) as f:
        data = json.load(f)

    include_adv = not args.no_adversarial
    questions = build_questions(data, include_adversarial=include_adv)
    counts = per_category_counts(include_adv, args.limit)
    sample = hc.stratified_sample(questions, counts, seed=ml.SAMPLE_SEED)

    mem_instances = ml.load_memory_stores(data)
    client = OpenAI(http_client=httpx.Client(timeout=httpx.Timeout(60, connect=10)))

    print(f"Honest LoCoMo run: answerer={args.answerer_model} judge={args.judge_model} "
          f"adversarial={'yes' if include_adv else 'no'} full_context={args.full_context} n={len(sample)}")

    retrieval_preds, baseline_preds = [], []
    for i, q in enumerate(sample, 1):
        predicted, _ = _answer_from_memories(q, mem_instances, client, args.answerer_model, args.top_k)
        judged = bool(_judge(q["question"], q["answer"], predicted, client, args.judge_model, args.judge_runs) >= 0.5)
        retrieval_preds.append({"category": q["category"],
                                "score": hc.score_question(q["category"], predicted, judged)})
        if args.full_context:
            conv_text = conversation_to_text(data[q["_conv_idx"]])
            fc = _answer_full_context(conv_text, q["question"], client, args.answerer_model)
            fc_judged = bool(_judge(q["question"], q["answer"], fc, client, args.judge_model, args.judge_runs) >= 0.5)
            baseline_preds.append({"category": q["category"],
                                   "score": hc.score_question(q["category"], fc, fc_judged)})
        print(f"  [{i}/{len(sample)}] {hc.CATEGORY_NAMES[q['category']]}")

    retrieval = hc.aggregate(retrieval_preds)
    report = {
        "answerer_model": args.answerer_model,
        "judge_model": args.judge_model,
        "judge_differs": True,
        "adversarial_included": include_adv,
        "judge_runs": args.judge_runs,
        "top_k": args.top_k,
        "retrieval": retrieval,
    }
    print(f"\nRetrieval J: {retrieval['overall_j']}  by category: {retrieval['by_category']}")
    if args.full_context:
        baseline = hc.aggregate(baseline_preds)
        report["full_context_baseline"] = baseline
        print(f"Full-context baseline J: {baseline['overall_j']}  by category: {baseline['by_category']}")
        print(f"widemem retrieval vs full-context ceiling: {retrieval['overall_j']} vs {baseline['overall_j']}")

    out_path = args.output or os.path.join(ml.RESULTS_DIR, f"honest_locomo_{ml.get_git_sha()}.json")
    os.makedirs(ml.RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
