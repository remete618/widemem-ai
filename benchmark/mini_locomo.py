#!/usr/bin/env python3
"""
widemem Mini-LoCoMo Regression Gate
====================================

A 50-question stratified subset of LoCoMo, run against the current widemem
code with the current default config. Used to catch regressions during
feature development without paying for the full 1,540-question / 13-hour /
$5 run.

What it measures
----------------
- Per-category J score (single-hop, multi-hop, open-domain, temporal)
- Overall J score
- Average tokens per query
- Average search latency
- Compared against a saved baseline (if provided)

What it does NOT measure
------------------------
- Extraction quality. Memory stores are reused from the v1 ingestion run,
  so the extraction prompt is v1.3 era. The gate measures retrieval and
  scoring code paths.
- For a full extraction+retrieval benchmark, use run_locomo.py (full
  re-ingestion, ~$5, ~13 hours).

Why a subset
------------
The full LoCoMo has 1,540 questions. A run-after-every-PR would cost ~$5
and take 5+ hours each. The 50-question subset is stratified (12-13 per
category) and runs in ~3 minutes for ~$0.05, making it cheap enough to
run on every feature branch.

Gate-pass criteria
------------------
A PR fails the gate if EITHER:
- Overall J drops by more than 3.0 points vs baseline
- Multi-hop J drops by more than 5.0 points vs baseline
  (multi-hop is widemem's strongest category; tighter tolerance)

Other categories are allowed to fluctuate freely. The gate is a regression
detector, not a quality bar.

Determinism
-----------
- Question sample is deterministic: stratified, sorted, then seeded shuffle.
- Same 50 questions on every run for the same SAMPLE_SEED.
- LLM judges are non-deterministic, so we run each question 3 times and
  average. This is the J score Mem0 / Zep / others use.

Usage
-----
    cd /Users/radu/widemem-ai
    set -a; source .env.local; set +a    # load OPENAI_API_KEY
    .venv/bin/python3 benchmark/mini_locomo.py

    # Compare against an existing baseline:
    .venv/bin/python3 benchmark/mini_locomo.py \\
        --baseline benchmark/results/mini_locomo_baseline.json

    # Save current run as the new baseline (e.g., after intentional improvement):
    .venv/bin/python3 benchmark/mini_locomo.py --save-as-baseline

    # Both: compare + save:
    .venv/bin/python3 benchmark/mini_locomo.py --baseline ... --save-as-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from subprocess import check_output

import httpx
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from widemem import MemoryConfig, WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    LLMConfig,
    ScoringConfig,
    VectorStoreConfig,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_FILE = "benchmark/locomo-data/data/locomo10.json"
STORES_DIR = "benchmark/results/widemem_stores"
RESULTS_DIR = "benchmark/results"
DEFAULT_BASELINE = "benchmark/results/mini_locomo_baseline.json"

SAMPLE_SEED = 42
SAMPLE_PER_CATEGORY = {
    1: 12,  # single-hop
    2: 13,  # temporal
    3: 12,  # open-domain
    4: 13,  # multi-hop
}
TOTAL_SAMPLE = sum(SAMPLE_PER_CATEGORY.values())  # 50

JUDGE_RUNS = 5  # 5-run averaging stabilizes single-question variance below the
                # gate-pass thresholds. Was 3; raised after observing 2-question
                # judge flips producing ~8-point swings on n=13 multi-hop sample.
EVAL_LLM = "gpt-4o-mini"
TOP_K = 10
API_TIMEOUT = 30
MAX_RETRIES = 3

# Gate criteria (points of J score, 0-100 scale).
# Calibrated empirically: with JUDGE_RUNS=5 and n=50 stratified, observed
# run-to-run variance on identical code stays within these tolerances.
GATE_OVERALL_MAX_DROP = 3.0
GATE_MULTIHOP_MAX_DROP = 5.0

CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "open-domain", 4: "multi-hop"}

ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories that contain information related to the question
2. Pay special attention to timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp
6. Always convert relative time references to specific dates, months, or years
7. Focus only on the content of the memories from both speakers
8. The answer should be less than 5-6 words.

Memories for speaker {speaker_a}:
{memories_a}

Memories for speaker {speaker_b}:
{memories_b}

Question: {question}

Answer:"""

TEMPORAL_ANSWER_PROMPT = """You are an intelligent memory assistant. Answer the temporal question below using ONLY the information in the provided memories.

CRITICAL RULES FOR TEMPORAL QUESTIONS:
1. Look for explicit dates, months, and years mentioned in the memories
2. If a memory mentions a relative time (e.g., "yesterday", "last week"), and includes a date context, calculate the actual date
3. Your answer MUST include a specific date, month, or year, NOT vague references like "yesterday" or "recently"
4. If you cannot determine a specific date from the memories, give your best estimate based on available context
5. The answer should be less than 5-6 words

Memories for speaker {speaker_a}:
{memories_a}

Memories for speaker {speaker_b}:
{memories_b}

Question: {question}

Answer (include specific date/month/year):"""

JUDGE_PROMPT = """Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading; as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading; as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_git_sha() -> str:
    try:
        return check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_widemem_version() -> str:
    try:
        import widemem
        return widemem.__version__
    except Exception:
        return "unknown"


def api_call_with_retry(client, model, messages, temperature=0.0, max_tokens=100):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=API_TIMEOUT,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err:
                wait = min(15 * (attempt + 1), 60)
                print(f"  429 rate limit, wait {wait}s (try {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"  error: {err[:80]} (try {attempt + 1}/{MAX_RETRIES})")
                time.sleep(3)
    return None


def judge_one(question, gold, predicted, client):
    prompt = JUDGE_PROMPT.format(
        question=str(question),
        gold_answer=str(gold),
        generated_answer=str(predicted),
    )
    text = api_call_with_retry(
        client,
        EVAL_LLM,
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
    )
    if text is None:
        return None
    if "{" in text:
        try:
            parsed = json.loads(text[text.index("{") : text.rindex("}") + 1])
            return parsed.get("label", "").upper() == "CORRECT"
        except (json.JSONDecodeError, ValueError):
            pass
    return "CORRECT" in text.upper() and "WRONG" not in text.upper()


def stratified_sample(all_questions, seed=SAMPLE_SEED):
    """Deterministic stratified sample.

    Sort each category by (sample_id, question) for stable ordering across
    runs, then seed-shuffle and take the first N per category.

    Returns the same 50 questions on every run for the same seed.
    """
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for q in all_questions:
        cat = q.get("category")
        if cat in CATEGORY_NAMES:
            by_cat[cat].append(q)

    sample = []
    for cat, count in SAMPLE_PER_CATEGORY.items():
        bucket = sorted(by_cat[cat], key=lambda q: (q["_sample_id"], q["question"]))
        rng.shuffle(bucket)
        sample.extend(bucket[:count])
    return sample


def load_memory_stores(data):
    """Open the v1-ingested stores read-only with current widemem code."""
    instances = {}
    for i in range(len(data)):
        storage_dir = os.path.join(STORES_DIR, f"conv_{i}")
        if not os.path.exists(storage_dir):
            raise RuntimeError(f"Store missing: {storage_dir}")
        config = MemoryConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.0),
            embedding=EmbeddingConfig(
                provider="openai", model="text-embedding-3-small"
            ),
            vector_store=VectorStoreConfig(
                provider="faiss", path=os.path.join(storage_dir, "faiss")
            ),
            scoring=ScoringConfig(
                decay_function="exponential",
                decay_rate=0.01,
                similarity_weight=0.5,
                importance_weight=0.3,
                recency_weight=0.2,
            ),
            history_db_path=os.path.join(storage_dir, "history.db"),
            enable_hierarchy=True,
        )
        instances[i] = WideMemory(config=config)
    return instances


def run_question(q_data, mem_instances, client):
    """Run a single question end to end: search + answer + judge x 3."""
    conv_idx = q_data["_conv_idx"]
    sample_id = q_data["_sample_id"]
    speaker_a = q_data["_speaker_a"]
    speaker_b = q_data["_speaker_b"]
    question = q_data["question"]
    gold = q_data["answer"]
    category = q_data["category"]

    mem = mem_instances[conv_idx]

    t0 = time.time()
    results_a = mem.search(
        query=question, user_id=f"{sample_id}_{speaker_a}", top_k=TOP_K
    )
    results_b = mem.search(
        query=question, user_id=f"{sample_id}_{speaker_b}", top_k=TOP_K
    )
    search_time = time.time() - t0

    memories_a = "\n".join(
        f"[importance={r.memory.importance:.1f}] {r.memory.content}" for r in results_a
    )
    memories_b = "\n".join(
        f"[importance={r.memory.importance:.1f}] {r.memory.content}" for r in results_b
    )
    memory_tokens = sum(
        len(r.memory.content.split()) for r in list(results_a) + list(results_b)
    )

    prompt_template = TEMPORAL_ANSWER_PROMPT if category == 2 else ANSWER_PROMPT
    prompt = prompt_template.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        memories_a=memories_a or "(no memories)",
        memories_b=memories_b or "(no memories)",
        question=question,
    )

    t1 = time.time()
    answer = api_call_with_retry(
        client, EVAL_LLM, [{"role": "user", "content": prompt}]
    )
    gen_time = time.time() - t1

    if answer is None:
        answer = "ERROR: API failed"

    correct = 0
    judge_fails = 0
    for _ in range(JUDGE_RUNS):
        result = judge_one(question, gold, answer, client)
        if result is None:
            judge_fails += 1
        elif result:
            correct += 1

    valid = JUDGE_RUNS - judge_fails
    j_score = correct / valid if valid > 0 else 0.0

    return {
        "sample_id": sample_id,
        "question": question,
        "gold": gold,
        "predicted": answer,
        "category": category,
        "category_name": CATEGORY_NAMES.get(category, "unknown"),
        "memory_tokens": memory_tokens,
        "search_latency": round(search_time, 4),
        "generation_latency": round(gen_time, 4),
        "j_score": j_score,
        "judge_valid_runs": valid,
    }


def summarize(predictions):
    by_cat = defaultdict(list)
    for p in predictions:
        by_cat[p["category_name"]].append(p)

    cat_j = {}
    cat_tokens = {}
    for cat in CATEGORY_NAMES.values():
        preds = by_cat.get(cat, [])
        cat_j[cat] = (
            sum(p["j_score"] for p in preds) / len(preds) * 100 if preds else 0.0
        )
        cat_tokens[cat] = (
            sum(p["memory_tokens"] for p in preds) / len(preds) if preds else 0
        )

    overall_j = (
        sum(p["j_score"] for p in predictions) / len(predictions) * 100
        if predictions
        else 0.0
    )
    avg_tokens = (
        sum(p["memory_tokens"] for p in predictions) / len(predictions)
        if predictions
        else 0
    )
    avg_search = (
        sum(p["search_latency"] for p in predictions) / len(predictions)
        if predictions
        else 0.0
    )

    return {
        "overall_j": round(overall_j, 2),
        "by_category": {cat: round(j, 2) for cat, j in cat_j.items()},
        "avg_memory_tokens": round(avg_tokens, 1),
        "avg_search_latency": round(avg_search, 4),
        "n_questions": len(predictions),
    }


def gate_check(current, baseline):
    """Return (passed, lines) describing pass/fail vs the baseline."""
    overall_delta = current["overall_j"] - baseline["overall_j"]
    multihop_delta = current["by_category"]["multi-hop"] - baseline["by_category"][
        "multi-hop"
    ]

    overall_ok = overall_delta >= -GATE_OVERALL_MAX_DROP
    multihop_ok = multihop_delta >= -GATE_MULTIHOP_MAX_DROP

    lines = [
        f"  overall:   {baseline['overall_j']:>6.2f} -> {current['overall_j']:>6.2f}  "
        f"({overall_delta:+.2f}) "
        f"[{'PASS' if overall_ok else 'FAIL'}, max drop {GATE_OVERALL_MAX_DROP}]",
        f"  multi-hop: {baseline['by_category']['multi-hop']:>6.2f} -> "
        f"{current['by_category']['multi-hop']:>6.2f}  ({multihop_delta:+.2f}) "
        f"[{'PASS' if multihop_ok else 'FAIL'}, max drop {GATE_MULTIHOP_MAX_DROP}]",
    ]
    for cat in ["single-hop", "open-domain", "temporal"]:
        delta = current["by_category"][cat] - baseline["by_category"][cat]
        lines.append(
            f"  {cat + ':':<11}{baseline['by_category'][cat]:>6.2f} -> "
            f"{current['by_category'][cat]:>6.2f}  ({delta:+.2f}) "
            "[informational, no gate]"
        )

    return overall_ok and multihop_ok, lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="widemem mini-LoCoMo regression gate")
    ap.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="path to baseline JSON to compare against (gate pass/fail)",
    )
    ap.add_argument(
        "--save-as-baseline",
        action="store_true",
        help=f"also write this run to {DEFAULT_BASELINE}",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="explicit output path; defaults to mini_locomo_<sha>_<timestamp>.json",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. `set -a; source .env.local; set +a` first.")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("widemem mini-LoCoMo gate")
    print(f"  widemem version: {get_widemem_version()}")
    print(f"  git sha:         {get_git_sha()}")
    print(f"  sample size:     {TOTAL_SAMPLE} questions (stratified)")
    print(f"  seed:            {SAMPLE_SEED}")

    with open(DATA_FILE) as f:
        data = json.load(f)

    mem_instances = load_memory_stores(data)
    print(f"  stores loaded:   {len(mem_instances)} conversations")

    all_questions = []
    for i, conv in enumerate(data):
        sample_id = conv["sample_id"]
        speaker_a = conv["conversation"]["speaker_a"]
        speaker_b = conv["conversation"]["speaker_b"]
        for q in conv["qa"]:
            if q["category"] == 5:
                continue
            q_copy = dict(q)
            q_copy["_conv_idx"] = i
            q_copy["_sample_id"] = sample_id
            q_copy["_speaker_a"] = speaker_a
            q_copy["_speaker_b"] = speaker_b
            all_questions.append(q_copy)

    sample = stratified_sample(all_questions)
    by_cat_count = defaultdict(int)
    for q in sample:
        by_cat_count[CATEGORY_NAMES[q["category"]]] += 1
    print(f"  question mix:    {dict(by_cat_count)}")

    http_client = httpx.Client(timeout=httpx.Timeout(API_TIMEOUT, connect=10))
    client = OpenAI(http_client=http_client)

    test = api_call_with_retry(
        client, EVAL_LLM, [{"role": "user", "content": "Say OK"}], max_tokens=5
    )
    if test is None:
        print("ERROR: API test failed")
        sys.exit(1)
    print(f"  api test:        {test}")
    print()

    start = time.time()
    predictions = []
    for i, q in enumerate(sample, 1):
        pred = run_question(q, mem_instances, client)
        predictions.append(pred)
        running_j = sum(p["j_score"] for p in predictions) / len(predictions) * 100
        print(
            f"  [{i:>2}/{TOTAL_SAMPLE}] {pred['category_name']:<11} "
            f"J={pred['j_score']:.2f}  running={running_j:.1f}%"
        )

    elapsed = time.time() - start
    summary = summarize(predictions)

    print()
    print("=" * 70)
    print(f"RESULTS  (elapsed {elapsed:.0f}s)")
    print("=" * 70)
    print(f"  overall J:        {summary['overall_j']:.2f}")
    for cat in ["single-hop", "multi-hop", "open-domain", "temporal"]:
        print(f"  {cat + ' J:':<18}{summary['by_category'][cat]:.2f}")
    print(f"  avg tokens:       {summary['avg_memory_tokens']:.1f}")
    print(f"  avg search lat:   {summary['avg_search_latency']:.4f}s")
    print()

    output = {
        "metadata": {
            "widemem_version": get_widemem_version(),
            "git_sha": get_git_sha(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_size": TOTAL_SAMPLE,
            "sample_seed": SAMPLE_SEED,
            "judge_runs": JUDGE_RUNS,
            "top_k_per_speaker": TOP_K,
            "eval_llm": EVAL_LLM,
            "elapsed_seconds": round(elapsed, 1),
        },
        "summary": summary,
        "predictions": predictions,
    }

    if args.output:
        out_path = args.output
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = os.path.join(
            RESULTS_DIR, f"mini_locomo_{get_git_sha()}_{ts}.json"
        )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  saved to: {out_path}")

    if args.save_as_baseline:
        with open(DEFAULT_BASELINE, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  baseline updated: {DEFAULT_BASELINE}")

    baseline_path = args.baseline
    if baseline_path is None and os.path.exists(DEFAULT_BASELINE) and not args.save_as_baseline:
        baseline_path = DEFAULT_BASELINE

    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        passed, lines = gate_check(summary, baseline_data["summary"])
        print()
        print("=" * 70)
        print(f"GATE  (baseline: {baseline_path})")
        print("=" * 70)
        for line in lines:
            print(line)
        print()
        if passed:
            print("  GATE PASS")
            sys.exit(0)
        else:
            print("  GATE FAIL")
            sys.exit(1)
    elif baseline_path:
        print(f"  WARNING: baseline not found at {baseline_path}")

    for mem in mem_instances.values():
        mem.close()


if __name__ == "__main__":
    main()
