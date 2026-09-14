#!/usr/bin/env python3
"""
Widemem LoCoMo WS1 -- structured event-time, Chunked, Resumable, Rate-Safe
=========================================================================
WS1 experiment runner. Re-ingests fresh stores with the [session_ts]
prefix so PR-B's parse_leading_datetime captures Memory.event_time, then
runs Q&A + Judge with the answer prompt surfacing each retrieved memory's
event date plus the conversation's most-recent-session anchor date.

The ONLY differences vs the validated clean baseline:
- stores re-ingested so event_time is populated (PR-A/PR-B, no code change
  to ingestion: the [session_ts] prefix is parsed automatically)
- answer prompt prefixes each memory with [event: DATE] and injects the
  conversation reference (anchor) date for relative-time resolution
Retrieval config is identical to the validated baseline (hierarchy on,
default scoring, no v1.6 flags), so non-temporal categories should stay
within judge noise of baseline (attribution guard) and temporal should
move. JUDGE_RUNS=3, single pass, no repair.

Config:
- enable_hybrid_search OFF, parse_temporal_hints OFF (matches baseline)
- JUDGE_RUNS=3 (scoped; rate-limit reality made 5 untenable)
- top_k=10 per speaker (Mem0 paper baseline)
- NO eval_repair.py pass
- Output: ws1_chunks/, locomo_ws1_final.json; stores in widemem_stores_ws1/

- Splits into chunks of 50, saves after each
- 30s timeout per API call, 5 retries with backoff
- Auto-pauses on consecutive failures
- Fully resumable: restart to continue from last chunk
- Merges all chunks at the end

Budget: ~6,160 API calls (fits in 10,000 RPD daily limit)
Time: ~5 hours at 20 q/min, ~$2 total

Usage:
    set -a; source .env.local; set +a    # load OPENAI_API_KEY
    cd widemem-ai
    .venv/bin/python3 benchmark/run_v3.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from widemem import WideMemory, MemoryConfig
from widemem.core.types import LLMConfig, EmbeddingConfig, VectorStoreConfig, ScoringConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_FILE = "benchmark/locomo-data/data/locomo10.json"
STORES_DIR = "benchmark/results/widemem_stores_ws1"
CHUNKS_DIR = "benchmark/results/ws1_chunks"
FINAL_OUTPUT = "benchmark/results/locomo_ws1_final.json"
CHUNK_SIZE = 50
JUDGE_RUNS = 3
EVAL_LLM = "gpt-4o-mini"
JUDGE_LLM = os.environ.get("WM_JUDGE_MODEL", EVAL_LLM)
API_TIMEOUT = 30
MAX_RETRIES = 5
MAX_CONSECUTIVE_API_FAILURES = 15  # All judges failing in a row = real API problem
PER_QUESTION_TIMEOUT_SEC = 180     # 3 min max per question (4 calls @ 30s + buffer)
MAX_TOTAL_COST_USD = 5.00          # Hard ceiling. Override with --max-cost.
MAX_WALL_CLOCK_HOURS = 8.0         # Override with --max-hours.
TOP_K = 10  # Matches Mem0 paper baseline.

# gpt-4o-mini pricing as of 2026-05; used to estimate spend from usage fields
GPT_4O_MINI_INPUT_PER_1M = 0.15   # USD per 1M input tokens
GPT_4O_MINI_OUTPUT_PER_1M = 0.60  # USD per 1M output tokens

# Module-level cost tracker. Updated by api_call_with_retry on every response.
_TOTAL_INPUT_TOKENS = 0
_TOTAL_OUTPUT_TOKENS = 0


def total_cost_usd() -> float:
    return (
        _TOTAL_INPUT_TOKENS / 1_000_000 * GPT_4O_MINI_INPUT_PER_1M
        + _TOTAL_OUTPUT_TOKENS / 1_000_000 * GPT_4O_MINI_OUTPUT_PER_1M
    )

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}

# conv_idx -> most-recent session date string (the answer-time anchor). Populated in main().
CONV_REFERENCE_DATE: dict = {}

# Control mode: same (re-ingested) stores, baseline prompt, no event/anchor.
NO_EVENT_SURFACE = False
ENTITY_BOOST_W = 0.0

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
8. If the question asks "how many" or asks for kinds/types/lists of things, first find EVERY matching memory, then answer with the complete count or the complete list of items. Do not stop at the first match.
9. Otherwise, the answer should be less than 5-6 words.

# REFERENCE DATE:
Each memory below is prefixed with [event: DATE], the date the statement was made. The most recent conversation date is {reference_date}. Resolve relative expressions ("last week", "N years ago", "the week before") against the relevant memory's event date and this reference date.

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
3. Your answer MUST include a specific date, month, or year — NOT vague references like "yesterday" or "recently"
4. If you cannot determine a specific date from the memories, give your best estimate based on available context
5. If the question asks "how many" or asks for kinds/types/lists of things, answer with the complete count or the complete list of items. Do not stop at the first match.
6. Otherwise, the answer should be less than 5-6 words

# REFERENCE DATE:
Each memory below is prefixed with [event: DATE], the date the statement was made. The most recent conversation date is {reference_date}. Compute the answer by anchoring relative expressions to the relevant memory's event date and this reference date (e.g. "the week before {reference_date}", "N years before the event date").

Memories for speaker {speaker_a}:
{memories_a}

Memories for speaker {speaker_b}:
{memories_b}

Question: {question}

Answer (include specific date/month/year):"""

# Exact validated-baseline prompts (no event-time / no anchor). Used by
# --no-event-surface to isolate the prompt effect from re-ingestion drift.
BASE_ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

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
8. If the question asks "how many" or asks for kinds/types/lists of things, first find EVERY matching memory, then answer with the complete count or the complete list of items. Do not stop at the first match.
9. Otherwise, the answer should be less than 5-6 words.

Memories for speaker {speaker_a}:
{memories_a}

Memories for speaker {speaker_b}:
{memories_b}

Question: {question}

Answer:"""

BASE_TEMPORAL_PROMPT = """You are an intelligent memory assistant. Answer the temporal question below using ONLY the information in the provided memories.

CRITICAL RULES FOR TEMPORAL QUESTIONS:
1. Look for explicit dates, months, and years mentioned in the memories
2. If a memory mentions a relative time (e.g., "yesterday", "last week"), and includes a date context, calculate the actual date
3. Your answer MUST include a specific date, month, or year, NOT vague references like "yesterday" or "recently"
4. If you cannot determine a specific date from the memories, give your best estimate based on available context
5. If the question asks "how many" or asks for kinds/types/lists of things, answer with the complete count or the complete list of items. Do not stop at the first match.
6. Otherwise, the answer should be less than 5-6 words

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
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


def api_call_with_retry(client, model, messages, temperature=0.0, max_tokens=100):
    """Make an API call with timeout and retry logic.

    Tracks usage on every successful response and updates the module-level
    token counters so cost can be tallied at any time via total_cost_usd().
    """
    global _TOTAL_INPUT_TOKENS, _TOTAL_OUTPUT_TOKENS
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens, timeout=API_TIMEOUT,
            )
            # OpenAI returns usage on chat.completions; track it.
            if response.usage is not None:
                _TOTAL_INPUT_TOKENS += response.usage.prompt_tokens or 0
                _TOTAL_OUTPUT_TOKENS += response.usage.completion_tokens or 0
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err:
                wait = min(20 * (attempt + 1), 120)
                print(f"      429, wait {wait}s ({attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            elif "timeout" in err.lower():
                print(f"      timeout ({attempt+1}/{MAX_RETRIES})")
                time.sleep(5)
            else:
                print(f"      error: {err[:80]} ({attempt+1}/{MAX_RETRIES})")
                time.sleep(5)
    return None


def judge_one(question, gold, predicted, client):
    prompt = JUDGE_PROMPT.format(
        question=str(question), gold_answer=str(gold), generated_answer=str(predicted),
    )
    text = api_call_with_retry(client, JUDGE_LLM, [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=200)
    if text is None:
        return None
    if "{" in text:
        try:
            parsed = json.loads(text[text.index("{"):text.rindex("}") + 1])
            return parsed.get("label", "").upper() == "CORRECT"
        except (json.JSONDecodeError, ValueError):
            pass
    return "CORRECT" in text.upper() and "WRONG" not in text.upper()


def compute_f1(prediction, gold):
    pred_tokens = str(prediction).lower().split()
    gold_tokens = str(gold).lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)


def compute_bleu1(prediction, gold):
    pred_tokens = str(prediction).lower().split()
    gold_tokens = str(gold).lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    return sum(1 for t in pred_tokens if t in gold_tokens) / len(pred_tokens)


def load_completed():
    """Load already-completed predictions from chunks."""
    done = {}
    if not os.path.exists(CHUNKS_DIR):
        return done
    for f in sorted(os.listdir(CHUNKS_DIR)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(CHUNKS_DIR, f)) as fh:
            chunk = json.load(fh)
        for p in chunk["predictions"]:
            key = f"{p['sample_id']}_{p['question'][:80]}"
            done[key] = p
    return done


def process_chunk(chunk_idx, questions, mem_instances, client):
    """Process a chunk: answer + judge each question."""
    chunk_path = os.path.join(CHUNKS_DIR, f"chunk_{chunk_idx:04d}.json")
    start = time.time()
    predictions = []
    consecutive_fail = 0

    for i, q_data in enumerate(questions):
        conv_idx = q_data["_conv_idx"]
        sample_id = q_data["_sample_id"]
        speaker_a = q_data["_speaker_a"]
        speaker_b = q_data["_speaker_b"]
        question = q_data["question"]
        gold = q_data["answer"]
        category = q_data["category"]

        mem = mem_instances[conv_idx]

        # Phase 2: Search + Answer
        t0 = time.time()
        results_a = mem.search(query=question, user_id=f"{sample_id}_{speaker_a}", top_k=TOP_K)
        results_b = mem.search(query=question, user_id=f"{sample_id}_{speaker_b}", top_k=TOP_K)
        search_time = time.time() - t0

        if NO_EVENT_SURFACE:
            def _fmt(r):
                return f"[importance={r.memory.importance:.1f}] {r.memory.content}"
        else:
            def _fmt(r):
                et = r.memory.event_time
                d = et.strftime("%d %B %Y") if et else "date unknown"
                return f"[event: {d}] {r.memory.content}"
        memories_a = "\n".join(_fmt(r) for r in results_a)
        memories_b = "\n".join(_fmt(r) for r in results_b)
        memory_tokens = sum(
            len(r.memory.content.split())
            for r in list(results_a) + list(results_b)
        )

        # Use temporal-specific prompt for temporal questions (category 2)
        if NO_EVENT_SURFACE:
            prompt_template = BASE_TEMPORAL_PROMPT if category == 2 else BASE_ANSWER_PROMPT
            prompt = prompt_template.format(
                speaker_a=speaker_a, speaker_b=speaker_b,
                memories_a=memories_a or "(no memories)", memories_b=memories_b or "(no memories)",
                question=question,
            )
        else:
            prompt_template = TEMPORAL_ANSWER_PROMPT if category == 2 else ANSWER_PROMPT
            prompt = prompt_template.format(
                speaker_a=speaker_a, speaker_b=speaker_b,
                memories_a=memories_a or "(no memories)", memories_b=memories_b or "(no memories)",
                question=question,
                reference_date=CONV_REFERENCE_DATE.get(conv_idx, "unknown"),
            )

        t1 = time.time()
        answer = api_call_with_retry(client, EVAL_LLM, [{"role": "user", "content": prompt}])
        gen_time = time.time() - t1

        if answer is None:
            answer = "ERROR: API failed"

        # Phase 3: Judge
        correct = 0
        api_failures = 0
        for r in range(JUDGE_RUNS):
            result = judge_one(question, gold, answer, client)
            if result is None:
                api_failures += 1
            elif result:
                correct += 1

        valid_runs = JUDGE_RUNS - api_failures
        j_score = correct / valid_runs if valid_runs > 0 else 0

        # Stuck-detection: ONLY count toward consecutive_fail when ALL judges
        # returned None (true API failure). A streak of legitimately-wrong
        # answers (J=0 with valid runs) is normal benchmark behavior and must
        # not abort the chunk. This was the bug that aborted v3 chunk 1 early.
        if api_failures == JUDGE_RUNS:
            consecutive_fail += 1
        else:
            consecutive_fail = 0

        all_results = list(results_a) + list(results_b)
        avg_imp = sum(r.memory.importance for r in all_results) / len(all_results) if all_results else 0

        pred = {
            "sample_id": sample_id, "question": question, "gold": gold,
            "predicted": answer, "category": category,
            "category_name": CATEGORY_NAMES.get(category, "unknown"),
            "memory_tokens_used": memory_tokens, "memories_retrieved": len(all_results),
            "avg_importance": round(avg_imp, 2),
            "search_latency": round(search_time, 4),
            "generation_latency": round(gen_time, 4),
            "total_latency": round(search_time + gen_time, 4),
            "f1": compute_f1(answer, gold), "bleu1": compute_bleu1(answer, gold),
            "j_score": j_score, "j_runs": valid_runs,
        }
        predictions.append(pred)

        # Progress every 10
        if (i + 1) % 10 == 0 or i == len(questions) - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed * 60
            j_avg = sum(p["j_score"] for p in predictions) / len(predictions) * 100
            print(f"    [{i+1}/{len(questions)}] J={j_avg:.1f}% | {rate:.0f} q/min | fail_streak={consecutive_fail}")

        if consecutive_fail >= MAX_CONSECUTIVE_API_FAILURES:
            print(
                f"    !! {MAX_CONSECUTIVE_API_FAILURES} consecutive ALL-judge API failures, "
                "stopping chunk (true API issue, not benchmark difficulty)"
            )
            break

    # Save chunk
    elapsed = time.time() - start
    chunk_data = {
        "chunk_idx": chunk_idx, "count": len(predictions),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions,
    }
    with open(chunk_path, "w") as f:
        json.dump(chunk_data, f, indent=2, default=str)

    j_avg = sum(p["j_score"] for p in predictions) / len(predictions) * 100 if predictions else 0
    print(f"    Chunk {chunk_idx} saved: {len(predictions)} preds, J={j_avg:.1f}%, {elapsed:.0f}s")

    return consecutive_fail < MAX_CONSECUTIVE_API_FAILURES


def merge_and_report():
    """Merge all chunks and print final results."""
    all_preds = []
    for f in sorted(os.listdir(CHUNKS_DIR)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(CHUNKS_DIR, f)) as fh:
            chunk = json.load(fh)
        all_preds.extend(chunk["predictions"])

    print(f"\n{'='*85}")
    print(f"FINAL RESULTS — WIDEMEM v2 (adaptive scoring, top_k={TOP_K})")
    print(f"{'='*85}")
    print(f"Total predictions: {len(all_preds)}")

    by_cat = defaultdict(list)
    for p in all_preds:
        cat = p.get("category")
        if cat in CATEGORY_NAMES:
            by_cat[CATEGORY_NAMES[cat]].append(p)

    if not all_preds:
        print("No predictions yet (ingestion may have consumed the time cap). "
              "Re-run to resume; ingested stores are cached and skipped.")
        return

    all_j = [p["j_score"] for p in all_preds]
    all_f1 = [p["f1"] for p in all_preds]
    all_b1 = [p["bleu1"] for p in all_preds]
    all_tokens = [p["memory_tokens_used"] for p in all_preds]

    overall_j = sum(all_j) / len(all_j) * 100

    baselines = {
        "Full-context": {"single-hop": 63.79, "multi-hop": 42.92, "open-domain": 62.29, "temporal": 21.71, "overall_j": 72.90, "tokens": 26031},
        "Mem0": {"single-hop": 67.13, "multi-hop": 51.15, "open-domain": 72.93, "temporal": 55.51, "overall_j": 66.88, "tokens": 1764},
        "Mem0^g": {"single-hop": 65.71, "multi-hop": 47.19, "open-domain": 75.71, "temporal": 58.13, "overall_j": 68.44, "tokens": 3616},
        "Zep": {"single-hop": 61.70, "multi-hop": 41.35, "open-domain": 76.60, "temporal": 49.31, "overall_j": 65.99, "tokens": 3911},
        "LangMem": {"single-hop": 62.23, "multi-hop": 47.92, "open-domain": 71.12, "temporal": 23.43, "overall_j": 58.10, "tokens": 127},
        "Widemem v1": {"single-hop": 41.25, "multi-hop": 53.31, "open-domain": 36.81, "temporal": 30.53, "overall_j": 45.32, "tokens": 157},
    }

    cats_j = {}
    for cat_name in ["single-hop", "multi-hop", "open-domain", "temporal"]:
        preds = by_cat.get(cat_name, [])
        js = [p["j_score"] for p in preds]
        cats_j[cat_name] = sum(js) / len(js) * 100 if js else 0

    wm_tokens = int(sum(all_tokens) / len(all_tokens))

    print(f"\n{'Method':<18} {'Single-Hop':>11} {'Multi-Hop':>10} {'Open-Dom':>9} {'Temporal':>9} {'Overall':>8} {'Tokens':>7}")
    print("-" * 80)
    for name, b in baselines.items():
        print(f"{name:<18} {b['single-hop']:>11.2f} {b['multi-hop']:>10.2f} {b['open-domain']:>9.2f} {b['temporal']:>9.2f} {b['overall_j']:>8.2f} {b['tokens']:>7}")
    print("-" * 80)
    print(f"{'** WIDEMEM v2 **':<18} {cats_j.get('single-hop',0):>11.2f} {cats_j.get('multi-hop',0):>10.2f} {cats_j.get('open-domain',0):>9.2f} {cats_j.get('temporal',0):>9.2f} {overall_j:>8.2f} {wm_tokens:>7}")

    # Efficiency
    print(f"\n--- Efficiency ---")
    jpt = overall_j / max(wm_tokens, 1)
    print(f"  Widemem v1: 45.32 J / 157 tokens = 0.2887 J/token")
    print(f"  Widemem v2: {overall_j:.2f} J / {wm_tokens} tokens = {jpt:.4f} J/token")
    print(f"  Mem0:       66.88 J / 1764 tokens = 0.0379 J/token")

    # Latency
    sorted_s = sorted(p["search_latency"] for p in all_preds)
    sorted_t = sorted(p["total_latency"] for p in all_preds)
    n = len(sorted_s)
    print(f"\n--- Latency ---")
    print(f"  Search p50: {sorted_s[n//2]:.3f}s  |  p95: {sorted_s[int(n*0.95)]:.3f}s")
    print(f"  Total  p50: {sorted_t[n//2]:.3f}s  |  p95: {sorted_t[int(n*0.95)]:.3f}s")

    # Delta from v1
    print(f"\n--- Delta from v1 ---")
    v1 = {"single-hop": 41.25, "multi-hop": 53.31, "open-domain": 36.81, "temporal": 30.53, "overall": 45.32}
    for cat in ["single-hop", "multi-hop", "open-domain", "temporal"]:
        delta = cats_j.get(cat, 0) - v1[cat]
        print(f"  {cat:<12}: {v1[cat]:.2f} -> {cats_j.get(cat,0):.2f}  ({'+' if delta>=0 else ''}{delta:.2f})")
    delta_all = overall_j - v1["overall"]
    print(f"  {'overall':<12}: {v1['overall']:.2f} -> {overall_j:.2f}  ({'+' if delta_all>=0 else ''}{delta_all:.2f})")

    # Save
    results = {
        "overall": {"count": len(all_preds), "f1": round(sum(all_f1)/len(all_f1)*100, 2),
                    "bleu1": round(sum(all_b1)/len(all_b1)*100, 2), "j_score": round(overall_j, 2)},
        "by_category": {cat: {"count": len(by_cat.get(cat, [])), "j_score": round(cats_j.get(cat, 0), 2)}
                       for cat in CATEGORY_NAMES.values()},
        "efficiency": {"avg_memory_tokens": wm_tokens, "j_per_token": round(jpt, 4)},
        "latency": {"search_p50": round(sorted_s[n//2], 4), "search_p95": round(sorted_s[int(n*0.95)], 4),
                    "total_p50": round(sorted_t[n//2], 4), "total_p95": round(sorted_t[int(n*0.95)], 4)},
    }
    final = {
        "metadata": {
            "benchmark": "LoCoMo", "system": "Widemem WS1 (event_time + answer-prompt anchor, re-ingested)",
            "changes": "default retrieval (no hybrid/temporal flags), single pass, no repair, JUDGE_RUNS=5",
            "eval_llm": EVAL_LLM, "judge_llm": JUDGE_LLM, "self_graded": JUDGE_LLM == EVAL_LLM, "judge_runs": JUDGE_RUNS,
            "timestamp": datetime.now().isoformat(), "total_questions": len(all_preds),
        },
        "results": results, "predictions": all_preds,
    }
    with open(FINAL_OUTPUT, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nSaved to {FINAL_OUTPUT}")


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="widemem LoCoMo v3 benchmark runner")
    ap.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Stop after processing this many questions total (for batch testing).",
    )
    ap.add_argument(
        "--max-cost",
        type=float,
        default=MAX_TOTAL_COST_USD,
        help=f"Hard cost ceiling in USD (default ${MAX_TOTAL_COST_USD}).",
    )
    ap.add_argument(
        "--max-hours",
        type=float,
        default=MAX_WALL_CLOCK_HOURS,
        help=f"Hard wall-clock ceiling in hours (default {MAX_WALL_CLOCK_HOURS}).",
    )
    ap.add_argument(
        "--max-convs",
        type=int,
        default=None,
        help="Only ingest/evaluate the first N conversations (mini probe).",
    )
    ap.add_argument(
        "--only-temporal",
        action="store_true",
        help="Evaluate only temporal questions (category 2).",
    )
    ap.add_argument(
        "--no-event-surface",
        action="store_true",
        help="Control: baseline prompt, no event_time/anchor. Same re-ingested "
             "stores as WS1; isolates the prompt effect. Separate output dir.",
    )
    ap.add_argument(
        "--entity-boost",
        type=float,
        default=0.0,
        help="Enable entity index + entity-aware re-rank at this weight. "
             "Backfills entities on the cached stores (no LLM, no re-embed). "
             "Separate output dir; compare per-category vs the control run.",
    )
    return ap.parse_args()


def get_sessions(conversation: dict) -> list:
    """Ordered (timestamp, turns) sessions for one conversation."""
    sessions = []
    i = 1
    while True:
        ts = conversation.get(f"session_{i}_date_time")
        turns = conversation.get(f"session_{i}")
        if ts is None and turns is None:
            break
        if turns:
            sessions.append((ts or "", turns))
        i += 1
    return sessions


def ingest_if_empty(mem, conv: dict, sample_id: str) -> None:
    """Phase 1: re-ingest with the [session_ts] prefix so PR-B's
    parse_leading_datetime captures event_time. Skipped if the store
    already holds memories (resumable)."""
    try:
        existing = json.loads(mem.export_json())
        if existing.get("count", 0) > 0:
            return
    except Exception:
        pass
    conversation = conv["conversation"]
    for session_ts, turns in get_sessions(conversation):
        for turn in turns:
            speaker = turn["speaker"]
            text = turn["text"]
            try:
                mem.add(
                    text=f"[{session_ts}] {speaker}: {text}",
                    user_id=f"{sample_id}_{speaker}",
                )
            except Exception as e:
                print(f"  ingest warn {turn.get('dia_id','?')}: {e}")


def main():
    global NO_EVENT_SURFACE, CHUNKS_DIR, FINAL_OUTPUT, ENTITY_BOOST_W
    args = parse_args()
    cost_cap = args.max_cost
    wall_cap_seconds = args.max_hours * 3600
    run_started_at = time.time()

    if args.no_event_surface:
        NO_EVENT_SURFACE = True
        CHUNKS_DIR = os.environ.get("WM_CHUNKS_DIR", "benchmark/results/ws1_control_chunks")
        FINAL_OUTPUT = os.environ.get("WM_FINAL_OUTPUT", "benchmark/results/locomo_ws1_control_final.json")
        print("CONTROL MODE: baseline prompt, no event/anchor, same re-ingested stores.")

    if args.entity_boost > 0:
        ENTITY_BOOST_W = args.entity_boost
        CHUNKS_DIR = f"benchmark/results/ws1_eb{args.entity_boost}_chunks"
        FINAL_OUTPUT = f"benchmark/results/locomo_ws1_eb{args.entity_boost}_final.json"
        print(f"ENTITY-BOOST MODE: W={args.entity_boost}, entities backfilled on cached stores (no re-ingest).")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    print(f"Budget: <=${cost_cap} or {args.max_hours}h wall-clock, whichever hits first.")
    if args.max_questions:
        print(f"Batch mode: stop after {args.max_questions} questions.")

    os.makedirs(CHUNKS_DIR, exist_ok=True)

    # Load LoCoMo data
    print("Loading LoCoMo data...")
    with open(DATA_FILE) as f:
        data = json.load(f)

    # WS1: re-ingest fresh stores so PR-B captures event_time from the
    # [session_ts] prefix. Config matches the validated clean baseline
    # (hierarchy on, default scoring, no v1.6 flags) so the ONLY variables
    # vs baseline are event_time capture + the answer-prompt surfacing.
    print("WS1: opening / re-ingesting memory stores (event_time enabled)...")
    mem_instances = {}
    for i, conv in enumerate(data):
        if args.max_convs is not None and i >= args.max_convs:
            break
        storage_dir = os.path.join(STORES_DIR, f"conv_{i}")
        config = MemoryConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.0),
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            vector_store=VectorStoreConfig(provider="faiss", path=os.path.join(storage_dir, "faiss")),
            scoring=ScoringConfig(
                decay_function="exponential", decay_rate=0.01,
                similarity_weight=0.5, importance_weight=0.3, recency_weight=0.2,
            ),
            history_db_path=os.path.join(storage_dir, "history.db"),
            enable_hierarchy=True,
            enable_entity_index=ENTITY_BOOST_W > 0,
            entity_boost_weight=ENTITY_BOOST_W,
        )
        mem = WideMemory(config=config)
        ingest_if_empty(mem, conv, conv["sample_id"])
        if ENTITY_BOOST_W > 0:
            n = mem.backfill_entities()
            print(f"  conv_{i}: backfilled entities on {n} memories (no LLM, no re-embed)")
        mem_instances[i] = mem
        sessions = get_sessions(conv["conversation"])
        CONV_REFERENCE_DATE[i] = sessions[-1][0] if sessions else "unknown"
    print(f"  {len(mem_instances)} stores ready (re-ingested with event_time)")
    print(f"  v1 DEFAULT retrieval (matches validated baseline); WS1 deltas = event_time + anchor only")

    # Build flat question list with metadata
    all_questions = []
    for i, conv in enumerate(data):
        if args.max_convs is not None and i >= args.max_convs:
            break
        sample_id = conv["sample_id"]
        speaker_a = conv["conversation"]["speaker_a"]
        speaker_b = conv["conversation"]["speaker_b"]
        for q in conv["qa"]:
            if q["category"] == 5:  # skip adversarial
                continue
            if args.only_temporal and q["category"] != 2:
                continue
            q_copy = dict(q)
            q_copy["_conv_idx"] = i
            q_copy["_sample_id"] = sample_id
            q_copy["_speaker_a"] = speaker_a
            q_copy["_speaker_b"] = speaker_b
            all_questions.append(q_copy)

    print(f"  Total questions: {len(all_questions)}")

    # Load completed chunks
    completed = load_completed()
    remaining = [q for q in all_questions if f"{q['_sample_id']}_{q['question'][:80]}" not in completed]
    print(f"  Already done: {len(completed)}")
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("All done! Merging...")
        merge_and_report()
        for mem in mem_instances.values():
            mem.close()
        return

    # Create client
    client = OpenAI(timeout=API_TIMEOUT)

    # Test API
    test = api_call_with_retry(client, EVAL_LLM, [{"role": "user", "content": "Say OK"}], max_tokens=5)
    print(f"  API test: {test}")

    # Apply --max-questions cap (for batch testing).
    if args.max_questions is not None:
        remaining = remaining[: args.max_questions]
        print(f"  Batch cap applied: {len(remaining)} questions will run")

    # Split into chunks
    chunks = [remaining[i:i + CHUNK_SIZE] for i in range(0, len(remaining), CHUNK_SIZE)]
    existing = len([f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json")])

    print(f"\n  Starting: {len(chunks)} chunks, ~{len(remaining) * 4} API calls")
    print(f"  ETA: ~{len(remaining) * 4 / 20:.0f} min at 20 q/min")

    total_start = time.time()
    aborted_reason = None
    for ci, chunk in enumerate(chunks):
        # Budget gates: check before each chunk so an in-flight chunk
        # is allowed to finish saving cleanly.
        wall_elapsed = time.time() - run_started_at
        spent = total_cost_usd()
        if spent >= cost_cap:
            aborted_reason = f"cost cap ${cost_cap} reached (spent ${spent:.2f})"
            print(f"\n!! ABORT: {aborted_reason}")
            break
        if wall_elapsed >= wall_cap_seconds:
            aborted_reason = (
                f"wall-clock cap {args.max_hours}h reached "
                f"({wall_elapsed/3600:.2f}h elapsed)"
            )
            print(f"\n!! ABORT: {aborted_reason}")
            break

        chunk_idx = existing + ci
        print(f"\n{'='*60}")
        print(
            f"CHUNK {chunk_idx} ({ci+1}/{len(chunks)}) - {len(chunk)} questions  "
            f"|  spent ${spent:.2f} / ${cost_cap}  "
            f"|  wall {wall_elapsed/60:.1f}m / {args.max_hours*60:.0f}m"
        )
        print(f"{'='*60}")

        healthy = process_chunk(chunk_idx, chunk, mem_instances, client)

        elapsed = time.time() - total_start
        done = (ci + 1) * CHUNK_SIZE
        rate = done / elapsed * 60 if elapsed > 0 else 0
        eta = (len(remaining) - done) / rate if rate > 0 else 0
        print(
            f"  Total: {done}/{len(remaining)} | {rate:.0f} q/min | "
            f"ETA: {eta:.0f} min | spent ${total_cost_usd():.2f}"
        )

        if not healthy:
            print(f"\n  Pausing 120s after failure streak...")
            time.sleep(120)

    if aborted_reason:
        print(f"\nRun aborted: {aborted_reason}")
        print(f"Final spend: ${total_cost_usd():.2f}")
        print(f"Chunks saved: {len([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.json')])}")
        print("Resume by running the script again; completed chunks are skipped.")

    # Merge
    merge_and_report()

    for mem in mem_instances.values():
        mem.close()


if __name__ == "__main__":
    main()
