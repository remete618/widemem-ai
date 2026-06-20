#!/usr/bin/env python3
"""
widemem Validation Harness — graph-layer improvement loop
=========================================================
A FRESH-INGEST validation eval over a fixed 3-conversation subset of LoCoMo,
with REAL per-category denominators (open-domain n=40) and 10 judge runs.

This replaces mini_locomo.py as the loop's fitness function. The mini gate
reuses v1.3-era stale stores and has n=4 open-domain — useless for measuring
the graph layer, whose entire job is open-domain. This harness fresh-ingests
with the CURRENT code so extraction + retrieval + graph changes are all
exercised, and reports per-category J the graph work can actually be steered by.

Two subcommands so the loop iterates fast:
    ingest   fresh-ingest VAL_CONVS into a store dir (slow; only when
             extraction / graph-extraction changes)
    eval     load stores, answer + judge all VAL questions (fast; rerun on
             every retrieval / traversal change without re-ingesting)

Config matches run_locomo.py (the full-1540 confirmation run) so the
validation number is directly comparable to the eventual full benchmark.

Usage:
    set -a; source .env.local; set +a
    cd /Users/radu/widemem-ai
    .venv/bin/python3 benchmark/val.py ingest --store-dir benchmark/results/val_stores_base
    .venv/bin/python3 benchmark/val.py eval   --store-dir benchmark/results/val_stores_base \
        --out benchmark/results/val_base.json
    # graph arm (once the layer exists):
    .venv/bin/python3 benchmark/val.py ingest --graph --store-dir benchmark/results/val_stores_graph
    .venv/bin/python3 benchmark/val.py eval --graph --store-dir benchmark/results/val_stores_graph \
        --out benchmark/results/val_graph.json --baseline benchmark/results/val_base.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
RESULTS_DIR = "benchmark/results"

# Fixed validation conversations: chosen for open-domain coverage (n=40 of 96
# total) and bounded ingest cost. idx into locomo10.json.
VAL_CONV_IDX = [0, 4, 8]  # conv-26, conv-43, conv-49

JUDGE_RUNS = 10          # paper standard; stable category means on n>=26
EVAL_LLM = "gpt-4o-mini"
TOP_K = 20               # per speaker, matches run_locomo.py full run
API_TIMEOUT = 30
MAX_RETRIES = 5

CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "open-domain", 4: "multi-hop"}

# gpt-4o-mini pricing (USD per 1M tokens), for spend tracking
IN_PER_1M, OUT_PER_1M = 0.15, 0.60
_IN_TOK = 0
_OUT_TOK = 0

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
# Helpers
# ---------------------------------------------------------------------------
def git_sha() -> str:
    try:
        return check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def total_cost() -> float:
    return _IN_TOK / 1_000_000 * IN_PER_1M + _OUT_TOK / 1_000_000 * OUT_PER_1M


def api_call(client, messages, temperature=0.0, max_tokens=100):
    global _IN_TOK, _OUT_TOK
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=EVAL_LLM, messages=messages, temperature=temperature,
                max_tokens=max_tokens, timeout=API_TIMEOUT,
            )
            if resp.usage:
                _IN_TOK += resp.usage.prompt_tokens
                _OUT_TOK += resp.usage.completion_tokens
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            wait = min(15 * (attempt + 1), 60) if ("429" in err or "rate_limit" in err) else 3
            print(f"  api retry {attempt + 1}/{MAX_RETRIES}: {err[:70]}", flush=True)
            time.sleep(wait)
    return None


def judge_one(question, gold, predicted, client):
    text = api_call(
        client,
        [{"role": "user", "content": JUDGE_PROMPT.format(
            question=str(question), gold_answer=str(gold), generated_answer=str(predicted))}],
        temperature=0.1, max_tokens=200,
    )
    if text is None:
        return None
    if "{" in text:
        try:
            parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
            return parsed.get("label", "").upper() == "CORRECT"
        except (json.JSONDecodeError, ValueError):
            pass
    return "CORRECT" in text.upper() and "WRONG" not in text.upper()


def make_config(storage_dir: str, graph: bool, hybrid: bool = False) -> MemoryConfig:
    """Config matches run_locomo.py. `graph` toggles the graph layer; `hybrid`
    toggles BM25 keyword blending (auto-zeroed for multi-hop, full for
    factual, so it targets single-hop without denting the moat)."""
    kwargs = dict(
        llm=LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.0),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreConfig(provider="faiss", path=os.path.join(storage_dir, "faiss")),
        scoring=ScoringConfig(
            decay_function="exponential", decay_rate=0.01,
            similarity_weight=0.5, importance_weight=0.3, recency_weight=0.2,
        ),
        history_db_path=os.path.join(storage_dir, "history.db"),
        enable_hierarchy=False,
    )
    # Forward-compatible: only set graph flags if the config supports them, so
    # this harness runs against current main (no graph) unchanged.
    if graph:
        supported = getattr(MemoryConfig, "model_fields", {})  # pydantic model
        if "enable_graph" in supported:
            kwargs["enable_graph"] = True
            kwargs["graph_db_path"] = os.path.join(storage_dir, "graph.db")
    if hybrid:
        kwargs["enable_hybrid_search"] = True
    return MemoryConfig(**kwargs)


def get_sessions(conversation: dict):
    sessions = []
    i = 1
    while f"session_{i}" in conversation:
        ts = conversation.get(f"session_{i}_date_time", "")
        sessions.append((ts, conversation[f"session_{i}"]))
        i += 1
    return sessions


def load_val_questions(data):
    qs = []
    for idx in VAL_CONV_IDX:
        conv = data[idx]
        sid = conv["sample_id"]
        sa = conv["conversation"]["speaker_a"]
        sb = conv["conversation"]["speaker_b"]
        for q in conv["qa"]:
            if q["category"] == 5:
                continue
            qc = dict(q)
            qc.update(_conv_idx=idx, _sample_id=sid, _speaker_a=sa, _speaker_b=sb)
            qs.append(qc)
    return qs


# ---------------------------------------------------------------------------
# Phase: ingest
# ---------------------------------------------------------------------------
def do_ingest(args):
    data = json.load(open(DATA_FILE))
    base = args.store_dir
    if os.path.exists(base) and args.fresh:
        shutil.rmtree(base)
    os.makedirs(base, exist_ok=True)
    print(f"ingest: graph={args.graph} store={base} convs={VAL_CONV_IDX}", flush=True)
    t0 = time.time()
    stats = []
    for idx in VAL_CONV_IDX:
        conv = data[idx]
        sid = conv["sample_id"]
        conversation = conv["conversation"]
        sdir = os.path.join(base, f"conv_{idx}")
        os.makedirs(sdir, exist_ok=True)
        mem = WideMemory(config=make_config(sdir, args.graph))
        n_turns = n_mem = 0
        ct = time.time()
        for session_ts, turns in get_sessions(conversation):
            for turn in turns:
                user_id = f"{sid}_{turn['speaker']}"
                try:
                    res = mem.add(text=f"[{session_ts}] {turn['speaker']}: {turn['text']}",
                                  user_id=user_id)
                    n_mem += len(res.memories)
                except Exception as e:
                    print(f"  warn add fail conv{idx} {turn.get('dia_id','?')}: {str(e)[:60]}", flush=True)
                n_turns += 1
        mem.close()
        dt = time.time() - ct
        stats.append(dict(conv_idx=idx, sample_id=sid, turns=n_turns, memories=n_mem,
                          seconds=round(dt, 1)))
        print(f"  conv-{idx}: {n_turns} turns -> {n_mem} mem in {dt/60:.1f}m", flush=True)
    meta = dict(git_sha=git_sha(), graph=args.graph, store_dir=base,
                timestamp=datetime.now(timezone.utc).isoformat(),
                elapsed_min=round((time.time() - t0) / 60, 1), cost_usd=round(total_cost(), 3),
                stats=stats)
    json.dump(meta, open(os.path.join(base, "ingest_meta.json"), "w"), indent=2)
    print(f"ingest done: {meta['elapsed_min']}m  ${meta['cost_usd']}", flush=True)


# ---------------------------------------------------------------------------
# Phase: eval
# ---------------------------------------------------------------------------
def answer_and_judge(q, mem, client):
    sid, sa, sb = q["_sample_id"], q["_speaker_a"], q["_speaker_b"]
    t0 = time.time()
    ra = mem.search(query=q["question"], user_id=f"{sid}_{sa}", top_k=TOP_K)
    rb = mem.search(query=q["question"], user_id=f"{sid}_{sb}", top_k=TOP_K)
    search_t = time.time() - t0
    ma = "\n".join(f"[importance={r.memory.importance:.1f}] {r.memory.content}" for r in ra)
    mb = "\n".join(f"[importance={r.memory.importance:.1f}] {r.memory.content}" for r in rb)
    tok = sum(len(r.memory.content.split()) for r in list(ra) + list(rb))
    tmpl = TEMPORAL_ANSWER_PROMPT if q["category"] == 2 else ANSWER_PROMPT
    prompt = tmpl.format(speaker_a=sa, speaker_b=sb,
                         memories_a=ma or "(no memories)", memories_b=mb or "(no memories)",
                         question=q["question"])
    answer = api_call(client, [{"role": "user", "content": prompt}]) or "ERROR"
    correct = valid = 0
    for _ in range(JUDGE_RUNS):
        r = judge_one(q["question"], q["answer"], answer, client)
        if r is not None:
            valid += 1
            correct += int(r)
    return dict(sample_id=sid, question=q["question"], gold=q["answer"], predicted=answer,
                category=q["category"], category_name=CATEGORY_NAMES[q["category"]],
                memory_tokens=tok, search_latency=round(search_t, 4),
                j_score=(correct / valid if valid else 0.0), judge_valid=valid)


def summarize(preds):
    by = defaultdict(list)
    for p in preds:
        by[p["category_name"]].append(p)
    cats = {c: round(sum(p["j_score"] for p in by[c]) / len(by[c]) * 100, 2) if by[c] else 0.0
            for c in CATEGORY_NAMES.values()}
    counts = {c: len(by[c]) for c in CATEGORY_NAMES.values()}
    overall = round(sum(p["j_score"] for p in preds) / len(preds) * 100, 2) if preds else 0.0
    return dict(overall_j=overall, by_category=cats, counts=counts,
                avg_memory_tokens=round(sum(p["memory_tokens"] for p in preds) / len(preds), 1) if preds else 0,
                avg_search_latency=round(sum(p["search_latency"] for p in preds) / len(preds), 4) if preds else 0,
                n_questions=len(preds))


def do_eval(args):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set")
    data = json.load(open(DATA_FILE))
    questions = load_val_questions(data)
    mix = defaultdict(int)
    for q in questions:
        mix[CATEGORY_NAMES[q["category"]]] += 1
    print(f"eval: graph={args.graph} store={args.store_dir} n={len(questions)} mix={dict(mix)} judges={JUDGE_RUNS}", flush=True)

    mems = {}
    for idx in VAL_CONV_IDX:
        sdir = os.path.join(args.store_dir, f"conv_{idx}")
        if not os.path.exists(sdir):
            sys.exit(f"ERROR: store missing {sdir} — run `ingest` first")
        mems[idx] = WideMemory(config=make_config(sdir, args.graph, hybrid=args.hybrid))

    client = OpenAI(http_client=httpx.Client(timeout=httpx.Timeout(API_TIMEOUT, connect=10)))
    t0 = time.time()
    preds = []
    for i, q in enumerate(questions, 1):
        preds.append(answer_and_judge(q, mems[q["_conv_idx"]], client))
        run = sum(p["j_score"] for p in preds) / len(preds) * 100
        print(f"  [{i:>3}/{len(questions)}] {preds[-1]['category_name']:<11} "
              f"J={preds[-1]['j_score']:.2f} running={run:.1f}%", flush=True)
    for m in mems.values():
        m.close()

    summary = summarize(preds)
    elapsed = time.time() - t0
    print("\n" + "=" * 64)
    print(f"VAL RESULTS  graph={args.graph}  ({elapsed/60:.1f}m  ${total_cost():.2f})")
    print("=" * 64)
    print(f"  overall J:   {summary['overall_j']:.2f}   (n={summary['n_questions']})")
    for c in ["single-hop", "multi-hop", "open-domain", "temporal"]:
        print(f"  {c+' J:':<16}{summary['by_category'][c]:>6.2f}  (n={summary['counts'][c]})")
    print(f"  avg tokens:  {summary['avg_memory_tokens']}")

    out = dict(metadata=dict(git_sha=git_sha(), graph=args.graph, store_dir=args.store_dir,
                             timestamp=datetime.now(timezone.utc).isoformat(),
                             judge_runs=JUDGE_RUNS, top_k=TOP_K, eval_llm=EVAL_LLM,
                             elapsed_min=round(elapsed / 60, 1), cost_usd=round(total_cost(), 3)),
               summary=summary, predictions=preds)
    json.dump(out, open(args.out, "w"), indent=2, default=str)
    print(f"  saved: {args.out}")

    if args.baseline and os.path.exists(args.baseline):
        b = json.load(open(args.baseline))["summary"]
        print("\n" + "=" * 64)
        print(f"VS BASELINE ({args.baseline})")
        print("=" * 64)
        od = summary["overall_j"] - b["overall_j"]
        md = summary["by_category"]["multi-hop"] - b["by_category"]["multi-hop"]
        print(f"  overall:   {b['overall_j']:>6.2f} -> {summary['overall_j']:>6.2f}  ({od:+.2f})")
        for c in ["single-hop", "multi-hop", "open-domain", "temporal"]:
            d = summary["by_category"][c] - b["by_category"][c]
            print(f"  {c+':':<13}{b['by_category'][c]:>6.2f} -> {summary['by_category'][c]:>6.2f}  ({d:+.2f})")
        keep = od > 0 and md >= -3.0
        print(f"\n  VERDICT: {'KEEP' if keep else 'REVERT'}  "
              f"(overall {'+' if od > 0 else ''}{od:.2f}, multi-hop floor {'held' if md >= -3.0 else 'BREACHED'})")


def main():
    ap = argparse.ArgumentParser(description="widemem validation harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest")
    pi.add_argument("--store-dir", required=True)
    pi.add_argument("--graph", action="store_true")
    pi.add_argument("--fresh", action="store_true", default=True)
    pe = sub.add_parser("eval")
    pe.add_argument("--store-dir", required=True)
    pe.add_argument("--graph", action="store_true")
    pe.add_argument("--hybrid", action="store_true")
    pe.add_argument("--out", required=True)
    pe.add_argument("--baseline", default=None)
    args = ap.parse_args()
    if args.cmd == "ingest":
        do_ingest(args)
    else:
        do_eval(args)


if __name__ == "__main__":
    main()
