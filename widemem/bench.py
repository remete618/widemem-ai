"""Reproducible benchmark reporting.

Turns a LoCoMo result file (from the benchmark harness) into a credibility
report: per-category J, token/latency, the exact config, the judge prompt, and
a reproducibility hash that fingerprints everything needed to reproduce the
number. The point is a benchmark claim a reader can verify, not dismiss.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

_CATEGORIES = ["single-hop", "multi-hop", "open-domain", "temporal"]


def _repro_hash(meta: Dict[str, Any], summary: Dict[str, Any], n_predictions: int) -> str:
    """Stable fingerprint of the run's reproducibility-relevant inputs. Same
    code + config + dataset + judge protocol -> same hash."""
    payload = {
        "git_sha": meta.get("git_sha"),
        "eval_llm": meta.get("eval_llm"),
        "judge_runs": meta.get("judge_runs"),
        "top_k": meta.get("top_k"),
        "graph": meta.get("graph"),
        "n_questions": summary.get("n_questions") or n_predictions,
        "categories": {c: summary.get("by_category", {}).get(c) for c in _CATEGORIES},
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def build_report(result: Dict[str, Any], judge_prompt: str | None = None) -> Tuple[str, Dict[str, Any]]:
    """Return (markdown_report, machine_report). Pure: no I/O, fully testable."""
    meta = result.get("metadata", {})
    summary = result.get("summary", {})
    preds = result.get("predictions", [])
    by_cat = summary.get("by_category", {})
    counts = summary.get("counts", {})
    rhash = _repro_hash(meta, summary, len(preds))

    rows = "\n".join(
        f"| {c} | {counts.get(c, '?')} | {by_cat.get(c, 0):.2f} |"
        for c in _CATEGORIES
    )
    # Built outside the f-string: backslash escapes inside f-string expressions
    # are only valid on Python 3.12+, and the package targets 3.10.
    fence = "```"
    judge_block = (
        f"{fence}\n{judge_prompt.strip()}\n{fence}"
        if judge_prompt
        else "_(embed the judge prompt used; see harness)_"
    )
    md = f"""# Widemem LoCoMo benchmark report

**Reproducibility hash:** `{rhash}`
**Overall J:** {summary.get('overall_j', 0):.2f}  (n={summary.get('n_questions', len(preds))})

| Category | n | J |
|---|---|---|
{rows}
| **Overall** | {summary.get('n_questions', len(preds))} | **{summary.get('overall_j', 0):.2f}** |

## Config (exact)
- git sha: `{meta.get('git_sha')}`
- eval LLM: `{meta.get('eval_llm')}`
- judge runs: {meta.get('judge_runs')}
- top_k per speaker: {meta.get('top_k')}
- graph layer: {meta.get('graph')}
- avg memory tokens/query: {summary.get('avg_memory_tokens')}
- avg search latency: {summary.get('avg_search_latency')}s
- run cost: ${meta.get('cost_usd')}

## Judge prompt
{judge_block}

## Raw predictions
{len(preds)} predictions included in the source result file (question, gold, predicted, per-question j_score).

_Reproduce: same git sha + config + dataset + judge protocol reproduces hash `{rhash}`._
"""
    machine = {
        "reproducibility_hash": rhash,
        "overall_j": summary.get("overall_j"),
        "by_category": {c: by_cat.get(c) for c in _CATEGORIES},
        "counts": {c: counts.get(c) for c in _CATEGORIES},
        "config": {
            "git_sha": meta.get("git_sha"),
            "eval_llm": meta.get("eval_llm"),
            "judge_runs": meta.get("judge_runs"),
            "top_k": meta.get("top_k"),
            "graph": meta.get("graph"),
        },
        "n_predictions": len(preds),
    }
    return md, machine


def report_from_file(path: str, judge_prompt: str | None = None) -> Tuple[str, Dict[str, Any]]:
    with open(path) as f:
        return build_report(json.load(f), judge_prompt=judge_prompt)
