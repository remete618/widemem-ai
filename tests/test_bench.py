"""Reproducible bench report: deterministic hash + correct table rendering."""
from __future__ import annotations

from widemem.bench import build_report

_RESULT = {
    "metadata": {
        "git_sha": "abc1234", "eval_llm": "gpt-4o-mini", "judge_runs": 10,
        "top_k": 20, "graph": False, "cost_usd": 0.38,
    },
    "summary": {
        "overall_j": 54.92,
        "by_category": {"single-hop": 45.6, "multi-hop": 58.56,
                        "open-domain": 45.0, "temporal": 59.27},
        "counts": {"single-hop": 100, "multi-hop": 250,
                   "open-domain": 40, "temporal": 96},
        "n_questions": 486, "avg_memory_tokens": 456.8, "avg_search_latency": 0.2,
    },
    "predictions": [{"j_score": 1.0}] * 486,
}


def test_report_contains_scores_and_hash():
    md, machine = build_report(_RESULT)
    assert "54.92" in md
    assert "open-domain | 40 | 45.00" in md
    assert machine["reproducibility_hash"] == md.split("`")[1]
    assert machine["overall_j"] == 54.92


def test_repro_hash_is_deterministic():
    h1 = build_report(_RESULT)[1]["reproducibility_hash"]
    h2 = build_report(_RESULT)[1]["reproducibility_hash"]
    assert h1 == h2 and len(h1) == 16


def test_repro_hash_changes_with_config():
    base = build_report(_RESULT)[1]["reproducibility_hash"]
    changed = {**_RESULT, "metadata": {**_RESULT["metadata"], "judge_runs": 3}}
    assert build_report(changed)[1]["reproducibility_hash"] != base


def test_judge_prompt_embedded_when_given():
    md, _ = build_report(_RESULT, judge_prompt="LABEL as CORRECT or WRONG")
    assert "LABEL as CORRECT or WRONG" in md
