"""Sanity checks on the committed LoCoMo dev/held-out split.

The split exists so tuning (dev) and publishable confirmation (held-out)
never share questions. These tests keep the two sets disjoint, complete,
and pinned to the harness that consumes them. Text-only: no package
install or dataset download required.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = ROOT / "benchmark" / "locomo_split.json"
LOCOMO10_CONVERSATIONS = 10


def load_split() -> dict:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def test_split_file_exists():
    assert SPLIT_PATH.exists(), "benchmark/locomo_split.json is missing"


def test_sets_are_disjoint():
    split = load_split()
    overlap = set(split["dev"]) & set(split["held_out"])
    assert not overlap, f"dev and held_out share conversations: {sorted(overlap)}"


def test_sets_cover_the_dataset():
    split = load_split()
    union = set(split["dev"]) | set(split["held_out"])
    assert union == set(range(LOCOMO10_CONVERSATIONS)), (
        f"split does not cover all {LOCOMO10_CONVERSATIONS} locomo10 conversations: {sorted(union)}"
    )


def test_dev_preserves_historical_subset():
    # Changing dev invalidates comparability with every earlier val.py
    # result. If a change is truly needed, record it in docs/HISTORY.md.
    assert load_split()["dev"] == [0, 4, 8]


def test_harness_uses_the_split():
    val_src = (ROOT / "benchmark" / "val.py").read_text(encoding="utf-8")
    assert "locomo_split.json" in val_src, "val.py no longer reads the committed split file"
    for flag in ["--dev-only", "--held-out"]:
        assert flag in val_src, f"val.py lost the {flag} flag"
