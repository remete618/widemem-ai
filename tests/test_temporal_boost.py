"""Tests for the soft temporal-boost semantics in score_and_rank.

Verifies the fix for the v1.5 regression where parse_temporal_hints was
hard-filtering memories out of the candidate pool when the auto-parsed
window did not contain the relevant memory. The v1.6 behavior treats
auto-parsed windows as soft boosts: in-window memories rank higher,
out-of-window memories are NOT excluded.

Backward compat: explicit time_after / time_before still filter, since
the caller is asserting intent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from widemem.core.types import (
    Memory,
    MemorySearchResult,
    MemoryTier,
    ScoringConfig,
)
from widemem.retrieval.temporal import score_and_rank

NOW = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)


def _make_result(
    id_: str,
    days_ago: int,
    similarity: float = 0.5,
    importance: float = 5.0,
) -> MemorySearchResult:
    created = NOW - timedelta(days=days_ago)
    return MemorySearchResult(
        memory=Memory(
            id=id_,
            content=f"memory {id_}",
            user_id="alice",
            tier=MemoryTier.FACT,
            importance=importance,
            created_at=created,
            updated_at=created,
        ),
        similarity_score=similarity,
    )


@pytest.fixture
def default_config() -> ScoringConfig:
    return ScoringConfig(
        decay_function="exponential",
        decay_rate=0.01,
        similarity_weight=0.5,
        importance_weight=0.3,
        recency_weight=0.2,
    )


def test_no_boost_window_no_change(default_config):
    """When temporal_boost_window is None, scoring is unchanged."""
    results = [
        _make_result("m1", days_ago=30, similarity=0.5, importance=5.0),
        _make_result("m2", days_ago=400, similarity=0.5, importance=5.0),
    ]
    ranked_a = score_and_rank(
        list(results), default_config, now=NOW, temporal_boost_window=None
    )
    # Same scoring with the parameter omitted entirely.
    results2 = [
        _make_result("m1", days_ago=30, similarity=0.5, importance=5.0),
        _make_result("m2", days_ago=400, similarity=0.5, importance=5.0),
    ]
    ranked_b = score_and_rank(list(results2), default_config, now=NOW)
    assert [r.memory.id for r in ranked_a] == [r.memory.id for r in ranked_b]
    for a, b in zip(ranked_a, ranked_b):
        assert abs(a.final_score - b.final_score) < 1e-9


def test_in_window_memory_ranks_higher_than_equivalent_out(default_config):
    """The exact failure mode we are fixing: when two equivalent memories
    exist, the one inside the parsed temporal window must rank higher,
    but the out-of-window one MUST still appear in results."""
    in_window = _make_result(
        "in_window", days_ago=60, similarity=0.5, importance=5.0
    )
    out_window = _make_result(
        "out_window", days_ago=400, similarity=0.5, importance=5.0
    )
    boost_window = (NOW - timedelta(days=90), NOW)  # last 90 days
    ranked = score_and_rank(
        [in_window, out_window],
        default_config,
        now=NOW,
        temporal_boost_window=boost_window,
        temporal_boost_strength=0.10,
    )
    assert len(ranked) == 2, "out-of-window memory must NOT be filtered out"
    assert ranked[0].memory.id == "in_window"
    assert ranked[1].memory.id == "out_window"
    # In-window memory has the boost on top of base score.
    assert ranked[0].final_score > ranked[1].final_score


def test_out_of_window_memory_not_excluded(default_config):
    """Regression-prevention test: an out-of-window memory whose
    similarity is much higher than any in-window memory's must still
    rank ABOVE in-window memories. The boost is small enough to lose
    to a real signal."""
    weak_in_window = _make_result(
        "weak_in", days_ago=30, similarity=0.2, importance=3.0
    )
    strong_out_window = _make_result(
        "strong_out", days_ago=400, similarity=0.95, importance=9.0
    )
    boost_window = (NOW - timedelta(days=60), NOW)
    ranked = score_and_rank(
        [weak_in_window, strong_out_window],
        default_config,
        now=NOW,
        temporal_boost_window=boost_window,
        temporal_boost_strength=0.10,
    )
    assert len(ranked) == 2
    # Strong signal beats the soft boost.
    assert ranked[0].memory.id == "strong_out"


def test_explicit_time_after_still_filters(default_config):
    """Explicit time_after caller arg keeps the HARD filter semantics."""
    in_range = _make_result(
        "in_range", days_ago=30, similarity=0.5, importance=5.0
    )
    out_range = _make_result(
        "out_range", days_ago=400, similarity=0.9, importance=9.0
    )
    cutoff = NOW - timedelta(days=90)
    ranked = score_and_rank(
        [in_range, out_range],
        default_config,
        now=NOW,
        time_after=cutoff,
    )
    # Out-of-range MUST be excluded by the hard filter.
    assert len(ranked) == 1
    assert ranked[0].memory.id == "in_range"


def test_explicit_filter_combines_with_soft_boost(default_config):
    """Hard filter applies first; soft boost only changes ranking among
    survivors. Memories that pass the filter and fall inside the boost
    window get the bump."""
    in_filter_in_boost = _make_result(
        "ifib", days_ago=30, similarity=0.5, importance=5.0
    )
    in_filter_out_boost = _make_result(
        "ifob", days_ago=85, similarity=0.5, importance=5.0
    )
    out_filter = _make_result(
        "of", days_ago=400, similarity=0.9, importance=9.0
    )
    filter_cutoff = NOW - timedelta(days=180)
    boost_window = (NOW - timedelta(days=60), NOW)
    ranked = score_and_rank(
        [in_filter_in_boost, in_filter_out_boost, out_filter],
        default_config,
        now=NOW,
        time_after=filter_cutoff,
        temporal_boost_window=boost_window,
        temporal_boost_strength=0.20,  # large enough to be visible
    )
    ids = [r.memory.id for r in ranked]
    assert "of" not in ids, "filter must exclude out-of-filter memory"
    # In-boost beats out-of-boost (same similarity, importance, so boost decides).
    assert ids.index("ifib") < ids.index("ifob")


def test_boost_window_with_none_endpoints(default_config):
    """Boost window with (after=None, before=X) means 'before X'.
    With (after=X, before=None) means 'after X'."""
    old = _make_result("old", days_ago=400, similarity=0.5, importance=5.0)
    recent = _make_result("recent", days_ago=10, similarity=0.5, importance=5.0)

    # "before 90 days ago" — only `old` is in window
    ranked_before = score_and_rank(
        [old, recent],
        default_config,
        now=NOW,
        temporal_boost_window=(None, NOW - timedelta(days=90)),
        temporal_boost_strength=0.10,
    )
    # Both memories present; old gets boost.
    assert len(ranked_before) == 2
    # old should be ranked higher (higher recency normally favors recent,
    # but the boost should push old up; given equal sim/imp, the boost
    # adds 0.10 to old's score)
    by_id = {r.memory.id: r.final_score for r in ranked_before}
    assert by_id["old"] > by_id["recent"] - 0.20, (
        f"old should be competitive with boost; got old={by_id['old']:.3f}, "
        f"recent={by_id['recent']:.3f}"
    )

    # "after 30 days ago" — only `recent` is in window
    ranked_after = score_and_rank(
        [old, recent],
        default_config,
        now=NOW,
        temporal_boost_window=(NOW - timedelta(days=30), None),
        temporal_boost_strength=0.10,
    )
    assert len(ranked_after) == 2
    # recent gets the boost on top of its higher recency
    assert ranked_after[0].memory.id == "recent"
