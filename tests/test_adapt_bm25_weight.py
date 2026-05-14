"""Tests for WideMemory._adapt_bm25_weight, the per-category BM25 weight scaler.

Verifies the fix for the v1.5 regression where hybrid BM25 was diluting
the importance-weighted signal that made multi-hop strong. The v1.6
behavior scales bm25_weight by detected query type: disabled for
multi-hop, full for factual, reduced for temporal and broad.

Locks in the contract:
  multi-hop  -> 0.0  (BM25 disabled)
  factual    -> configured_weight  (full)
  temporal   -> configured_weight * 0.4  (reduced)
  broad      -> configured_weight * 0.6  (moderate)
"""

from __future__ import annotations

import pytest

from widemem.core.memory import WideMemory

DEFAULT_W = 0.5  # the typical configured weight


# ---------------------------------------------------------------------------
# Multi-hop: BM25 disabled (0.0)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "How does Alice relate to Bob?",
        "Compare Alice and Bob",
        "Contrast the two trips",
        "What is the relationship between Alice and Bob?",
        "Connection between her job and her move",
        "What do they have in common?",
    ],
)
def test_multi_hop_queries_return_zero(query):
    assert WideMemory._adapt_bm25_weight(query, DEFAULT_W) == 0.0


# ---------------------------------------------------------------------------
# Factual: full configured weight
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "Where does Alice live?",
        "Who is Caroline?",
        "What is her job?",
        "What was the address?",
        "What does she do for work?",
        "How old is Alice?",
        "How many siblings does she have?",
        "Which school did she attend?",
        "Name her parents",
    ],
)
def test_factual_queries_return_full_weight(query):
    assert WideMemory._adapt_bm25_weight(query, DEFAULT_W) == DEFAULT_W


def test_short_what_query_factual():
    """'what ...' under 10 words is factual."""
    assert WideMemory._adapt_bm25_weight(
        "What about her hobbies", DEFAULT_W
    ) == DEFAULT_W


# ---------------------------------------------------------------------------
# Temporal: reduced weight (0.4 * configured)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "When did Alice move?",
        "What time was the meeting?",
        "What date is the deadline?",
        "How long ago was that?",
        "Last time we met",
        "Recently she changed jobs",
        "Before the camping trip what happened",
        "How recent is that record",
        "What year did she graduate?",
        "What month is the deadline?",
    ],
)
def test_temporal_queries_get_reduced_weight(query):
    expected = DEFAULT_W * 0.4
    actual = WideMemory._adapt_bm25_weight(query, DEFAULT_W)
    assert abs(actual - expected) < 1e-9, (
        f"temporal query {query!r}: got {actual}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Broad / unknown: moderate weight (0.6 * configured)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "Tell me about her hobbies",
        "Describe the project",
        "Summarize what we know about the move",
        "Give me an overview of her career",
    ],
)
def test_broad_queries_get_moderate_weight(query):
    expected = DEFAULT_W * 0.6
    actual = WideMemory._adapt_bm25_weight(query, DEFAULT_W)
    assert abs(actual - expected) < 1e-9, (
        f"broad query {query!r}: got {actual}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Override precedence: multi-hop signals beat factual starts
# ---------------------------------------------------------------------------
def test_multi_hop_overrides_factual_start():
    """Query starts with 'how' (factual-shaped) but contains 'how does'
    (multi-hop signal). Multi-hop wins."""
    assert WideMemory._adapt_bm25_weight(
        "How does Alice relate to Bob", DEFAULT_W
    ) == 0.0


def test_multi_hop_overrides_short_what():
    """Short 'what' query that contains 'in common' is multi-hop."""
    assert WideMemory._adapt_bm25_weight(
        "What do they have in common?", DEFAULT_W
    ) == 0.0


# ---------------------------------------------------------------------------
# Weight scaling behavior
# ---------------------------------------------------------------------------
def test_zero_configured_weight_returns_zero_always():
    """If user explicitly sets hybrid_bm25_weight=0, BM25 is always off."""
    assert WideMemory._adapt_bm25_weight("Where does Alice live?", 0.0) == 0.0
    assert WideMemory._adapt_bm25_weight("When did Alice move?", 0.0) == 0.0
    assert WideMemory._adapt_bm25_weight("Describe the project", 0.0) == 0.0


def test_high_configured_weight_scales_through():
    """If user sets hybrid_bm25_weight=0.8, factual queries get 0.8,
    temporal 0.32, broad 0.48."""
    assert WideMemory._adapt_bm25_weight("Where does Alice live?", 0.8) == 0.8
    assert (
        abs(WideMemory._adapt_bm25_weight("When did Alice move?", 0.8) - 0.32)
        < 1e-9
    )
    assert (
        abs(WideMemory._adapt_bm25_weight("Describe project", 0.8) - 0.48)
        < 1e-9
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_query_returns_broad_weight():
    """Empty query is broad/unknown."""
    expected = DEFAULT_W * 0.6
    assert abs(WideMemory._adapt_bm25_weight("", DEFAULT_W) - expected) < 1e-9


def test_whitespace_only_returns_broad_weight():
    expected = DEFAULT_W * 0.6
    assert (
        abs(WideMemory._adapt_bm25_weight("   \n\t  ", DEFAULT_W) - expected)
        < 1e-9
    )


def test_mixed_case_detection():
    """Detection should be case-insensitive."""
    assert WideMemory._adapt_bm25_weight("WHERE does Alice live?", DEFAULT_W) == DEFAULT_W
    assert WideMemory._adapt_bm25_weight("HOW DOES Alice relate to Bob", DEFAULT_W) == 0.0
