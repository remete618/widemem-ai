"""Unit tests for WideMemory._adapt_scoring().

Documents and locks in the v1.4 query-type detection that drives per-query
scoring weight selection. Each test asserts the three weight components and
the similarity_first flag for a representative query of its category.

The current behavior in plain terms:
  - Temporal queries  -> recency-heavy   (sim 0.40, imp 0.10, rec 0.50, sim_first False)
  - Factual queries   -> similarity-heavy (sim 0.75, imp 0.10, rec 0.15, sim_first True)
  - Multi-hop queries -> caller defaults  (default config returned, sim_first False)
  - Broad / unknown   -> caller defaults  (default config returned, sim_first False)

Tests here are pure coverage. They do not change behavior. If any test
starts failing, it means _adapt_scoring's contract changed; update the
test and bump the mini-LoCoMo baseline together.
"""

from __future__ import annotations

import pytest

from widemem.core.memory import WideMemory
from widemem.core.types import ScoringConfig


@pytest.fixture
def default_config() -> ScoringConfig:
    return ScoringConfig(
        decay_function="exponential",
        decay_rate=0.01,
        similarity_weight=0.5,
        importance_weight=0.3,
        recency_weight=0.2,
    )


def _is_temporal_profile(cfg: ScoringConfig) -> bool:
    return (
        cfg.similarity_weight == 0.40
        and cfg.importance_weight == 0.10
        and cfg.recency_weight == 0.50
    )


def _is_factual_profile(cfg: ScoringConfig) -> bool:
    return (
        cfg.similarity_weight == 0.75
        and cfg.importance_weight == 0.10
        and cfg.recency_weight == 0.15
    )


def _is_default_profile(cfg: ScoringConfig, default: ScoringConfig) -> bool:
    return (
        cfg.similarity_weight == default.similarity_weight
        and cfg.importance_weight == default.importance_weight
        and cfg.recency_weight == default.recency_weight
    )


# ---------------------------------------------------------------------------
# Temporal classification
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
        "After the meeting where did she go",
        "How recent is that record",
        "What year did she graduate?",
        "What month is the deadline?",
    ],
)
def test_temporal_queries_get_recency_boost(query, default_config):
    cfg, sim_first = WideMemory._adapt_scoring(query, default_config)
    assert _is_temporal_profile(cfg), f"Query failed to classify as temporal: {query!r}"
    assert sim_first is False


def test_temporal_preserves_decay_settings(default_config):
    """Temporal classification keeps the caller's decay function/rate."""
    cfg, _ = WideMemory._adapt_scoring("When did this happen?", default_config)
    assert cfg.decay_function == default_config.decay_function
    assert cfg.decay_rate == default_config.decay_rate


# ---------------------------------------------------------------------------
# Factual classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "Where does Alice live?",
        "Who is Caroline?",
        "What is her job?",
        "What was the address?",
        "What does she do for work?",
        "What did he say?",
        "What are her hobbies?",
        "What were the names?",
        "What do they prefer?",
        "How old is Alice?",
        "How much was the bill?",
        "How many siblings does she have?",
        "Which school did she attend?",
        "Name her parents",
        "Is she married?",
        "Was the meeting yesterday?",
        "Does she have a dog?",
        "Did they go?",
        "Has she finished?",
        "Have they met?",
    ],
)
def test_factual_queries_get_similarity_boost(query, default_config):
    cfg, sim_first = WideMemory._adapt_scoring(query, default_config)
    assert _is_factual_profile(cfg), f"Query failed to classify as factual: {query!r}"
    assert sim_first is True


def test_short_what_query_treated_as_factual(default_config):
    """A 'what ...' query under 10 words is factual even without an explicit subform."""
    cfg, sim_first = WideMemory._adapt_scoring(
        "What about her childhood pets", default_config
    )
    assert _is_factual_profile(cfg)
    assert sim_first is True


def test_long_what_query_not_factual(default_config):
    """A 'what ...' query over 10 words is NOT auto-classified as factual."""
    long_what = (
        "What can you tell me about everything that happened "
        "during her childhood and her schooling"
    )
    assert len(long_what.split()) > 10
    cfg, sim_first = WideMemory._adapt_scoring(long_what, default_config)
    assert _is_default_profile(cfg, default_config)
    assert sim_first is False


# ---------------------------------------------------------------------------
# Multi-hop classification (overrides factual)
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
def test_multi_hop_overrides_factual(query, default_config):
    """Multi-hop signals win over factual starts: returns default config, sim_first False."""
    cfg, sim_first = WideMemory._adapt_scoring(query, default_config)
    assert _is_default_profile(cfg, default_config), (
        f"Multi-hop query incorrectly classified as factual: {query!r}"
    )
    assert sim_first is False


# ---------------------------------------------------------------------------
# Default fall-through
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
def test_broad_queries_get_default_weights(query, default_config):
    cfg, sim_first = WideMemory._adapt_scoring(query, default_config)
    assert _is_default_profile(cfg, default_config)
    assert sim_first is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_query_returns_default(default_config):
    cfg, sim_first = WideMemory._adapt_scoring("", default_config)
    assert _is_default_profile(cfg, default_config)
    assert sim_first is False


def test_whitespace_only_query_returns_default(default_config):
    cfg, sim_first = WideMemory._adapt_scoring("    \n\t  ", default_config)
    assert _is_default_profile(cfg, default_config)
    assert sim_first is False


def test_mixed_case_temporal_detection(default_config):
    """Detection should be case-insensitive."""
    cfg, sim_first = WideMemory._adapt_scoring("WHEN did Alice move?", default_config)
    assert _is_temporal_profile(cfg)
    assert sim_first is False


def test_mixed_case_factual_detection(default_config):
    cfg, sim_first = WideMemory._adapt_scoring("WHERE does Alice live?", default_config)
    assert _is_factual_profile(cfg)
    assert sim_first is True


def test_caller_default_passes_through_unchanged(default_config):
    """When the query doesn't match a category, the exact ScoringConfig instance
    passed in is returned (not a copy). Tested because copying would silently
    drop any custom attributes a future ScoringConfig might add."""
    cfg, _ = WideMemory._adapt_scoring("describe the project", default_config)
    assert cfg is default_config


def test_factual_classification_creates_new_config(default_config):
    """Factual / temporal branches must NOT mutate the caller's default config."""
    cfg, _ = WideMemory._adapt_scoring("Where does Alice live?", default_config)
    assert cfg is not default_config
    # Original config untouched
    assert default_config.similarity_weight == 0.5
    assert default_config.importance_weight == 0.3
    assert default_config.recency_weight == 0.2
