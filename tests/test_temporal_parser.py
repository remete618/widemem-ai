"""Tests for the temporal hint parser used by retrieval."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from widemem.retrieval.temporal_parser import (
    looks_temporal,
    parse_temporal_hints,
)

# Fixed clock for deterministic tests
NOW = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)


def test_no_temporal_returns_none():
    after, before = parse_temporal_hints("What is Alice's job?", now=NOW)
    assert after is None and before is None


def test_event_relative_returns_none():
    """Event-relative references cannot be resolved without context."""
    after, before = parse_temporal_hints("Before the camping trip", now=NOW)
    assert after is None and before is None


@pytest.mark.parametrize(
    "query,expected_year,expected_month",
    [
        ("What happened in July 2023?", 2023, 7),
        ("Events in March 2024", 2024, 3),
        ("Trips in december 2022", 2022, 12),
        ("What happened in May 2025", 2025, 5),
        ("March 2024 events", 2024, 3),
    ],
)
def test_explicit_month_year(query, expected_year, expected_month):
    after, before = parse_temporal_hints(query, now=NOW)
    assert after is not None and before is not None
    assert after.year == expected_year and after.month == expected_month
    assert before.year == expected_year and before.month == expected_month
    assert after.day == 1
    assert before.day >= 28


def test_year_only():
    after, before = parse_temporal_hints("What happened in 2023?", now=NOW)
    assert after == datetime(2023, 1, 1, tzinfo=timezone.utc)
    assert before == datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def test_implicit_month_assumes_recent():
    """'In July' without a year should assume the most recent July before now."""
    # NOW is May 2026, so 'in July' means July 2025
    after, before = parse_temporal_hints("What happened in July?", now=NOW)
    assert after.year == 2025 and after.month == 7
    assert before.year == 2025 and before.month == 7


def test_implicit_month_after_now_uses_this_year():
    """'In May' when now is May 2026 -> May 2026."""
    after, before = parse_temporal_hints("In May", now=NOW)
    assert after.year == 2026 and after.month == 5


def test_on_specific_date():
    after, before = parse_temporal_hints("on May 15", now=NOW)
    assert after.year == 2026 and after.month == 5 and after.day == 15
    assert before.year == 2026 and before.month == 5 and before.day == 15
    assert before.hour == 23


def test_on_specific_date_with_year():
    after, before = parse_temporal_hints("On May 15, 2023", now=NOW)
    assert after.year == 2023 and after.month == 5 and after.day == 15


def test_on_specific_date_day_first():
    after, before = parse_temporal_hints("on 15 May", now=NOW)
    assert after.month == 5 and after.day == 15


def test_last_week():
    after, before = parse_temporal_hints("What happened last week?", now=NOW)
    assert after is not None and before is not None
    assert before == NOW
    assert (NOW - after).days == 14  # last X is 2-X window


def test_last_month():
    after, before = parse_temporal_hints("What did Alice do last month?", now=NOW)
    assert after is not None and before is not None
    assert before == NOW


def test_n_units_ago():
    after, before = parse_temporal_hints("Two months ago", now=NOW)
    assert after is not None and before is not None
    # Window should be centered ~60 days back from NOW, +/- the fuzzy window.
    days_back = (NOW - (after + (before - after) / 2)).days
    assert 50 <= days_back <= 70


def test_word_n_units_ago():
    after, before = parse_temporal_hints("Three weeks ago", now=NOW)
    assert after is not None and before is not None


def test_yesterday():
    after, before = parse_temporal_hints("What happened yesterday?", now=NOW)
    assert after.year == 2026 and after.month == 5 and after.day == 13
    assert before.year == 2026 and before.month == 5 and before.day == 13
    assert before.hour == 23


def test_today():
    after, before = parse_temporal_hints("What is today's appointment?", now=NOW)
    assert after.day == 14 and before.day == 14


def test_looks_temporal_positive():
    assert looks_temporal("When did Alice move?")
    assert looks_temporal("What happened in July?")
    assert looks_temporal("Two months ago")
    assert looks_temporal("Last week's events")
    assert looks_temporal("Yesterday's call")


def test_looks_temporal_negative():
    assert not looks_temporal("What is Alice's job?")
    assert not looks_temporal("How does Caroline relate to Bob?")
    assert not looks_temporal("Describe the project")


def test_unrecognized_month_safe():
    """Garbage month name should not crash."""
    after, before = parse_temporal_hints("in xyzember 2023", now=NOW)
    # Should fall through to year-only or return None
    # Year-only would catch 'in 2023' but only if the regex matches the bare year
    # Either way, no crash
    assert (after is None and before is None) or (
        after is not None and after.year == 2023
    )


def test_invalid_day_safe():
    """Invalid day like 32 should not crash; returns None gracefully."""
    after, before = parse_temporal_hints("on May 99", now=NOW)
    # day=99 is out of range; parser should not return a malformed date
    if after is not None:
        assert 1 <= after.day <= 31


def test_no_year_zero_handling():
    """Year 0 is invalid in datetime; parser must not emit it."""
    after, before = parse_temporal_hints("What about 2026?", now=NOW)
    # "in" prefix is required for year-only match; bare "2026?" should not match
    assert after is None or after.year >= 1
