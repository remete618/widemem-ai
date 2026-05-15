"""Tests for the conservative leading-date parser used at ingest."""

from __future__ import annotations

from datetime import timezone

import pytest

from widemem.extraction.datetime_parse import parse_leading_datetime


@pytest.mark.parametrize("text,expected", [
    ("[2023-05-08] Caroline went to the group", (2023, 5, 8)),
    ("[2023-05-08T13:56] message", (2023, 5, 8)),
    ("[8 May 2023] hello", (2023, 5, 8)),
    ("[May 8, 2023] hello", (2023, 5, 8)),
    ("(2023-05-08) hello", (2023, 5, 8)),
    ("[1:56 pm on 8 May, 2023] Hey Mel!", (2023, 5, 8)),
    ("2023-05-08 plain leading iso", (2023, 5, 8)),
    ("2023-05-08T13:56:00 leading iso ts", (2023, 5, 8)),
])
def test_parses_leading_date(text, expected):
    dt = parse_leading_datetime(text)
    assert dt is not None
    assert (dt.year, dt.month, dt.day) == expected
    assert dt.tzinfo is not None
    assert dt.utcoffset() == timezone.utc.utcoffset(None)


@pytest.mark.parametrize("text", [
    "",
    None,
    "Caroline likes hiking",
    "We met on 8 May 2023 at the cafe",        # date not leading
    "[not a date] hello",
    "[hi] hello",
    "yesterday we went out",
    "[2023-13-45] impossible date",
    "The year was 2023 and all was well",
])
def test_returns_none_when_no_clear_leading_date(text):
    assert parse_leading_datetime(text) is None
