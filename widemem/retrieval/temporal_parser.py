"""Parse temporal hints from natural-language queries into time-range filters.

Examples:
  "What happened in July 2023?"  -> (2023-07-01, 2023-07-31)
  "Last month?"                  -> (now - 30d, now)
  "Two years ago"                -> (now - 2y - 30d, now - 2y + 30d)
  "Before the camping trip"      -> (None, None)  (event-relative, can't resolve)
  "What is Alice's job?"         -> (None, None)  (not temporal)

Conservative by design. Returns (None, None) for any query we cannot confidently
bound. Better to leave the search unfiltered than over-filter and miss the answer.

Usage:
    from widemem.retrieval.temporal_parser import parse_temporal_hints
    time_after, time_before = parse_temporal_hints("What happened last July?")
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

# Conservative window for fuzzy references. "Two years ago" gets +/- 30 days
# rather than a single day, because users speaking in years rarely mean
# exactly 365 days.
FUZZY_DAYS = 30

MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

UNIT_DAYS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
}

# Patterns we recognize, in priority order. First match wins.

# "in May 2023" / "in July of 2023" / "in May" / "in 2023"
_MONTH_YEAR = re.compile(
    r"\bin\s+(?:the\s+(?:month\s+of\s+)?)?(\w+)(?:\s+(?:of\s+)?(\d{4}))?\b",
    re.IGNORECASE,
)

# "March 2024" / "March, 2024" without leading "in"
_MONTH_YEAR_BARE = re.compile(
    r"\b(\w+)[\s,]+(\d{4})\b",
    re.IGNORECASE,
)

# "on May 15", "on 15 May", "on May 15 2023", "on May 15, 2023"
_ON_DATE = re.compile(
    r"\bon\s+(?:(\w+)\s+(\d{1,2})|(\d{1,2})\s+(\w+))(?:[\s,]+(\d{4}))?\b",
    re.IGNORECASE,
)

# "last week", "last month", "last year"
_LAST_UNIT = re.compile(
    r"\blast\s+(week|weeks|month|months|year|years|day|days)\b",
    re.IGNORECASE,
)

# "two weeks ago", "3 months ago", "5 years ago", "yesterday", "a week ago"
_N_UNITS_AGO = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+"
    r"(day|days|week|weeks|month|months|year|years)\s+ago\b",
    re.IGNORECASE,
)

# "yesterday", "today", "tomorrow"
_RELATIVE_DAY = re.compile(r"\b(yesterday|today|tonight)\b", re.IGNORECASE)

# "in 2023" by itself (handled by _MONTH_YEAR but make explicit)
_YEAR_ONLY = re.compile(r"\bin\s+(\d{4})\b", re.IGNORECASE)

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a": 1, "an": 1,
}


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    next_month = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return (next_month - timedelta(days=1)).day


def _now() -> datetime:
    """Hook for tests to override."""
    return datetime.now(timezone.utc)


def _month_to_int(name: str) -> Optional[int]:
    return MONTH_NAMES.get(name.lower().strip(".,"))


def _to_int(token: str) -> Optional[int]:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return WORD_NUMBERS.get(token)


def parse_temporal_hints(
    query: str, now: Optional[datetime] = None
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Parse a natural-language query into (time_after, time_before).

    Returns (None, None) when no confident temporal bound is found.

    The parser is conservative: only emits filters when the language is
    explicit enough to bound. Event-relative references like "before the
    camping trip" cannot be resolved without context and return (None, None).
    """
    if now is None:
        now = _now()
    q = query.lower()

    # 1. Explicit "in <month> <year>" / "in <month>" / "in <year>"
    m = _MONTH_YEAR.search(q)
    if m:
        word, year_str = m.group(1), m.group(2)
        month = _month_to_int(word)
        year = int(year_str) if year_str else None
        if month and year:
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            end = datetime(
                year, month, _last_day_of_month(year, month),
                23, 59, 59, tzinfo=timezone.utc,
            )
            return start, end
        if month and not year:
            # Disambiguate: "in July" -> assume most recent July
            year_guess = now.year if now.month >= month else now.year - 1
            start = datetime(year_guess, month, 1, tzinfo=timezone.utc)
            end = datetime(
                year_guess, month, _last_day_of_month(year_guess, month),
                23, 59, 59, tzinfo=timezone.utc,
            )
            return start, end
        if year_str and year_str.isdigit() and len(year_str) == 4:
            year = int(year_str)
            return (
                datetime(year, 1, 1, tzinfo=timezone.utc),
                datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            )

    # 2. "March 2024" without leading "in"
    m = _MONTH_YEAR_BARE.search(q)
    if m:
        word, year_str = m.group(1), m.group(2)
        month = _month_to_int(word)
        if month and year_str:
            year = int(year_str)
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            end = datetime(
                year, month, _last_day_of_month(year, month),
                23, 59, 59, tzinfo=timezone.utc,
            )
            return start, end

    # 3. "on <date>"
    m = _ON_DATE.search(q)
    if m:
        # Either (month, day) or (day, month) groups depending on order
        month_str = m.group(1) or m.group(4)
        day_str = m.group(2) or m.group(3)
        year_str = m.group(5)
        month = _month_to_int(month_str) if month_str else None
        try:
            day = int(day_str) if day_str else None
        except (TypeError, ValueError):
            day = None
        if month and day and 1 <= day <= 31:
            year = int(year_str) if year_str else now.year
            try:
                target = datetime(year, month, day, tzinfo=timezone.utc)
                start = target.replace(hour=0, minute=0, second=0)
                end = target.replace(hour=23, minute=59, second=59)
                return start, end
            except ValueError:
                pass

    # 4. "last <unit>"
    m = _LAST_UNIT.search(q)
    if m:
        unit = m.group(1).lower()
        days = UNIT_DAYS[unit]
        # "last week" = the 7 days before now; same shape for month / year.
        end = now
        start = now - timedelta(days=days * 2)  # window through "last whatever"
        return start, end

    # 5. "<N> <units> ago"
    m = _N_UNITS_AGO.search(q)
    if m:
        n = _to_int(m.group(1))
        unit = m.group(2).lower()
        if n is not None and unit in UNIT_DAYS:
            offset_days = n * UNIT_DAYS[unit]
            center = now - timedelta(days=offset_days)
            return (
                center - timedelta(days=FUZZY_DAYS),
                center + timedelta(days=FUZZY_DAYS),
            )

    # 6. "yesterday" / "today" / "tonight"
    m = _RELATIVE_DAY.search(q)
    if m:
        word = m.group(1).lower()
        if word == "yesterday":
            target = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
            return (
                target,
                target.replace(hour=23, minute=59, second=59),
            )
        if word in ("today", "tonight"):
            return (
                now.replace(hour=0, minute=0, second=0),
                now.replace(hour=23, minute=59, second=59),
            )

    # 7. "in 2023" — year-only fallback (caught earlier but explicit)
    m = _YEAR_ONLY.search(q)
    if m:
        year = int(m.group(1))
        return (
            datetime(year, 1, 1, tzinfo=timezone.utc),
            datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        )

    return None, None


def looks_temporal(query: str) -> bool:
    """Cheap heuristic to detect temporal queries that may benefit from
    timestamp parsing. Used to gate parsing off when the query clearly
    is not about time."""
    q = query.lower()
    triggers = (
        "when ", "what time", "what date", "how long ago", "last ",
        "recently", "before the", "after the", "how recent",
        "what year", "what month", "what day", "yesterday",
        "today", "tonight", "ago", " in 19", " in 20",
    )
    return any(t in q for t in triggers) or any(
        m in q for m in MONTH_NAMES
    )
