from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from widemem.core._time import as_utc

# Deterministic leading-date extraction. Conservative on purpose: a wrong
# event time is worse than no event time, so this only matches a date that
# is clearly at the very start of the text, either wrapped in [...] / (...)
# or as a leading ISO token. No freeform scanning of the body.

_BRACKET = re.compile(r"^\s*[\[(]\s*([^\])]{4,48}?)\s*[\])]")
_LEADING_ISO = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)")

_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p on %d %b, %Y",
    "%I:%M %p on %d %B %Y",
)


def _try_formats(token: str) -> Optional[datetime]:
    token = token.strip()
    for fmt in _FORMATS:
        try:
            return as_utc(datetime.strptime(token, fmt))
        except ValueError:
            continue
    return None


def parse_leading_datetime(text: Optional[str]) -> Optional[datetime]:
    """Return a tz-aware UTC datetime if text starts with a recognizable
    date token, else None. Never guesses."""
    if not text:
        return None
    m = _BRACKET.match(text)
    if m:
        return _try_formats(m.group(1))
    m = _LEADING_ISO.match(text)
    if m:
        return _try_formats(m.group(1))
    return None
