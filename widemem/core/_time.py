from __future__ import annotations

from datetime import datetime, timezone


# Naive input is stamped UTC, not converted. Safe because utcnow() was widemem's only naive source. Don't pass local-time naive datetimes.
def as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
