"""Times in the loop: aware datetimes in code, UTC ISO strings on disk.

A UTC ISO string with a fixed shape sorts as text in time order, so the store
can compare and index times without parsing them.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

_EXCESS_DIGITS = re.compile(r"(\.\d{6})\d+")


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(moment: datetime) -> str:
    """``moment`` as ``2026-11-11T15:35:00Z``. A naive datetime is refused: its zone is a guess."""
    if moment.tzinfo is None:
        raise ValueError("a naive datetime has no zone; pass an aware one")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(text: str) -> datetime:
    """An aware UTC datetime from an ISO string, with or without fractional seconds."""
    # Alpaca sends nanoseconds; datetime reads at most microseconds.
    cleaned = _EXCESS_DIGITS.sub(r"\1", text.strip()).replace("Z", "+00:00")
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)
