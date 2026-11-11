"""How the price moved after each decision. Every analysis gets four marks: an hour
on, the next close, and one and five trading days after that. They are prices and
returns, nothing more; whether a decision was right is read from them elsewhere,
against what the decision asked for.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

from tradeloop import timeutil
from tradeloop.alpaca import Bar
from tradeloop.schedule import Session
from tradeloop.store import Store

HORIZONS = ("60m", "close", "1d", "5d")
_BAR, _BAR_WIDTH = "15Min", timedelta(minutes=15)


class Bars(Protocol):
    def bars(self, symbols, timeframe: str, start: datetime, end: datetime) -> dict[str, list[Bar]]: ...


def plan(run_id: str, started: datetime, sessions: list[Session]) -> list[dict]:
    """The outcome rows of one analysis: each horizon and when its price is due.

    "close" is the first close after the analysis started; the day horizons count
    trading days from that close, so a holiday or a weekend lengthens none of them.
    A horizon past the known calendar is left out and can be planned again later.
    """
    rows = [{"run_id": run_id, "horizon": "60m", "due_at": timeutil.iso(started + timedelta(minutes=60))}]
    closes = [s.close for s in sessions if s.close > started]
    for horizon, index in (("close", 0), ("1d", 1), ("5d", 5)):
        if index < len(closes):
            rows.append({"run_id": run_id, "horizon": horizon, "due_at": timeutil.iso(closes[index])})
    return rows


def price_at(bars: list[Bar], moment: datetime) -> float | None:
    """The close of the last bar that had finished by ``moment``, from bars oldest first."""
    price = None
    for bar in bars:
        if bar.at + _BAR_WIDTH > moment:
            break
        price = bar.close
    return price


def mark_due(store: Store, market: Bars, now: datetime) -> int:
    """Price every outcome whose time has come; the number marked.

    One request per ticker covers all of its due outcomes. An outcome with no
    finished bar yet stays due and is tried again on a later tick.
    """
    due = store.due_outcomes(timeutil.iso(now - _BAR_WIDTH))
    by_ticker: dict[str, list[dict]] = {}
    for row in due:
        by_ticker.setdefault(row["ticker"], []).append(row)
    marked = 0
    for ticker, rows in by_ticker.items():
        moments = [timeutil.parse(r["due_at"]) for r in rows]
        # Far enough back to reach the last trade before a due time that falls
        # outside trading hours: a long weekend, or an hour after the close.
        bars = market.bars([ticker], _BAR, min(moments) - timedelta(days=5), max(moments)).get(ticker, [])
        for row, moment in zip(rows, moments, strict=True):
            price = price_at(bars, moment)
            if price is None:
                continue
            store.mark_outcome(row["run_id"], row["horizon"], price, price / row["ref_price"] - 1, timeutil.iso(now))
            marked += 1
    return marked
