"""Five-minute prices, kept so the dashboard can draw each ticker's day and mark the
trades on it. The pages are built from the store alone, so the prices have to be in it."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Protocol

from tradeloop import timeutil
from tradeloop.alpaca import Bar
from tradeloop.store import Store

TIMEFRAME, WIDTH = "5Min", timedelta(minutes=5)
FIRST_FETCH = timedelta(days=7)      # how far back the first fetch reaches, at most


class Bars(Protocol):
    def bars(self, symbols, timeframe: str, start: datetime, end: datetime) -> dict[str, list[Bar]]: ...


def update(store: Store, market: Bars, tickers: Iterable[str], now: datetime) -> int:
    """Fetch and keep the bars that have appeared since the newest one held; returns how many came back.

    The first fetch reaches back to the loop's first tick, a week at most. Each
    later one starts two bars before the newest held, so a bar that was still
    forming when it was last fetched is replaced by its final form.
    """
    latest = store.latest_bar_at()
    if latest:
        start = timeutil.parse(latest) - 2 * WIDTH
    else:
        first = store.one("SELECT MIN(scheduled_at) AS at FROM ticks")["at"]
        start = max(timeutil.parse(first) if first else now - WIDTH, now - FIRST_FETCH)
    if start >= now:
        return 0
    fetched = market.bars(list(tickers), TIMEFRAME, start, now)
    rows = [{"ticker": ticker, "at": timeutil.iso(bar.at), "open": bar.open, "high": bar.high, "low": bar.low,
             "close": bar.close, "volume": bar.volume} for ticker, bars in fetched.items() for bar in bars]
    store.add_bars(rows)
    return len(rows)
