"""Prices kept for the dashboard's charts: one-minute bars for the days the loop has
run and the week before them, and daily bars for the year before. The pages are built
from the store alone, so the prices have to be in it.

Alpaca's free plan gives every exchange's trades ("sip") once they are 15 minutes old,
and one exchange's ("iex") up to the moment. Bars are taken from the first as far as it
goes and from the second for the last minutes; the next fetch replaces those with the
full ones.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Protocol

from tradeloop import timeutil
from tradeloop.alpaca import Bar
from tradeloop.store import Store

MINUTE, DAY = "1Min", "1Day"
SETTLED_AFTER = timedelta(minutes=16)    # how old a bar must be for every exchange's version of it to be readable
FIRST_FETCH = {MINUTE: timedelta(days=7), DAY: timedelta(days=400)}
# How far behind the newest bar held each later fetch starts: far enough to replace
# the last minutes taken from one exchange, and a day bar that was still being made.
OVERLAP = {MINUTE: timedelta(minutes=45), DAY: timedelta(days=5)}


class Bars(Protocol):
    def bars(self, symbols, timeframe: str, start: datetime, end: datetime,
             feed: str | None = None) -> dict[str, list[Bar]]: ...


def update(store: Store, market: Bars, tickers: Iterable[str], now: datetime) -> int:
    """Fetch and keep the bars that have appeared since the newest held; returns how many came back."""
    tickers = list(tickers)
    return sum(_update(store, market, tickers, timeframe, now) for timeframe in (MINUTE, DAY))


def _update(store: Store, market: Bars, tickers: list[str], timeframe: str, now: datetime) -> int:
    latest = store.latest_bar_at(timeframe)
    start = timeutil.parse(latest) - OVERLAP[timeframe] if latest else now - FIRST_FETCH[timeframe]
    settled = now - SETTLED_AFTER
    fetched: dict[str, dict[datetime, Bar]] = {ticker: {} for ticker in tickers}

    def take(got: dict[str, list[Bar]]) -> None:
        for ticker, bars in got.items():
            fetched.setdefault(ticker, {}).update({bar.at: bar for bar in bars})

    if timeframe == MINUTE and start < now:
        take(market.bars(tickers, timeframe, max(start, settled), now))        # the last minutes, from one exchange
    if start < settled:
        try:
            take(market.bars(tickers, timeframe, start, settled, feed="sip"))  # every exchange, and it wins
        except Exception:  # noqa: BLE001 - a plan without that feed still has the other
            take(market.bars(tickers, timeframe, start, settled))
    rows = [{"ticker": ticker, "timeframe": timeframe, "at": timeutil.iso(bar.at), "open": bar.open, "high": bar.high,
             "low": bar.low, "close": bar.close, "volume": bar.volume}
            for ticker, bars in fetched.items() for bar in bars.values()]
    store.add_bars(rows)
    return len(rows)
