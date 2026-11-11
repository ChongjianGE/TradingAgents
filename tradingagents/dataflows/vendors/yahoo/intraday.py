"""Intraday price bars from Yahoo, for a run that needs to see inside the trading day.

The daily tools show a day only as one bar, and while the market is open that bar
is still forming. A run that decides during the session, or before the open after
overnight news, reads these to see how the price actually moved, including in
pre-market and after-hours trading.

Point in time: bars are served up to the end of the analysis date and no later.
Yahoo keeps intraday bars for a limited time, so a date past that is reported as
unavailable rather than answered from daily data.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import yfinance as yf

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import price_text, raise_for_empty, yf_retry

# Each bar size, and how many days back Yahoo serves it.
INTERVALS = {"5m": 60, "15m": 60, "30m": 60, "60m": 730}
MAX_LOOK_BACK_DAYS = 10
_REGULAR_OPEN, _REGULAR_CLOSE = time(9, 30), time(16, 0)


def _session(moment: datetime) -> str:
    """Which part of the US trading day a bar belongs to, by its exchange-local start time."""
    if moment.time() < _REGULAR_OPEN:
        return "pre-market"
    return "regular" if moment.time() < _REGULAR_CLOSE else "after-hours"


def get_intraday_bars(symbol: str, interval: str, look_back_days: int, as_of_date: str) -> str:
    """Intraday bars for ``symbol`` over the ``look_back_days`` up to and including ``as_of_date``."""
    if interval not in INTERVALS:
        raise ValueError(f"interval {interval!r} is not served; choose one of {', '.join(INTERVALS)}")
    end_day = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    look_back = max(1, min(int(look_back_days), MAX_LOOK_BACK_DAYS))
    start_day = end_day - timedelta(days=look_back)
    canonical = normalize_symbol(symbol)

    kept = INTERVALS[interval]
    if (date.today() - start_day).days > kept:
        raise NoMarketDataError(
            symbol, canonical,
            f"Yahoo keeps {interval} bars for {kept} days, and {as_of_date} is further back than that")

    data = yf_retry(lambda: yf.Ticker(canonical).history(
        start=start_day.isoformat(), end=(end_day + timedelta(days=1)).isoformat(),
        interval=interval, prepost=True))
    what = f"{interval} bars between {start_day} and {end_day}"
    if data is None or data.empty:
        raise_for_empty(symbol, canonical, what)

    # The index is in the exchange's own time; a bar belongs to the day it starts on there.
    data = data[[stamp.date() <= end_day for stamp in data.index]]
    if data.empty:
        raise_for_empty(symbol, canonical, what)

    us_hours = str(data.index.tz) == "America/New_York"
    lines = ["Datetime,Open,High,Low,Close,Volume" + (",Session" if us_hours else "")]
    for stamp, row in data.iterrows():
        cells = [f"{stamp:%Y-%m-%d %H:%M}", *(price_text(float(row[c])) for c in ("Open", "High", "Low", "Close")),
                 str(int(row["Volume"]))]
        if us_hours:
            cells.append(_session(stamp))
        lines.append(",".join(cells))

    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = (
        f"# Intraday {interval} bars for {label} from {start_day} to {end_day}\n"
        f"# Total records: {len(data)}. Times are the exchange's own ({data.index.tz}).\n"
        "# Bars outside the regular session are pre-market and after-hours trading, on thin volume.\n"
        "# The last bar may still be forming.\n\n"
    )
    return header + "\n".join(lines) + "\n"
