"""Keeping prices for the charts: what is asked of which feed, and what is kept."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradeloop import pricebars, timeutil
from tradeloop.alpaca import AlpacaError, Bar
from tradeloop.store import Store

NOW = datetime(2026, 11, 11, 15, 37, tzinfo=UTC)
SETTLED = NOW - timedelta(minutes=16)


class Feed:
    """Two feeds over the same clock: "sip" has every bar up to 15 minutes ago, the default has them all, thinner."""

    def __init__(self, minute=(), day=(), sip=True):
        self.minute, self.day, self.sip, self.asked = list(minute), list(day), sip, []

    def bars(self, symbols, timeframe, start, end, feed=None):
        self.asked.append((timeframe, feed, start, end))
        if feed == "sip" and not self.sip:
            raise AlpacaError(403, "GET /v2/stocks/bars: subscription does not permit querying recent SIP data")
        held = self.day if timeframe == "1Day" else self.minute
        scale = 1.0 if feed == "sip" else 0.1          # one exchange sees a tenth of the volume
        return {s: [Bar(b.at, b.open, b.high, b.low, b.close, b.volume * scale) for b in held if start <= b.at <= end]
                for s in symbols}


def bar(minutes_ago, close, volume=1000.0):
    return Bar(NOW - timedelta(minutes=minutes_ago), close, close + 0.2, close - 0.2, close, volume)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "loop.db") as s:
        yield s


def kept(store, timeframe="1Min"):
    return [(timeutil.parse(b["at"]), b["close"], b["volume"]) for b in store.bars(timeframe)]


@pytest.mark.unit
class TestKeepingBars:
    def test_the_first_fetch_takes_a_week_of_minutes_and_a_year_of_days(self, store):
        feed = Feed(minute=[bar(60, 100.0)], day=[bar(60 * 24 * 30, 90.0)])
        assert pricebars.update(store, feed, ["NVDA"], NOW) == 2
        starts = {(timeframe, source): start for timeframe, source, start, _ in feed.asked}
        assert starts[("1Min", "sip")] == NOW - timedelta(days=7) and starts[("1Day", "sip")] == NOW - timedelta(days=400)
        assert [c for _, c, _ in kept(store)] == [100.0] and [c for _, c, _ in kept(store, "1Day")] == [90.0]

    def test_every_exchanges_bars_are_used_as_far_as_they_go_and_one_exchanges_for_the_last_minutes(self, store):
        feed = Feed(minute=[bar(40, 100.0), bar(20, 101.0), bar(10, 102.0), bar(2, 103.0)])
        pricebars.update(store, feed, ["NVDA"], NOW)
        asked = {(timeframe, source): (start, end) for timeframe, source, start, end in feed.asked}
        assert asked[("1Min", "sip")][1] == SETTLED and asked[("1Min", None)] == (SETTLED, NOW)
        assert [(c, v) for _, c, v in kept(store)] == [(100.0, 1000.0), (101.0, 1000.0), (102.0, 100.0), (103.0, 100.0)]

    def test_the_next_fetch_replaces_the_thin_bars_with_the_full_ones(self, store):
        feed = Feed(minute=[bar(40, 100.0), bar(10, 102.0)])
        pricebars.update(store, feed, ["NVDA"], NOW)
        assert kept(store)[-1][2] == 100.0                           # ten minutes old: one exchange's
        later = NOW + timedelta(minutes=30)
        pricebars.update(store, feed, ["NVDA"], later)
        assert [v for _, _, v in kept(store)] == [1000.0, 1000.0]
        again = [(start, end) for timeframe, source, start, end in feed.asked if (timeframe, source) == ("1Min", "sip")][-1]
        assert again == (NOW - timedelta(minutes=10) - timedelta(minutes=45), later - timedelta(minutes=16))

    def test_a_plan_without_every_exchange_falls_back_to_the_one(self, store):
        feed = Feed(minute=[bar(40, 100.0), bar(2, 103.0)], sip=False)
        assert pricebars.update(store, feed, ["NVDA"], NOW) == 2
        assert [(c, v) for _, c, v in kept(store)] == [(100.0, 100.0), (103.0, 100.0)]

    def test_daily_bars_are_refreshed_from_a_few_days_back(self, store):
        feed = Feed(day=[bar(60 * 24 * 3, 90.0), bar(60 * 24 * 2, 91.0)])
        pricebars.update(store, feed, ["NVDA"], NOW)
        feed.day[-1] = bar(60 * 24 * 2, 91.5)                        # the day's bar, now final
        pricebars.update(store, feed, ["NVDA"], NOW + timedelta(hours=1))
        assert [c for _, c, _ in kept(store, "1Day")] == [90.0, 91.5]
        day_asks = [start for timeframe, _, start, _ in feed.asked if timeframe == "1Day"]
        assert day_asks[-1] == NOW - timedelta(days=2) - timedelta(days=5)

    def test_a_feed_that_is_down_is_the_callers_to_handle(self, store):
        class Down:
            def bars(self, *args, **kwargs):
                raise ConnectionError("no route")

        with pytest.raises(ConnectionError):
            pricebars.update(store, Down(), ["NVDA"], NOW)
        assert store.latest_bar_at("1Min") is None
