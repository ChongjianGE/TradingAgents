"""Keeping five-minute prices for the charts: what is asked for, and what is kept."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradeloop import pricebars, timeutil
from tradeloop.alpaca import Bar
from tradeloop.store import Store

NOW = datetime(2026, 11, 11, 15, 37, tzinfo=UTC)


class Feed:
    def __init__(self, bars):
        self.held, self.asked = bars, []

    def bars(self, symbols, timeframe, start, end):
        self.asked.append((list(symbols), timeframe, start, end))
        return {s: [b for b in self.held.get(s, []) if start <= b.at <= end] for s in symbols}


def bar(minutes_ago, close):
    return Bar(NOW - timedelta(minutes=minutes_ago), close, close + 0.2, close - 0.2, close, 500.0)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "loop.db") as s:
        yield s


def _first_tick(store, at):
    store.begin_tick("t1", "core", timeutil.iso(at), timeutil.iso(at), "simulate")


@pytest.mark.unit
class TestKeepingBars:
    def test_the_first_fetch_reaches_back_to_the_loops_first_tick(self, store):
        _first_tick(store, NOW - timedelta(hours=3))
        feed = Feed({"NVDA": [bar(200, 99.0), bar(60, 100.0), bar(5, 101.0)], "CSCO": [bar(5, 50.0)]})
        assert pricebars.update(store, feed, ["NVDA", "CSCO"], NOW) == 3
        assert feed.asked == [(["NVDA", "CSCO"], "5Min", NOW - timedelta(hours=3), NOW)]
        assert store.query("SELECT ticker, close FROM bars ORDER BY ticker, at") == [
            {"ticker": "CSCO", "close": 50.0}, {"ticker": "NVDA", "close": 100.0}, {"ticker": "NVDA", "close": 101.0}]

    def test_but_no_further_than_a_week(self, store):
        _first_tick(store, NOW - timedelta(days=40))
        feed = Feed({})
        pricebars.update(store, feed, ["NVDA"], NOW)
        assert feed.asked[0][2] == NOW - timedelta(days=7)

    def test_a_later_fetch_starts_two_bars_before_the_newest_held(self, store):
        _first_tick(store, NOW - timedelta(hours=3))
        feed = Feed({"NVDA": [bar(60, 100.0), bar(7, 101.0)]})
        pricebars.update(store, feed, ["NVDA"], NOW)
        feed.held["NVDA"] = [bar(60, 100.0), bar(7, 101.6), bar(2, 102.0)]      # the 15:30 bar has since closed higher
        assert pricebars.update(store, feed, ["NVDA"], NOW + timedelta(minutes=5)) == 2
        assert feed.asked[1][2] == NOW - timedelta(minutes=7) - timedelta(minutes=10)
        assert [b["close"] for b in store.query("SELECT close FROM bars ORDER BY at")] == [100.0, 101.6, 102.0]

    def test_with_no_tick_and_no_bars_there_is_nothing_to_ask_for_yet(self, store):
        feed = Feed({"NVDA": [bar(2, 100.0)]})
        assert pricebars.update(store, feed, ["NVDA"], NOW) == 1      # the last five minutes, no more
        assert feed.asked[0][2] == NOW - timedelta(minutes=5)

    def test_a_feed_that_fails_is_the_callers_to_handle(self, store):
        class Down:
            def bars(self, *args):
                raise ConnectionError("no route")

        _first_tick(store, NOW - timedelta(hours=1))
        with pytest.raises(ConnectionError):
            pricebars.update(store, Down(), ["NVDA"], NOW)
        assert store.latest_bar_at() is None
