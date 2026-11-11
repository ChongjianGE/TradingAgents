"""Whole ticks, every module together: a scripted market and scripted analyses on one
side, the real store, scheduler, gate, risk rules, executor and simulated book on the other."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tradeloop import timeutil
from tradeloop.alpaca import AlpacaError, Bar, Quote
from tradeloop.analysis import AnalysisResult
from tradeloop.broker import OrderState, SimBroker
from tradeloop.config import LoopConfig
from tradeloop.engine import Engine, run_forever
from tradeloop.schedule import Scheduler, Tick
from tradeloop.store import Store

ET = ZoneInfo("America/New_York")
TICKERS = ["NVDA", "CSCO", "HD", "DIS", "GEV"]
CALENDAR = [{"date": f"2026-11-{d:02d}", "open": "09:30", "close": "16:00"} for d in (10, 11, 12, 13, 16, 17, 18, 19, 20)]


def et(day, hhmm):
    return datetime.fromisoformat(f"2026-11-{day:02d}T{hhmm}").replace(tzinfo=ET)


class FakeMarket:
    def __init__(self):
        self.prices = dict.fromkeys(TICKERS, 100.0)
        self.headlines: dict[str, list[datetime]] = {}
        self.series: dict[str, list[Bar]] = {}
        self.down = False
        self.quote_calls = 0

    def quotes(self, symbols):
        self.quote_calls += 1
        if self.down:
            raise AlpacaError(0, "GET /v2/stocks/snapshots did not reach Alpaca (ConnectionError)")
        return {s: Quote(s, self.prices[s], et(11, "09:30"), 99.0) for s in symbols if s in self.prices}

    def news(self, symbols, start=None, limit=50):
        return [{"id": i, "at": at, "headline": "h", "symbols": list(symbols)}
                for i, at in enumerate(self.headlines.get(symbols[0], [])) if start is None or at > start]

    def bars(self, symbols, timeframe, start, end):
        return {s: [b for b in self.series.get(s, []) if start <= b.at <= end] for s in symbols}


class FakeAnalysts:
    """Answers each request from a per-ticker script; a ticker with no script left says Hold."""

    def __init__(self, clock):
        self.clock, self.script, self.calls, self.takes = clock, {}, [], timedelta(0)

    def say(self, ticker, *answers):
        self.script.setdefault(ticker, []).extend(answers)

    def run_many(self, requests):
        self.calls.append(list(requests))
        self.clock[0] += self.takes
        results = []
        for request in requests:
            answer = self.script[request.ticker].pop(0) if self.script.get(request.ticker) else "Hold"
            if isinstance(answer, Exception):
                results.append(AnalysisResult(run_id=request.run_id, ticker=request.ticker, status="error",
                                              error=f"{type(answer).__name__}: {answer}"))
                continue
            rating, fields = answer if isinstance(answer, tuple) else (answer, {})
            results.append(AnalysisResult(
                run_id=request.run_id, ticker=request.ticker, status="ok", rating=rating,
                decision={"rating": rating, "executive_summary": f"{rating} it", **fields}, summary=f"{rating} it",
                elapsed_seconds=380.0, input_tokens=190_000, output_tokens=32_000, cost_usd=1.0,
                llm_calls=17, tool_calls=28, recorded_to_memory=request.record_to_memory,
                steps=[{"seq": 0, "agent": "Market Analyst", "kind": "tool", "name": "get_stock_data",
                        "started_at": timeutil.iso(self.clock[0]), "elapsed_seconds": 0.4, "input_tokens": 0,
                        "output_tokens": 0, "output_chars": 2992, "error": None}]))
        return results


class Loop:
    def __init__(self, tmp_path, broker=SimBroker, **settings):
        self.cfg = LoopConfig(data_dir=tmp_path / "loop", tickers=TICKERS, **settings)
        self.store = Store(self.cfg.db_path)
        self.clock = [et(11, "08:45")]
        self.market, self.analysts = FakeMarket(), FakeAnalysts(self.clock)
        self.broker = broker(self.store, self.cfg)
        self.scheduler = Scheduler(lambda start, end: CALENDAR, self.cfg)
        self.built, self.slept = [], []
        self.engine = Engine(self.cfg, self.store, self.broker, self.market, self.analysts, self.scheduler,
                             now=lambda: self.clock[0], today=lambda: self.clock[0].astimezone(ET).date(),
                             after_tick=lambda: self.built.append(1), sleep=self.slept.append)

    def tick(self, kind, day, hhmm):
        self.clock[0] = et(day, hhmm)
        return self.engine.run_tick(Tick(et(day, hhmm), kind))

    def targets(self, tick_id):
        return {r["ticker"]: r for r in self.store.query("SELECT * FROM targets WHERE tick_id = ?", (tick_id,))}

    def analysed(self, call=-1):
        return [r.ticker for r in self.analysts.calls[call]]

    def qty(self, ticker="NVDA"):
        return self.broker.account(self.market.prices).qty(ticker)


@pytest.fixture
def loop(tmp_path):
    made = Loop(tmp_path)
    yield made
    made.store.close()


@pytest.mark.unit
class TestThePreMarketTick:
    def test_every_ticker_is_analysed_at_once_and_nothing_is_ordered(self, loop):
        loop.analysts.say("NVDA", ("Buy", {"target_weight": 0.5, "stop_loss": 92.0, "price_target": 120.0}))
        report = loop.tick("premarket", 11, "08:45")

        assert (report.status, report.analysed, report.failed, report.orders) == ("ok", TICKERS, [], 0)
        assert len(loop.analysts.calls) == 1 and loop.analysed() == TICKERS      # one batch of five
        tick = loop.store.tick("20261111T0845-premarket")
        assert tick["status"] == "ok" and tick["execution"] == "simulate" and "5 analysed" in tick["note"]

        (nvda,) = loop.store.query("SELECT * FROM analyses WHERE ticker = 'NVDA'")
        assert (nvda["rating"], nvda["target_weight"], nvda["stop_loss"], nvda["price_target"]) == ("Buy", 0.5, 92.0, 120.0)
        assert (nvda["kind"], nvda["ref_price"], nvda["cost_usd"], nvda["tool_calls"]) == ("premarket", 100.0, 1.0, 28)
        assert nvda["run_id"] == "20261111T0845-premarket-NVDA" and nvda["trade_date"] == "2026-11-11"
        assert loop.store.query("SELECT * FROM orders") == []

    def test_the_decision_is_sized_and_ruled_on_but_deferred(self, loop):
        loop.analysts.say("NVDA", ("Buy", {"target_weight": 0.5}))
        loop.tick("premarket", 11, "08:45")
        nvda = loop.targets("20261111T0845-premarket")["NVDA"]
        assert nvda["verdict"] == "deferred:blocked" and nvda["raw_weight"] == pytest.approx(0.1)
        assert [c["name"] for c in nvda["checks"] if not c["passed"]] == ["confirmation", "trading_window"]
        hold = loop.targets("20261111T0845-premarket")["CSCO"]
        assert hold["verdict"] == "deferred:keep" and hold["raw_weight"] is None

    def test_what_was_seen_is_recorded_with_the_gates_reason(self, loop):
        loop.tick("premarket", 11, "08:45")
        rows = loop.store.query("SELECT * FROM observations ORDER BY ticker")
        assert len(rows) == 5 and {r["gate_action"] for r in rows} == {"analyze"}
        assert rows[0]["gate_reasons"] == ["scheduled premarket analysis"] and rows[0]["price"] == 100.0
        (snapshot,) = loop.store.query("SELECT * FROM snapshots")
        assert (snapshot["equity"], snapshot["cash"], snapshot["positions"]) == (100_000.0, 100_000.0, [])

    def test_each_request_carries_the_book_the_time_and_the_date(self, loop):
        loop.tick("premarket", 11, "08:45")
        request = loop.analysts.calls[0][0]
        assert request.trade_date == "2026-11-11" and request.as_of_time == "2026-11-11 08:45 EST (before the open)"
        assert request.portfolio.cash == 100_000.0 and request.portfolio.positions == []
        assert request.record_to_memory is False
        told = request.session_context          # nothing happened yet this session; the size of a position is still said
        assert told.startswith("How a target weight becomes a position in this account:")
        assert "A full position is 20% of the account's equity: about $20,000 now, about 200 shares of NVDA at 100.00." in told
        assert "Held now: nothing" in told and "Earlier in this session" not in told

    def test_each_analysis_gets_its_outcome_marks_planned(self, loop):
        loop.tick("premarket", 11, "08:45")
        rows = loop.store.query("SELECT horizon, due_at FROM outcomes WHERE run_id = '20261111T0845-premarket-NVDA'")
        assert {r["horizon"]: r["due_at"] for r in rows} == {
            "60m": "2026-11-11T14:45:00Z", "close": "2026-11-11T21:00:00Z",
            "1d": "2026-11-12T21:00:00Z", "5d": "2026-11-18T21:00:00Z"}

    def test_the_after_tick_step_runs_and_its_failure_is_not_the_ticks(self, loop):
        loop.tick("premarket", 11, "08:45")
        assert loop.built == [1]
        loop.engine._after_tick = lambda: 1 / 0
        assert loop.tick("core", 11, "09:35").status == "ok"


@pytest.mark.unit
class TestTheSession:
    def _confirmed_buy(self, loop, **fields):
        """A Buy before the open, then a second one after a move at 10:35 confirms it."""
        loop.analysts.say("NVDA", ("Buy", fields), ("Buy", fields))
        loop.tick("premarket", 11, "08:45")
        loop.tick("core", 11, "09:35")
        loop.market.prices["NVDA"] = 102.0
        return loop.tick("core", 11, "10:35")

    def test_a_quiet_hour_calls_no_model_and_orders_nothing(self, loop):
        loop.analysts.say("NVDA", "Buy")
        loop.tick("premarket", 11, "08:45")
        report = loop.tick("core", 11, "09:35")
        assert report.analysed == [] and report.orders == 0 and len(loop.analysts.calls) == 1
        nvda = loop.targets("20261111T0935-core")["NVDA"]
        assert nvda["verdict"] == "blocked" and "1 so far" in nvda["checks"][0]["detail"]
        assert {r["gate_action"] for r in loop.store.query(
            "SELECT gate_action FROM observations WHERE tick_id = '20261111T0935-core'")} == {"reuse"}

    def test_a_move_triggers_one_tickers_analysis_and_a_second_buy_is_executed(self, loop):
        report = self._confirmed_buy(loop)
        assert report.analysed == ["NVDA"] and loop.analysed() == ["NVDA"] and report.orders == 1
        (order,) = loop.store.query("SELECT * FROM orders")
        # A full position is 20% of 100,000 at the price after the analysis: 196 shares at 102.
        assert (order["side"], order["qty"], order["type"], order["limit_price"], order["status"]) == (
            "buy", 196.0, "limit", 102.1, "filled")
        assert order["client_order_id"] == "tl-20261111T1035-core-NVDA-buy"
        assert loop.qty() == 196.0
        nvda = loop.targets("20261111T1035-core")["NVDA"]
        assert nvda["verdict"] == "increase" and all(c["passed"] for c in nvda["checks"])
        (obs,) = loop.store.query("SELECT gate_reasons FROM observations WHERE tick_id = '20261111T1035-core' "
                                  "AND ticker = 'NVDA'")
        assert obs["gate_reasons"] == ["price moved +2.0% since the last analysis"]

    def test_prices_are_read_again_after_the_analysis(self, loop):
        loop.analysts.say("NVDA", "Buy", "Buy")
        loop.tick("premarket", 11, "08:45")
        before = loop.market.quote_calls
        loop.market.prices["NVDA"] = 102.0
        loop.tick("core", 11, "10:35")
        assert loop.market.quote_calls == before + 2

    def test_the_second_analysis_is_told_what_the_first_decided(self, loop):
        self._confirmed_buy(loop)
        context = loop.analysts.calls[-1][0].session_context
        assert "- 08:45 premarket analysis rated it Buy" in context and "say what has changed" in context

    def test_a_later_analysis_is_told_what_was_traded_and_sees_the_position(self, loop):
        self._confirmed_buy(loop)
        loop.market.prices["NVDA"] = 104.5
        loop.tick("core", 11, "11:35")
        request = loop.analysts.calls[-1][0]
        assert "bought 196 at 102.03" in request.session_context
        told = request.session_context
        assert "- Held now: 196 shares, about $20,482, which is a target weight of 1.019." in told
        assert "(Buy) asked for a target weight of 1, about 192 shares. 4 shares more than that are still held." in told
        assert "The position is larger than a full one." in told       # it rose after it was bought
        (position,) = request.portfolio.positions
        assert (position.ticker, position.quantity) == ("NVDA", 196.0)
        assert request.as_of_time.endswith("(regular session)")

    def test_an_analysis_is_told_when_its_last_target_was_not_bought(self, loop):
        """A larger target held back for want of a second opinion must not read as already in place."""
        loop.analysts.say("NVDA", "Overweight", "Overweight", "Buy", "Buy")
        loop.tick("premarket", 11, "08:45")
        loop.tick("core", 11, "09:35")
        loop.market.prices["NVDA"] = 102.0
        loop.tick("core", 11, "10:35")               # the second Overweight: 0.6 of a full position is bought
        held = loop.qty()
        loop.market.prices["NVDA"] = 104.0
        loop.tick("core", 11, "11:35")               # Buy, the first to ask for more: held back
        assert loop.qty() == held
        loop.market.prices["NVDA"] = 106.0
        loop.tick("core", 11, "12:35")
        told = loop.analysts.calls[-1][0].session_context
        assert f"- Held now: {held:g} shares" in told
        assert re.search(r"\(Buy\) asked for a target weight of 1, about \d+ shares\. \d+ shares of that have not been bought\.", told)
        assert told.index("How a target weight") < told.index("Earlier in this session")

    def test_a_held_position_with_a_stop_gets_a_stop_order(self, loop):
        self._confirmed_buy(loop, stop_loss=95.0)
        (stop,) = loop.store.query("SELECT * FROM orders WHERE type = 'stop'")
        assert (stop["side"], stop["qty"], stop["stop_price"], stop["status"]) == ("sell", 196.0, 95.0, "new")

    def test_a_fall_through_the_stop_sells_and_the_decision_does_not_buy_back(self, loop):
        self._confirmed_buy(loop, stop_loss=95.0)
        # The fall triggers a new analysis each tick; both fail, so the old Buy still stands.
        loop.analysts.say("NVDA", RuntimeError("provider unavailable"), RuntimeError("provider unavailable"))
        loop.market.prices["NVDA"] = 94.0
        loop.tick("core", 11, "11:35")
        assert loop.qty() == 0.0
        (sold,) = loop.store.query("SELECT * FROM fills WHERE side = 'sell'")
        assert sold["qty"] == 196.0 and sold["price"] == pytest.approx(94.0 * (1 - 0.0003))

        loop.market.prices["NVDA"] = 96.5          # back above the stop: the old Buy must not re-enter
        loop.tick("core", 11, "12:35")
        assert loop.qty() == 0.0
        nvda = loop.targets("20261111T1235-core")["NVDA"]
        assert nvda["verdict"] == "blocked" and "already stopped out" in str(nvda["checks"])

    def test_a_sell_decision_closes_the_position_without_waiting(self, loop):
        self._confirmed_buy(loop)
        loop.analysts.say("NVDA", "Sell")
        loop.market.prices["NVDA"] = 104.5
        report = loop.tick("core", 11, "11:35")
        assert report.orders == 1 and loop.qty() == 0.0
        assert loop.targets("20261111T1135-core")["NVDA"]["verdict"] == "reduce"

    def test_one_tickers_failed_analysis_leaves_the_others_and_its_standing_decision(self, loop):
        loop.analysts.say("NVDA", "Buy")
        loop.analysts.say("HD", RuntimeError("rate limited"))
        report = loop.tick("premarket", 11, "08:45")
        assert report.status == "ok" and report.failed == ["HD"] and len(report.analysed) == 4
        (failed,) = loop.store.query("SELECT * FROM analyses WHERE status = 'error'")
        assert failed["ticker"] == "HD" and "rate limited" in failed["error"]
        (event,) = loop.store.query("SELECT * FROM events WHERE level = 'error'")
        assert (event["source"], event["ticker"]) == ("analysis", "HD")
        assert loop.targets("20261111T0845-premarket")["HD"]["verdict"] == "deferred:keep"

        # The next session tick has no decision for HD yet, so it tries again; the others are reused.
        loop.tick("core", 11, "09:35")
        assert loop.analysed() == ["HD"]

    def test_headlines_since_the_last_analysis_trigger_only_their_ticker(self, loop):
        loop.tick("premarket", 11, "08:45")
        loop.market.headlines["GEV"] = [et(11, "09:00") + timedelta(minutes=i) for i in range(8)]
        loop.market.headlines["DIS"] = [et(11, "08:00")] * 20        # before the analysis: old news
        loop.tick("core", 11, "09:35")
        assert loop.analysed() == ["GEV"]
        rows = {r["ticker"]: r for r in loop.store.query(
            "SELECT * FROM observations WHERE tick_id = '20261111T0935-core'")}
        assert rows["GEV"]["new_headlines"] == 8 and rows["DIS"]["new_headlines"] == 0

    def test_a_decision_goes_stale_after_three_hours(self, loop):
        loop.tick("premarket", 11, "08:45")
        loop.tick("core", 11, "11:35")
        assert len(loop.analysts.calls) == 1
        loop.tick("core", 11, "12:35")
        assert loop.analysed() == TICKERS

    def test_the_last_minutes_before_the_close_place_no_orders(self, tmp_path):
        loop = Loop(tmp_path, confirm_rounds=1)
        loop.analysts.say("NVDA", "Buy", "Buy")              # the stale decision is analysed again at 15:55
        loop.tick("premarket", 11, "08:45")
        report = loop.tick("core", 11, "15:55")
        nvda = loop.targets("20261111T1555-core")["NVDA"]
        assert report.orders == 0 and nvda["verdict"] == "deferred:increase"
        assert nvda["checks"][-1]["detail"] == "within 10 minutes of the close"
        loop.store.close()

    def test_an_analysis_that_runs_past_the_window_places_no_orders(self, tmp_path):
        loop = Loop(tmp_path, confirm_rounds=1)
        loop.analysts.takes = timedelta(minutes=20)
        loop.analysts.say("NVDA", "Buy")
        report = loop.tick("core", 11, "15:35")              # the analysis ends at 15:55
        assert report.analysed == TICKERS and report.orders == 0
        loop.store.close()

    def test_five_confirmed_buys_fill_the_book_and_no_more(self, tmp_path):
        loop = Loop(tmp_path, confirm_rounds=1)
        for ticker in TICKERS:
            loop.analysts.say(ticker, "Buy")
        loop.tick("core", 11, "09:35")
        account = loop.broker.account(loop.market.prices)
        assert [account.qty(t) for t in TICKERS] == [200.0] * 4 + [199.0]   # the fifth buy has the cash left after costs
        assert 0 <= account.cash < 200 and account.equity == pytest.approx(100_000 - 99_900 * 0.0003, abs=0.5)
        assert {o["status"] for o in loop.store.query("SELECT status FROM orders")} == {"filled"}   # none refused
        loop.store.close()


@pytest.mark.unit
class TestOutsideTheSession:
    def test_the_post_close_run_is_the_one_recorded_in_the_memory_log(self, loop):
        loop.tick("premarket", 11, "08:45")
        report = loop.tick("postclose", 11, "16:30")
        assert report.analysed == TICKERS and report.orders == 0
        assert all(r.record_to_memory for r in loop.analysts.calls[-1])
        assert all(r.as_of_time.endswith("(after the close)") for r in loop.analysts.calls[-1])
        recorded = loop.store.query("SELECT kind, recorded_to_memory FROM analyses ORDER BY started_at")
        assert {(r["kind"], r["recorded_to_memory"]) for r in recorded} == {("premarket", 0), ("postclose", 1)}

    def test_the_watch_does_nothing_without_a_reason(self, loop):
        loop.tick("postclose", 11, "16:30")
        report = loop.tick("offhours", 11, "18:00")
        assert report.analysed == [] and len(loop.analysts.calls) == 1
        rows = loop.store.query("SELECT gate_action, gate_reasons FROM observations WHERE tick_id = '20261111T1800-offhours'")
        assert {r["gate_action"] for r in rows} == {"skip"}

    def test_the_watch_analyses_a_ticker_that_moves_but_only_once_a_night(self, loop):
        loop.tick("postclose", 11, "16:30")
        loop.market.prices["NVDA"] = 108.0                       # results after the close
        report = loop.tick("offhours", 11, "18:00")
        assert report.analysed == ["NVDA"] and report.orders == 0
        assert loop.analysts.calls[-1][0].as_of_time.endswith("(market closed)")

        loop.market.prices["NVDA"] = 112.0
        again = loop.tick("offhours", 11, "19:00")
        assert again.analysed == []
        (obs,) = loop.store.query("SELECT gate_reasons FROM observations WHERE tick_id = '20261111T1900-offhours' "
                                  "AND ticker = 'NVDA'")
        assert "used its analyses for tonight" in obs["gate_reasons"][-1]

    def test_after_midnight_at_the_exchange_the_run_is_dated_by_the_earlier_calendar(self, loop):
        """At 01:00 Eastern it is still the day before on a machine in Pacific time, and
        TradingAgents refuses a date its own clock calls the future."""
        loop.tick("postclose", 11, "16:30")
        loop.market.prices["NVDA"] = 108.0
        loop.engine._today = lambda: date(2026, 11, 11)
        loop.tick("offhours", 12, "01:00")
        assert loop.analysts.calls[-1][0].trade_date == "2026-11-11"


@pytest.mark.unit
class TestLimitsAcrossTheDay:
    def test_a_spent_budget_stops_the_model_but_not_the_loop(self, tmp_path):
        loop = Loop(tmp_path, daily_llm_budget_usd=5.0)
        loop.tick("premarket", 11, "08:45")                       # five analyses at one dollar each
        loop.market.prices["NVDA"] = 105.0
        report = loop.tick("core", 11, "09:35")
        assert report.status == "ok" and report.analysed == [] and len(loop.analysts.calls) == 1
        (obs,) = loop.store.query("SELECT gate_reasons FROM observations WHERE tick_id = '20261111T0935-core' "
                                  "AND ticker = 'NVDA'")
        assert obs["gate_reasons"] == ["the day's model budget is spent"]

        loop.tick("premarket", 12, "08:45")                       # a new day, a new budget
        assert len(loop.analysts.calls) == 2
        loop.store.close()

    def test_past_the_days_loss_limit_a_confirmed_buy_is_blocked(self, tmp_path):
        loop = Loop(tmp_path, confirm_rounds=1, max_position_weight=1.0)
        loop.analysts.say("NVDA", "Buy")
        loop.tick("core", 11, "09:35")                            # all in on NVDA at 100
        loop.market.prices["NVDA"] = 97.0                         # the book is down 3%
        loop.market.prices["CSCO"] = 102.0                        # enough of a move to analyse CSCO again
        loop.analysts.say("CSCO", "Buy")
        loop.tick("core", 11, "10:35")
        csco = loop.targets("20261111T1035-core")["CSCO"]
        assert csco["verdict"] == "blocked"
        assert any(c["name"] == "daily_loss" and not c["passed"] for c in csco["checks"])
        loop.store.close()


@pytest.mark.unit
class TestFailuresAndRestarts:
    def test_a_tick_runs_once(self, loop):
        loop.tick("premarket", 11, "08:45")
        again = loop.tick("premarket", 11, "08:45")
        assert again.status == "skipped" and len(loop.analysts.calls) == 1

    def test_a_tick_that_crashed_midway_is_run_again_under_fresh_run_ids(self, loop):
        loop.store.begin_tick("20261111T0845-premarket", "premarket", "x", "x", "simulate")
        loop.store.begin_analysis(run_id="20261111T0845-premarket-NVDA", tick_id="20261111T0845-premarket",
                                  ticker="NVDA", trade_date="2026-11-11", kind="premarket", started_at="x",
                                  ref_price=100.0, artifacts_dir="")
        report = loop.tick("premarket", 11, "08:45")
        assert report.status == "ok" and report.analysed == TICKERS
        ids = {r["run_id"] for r in loop.store.query("SELECT run_id FROM analyses WHERE ticker = 'NVDA'")}
        assert ids == {"20261111T0845-premarket-NVDA", "20261111T0845-premarket-NVDA-retry2"}

    def test_a_market_data_outage_fails_the_tick_and_the_next_one_runs(self, loop):
        loop.market.down = True
        report = loop.tick("premarket", 11, "08:45")
        assert report.status == "error" and "did not reach Alpaca" in report.note
        assert loop.store.tick("20261111T0845-premarket")["status"] == "error"
        assert loop.analysts.calls == [] and loop.built == [1]

        loop.market.down = False
        assert loop.tick("core", 11, "09:35").status == "ok"

    def test_a_ticker_with_no_price_is_left_alone(self, loop):
        del loop.market.prices["GEV"]
        report = loop.tick("core", 11, "09:35")
        assert report.status == "ok" and "GEV" not in report.analysed
        gev = loop.targets("20261111T0935-core")["GEV"]
        assert gev["verdict"] == "keep" and gev["price"] is None


@pytest.mark.unit
class TestOutcomes:
    def test_a_due_outcome_is_marked_on_a_later_tick(self, loop):
        loop.tick("premarket", 11, "08:45")
        loop.market.series["NVDA"] = [Bar(et(11, "09:30"), 100.0, 101.0, 99.5, 100.8, 1000),
                                      Bar(et(11, "09:45"), 100.8, 102.5, 100.6, 102.0, 1000)]
        loop.tick("core", 11, "10:35")
        (hour,) = loop.store.query("SELECT * FROM outcomes WHERE run_id = '20261111T0845-premarket-NVDA' AND horizon = '60m'")
        # Due at 09:45: the last bar finished by then is 09:30's.
        assert hour["price"] == 100.8 and hour["ret"] == pytest.approx(0.008) and hour["marked_at"]
        pending = loop.store.query("SELECT horizon FROM outcomes WHERE run_id = '20261111T0845-premarket-NVDA' "
                                   "AND marked_at IS NULL")
        assert {r["horizon"] for r in pending} == {"close", "1d", "5d"}


@pytest.mark.unit
class TestRunningForever:
    def _run(self, loop, start, ticks):
        loop.clock[0] = start
        seen = []

        def sleep(seconds):
            loop.clock[0] += timedelta(seconds=seconds)

        run_forever(loop.engine, sleep=sleep, stop=lambda: len(seen) >= ticks, on_tick=seen.append)
        return seen

    def test_ticks_run_in_order_as_their_time_comes(self, loop):
        seen = self._run(loop, et(11, "07:30"), 4)
        assert [r.tick_id for r in seen] == ["20261111T0800-offhours", "20261111T0845-premarket",
                                             "20261111T0935-core", "20261111T1035-core"]
        assert loop.clock[0] == et(11, "10:35")

    def test_a_tick_missed_while_the_last_one_ran_long_is_skipped(self, loop):
        loop.analysts.takes = timedelta(minutes=100)             # the 08:45 analysis ends at 10:25
        seen = self._run(loop, et(11, "08:30"), 2)
        assert [r.tick_id for r in seen] == ["20261111T0845-premarket", "20261111T1035-core"]
        (skipped,) = loop.store.query("SELECT message FROM events WHERE source = 'scheduler'")
        assert "20261111T0935-core skipped: 50 minutes late" in skipped["message"]

    def test_stopping_while_waiting_returns_at_once(self, loop):
        loop.clock[0] = et(11, "07:30")
        calls = []
        run_forever(loop.engine, sleep=lambda s: calls.append(s), stop=lambda: bool(calls))
        assert len(calls) == 1 and loop.analysts.calls == []


@pytest.mark.unit
class TestABrokerThatFillsAMomentLater:
    """An exchange fills a marketable order within moments, not within the call that placed it."""

    def _loop(self, tmp_path):
        from tradeloop.broker import OrderState

        loop = Loop(tmp_path, confirm_rounds=1)

        def accept_only(order, price, at):
            if order.type == "stop":
                return OrderState(status="new", broker_order_id=f"sim-{order.client_order_id}")
            return OrderState(status="accepted", broker_order_id=f"sim-{order.client_order_id}")

        loop.broker.submit = accept_only        # the fill comes on the next look, as at a real broker
        return loop

    def test_the_tick_waits_once_then_records_the_fill_and_places_the_stop(self, tmp_path):
        loop = self._loop(tmp_path)
        loop.analysts.say("NVDA", ("Buy", {"stop_loss": 95.0}))
        report = loop.tick("core", 11, "09:35")

        assert report.orders == 1 and loop.slept == [3.0]      # one look was enough
        (buy,) = loop.store.query("SELECT * FROM orders WHERE type = 'limit'")
        assert buy["status"] == "filled" and buy["tick_id"] == "20261111T0935-core"
        assert loop.qty() == 200.0
        (stop,) = loop.store.query("SELECT * FROM orders WHERE type = 'stop'")
        assert (stop["qty"], stop["stop_price"], stop["tick_id"]) == (200.0, 95.0, "20261111T0935-core")
        loop.store.close()

    def test_no_wait_when_nothing_is_left_open(self, loop):
        loop.analysts.say("NVDA", "Hold")
        loop.tick("core", 11, "09:35")
        assert loop.slept == []


class PartBroker(SimBroker):
    """Fills as an exchange can: accepted at once, then filled in parts, a scripted number of shares per look.

    Like Alpaca it refuses a stop while a buy of the same name is working, and a
    buy while a stop of that name rests ("potential wash trade").
    """

    def __init__(self, store, cfg):
        super().__init__(store, cfg)
        self.script: dict[str, list[float]] = {}
        self.cancelled: list[str] = []

    def _open(self, ticker, stops):
        return [r for r in self.store.open_orders(self.name)
                if r["ticker"] == ticker and (r["type"] == "stop") == stops and r["status"] != "pending_new"]

    def submit(self, order, price, at):
        if self._open(order.ticker, stops=order.type != "stop"):
            return OrderState(status="rejected", error="403 potential wash trade detected")
        status = "new" if order.type == "stop" else "accepted"
        return OrderState(status=status, broker_order_id=f"sim-{order.client_order_id}")

    def cancel(self, order):
        self.cancelled.append(order["client_order_id"])

    def refresh(self, order, price, at):
        if order["type"] == "stop":
            return super().refresh(order, price, at)
        done, paid = order["filled_qty"] or 0.0, order["filled_avg_price"] or 0.0
        if order["client_order_id"] in self.cancelled:
            return OrderState("canceled", done, paid or None, order["broker_order_id"])
        steps = self.script.get(order["ticker"], [])
        part = min(steps.pop(0), order["qty"] - done) if steps and price is not None else 0.0
        if part <= 0:
            return OrderState("partially_filled" if done else "accepted", done, paid or None, order["broker_order_id"])
        fill = self._try_fill(order["ticker"], order["side"], part, "market", None, price, order["broker_order_id"])
        total = done + part
        return OrderState("filled" if total >= order["qty"] else "partially_filled", total,
                          (done * paid + part * fill.filled_avg_price) / total, order["broker_order_id"])


@pytest.mark.unit
class TestOrdersThatFillInPartsOrNotAtAll:
    """Every order ends, filled or withdrawn, before the stops go in; what is missing is the next tick's."""

    TICK = "20261111T0935-core"

    def _loop(self, tmp_path, script, **settings):
        loop = Loop(tmp_path, broker=PartBroker, confirm_rounds=1, **settings)
        loop.broker.script["NVDA"] = list(script)
        loop.analysts.say("NVDA", ("Buy", {"stop_loss": 95.0}))
        return loop

    def _orders(self, loop, kind):
        return loop.store.query("SELECT * FROM orders WHERE type = ? ORDER BY submitted_at, client_order_id", (kind,))

    def test_parts_are_waited_for_and_one_stop_guards_the_whole_fill(self, tmp_path):
        loop = self._loop(tmp_path, [80, 0, 120])
        report = loop.tick("core", 11, "09:35")

        assert report.status == "ok" and loop.slept == [3.0, 3.0, 3.0]
        (buy,) = self._orders(loop, "limit")
        assert (buy["status"], buy["filled_qty"]) == ("filled", 200.0)
        assert [f["qty"] for f in loop.store.query("SELECT qty FROM fills ORDER BY rowid")] == [80.0, 120.0]
        (stop,) = self._orders(loop, "stop")
        assert (stop["status"], stop["qty"], stop["stop_price"], stop["tick_id"]) == ("new", 200.0, 95.0, self.TICK)
        loop.store.close()

    def test_what_has_not_filled_in_time_is_withdrawn_and_the_stop_covers_what_did(self, tmp_path):
        loop = self._loop(tmp_path, [80])
        report = loop.tick("core", 11, "09:35")

        assert report.status == "ok" and sum(loop.slept) == 120.0 and len(loop.slept) == 40
        (buy,) = self._orders(loop, "limit")
        assert (buy["status"], buy["filled_qty"], buy["qty"]) == ("canceled", 80.0, 200.0)
        assert loop.qty() == 80.0
        (stop,) = self._orders(loop, "stop")          # placed, not refused: nothing of NVDA's was working any more
        assert (stop["status"], stop["qty"], stop["tick_id"]) == ("new", 80.0, self.TICK)
        (event,) = loop.store.query("SELECT message FROM events WHERE source = 'executor'")
        assert "80 of 200 filled in time" in event["message"]
        loop.store.close()

    def test_the_next_tick_orders_what_is_missing_and_the_stop_grows_with_it(self, tmp_path):
        loop = self._loop(tmp_path, [80])
        loop.tick("core", 11, "09:35")
        loop.broker.script["NVDA"] = [120]
        report = loop.tick("core", 11, "10:35")

        assert report.orders == 1 and report.analysed == [] and len(loop.analysts.calls) == 1   # on the standing decision
        first, second = self._orders(loop, "limit")
        # 119, not 120: the first 80 cost their spread, and a fifth of what is left buys 199 in all.
        assert (second["qty"], second["status"], second["tick_id"]) == (119.0, "filled", "20261111T1035-core")
        assert loop.qty() == 199.0
        old, new = self._orders(loop, "stop")
        assert (old["status"], old["qty"]) == ("canceled", 80.0)
        assert (new["status"], new["qty"]) == ("new", 199.0)
        assert loop.store.query("SELECT * FROM orders WHERE status = 'rejected'") == []
        loop.store.close()

    def test_an_add_that_fills_in_part_still_ends_with_the_whole_position_guarded(self, tmp_path):
        loop = self._loop(tmp_path, [80])
        loop.tick("core", 11, "09:35")
        loop.broker.script["NVDA"] = [50]              # of the 119 asked for
        loop.tick("core", 11, "10:35")

        assert loop.qty() == 130.0
        stops = self._orders(loop, "stop")
        assert [(s["status"], s["qty"]) for s in stops] == [("canceled", 80.0), ("new", 130.0)]
        assert loop.store.query("SELECT * FROM orders WHERE status = 'rejected'") == []
        loop.store.close()

    def test_an_order_the_price_ran_away_from_leaves_no_position_and_no_stop(self, tmp_path):
        loop = self._loop(tmp_path, [])
        report = loop.tick("core", 11, "09:35")

        assert report.status == "ok" and report.orders == 1
        (buy,) = self._orders(loop, "limit")
        assert (buy["status"], buy["filled_qty"]) == ("canceled", 0.0)
        assert loop.qty() == 0 and self._orders(loop, "stop") == []
        assert loop.store.query("SELECT * FROM fills") == []
        loop.store.close()

    def test_the_wait_ends_on_time_when_it_is_not_a_whole_number_of_looks(self, tmp_path):
        loop = self._loop(tmp_path, [], fill_wait_seconds=5.0)
        loop.tick("core", 11, "09:35")
        assert loop.slept == [3.0, 2.0]
        loop.store.close()

    def test_no_wait_at_all_withdraws_at_once(self, tmp_path):
        loop = self._loop(tmp_path, [200], fill_wait_seconds=0.0)
        loop.tick("core", 11, "09:35")
        assert loop.slept == [] and loop.qty() == 0
        assert self._orders(loop, "limit")[0]["status"] == "canceled"
        loop.store.close()

    def test_prices_that_cannot_be_read_while_waiting_do_not_fail_the_tick(self, tmp_path):
        loop = self._loop(tmp_path, [0, 200])

        def sleep(seconds):
            loop.slept.append(seconds)
            loop.market.down = True                    # the data feed drops once the orders are out

        loop.engine._sleep = sleep
        report = loop.tick("core", 11, "09:35")
        assert report.status == "ok" and loop.slept == [3.0, 3.0]
        assert loop.qty() == 200.0 and self._orders(loop, "stop")[0]["qty"] == 200.0
        loop.store.close()

    def test_while_waiting_only_the_names_with_orders_out_are_priced(self, tmp_path):
        loop = self._loop(tmp_path, [0, 200])
        asked = []
        quotes = loop.market.quotes
        loop.market.quotes = lambda symbols: asked.append(list(symbols)) or quotes(symbols)
        loop.tick("core", 11, "09:35")
        assert asked[-2:] == [["NVDA"], ["NVDA"]]
        loop.store.close()


@pytest.mark.unit
class TestSurvivingOutages:
    def test_a_calendar_that_cannot_be_read_today_falls_back_to_yesterdays(self, loop):
        calls = []

        def flaky(start, end):
            calls.append(start)
            if len(calls) > 1:
                raise AlpacaError(0, "calendar unreachable")
            return CALENDAR

        scheduler = Scheduler(flaky, loop.cfg)
        assert scheduler.next_tick(et(11, "08:00")).kind == "premarket"
        tomorrow = scheduler.next_tick(et(12, "08:00"))           # the fetch fails; the cached calendar serves
        assert (tomorrow.kind, tomorrow.at) == ("premarket", et(12, "08:45"))
        assert len(calls) == 2
        scheduler.next_tick(et(12, "09:00"))                       # and it is asked for again
        assert len(calls) == 3

    def test_with_no_calendar_at_all_the_loop_waits_and_tries_again(self, loop):
        attempts, slept = [], []
        real = loop.scheduler.next_tick

        def next_tick(after):
            attempts.append(after)
            if len(attempts) == 1:
                raise AlpacaError(0, "calendar unreachable")
            return real(after)

        loop.engine.scheduler.next_tick = next_tick
        loop.clock[0] = et(11, "08:30")
        seen = []

        def sleep(seconds):
            slept.append(seconds)
            loop.clock[0] += timedelta(seconds=seconds)

        run_forever(loop.engine, sleep=sleep, stop=lambda: len(seen) >= 1, on_tick=seen.append)
        assert slept[0] == 60.0 and [r.tick_id for r in seen] == ["20261111T0845-premarket"]
        (event,) = loop.store.query("SELECT message FROM events WHERE source = 'scheduler'")
        assert "next tick not known" in event["message"]
