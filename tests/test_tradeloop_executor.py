"""From rulings to orders and fills, against the simulated book and a scripted broker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradeloop.broker import Account, OrderState, Position, SimBroker
from tradeloop.config import LoopConfig
from tradeloop.executor import Executor, plan_orders
from tradeloop.risk import Ruling
from tradeloop.store import Store

NOW = datetime(2026, 11, 11, 15, 35, tzinfo=UTC)
CFG = LoopConfig(sim_starting_cash=100_000.0)
TICK = "20261111T1035-core"


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "loop.db") as s:
        yield s


def tick(store, tick_id=TICK):
    store.begin_tick(tick_id, "core", "2026-11-11T15:35:00Z", "2026-11-11T15:35:00Z", "simulate")
    return tick_id


def ruling(ticker="NVDA", price=100.0, current=0.0, target_qty=200.0, verdict="increase", actionable=True):
    return Ruling(ticker=ticker, price=price, current_qty=current, current_weight=0.0, raw_weight=None,
                  target_weight=0.0, target_qty=target_qty, verdict=verdict, actionable=actionable)


@pytest.mark.unit
class TestPlanningOrders:
    def test_a_buy_is_a_limit_a_little_above_the_price(self):
        (order,) = plan_orders([ruling()], TICK, CFG)
        assert (order.side, order.qty, order.type, order.limit_price) == ("buy", 200.0, "limit", 100.1)
        assert order.client_order_id == "tl-20261111T1035-core-NVDA-buy" and order.reason == "increase"

    def test_a_sell_is_a_limit_a_little_below_it(self):
        (order,) = plan_orders([ruling(current=200.0, target_qty=60.0, verdict="reduce")], TICK, CFG)
        assert (order.side, order.qty, order.limit_price) == ("sell", 140.0, 99.9)

    def test_sells_come_before_buys(self):
        orders = plan_orders([ruling("NVDA"), ruling("CSCO", current=50.0, target_qty=0.0, verdict="reduce"),
                              ruling("HD"), ruling("DIS", current=10.0, target_qty=0.0, verdict="reduce")], TICK, CFG)
        assert [(o.ticker, o.side) for o in orders] == [("CSCO", "sell"), ("DIS", "sell"), ("NVDA", "buy"), ("HD", "buy")]

    def test_a_change_too_small_to_be_worth_its_cost_is_left(self):
        assert plan_orders([ruling(current=199.0, target_qty=200.0)], TICK, CFG) == []      # 100 dollars

    def test_a_full_exit_is_always_placed(self):
        (order,) = plan_orders([ruling(current=1.0, target_qty=0.0, verdict="reduce")], TICK, CFG)
        assert (order.side, order.qty) == ("sell", 1.0)

    def test_a_stop_exit_goes_at_market(self):
        (order,) = plan_orders([ruling(current=100.0, target_qty=0.0, verdict="exit_stop")], TICK, CFG)
        assert order.type == "market" and order.limit_price is None

    def test_nothing_is_ordered_outside_the_trading_window_or_without_a_price(self):
        assert plan_orders([ruling(actionable=False), ruling("CSCO", price=None)], TICK, CFG) == []

    def test_buys_are_cut_to_the_cash_there_is(self):
        rulings = [ruling(t, target_qty=200.0) for t in ("NVDA", "CSCO", "HD")]
        orders = plan_orders(rulings, TICK, CFG, cash=45_000.0)
        # 200 shares at the 100.10 limit is 20,020; the third buy gets what is left.
        assert [(o.ticker, o.qty) for o in orders] == [("NVDA", 200.0), ("CSCO", 200.0), ("HD", 49.0)]

    def test_this_ticks_sells_pay_for_its_buys(self):
        rulings = [ruling("NVDA", target_qty=200.0), ruling("CSCO", current=300.0, target_qty=0.0, verdict="reduce")]
        orders = plan_orders(rulings, TICK, CFG, cash=1_000.0)
        assert [(o.ticker, o.side, o.qty) for o in orders] == [("CSCO", "sell", 300.0), ("NVDA", "buy", 200.0)]

    def test_a_buy_cut_below_the_minimum_is_dropped(self):
        assert plan_orders([ruling(target_qty=200.0)], TICK, CFG, cash=150.0) == []
        assert plan_orders([ruling(target_qty=200.0)], TICK, CFG, cash=0.0) == []

    def test_a_fractional_difference_is_rounded_down_to_whole_shares(self):
        (order,) = plan_orders([ruling(current=10.5, target_qty=20.0)], TICK, CFG)
        assert order.qty == 9.0
        (exact,) = plan_orders([ruling(current=10.5, target_qty=20.0)], TICK, LoopConfig(allow_fractional=True))
        assert exact.qty == 9.5


@pytest.mark.unit
class TestPlacingOrders:
    def test_a_filled_order_is_recorded_with_its_fill_and_cost(self, store):
        broker, tick_id = SimBroker(store, CFG), tick(store)
        executor = Executor(broker, store, CFG)
        (row,) = executor.place(plan_orders([ruling()], tick_id, CFG), {"NVDA": 100.0}, NOW, tick_id)

        assert (row["status"], row["filled_qty"], row["broker"]) == ("filled", 200.0, "simulate")
        assert row["filled_avg_price"] == pytest.approx(100.03)
        assert row["est_cost_usd"] == pytest.approx(6.0)                 # 3 bps of 20,000
        (fill,) = store.query("SELECT * FROM fills")
        assert (fill["qty"], fill["side"], fill["client_order_id"]) == (200.0, "buy", row["client_order_id"])
        assert fill["price"] == pytest.approx(100.03) and fill["cost_usd"] == pytest.approx(6.0018)
        assert broker.account({"NVDA": 100.0}).qty("NVDA") == 200.0

    def test_placing_the_same_ticks_orders_again_changes_nothing(self, store):
        broker, tick_id = SimBroker(store, CFG), tick(store)
        executor = Executor(broker, store, CFG)
        orders = plan_orders([ruling()], tick_id, CFG)
        executor.place(orders, {"NVDA": 100.0}, NOW, tick_id)
        executor.place(orders, {"NVDA": 100.0}, NOW, tick_id)
        assert broker.account({}).qty("NVDA") == 200.0
        assert len(store.query("SELECT * FROM orders")) == 1 and len(store.query("SELECT * FROM fills")) == 1

    def test_an_order_recorded_but_never_sent_is_sent_on_the_retry(self, store):
        """A crash between recording and sending leaves a pending row; the retry finishes it."""
        broker, tick_id = SimBroker(store, CFG), tick(store)
        (order,) = plan_orders([ruling()], tick_id, CFG)
        store.add_order(client_order_id=order.client_order_id, tick_id=tick_id, ticker="NVDA", side="buy", qty=200.0,
                        type="limit", limit_price=100.1, stop_price=None, time_in_force="day", reason="increase",
                        status="pending_new", broker="simulate", broker_order_id=None,
                        submitted_at="2026-11-11T15:35:00Z", updated_at="2026-11-11T15:35:00Z")
        Executor(broker, store, CFG).place([order], {"NVDA": 100.0}, NOW, tick_id)
        assert store.order(order.client_order_id)["status"] == "filled" and broker.account({}).qty("NVDA") == 200.0

    def test_a_rejected_order_keeps_the_brokers_reason(self, store):
        broker, tick_id = SimBroker(store, LoopConfig(sim_starting_cash=1_000.0)), tick(store)
        (row,) = Executor(broker, store, CFG).place(plan_orders([ruling()], tick_id, CFG), {"NVDA": 100.0}, NOW, tick_id)
        assert row["status"] == "rejected" and "no margin" in row["error"]
        assert store.query("SELECT * FROM fills") == []

    def test_sells_free_the_cash_the_buys_need(self, store):
        cfg = LoopConfig(sim_starting_cash=20_010.0)
        broker, tick_id = SimBroker(store, cfg), tick(store)
        executor = Executor(broker, store, cfg)
        executor.place(plan_orders([ruling("NVDA")], tick_id, cfg), {"NVDA": 100.0}, NOW, tick_id)

        later = tick(store, "20261111T1135-core")
        rulings = [ruling("CSCO", target_qty=150.0), ruling("NVDA", current=200.0, target_qty=0.0, verdict="reduce")]
        rows = executor.place(plan_orders(rulings, later, cfg), {"NVDA": 100.0, "CSCO": 100.0}, NOW, later)
        assert [(r["ticker"], r["status"]) for r in rows] == [("NVDA", "filled"), ("CSCO", "filled")]


@pytest.mark.unit
class TestSettlingOpenOrders:
    def test_a_waiting_limit_order_fills_when_the_price_comes_to_it(self, store):
        broker, tick_id = SimBroker(store, CFG), tick(store)
        executor = Executor(broker, store, CFG)
        executor.place(plan_orders([ruling()], tick_id, CFG), {"NVDA": 100.2}, NOW, tick_id)     # limit 100.1 is under the ask
        assert store.open_orders("simulate")[0]["status"] == "new"

        assert executor.settle_open_orders({"NVDA": 99.9}, NOW + timedelta(minutes=5), tick_id) == 1
        assert store.open_orders("simulate") == [] and broker.account({}).qty("NVDA") == 200.0

    def test_an_order_left_from_an_earlier_tick_is_cancelled(self, store):
        broker, tick_id = SimBroker(store, CFG), tick(store)
        executor = Executor(broker, store, CFG)
        (row,) = executor.place(plan_orders([ruling()], tick_id, CFG), {"NVDA": 100.2}, NOW, tick_id)

        later = tick(store, "20261111T1135-core")
        assert executor.settle_open_orders({"NVDA": 100.5}, NOW + timedelta(hours=1), later) == 0
        assert store.order(row["client_order_id"])["status"] == "canceled"
        assert broker.account({}).qty("NVDA") == 0

    def test_a_partial_fill_is_recorded_part_by_part(self, store):
        """A scripted broker fills 80 of 200, then the rest at another price."""
        states = [OrderState("partially_filled", 80.0, 100.0, "o1"), OrderState("filled", 200.0, 100.06, "o1")]

        class Scripted:
            name = "alpaca_paper"
            cancelled: list = []

            def submit(self, order, price, at):
                return OrderState("new", broker_order_id="o1")

            def refresh(self, row, price, at):
                return states.pop(0)

            def cancel(self, row):
                self.cancelled.append(row["client_order_id"])

        tick_id = tick(store)
        executor = Executor(Scripted(), store, CFG)
        (row,) = executor.place(plan_orders([ruling()], tick_id, CFG), {"NVDA": 100.0}, NOW, tick_id)
        assert row["status"] == "new" and row["broker_order_id"] == "o1"

        executor.settle_open_orders({"NVDA": 100.0}, NOW, tick_id)
        executor.settle_open_orders({"NVDA": 100.0}, NOW, tick_id)
        fills = store.query("SELECT qty, price FROM fills ORDER BY at, qty")
        assert [f["qty"] for f in fills] == [80.0, 120.0]
        assert fills[1]["price"] == pytest.approx(100.10)        # (200 × 100.06 − 80 × 100) / 120
        assert store.order(row["client_order_id"])["status"] == "filled"

    def test_a_fill_is_timed_by_the_broker_when_it_says_when(self, store):
        filled_at = datetime(2026, 11, 11, 15, 35, 2, tzinfo=UTC)

        class Timed:
            name = "alpaca_paper"

            def submit(self, order, price, at):
                return OrderState("filled", 200.0, 100.02, "o1", filled_at=filled_at)

        tick_id = tick(store)
        Executor(Timed(), store, CFG).place(plan_orders([ruling()], tick_id, CFG), {"NVDA": 100.0},
                                            NOW + timedelta(minutes=7), tick_id)
        (fill,) = store.query("SELECT at FROM fills")
        assert fill["at"] == "2026-11-11T15:35:02Z"            # not the time the tick looked


def OrderRequestFor(ticker, side, qty):  # noqa: N802 - reads as a constructor at the call site
    from tradeloop.broker import OrderRequest

    return OrderRequest(ticker=ticker, side=side, qty=qty, type="market", client_order_id=f"direct-{side}-{qty:g}")


def held(qty=200.0):
    return Account(equity=100_000.0, cash=80_000.0, positions={"NVDA": Position("NVDA", qty, 100.03)})


@pytest.mark.unit
class TestProtectiveStops:
    def _bought(self, store):
        broker, tick_id = SimBroker(store, CFG), tick(store)
        executor = Executor(broker, store, CFG)
        executor.place(plan_orders([ruling()], tick_id, CFG), {"NVDA": 100.0}, NOW, tick_id)
        return broker, executor, tick_id

    def test_a_held_position_gets_one_stop_at_the_decisions_level(self, store):
        broker, executor, tick_id = self._bought(store)
        executor.sync_stops(held(), {"NVDA": 92.0}, {"NVDA": 100.0}, NOW, tick_id)
        (stop,) = [o for o in store.open_orders("simulate") if o["type"] == "stop"]
        assert (stop["side"], stop["qty"], stop["stop_price"], stop["time_in_force"]) == ("sell", 200.0, 92.0, "gtc")

        executor.sync_stops(held(), {"NVDA": 92.0}, {"NVDA": 100.0}, NOW, tick_id)      # unchanged: left alone
        assert len(store.query("SELECT * FROM orders WHERE type = 'stop'")) == 1

    def test_a_new_level_or_size_replaces_the_stop(self, store):
        broker, executor, tick_id = self._bought(store)
        executor.sync_stops(held(), {"NVDA": 92.0}, {"NVDA": 100.0}, NOW, tick_id)
        later = tick(store, "20261111T1135-core")
        executor.sync_stops(held(120.0), {"NVDA": 95.0}, {"NVDA": 100.0}, NOW, later)
        stops = store.query("SELECT status, qty, stop_price FROM orders WHERE type = 'stop' ORDER BY submitted_at, status")
        assert sorted((s["status"], s["qty"], s["stop_price"]) for s in stops) == [("canceled", 200.0, 92.0), ("new", 120.0, 95.0)]

    def test_no_position_or_no_level_means_no_stop(self, store):
        broker, executor, tick_id = self._bought(store)
        executor.sync_stops(held(), {"NVDA": 92.0}, {"NVDA": 100.0}, NOW, tick_id)
        executor.sync_stops(Account(100_000.0, 100_000.0, {}), {"NVDA": 92.0}, {"NVDA": 100.0}, NOW, tick_id)
        assert [o for o in store.open_orders("simulate") if o["type"] == "stop"] == []

        executor.sync_stops(held(), {"NVDA": None}, {"NVDA": 100.0}, NOW, tick_id)
        assert [o for o in store.open_orders("simulate") if o["type"] == "stop"] == []

    def test_a_stop_at_or_above_the_price_is_not_placed(self, store):
        """The risk rules close such a position at market; a stop there would fire at once."""
        broker, executor, tick_id = self._bought(store)
        executor.sync_stops(held(), {"NVDA": 101.0}, {"NVDA": 100.0}, NOW, tick_id)
        assert store.query("SELECT * FROM orders WHERE type = 'stop'") == []

    def test_the_stop_sells_when_the_price_falls_through_it(self, store):
        broker, executor, tick_id = self._bought(store)
        executor.sync_stops(held(), {"NVDA": 92.0}, {"NVDA": 100.0}, NOW, tick_id)
        later = tick(store, "20261111T1135-core")
        assert executor.settle_open_orders({"NVDA": 91.0}, NOW + timedelta(hours=1), later) == 1
        assert broker.account({}).positions == {}
        (fill,) = store.query("SELECT * FROM fills WHERE side = 'sell'")
        assert fill["qty"] == 200.0 and fill["price"] == pytest.approx(91.0 * (1 - 0.0003))

    def test_adding_to_a_position_takes_its_stop_off_first_and_puts_a_larger_one_back(self, store):
        """A broker refuses a buy that could trade against one's own resting stop."""
        broker, executor, tick_id = self._bought(store)
        executor.sync_stops(held(), {"NVDA": 92.0}, {"NVDA": 100.0}, NOW, tick_id)
        later = tick(store, "20261111T1135-core")
        order_log = []
        real_submit, real_cancel = broker.submit, broker.cancel
        broker.submit = lambda o, p, a: order_log.append(("submit", o.type)) or real_submit(o, p, a)
        broker.cancel = lambda row: order_log.append(("cancel", row["type"])) or real_cancel(row)

        executor.place(plan_orders([ruling(current=200.0, target_qty=300.0)], later, CFG), {"NVDA": 100.0}, NOW, later)
        executor.sync_stops(broker.account({"NVDA": 100.0}), {"NVDA": 92.0}, {"NVDA": 100.0}, NOW, later)
        assert order_log == [("cancel", "stop"), ("submit", "limit"), ("submit", "stop")]
        (stop,) = [o for o in store.open_orders("simulate") if o["type"] == "stop"]
        assert stop["qty"] == 300.0

    def test_a_sell_is_cut_to_what_is_still_held(self, store):
        """A stop can sell the shares while an analysis runs; the sell planned before must not go short."""
        broker, executor, tick_id = self._bought(store)                # 200 held
        later = tick(store, "20261111T1135-core")
        broker.submit(OrderRequestFor("NVDA", "sell", 150.0), 100.0, NOW)     # 50 left, behind the planner's back
        (row,) = executor.place(plan_orders([ruling(current=200.0, target_qty=0.0, verdict="reduce")], later, CFG),
                                {"NVDA": 100.0}, NOW, later)
        assert (row["qty"], row["status"]) == (50.0, "filled") and broker.account({}).positions == {}

    def test_a_sell_of_shares_already_gone_is_dropped_and_logged(self, store):
        broker, executor, tick_id = self._bought(store)
        later = tick(store, "20261111T1135-core")
        broker.submit(OrderRequestFor("NVDA", "sell", 200.0), 100.0, NOW)
        placed = executor.place(plan_orders([ruling(current=200.0, target_qty=0.0, verdict="reduce")], later, CFG),
                                {"NVDA": 100.0}, NOW, later)
        assert placed == [] and store.query("SELECT * FROM orders WHERE tick_id = ?", (later,)) == []
        (event,) = store.query("SELECT * FROM events")
        assert "sell of 200 NVDA dropped: 0 held now" in event["message"]

    def test_selling_cancels_the_stop_that_holds_the_shares(self, store):
        broker, executor, tick_id = self._bought(store)
        executor.sync_stops(held(), {"NVDA": 92.0}, {"NVDA": 100.0}, NOW, tick_id)
        later = tick(store, "20261111T1135-core")
        sell = plan_orders([ruling(current=200.0, target_qty=0.0, verdict="reduce")], later, CFG)
        executor.place(sell, {"NVDA": 100.0}, NOW, later)
        assert store.open_orders("simulate") == [] and broker.account({}).positions == {}
