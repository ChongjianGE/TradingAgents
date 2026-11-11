"""The loop's record: what is written comes back, a tick runs once, and threads share it safely."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from tradeloop.store import SCHEMA_VERSION, Store

T0, T1, T2 = "2026-11-11T14:35:00Z", "2026-11-11T15:35:00Z", "2026-11-11T16:35:00Z"


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "loop.db") as s:
        yield s


def _tick(store, tick_id="20261111T0935-core", at=T0):
    assert store.begin_tick(tick_id, "core", at, at, "simulate")
    return tick_id


def _analysis(store, run_id, ticker="NVDA", started_at=T0, tick_id=None, **done):
    store.begin_analysis(run_id=run_id, tick_id=tick_id or _tick(store, f"t-{run_id}", started_at),
                         ticker=ticker, trade_date="2026-11-11", kind="core", started_at=started_at,
                         ref_price=168.89, artifacts_dir=f"runs/{run_id}")
    if done:
        store.finish_analysis(run_id, **done)


@pytest.mark.unit
class TestTicks:
    def test_a_finished_tick_does_not_run_again(self, store):
        tick_id = _tick(store)
        store.finish_tick(tick_id, "ok", T1, note="5 analysed")
        assert store.begin_tick(tick_id, "core", T0, T2, "simulate") is False
        assert store.tick(tick_id)["note"] == "5 analysed"

    def test_a_tick_left_running_by_a_crash_is_taken_over(self, store):
        tick_id = _tick(store)
        assert store.begin_tick(tick_id, "core", T0, T1, "simulate") is True
        assert store.tick(tick_id)["started_at"] == T1

    def test_a_failed_tick_is_not_retried(self, store):
        tick_id = _tick(store)
        store.finish_tick(tick_id, "error", T1, note="broker unreachable")
        assert store.begin_tick(tick_id, "core", T0, T2, "simulate") is False


@pytest.mark.unit
class TestTheBook:
    def test_a_snapshot_keeps_its_positions(self, store):
        tick_id = _tick(store)
        positions = [{"ticker": "NVDA", "qty": 10, "avg_price": 168.5, "price": 169.0}]
        store.add_snapshot(tick_id, T0, 100_005.0, 98_315.0, positions)
        (row,) = store.query("SELECT * FROM snapshots")
        assert row["positions"] == positions and row["equity"] == 100_005.0

    def test_the_days_starting_equity_is_the_last_one_before_it(self, store):
        store.add_snapshot(_tick(store, "a", T0), T0, 100_000.0, 100_000.0, [])
        store.add_snapshot(_tick(store, "b", T1), T1, 100_500.0, 100_500.0, [])
        assert store.equity_at_or_before(T1) == 100_500.0
        assert store.equity_at_or_before("2026-11-11T15:00:00Z") == 100_000.0
        assert store.equity_at_or_before("2026-11-10T00:00:00Z") is None

    def test_observations_keep_the_gates_reasons(self, store):
        tick_id = _tick(store)
        store.add_observations([{"tick_id": tick_id, "ticker": "NVDA", "price": 169.0, "price_at": T0,
                                 "prev_close": 178.64, "position_qty": 0, "new_headlines": 4,
                                 "gate_action": "analyze", "gate_reasons": ["4 new headlines"]}])
        (row,) = store.query("SELECT * FROM observations")
        assert row["gate_reasons"] == ["4 new headlines"]


@pytest.mark.unit
class TestAnalyses:
    def test_the_latest_decision_is_the_newest_rated_one(self, store):
        _analysis(store, "r1", started_at=T0, status="ok", rating="Hold")
        _analysis(store, "r2", started_at=T1, status="ok", rating="Overweight", target_weight=0.6,
                  decision={"rating": "Overweight", "target_weight": 0.6})
        _analysis(store, "r3", started_at=T2, status="error", error="provider unavailable")
        latest = store.latest_decision("NVDA")
        assert latest["run_id"] == "r2"
        assert latest["decision"] == {"rating": "Overweight", "target_weight": 0.6}
        assert store.latest_decision("NVDA", since=T2) is None
        assert store.latest_decision("AAPL") is None

    def test_recent_decisions_come_newest_first_and_skip_failures(self, store):
        _analysis(store, "r1", started_at=T0, status="ok", rating="Hold")
        _analysis(store, "r2", started_at=T1, status="error")
        _analysis(store, "r3", started_at=T2, status="ok", rating="Buy")
        assert [d["run_id"] for d in store.recent_decisions("NVDA", 5)] == ["r3", "r1"]
        assert [d["run_id"] for d in store.recent_decisions("NVDA", 1)] == ["r3"]

    def test_a_window_of_analyses_includes_failures_and_filters_by_kind(self, store):
        _analysis(store, "r1", started_at=T0, status="ok")
        _analysis(store, "r2", started_at=T1, status="error")
        assert [a["run_id"] for a in store.analyses_between("NVDA", T0, T2)] == ["r1", "r2"]
        assert store.analyses_between("NVDA", T1, T1) == []          # the end is exclusive
        assert store.analyses_between("NVDA", T0, T2, kind="offhours") == []

    def test_model_spend_is_summed_from_a_point_in_time(self, store):
        _analysis(store, "r1", started_at=T0, status="ok", cost_usd=0.9)
        _analysis(store, "r2", started_at=T1, status="ok", cost_usd=1.1)
        _analysis(store, "r3", started_at=T2, status="error")       # failed before any usage
        assert store.llm_cost_since(T0) == pytest.approx(2.0)
        assert store.llm_cost_since(T1) == pytest.approx(1.1)
        assert store.llm_cost_since("2026-11-12T00:00:00Z") == 0.0

    def test_agent_steps_are_kept_in_order(self, store):
        _analysis(store, "r1")
        steps = [{"seq": i, "agent": agent, "kind": kind, "name": name, "started_at": T0,
                  "elapsed_seconds": 1.5, "input_tokens": 100, "output_tokens": 20, "output_chars": 80, "error": None}
                 for i, (agent, kind, name) in enumerate([("Market Analyst", "llm", "claude-sonnet-5"),
                                                           ("Market Analyst", "tool", "get_stock_data")])]
        store.add_agent_steps("r1", steps)
        rows = store.query("SELECT agent, kind, name FROM agent_steps WHERE run_id = 'r1' ORDER BY seq")
        assert [r["name"] for r in rows] == ["claude-sonnet-5", "get_stock_data"]


@pytest.mark.unit
class TestOrdersAndFills:
    def _order(self, store, client_order_id="tl-1-NVDA-buy", status="new", broker="simulate", **more):
        store.add_order(client_order_id=client_order_id, tick_id=_tick(store, f"t-{client_order_id}"),
                        ticker="NVDA", side="buy", qty=10, type="limit", limit_price=169.2, stop_price=None,
                        time_in_force="day", reason="target 0.12", status=status, broker=broker,
                        broker_order_id=None, submitted_at=T0, updated_at=T0, **more)

    def test_an_order_is_updated_as_it_fills(self, store):
        self._order(store)
        store.update_order("tl-1-NVDA-buy", status="filled", filled_qty=10, filled_avg_price=169.05, updated_at=T1)
        order = store.order("tl-1-NVDA-buy")
        assert (order["status"], order["filled_qty"], order["filled_avg_price"]) == ("filled", 10, 169.05)

    def test_the_same_client_order_id_cannot_be_recorded_twice(self, store):
        self._order(store)
        with pytest.raises(sqlite3.IntegrityError):
            store.add_order(client_order_id="tl-1-NVDA-buy", tick_id="t-tl-1-NVDA-buy", ticker="NVDA", side="buy",
                            qty=10, type="limit", limit_price=1.0, stop_price=None, time_in_force="day",
                            reason="", status="new", broker="simulate", broker_order_id=None,
                            submitted_at=T0, updated_at=T0)

    def test_open_orders_are_this_brokers_unfinished_ones(self, store):
        self._order(store, "a", status="new")
        self._order(store, "b", status="filled")
        self._order(store, "c", status="partially_filled")
        self._order(store, "d", status="new", broker="alpaca_paper")
        assert [o["client_order_id"] for o in store.open_orders("simulate")] == ["a", "c"]

    def test_a_fill_seen_twice_is_recorded_once(self, store):
        fill = {"fill_id": "f1", "broker_order_id": "o1", "client_order_id": "tl-1-NVDA-buy", "ticker": "NVDA",
                "side": "buy", "qty": 10, "price": 169.05, "at": T1, "fee_usd": 0.0, "cost_usd": 0.51}
        assert store.add_fills([fill]) == 1
        assert store.add_fills([fill, {**fill, "fill_id": "f2"}]) == 1
        assert len(store.query("SELECT * FROM fills")) == 2


@pytest.mark.unit
class TestOutcomes:
    def test_only_outcomes_whose_time_has_come_are_due(self, store):
        _analysis(store, "r1", status="ok")
        store.add_outcomes([{"run_id": "r1", "horizon": "60m", "due_at": T1},
                            {"run_id": "r1", "horizon": "1d", "due_at": "2026-11-12T21:00:00Z"}])
        (due,) = store.due_outcomes(T1)
        assert (due["horizon"], due["ticker"], due["ref_price"]) == ("60m", "NVDA", 168.89)

        store.mark_outcome("r1", "60m", 170.58, 0.01, T2)
        assert store.due_outcomes(T2) == []
        (row,) = store.query("SELECT * FROM outcomes WHERE horizon = '60m'")
        assert row["ret"] == 0.01 and row["marked_at"] == T2

    def test_adding_the_same_outcome_again_keeps_the_first(self, store):
        _analysis(store, "r1", status="ok")
        row = {"run_id": "r1", "horizon": "60m", "due_at": T1}
        store.add_outcomes([row])
        store.mark_outcome("r1", "60m", 170.0, 0.0066, T2)
        store.add_outcomes([row])
        assert store.query("SELECT price FROM outcomes")[0]["price"] == 170.0


@pytest.mark.unit
class TestTheRest:
    def test_small_values_outlive_the_connection(self, tmp_path):
        with Store(tmp_path / "loop.db") as s:
            s.put("sim_book", {"cash": 98_000.5, "positions": {"NVDA": {"qty": 10, "avg_price": 168.5}}})
            s.put("sim_book", {"cash": 97_000.0, "positions": {}})
        with Store(tmp_path / "loop.db") as s:
            assert s.get("sim_book") == {"cash": 97_000.0, "positions": {}}
            assert s.get("missing", default=[]) == []
            assert s.get("schema_version") == SCHEMA_VERSION

    def test_a_file_from_another_schema_is_refused(self, tmp_path):
        with Store(tmp_path / "loop.db") as s:
            s.put("schema_version", SCHEMA_VERSION + 1)
        with pytest.raises(RuntimeError, match="schema version"):
            Store(tmp_path / "loop.db")

    def test_events_keep_their_data(self, store):
        store.log("warning", "gate", "budget reached", at=T0, ticker="NVDA", data={"spent": 61.2})
        (event,) = store.query("SELECT * FROM events")
        assert event["data"] == {"spent": 61.2} and event["level"] == "warning"

    def test_a_failed_batch_writes_nothing(self, store):
        tick_id = _tick(store)
        good = {"tick_id": tick_id, "ticker": "NVDA", "price": 1.0, "price_at": T0, "prev_close": 1.0,
                "position_qty": 0, "new_headlines": 0, "gate_action": "reuse", "gate_reasons": []}
        with pytest.raises(sqlite3.IntegrityError):
            store.add_observations([good, {**good, "ticker": "CSCO", "gate_action": None}])
        assert store.query("SELECT * FROM observations") == []

    def test_parallel_writers_lose_nothing(self, store):
        tick_id = _tick(store)

        def write(i):
            store.begin_analysis(run_id=f"r{i}", tick_id=tick_id, ticker=f"T{i % 5}", trade_date="2026-11-11",
                                 kind="core", started_at=T0, ref_price=1.0, artifacts_dir="")
            store.finish_analysis(f"r{i}", status="ok", cost_usd=0.01)

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(write, range(200)))
        assert store.one("SELECT COUNT(*) AS n FROM analyses WHERE status = 'ok'")["n"] == 200
        assert store.llm_cost_since(T0) == pytest.approx(2.0)
