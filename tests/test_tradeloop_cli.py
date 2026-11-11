"""The loop's commands, driven without a network: a scripted Alpaca, scripted analyses."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tests.test_tradeloop_engine import CALENDAR, TICKERS, FakeAnalysts, FakeMarket, et
from tradeloop import cli
from tradeloop.alpaca import AlpacaError, Bar
from tradeloop.config import LoopConfig
from tradeloop.store import Store

runner = CliRunner()


class FakeClient(FakeMarket):
    """An Alpaca that is up, has an active paper account and knows the calendar."""

    def __init__(self):
        super().__init__()
        self.account_payload = {"status": "ACTIVE", "trading_blocked": False, "equity": "100000", "cash": "100000"}

    def account(self):
        return self.account_payload

    def clock(self):
        return {"is_open": False}

    def calendar(self, start, end):
        return [row for row in CALENDAR if start <= row["date"] <= end] if start == end else CALENDAR

    def positions(self):
        return []


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A data directory of the test's own, and keys that are 'there'."""
    monkeypatch.setenv("TRADELOOP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APCA_API_KEY_ID", "PKTEST")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    return tmp_path


@pytest.fixture
def wired(home, monkeypatch):
    """``build_loop`` with a scripted client and scripted analyses; returns them."""
    client, clock = FakeClient(), [et(11, "10:35")]
    analysts = FakeAnalysts(clock)
    real = cli.build_loop
    monkeypatch.setattr(cli, "build_loop", lambda cfg: real(cfg, client=client, analysts=analysts))
    monkeypatch.setattr(cli, "AlpacaClient", lambda key, secret: client)     # no command reaches the network
    monkeypatch.setattr(cli.timeutil, "utcnow", lambda: clock[0])
    return client, analysts, clock


@pytest.mark.unit
class TestInit:
    def test_it_writes_a_config_that_loads(self, home):
        result = runner.invoke(cli.app, ["init"])
        assert result.exit_code == 0 and (home / "config.toml").is_file()
        assert LoopConfig.load().tickers == TICKERS

    def test_it_does_not_overwrite_an_existing_file_unless_forced(self, home):
        (home / "config.toml").write_text('tickers = ["SPY"]\n')
        assert runner.invoke(cli.app, ["init"]).exit_code == 1
        assert LoopConfig.load().tickers == ["SPY"]
        assert runner.invoke(cli.app, ["init", "--force"]).exit_code == 0
        assert LoopConfig.load().tickers == TICKERS


@pytest.mark.unit
class TestCheck:
    def test_everything_in_place_passes(self, home, monkeypatch):
        monkeypatch.setattr(cli, "AlpacaClient", lambda key, secret: FakeClient())
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = runner.invoke(cli.app, ["check"])
        assert result.exit_code == 0, result.output
        assert "Alpaca paper account" in result.output and "status ACTIVE" in result.output
        assert "prices for every ticker" in result.output and "FAIL" not in result.output
        assert "OPENAI_API_KEY is set" in result.output

    def test_a_missing_model_key_fails_and_names_the_variable(self, home, monkeypatch):
        monkeypatch.setattr(cli, "AlpacaClient", lambda key, secret: FakeClient())
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = runner.invoke(cli.app, ["check"])
        assert result.exit_code == 1 and "OPENAI_API_KEY is not set" in result.output

    def test_a_blocked_account_fails(self, home, monkeypatch):
        client = FakeClient()
        client.account_payload["trading_blocked"] = True
        monkeypatch.setattr(cli, "AlpacaClient", lambda key, secret: client)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert runner.invoke(cli.app, ["check"]).exit_code == 1

    def test_missing_alpaca_keys_are_reported_not_raised(self, home, monkeypatch):
        monkeypatch.delenv("APCA_API_KEY_ID")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = runner.invoke(cli.app, ["check"])
        assert result.exit_code == 1 and "no Alpaca paper keys found" in result.output

    def test_a_ticker_with_no_price_fails(self, home, monkeypatch):
        client = FakeClient()
        del client.prices["GEV"]
        monkeypatch.setattr(cli, "AlpacaClient", lambda key, secret: client)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = runner.invoke(cli.app, ["check"])
        assert result.exit_code == 1 and "no price for GEV" in result.output

    def test_a_bad_config_file_is_named(self, home):
        (home / "config.toml").write_text("interval_minutes = 1\n")
        result = runner.invoke(cli.app, ["check"])
        assert result.exit_code == 1 and "no time for an analysis" in result.output


@pytest.mark.unit
class TestTick:
    def test_one_tick_runs_records_and_builds_the_dashboard(self, home, wired):
        client, analysts, clock = wired
        result = runner.invoke(cli.app, ["tick", "--kind", "premarket"])
        assert result.exit_code == 0, result.output
        assert "20261111T1035-premarket  ok" in result.output and "5 analysed" in result.output
        assert len(analysts.calls) == 1 and len(analysts.calls[0]) == 5
        assert (home / "dashboard" / "index.html").is_file() and (home / "dashboard" / "day-2026-11-11.html").is_file()
        with Store(home / "tradeloop.db") as store:
            assert store.tick("20261111T1035-premarket")["status"] == "ok"

    def test_a_core_tick_during_the_session_trades_on_the_simulated_book(self, home, wired):
        client, analysts, clock = wired
        (home / "config.toml").write_text("confirm_rounds = 1\n")
        analysts.say("NVDA", "Buy")
        result = runner.invoke(cli.app, ["tick"])
        assert result.exit_code == 0 and "1 orders" in result.output
        with Store(home / "tradeloop.db") as store:
            (order,) = store.query("SELECT * FROM orders")
            assert (order["ticker"], order["side"], order["status"], order["broker"]) == ("NVDA", "buy", "filled", "simulate")

    def test_an_unknown_kind_is_refused(self, home, wired):
        result = runner.invoke(cli.app, ["tick", "--kind", "lunch"])
        assert result.exit_code == 1 and "kind must be one of" in result.output

    def test_a_failed_tick_exits_non_zero(self, home, wired):
        client, analysts, clock = wired
        client.down = True
        result = runner.invoke(cli.app, ["tick"])
        assert result.exit_code == 1 and "error" in result.output


@pytest.mark.unit
class TestStatusAndDashboard:
    def test_status_shows_the_book_the_decisions_and_the_next_tick(self, home, wired):
        client, analysts, clock = wired
        analysts.say("NVDA", ("Overweight", {"target_weight": 0.3, "stop_loss": 95.0}))
        runner.invoke(cli.app, ["tick", "--kind", "premarket"])
        result = runner.invoke(cli.app, ["status"])
        assert result.exit_code == 0, result.output
        assert "next tick 11-11 11:35 (core)" in result.output and "market open" in result.output
        assert "equity 100,000.00" in result.output and "model spend today 5.00 of 60.00" in result.output
        assert "Overweight at 11-11 10:35, target 0.3, stop 95" in result.output
        assert "CSCO" in result.output and "Hold at 11-11 10:35" in result.output

    def test_the_dashboard_command_builds_every_day(self, home, wired):
        runner.invoke(cli.app, ["tick", "--kind", "premarket"])
        (home / "dashboard" / "day-2026-11-11.html").unlink()
        result = runner.invoke(cli.app, ["dashboard"])
        assert result.exit_code == 0 and "index.html" in result.output
        assert (home / "dashboard" / "day-2026-11-11.html").is_file()

    def test_the_dashboard_command_first_brings_the_charts_prices_up_to_date(self, home, wired):
        client, _, clock = wired
        runner.invoke(cli.app, ["tick", "--kind", "premarket"])
        client.series["NVDA"] = [Bar(et(11, "10:40"), 100.0, 100.5, 99.8, 100.2, 900),
                                 Bar(et(11, "10:45"), 100.2, 100.9, 100.1, 100.7, 700)]
        clock[0] = et(11, "10:50")
        result = runner.invoke(cli.app, ["dashboard"])
        assert result.exit_code == 0, result.output
        with Store(home / "tradeloop.db") as store:
            assert [b["close"] for b in store.query("SELECT close FROM bars WHERE ticker = 'NVDA' ORDER BY at")] == [100.2, 100.7]

    def test_the_dashboard_still_builds_when_the_prices_cannot_be_fetched(self, home, wired):
        client, _, clock = wired
        runner.invoke(cli.app, ["tick", "--kind", "premarket"])

        def down(*args, **kwargs):
            raise AlpacaError(0, "GET /v2/stocks/bars did not reach Alpaca (ConnectionError)")

        client.bars = down
        clock[0] = et(11, "10:50")
        result = runner.invoke(cli.app, ["dashboard"])
        assert result.exit_code == 0 and "not brought up to date (AlpacaError)" in result.output
        assert (home / "dashboard" / "index.html").is_file()


@pytest.mark.unit
class TestRun:
    def test_run_goes_tick_by_tick_until_interrupted(self, home, wired, monkeypatch):
        client, analysts, clock = wired
        clock[0] = et(11, "08:30")
        ticks = []

        def sleep(seconds):
            from datetime import timedelta

            clock[0] += timedelta(seconds=seconds)
            if len(ticks) >= 2:
                raise KeyboardInterrupt

        monkeypatch.setattr(cli.time, "sleep", sleep)
        monkeypatch.setattr(cli, "_print_report", lambda report: ticks.append(report.tick_id))
        result = runner.invoke(cli.app, ["run"])
        assert result.exit_code == 0 and "Stopped." in result.output
        assert ticks == ["20261111T0845-premarket", "20261111T0935-core"]

    def test_it_stops_by_itself_after_the_last_sessions_post_close_tick(self, home, wired, monkeypatch):
        from datetime import timedelta

        client, analysts, clock = wired
        clock[0] = et(11, "15:40")
        ticks = []
        monkeypatch.setattr(cli.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + timedelta(seconds=seconds)))
        monkeypatch.setattr(cli, "_print_report", lambda report: ticks.append(report.tick_id))
        result = runner.invoke(cli.app, ["run", "--sessions", "2"])
        assert result.exit_code == 0, result.output
        assert "Stops after the tick at 11-12 16:30 EST" in result.output
        assert "The last session's post-close tick is done." in result.output
        assert ticks[0] == "20261111T1630-postclose" and ticks[-1] == "20261112T1630-postclose"
        assert not (home / "run.pid").exists()                  # the lock is released

    def test_more_sessions_than_the_calendar_knows_is_refused(self, home, wired):
        result = runner.invoke(cli.app, ["run", "--sessions", "99"])
        assert result.exit_code == 1 and "--sessions must be between 1 and" in result.output

    def test_a_second_loop_on_the_same_data_is_refused(self, home, wired):
        import os

        (home / "run.pid").write_text(str(os.getpid()))         # a process that is certainly alive
        result = runner.invoke(cli.app, ["run"])
        assert result.exit_code == 1 and "already running" in result.output

    def test_a_stale_lock_from_a_dead_process_does_not_block_a_start(self, home, wired, monkeypatch):
        (home / "run.pid").write_text("999999")
        monkeypatch.setattr(cli.time, "sleep", lambda s: (_ for _ in ()).throw(KeyboardInterrupt))
        result = runner.invoke(cli.app, ["run"])
        assert result.exit_code == 0 and "Stopped." in result.output

    def test_stop_signals_the_running_loop_and_says_so_when_there_is_none(self, home, wired, monkeypatch):
        import os
        import signal

        assert "No loop is running." in runner.invoke(cli.app, ["stop"]).output
        (home / "run.pid").write_text(str(os.getpid()))
        sent = []
        monkeypatch.setattr(cli.os, "kill", lambda pid, sig: sent.append((pid, sig)))
        result = runner.invoke(cli.app, ["stop"])
        assert sent[-1] == (os.getpid(), signal.SIGTERM) and "Asked the loop" in result.output

    def test_missing_keys_stop_it_before_anything_runs(self, home, monkeypatch):
        monkeypatch.delenv("APCA_API_KEY_ID")
        result = runner.invoke(cli.app, ["run"])
        assert result.exit_code == 1 and "no Alpaca paper keys found" in result.output


@pytest.mark.unit
def test_the_alpaca_error_type_is_what_the_commands_catch():
    assert issubclass(AlpacaError, RuntimeError)
