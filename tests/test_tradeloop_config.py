"""The loop's settings: defaults that load, files that are read, and mistakes that are refused."""

from __future__ import annotations

import json
import tomllib
from datetime import UTC, datetime, timedelta, timezone

import pytest

from tradeloop import timeutil
from tradeloop.config import CONFIG_TEMPLATE, KEEP, LoopConfig


@pytest.mark.unit
class TestDefaults:
    def test_the_defaults_are_a_valid_config(self):
        cfg = LoopConfig()
        assert cfg.tickers == ["NVDA", "CSCO", "HD", "DIS", "GEV"]
        assert cfg.interval_minutes == 60
        assert cfg.execution == "simulate"          # nothing leaves the machine by default
        assert cfg.rating_weights["Hold"] == KEEP
        assert cfg.workers == 5                     # every ticker at once

    def test_the_written_template_loads_as_the_defaults(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        loaded = LoopConfig.load(path)
        assert loaded.model_dump(exclude={"data_dir"}) == LoopConfig().model_dump(exclude={"data_dir"})
        assert tomllib.loads(CONFIG_TEMPLATE)["execution"] == "simulate"

    def test_paths_hang_off_the_data_directory(self, tmp_path):
        cfg = LoopConfig(data_dir=tmp_path)
        assert cfg.db_path == tmp_path / "tradeloop.db"
        assert cfg.runs_dir == tmp_path / "runs"
        assert cfg.dashboard_dir == tmp_path / "dashboard"

    def test_a_round_trip_costs_the_spread_and_slippage_twice(self):
        cfg = LoopConfig(half_spread_bps=2.0, slippage_bps=1.0)
        assert cfg.round_trip_cost == pytest.approx(0.0006)

    def test_the_analysis_config_takes_the_llm_overrides(self):
        cfg = LoopConfig(llm={"quick_think_llm": "claude-sonnet-5", "max_debate_rounds": 2})
        built = cfg.tradingagents_config()
        assert built["quick_think_llm"] == "claude-sonnet-5" and built["max_debate_rounds"] == 2
        assert "data_vendors" in built              # the rest is TradingAgents' own defaults


@pytest.mark.unit
class TestLoading:
    def test_a_json_file_is_read_too(self, tmp_path):
        path = tmp_path / "loop.json"
        path.write_text(json.dumps({"tickers": ["aapl", "AAPL", " msft "], "interval_minutes": 30}))
        cfg = LoopConfig.load(path)
        assert cfg.tickers == ["AAPL", "MSFT"]      # upper-cased, once each
        assert cfg.interval_minutes == 30

    def test_no_file_means_the_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADELOOP_DATA_DIR", str(tmp_path))
        assert LoopConfig.load().tickers == LoopConfig().tickers

    def test_the_data_directory_file_is_found_without_a_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADELOOP_DATA_DIR", str(tmp_path))
        (tmp_path / "config.toml").write_text('tickers = ["SPY"]\n')
        assert LoopConfig.load().tickers == ["SPY"]

    def test_an_unreadable_file_names_itself(self, tmp_path):
        path = tmp_path / "broken.toml"
        path.write_text("tickers = [")
        with pytest.raises(ValueError, match="broken.toml"):
            LoopConfig.load(path)


@pytest.mark.unit
class TestRefusals:
    @pytest.mark.parametrize("fields, message", [
        ({"tickers": []}, "at least one ticker"),
        ({"tickers": ["../etc"]}, None),
        ({"analysts": ["market", "macro"]}, "analysts must be chosen"),
        ({"analysts": []}, "analysts must be chosen"),
        ({"interval_minutes": 2}, "no time for an analysis"),
        ({"max_position_weight": 1.5}, "does not use margin"),
        ({"max_gross_exposure": 2.0}, "does not use margin"),
        ({"confirm_rounds": 0}, "must be positive"),
        ({"max_workers": 0}, "at least 1"),
        ({"rating_weights": {"Buy": 1.0}}, "exactly these ratings"),
        ({"rating_weights": {"Buy": 1.5, "Overweight": 0.6, "Hold": "keep", "Underweight": 0.3, "Sell": 0}},
         "between 0 and 1"),
        ({"execution": "live"}, None),
        ({"intervall_minutes": 30}, None),         # a misspelt key is not silently ignored
    ])
    def test_a_bad_setting_is_refused(self, fields, message):
        with pytest.raises(ValueError, match=message):
            LoopConfig(**fields)

    def test_analysts_keep_the_pipeline_order(self):
        assert LoopConfig(analysts=["news", "market"]).analysts == ["market", "news"]


@pytest.mark.unit
class TestTimes:
    def test_times_are_stored_as_sortable_utc(self):
        eastern = timezone(timedelta(hours=-5))
        assert timeutil.iso(datetime(2026, 11, 11, 10, 35, tzinfo=eastern)) == "2026-11-11T15:35:00Z"
        assert timeutil.iso(datetime(2026, 11, 11, 9, 0, tzinfo=UTC)) < timeutil.iso(datetime(2026, 11, 11, 10, 0, tzinfo=UTC))

    def test_a_naive_time_is_refused(self):
        with pytest.raises(ValueError, match="naive"):
            timeutil.iso(datetime(2026, 11, 11, 10, 35))

    @pytest.mark.parametrize("text, expected", [
        ("2026-11-10T20:59:59.901302182Z", datetime(2026, 11, 10, 20, 59, 59, 901302, tzinfo=UTC)),  # nanoseconds
        ("2026-11-10T21:00:00Z", datetime(2026, 11, 10, 21, 0, tzinfo=UTC)),
        ("2026-11-10T16:00:00-05:00", datetime(2026, 11, 10, 21, 0, tzinfo=UTC)),
        ("2026-11-10T21:00:00", datetime(2026, 11, 10, 21, 0, tzinfo=UTC)),                          # no zone reads as UTC
    ])
    def test_broker_timestamps_are_read(self, text, expected):
        assert timeutil.parse(text) == expected

    def test_a_time_survives_the_round_trip(self):
        moment = datetime(2026, 11, 11, 15, 35, tzinfo=UTC)
        assert timeutil.parse(timeutil.iso(moment)) == moment


@pytest.mark.unit
class TestTheChartSite:
    def test_it_defaults_to_a_quote_page_and_can_be_changed_or_turned_off(self):
        assert LoopConfig().chart_url == "https://finance.yahoo.com/quote/{ticker}"
        assert LoopConfig(chart_url=" https://charts.example/{ticker} ").chart_url == "https://charts.example/{ticker}"
        assert LoopConfig(chart_url="").chart_url == ""

    @pytest.mark.parametrize("value", ["finance.yahoo.com/quote/{ticker}", "https://finance.yahoo.com/quote/NVDA",
                                       "javascript:alert('{ticker}')"])
    def test_anything_but_a_web_address_with_the_ticker_in_it_is_refused(self, value):
        with pytest.raises(ValueError, match="chart_url"):
            LoopConfig(chart_url=value)
