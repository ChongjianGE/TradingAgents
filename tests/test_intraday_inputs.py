"""What a run made during the day needs from the pipeline: intraday bars, the
current time, what was already decided this session, and a decision in numbers."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from tradingagents.agents import tools
from tradingagents.agents.context import build_instrument_context, get_portfolio_context_from_state
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating, render_pm_decision
from tradingagents.dataflows import router
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors.yahoo import intraday
from tradingagents.graph.propagation import Propagator

TODAY = date.today()


def _bars(day: date, times=("04:00", "09:30", "10:00", "16:00"), tz="America/New_York"):
    index = pd.DatetimeIndex([pd.Timestamp(f"{day} {t}", tz=tz) for t in times])
    return pd.DataFrame({"Open": 100.0, "High": 101.5, "Low": 99.25, "Close": 100.75, "Volume": 12345}, index=index)


class _Ticker:
    """Stands in for yfinance.Ticker; keeps what history was asked for."""

    asked: dict = {}
    frame: pd.DataFrame | None = None

    def __init__(self, symbol):
        type(self).asked["symbol"] = symbol

    def history(self, **kwargs):
        type(self).asked.update(kwargs)
        return type(self).frame


@pytest.fixture
def yahoo(monkeypatch):
    _Ticker.asked, _Ticker.frame = {}, _bars(TODAY)
    monkeypatch.setattr(intraday.yf, "Ticker", _Ticker)
    return _Ticker


@pytest.mark.unit
class TestIntradayBars:
    def test_bars_are_listed_with_their_session(self, yahoo):
        text = intraday.get_intraday_bars("NVDA", "30m", 3, TODAY.isoformat())
        lines = text.strip().splitlines()
        assert lines[0].startswith("# Intraday 30m bars for NVDA")
        assert "Datetime,Open,High,Low,Close,Volume,Session" in lines
        assert f"{TODAY} 04:00,100.00,101.50,99.25,100.75,12345,pre-market" in lines
        assert lines[-3].endswith("regular") and lines[-1].endswith("after-hours")
        assert "Total records: 4" in text and "may still be forming" in text

    def test_the_request_covers_the_window_and_extended_hours(self, yahoo):
        intraday.get_intraday_bars("NVDA", "5m", 2, TODAY.isoformat())
        assert yahoo.asked == {"symbol": "NVDA", "start": (TODAY - timedelta(days=2)).isoformat(),
                               "end": (TODAY + timedelta(days=1)).isoformat(), "interval": "5m", "prepost": True}

    def test_the_window_is_kept_between_one_and_ten_days(self, yahoo):
        intraday.get_intraday_bars("NVDA", "30m", 500, TODAY.isoformat())
        assert yahoo.asked["start"] == (TODAY - timedelta(days=10)).isoformat()
        intraday.get_intraday_bars("NVDA", "30m", 0, TODAY.isoformat())
        assert yahoo.asked["start"] == (TODAY - timedelta(days=1)).isoformat()

    def test_bars_after_the_analysis_date_are_never_served(self, yahoo):
        yesterday = TODAY - timedelta(days=1)
        yahoo.frame = pd.concat([_bars(yesterday), _bars(TODAY)])
        text = intraday.get_intraday_bars("NVDA", "30m", 3, yesterday.isoformat())
        assert str(TODAY) not in text.split("\n\n", 1)[1] and "Total records: 4" in text

    def test_a_date_older_than_yahoo_keeps_is_unavailable_not_guessed(self, yahoo):
        old = (TODAY - timedelta(days=90)).isoformat()
        with pytest.raises(NoMarketDataError, match="keeps 30m bars for 60 days"):
            intraday.get_intraday_bars("NVDA", "30m", 3, old)
        assert yahoo.asked == {}                       # nothing was requested

    def test_an_unknown_interval_is_refused(self, yahoo):
        with pytest.raises(ValueError, match="choose one of 5m, 15m, 30m, 60m"):
            intraday.get_intraday_bars("NVDA", "1m", 3, TODAY.isoformat())

    def test_an_empty_answer_is_an_absence(self, yahoo, monkeypatch):
        yahoo.frame = pd.DataFrame()
        monkeypatch.setattr("tradingagents.dataflows.vendors.yahoo.common.vendor_reachable", lambda host: True)
        with pytest.raises(NoMarketDataError):
            intraday.get_intraday_bars("NVDA", "30m", 3, TODAY.isoformat())

    def test_another_exchanges_bars_carry_no_us_session_label(self, yahoo):
        yahoo.frame = _bars(TODAY, times=("09:30", "10:00"), tz="Asia/Hong_Kong")
        text = intraday.get_intraday_bars("0700.HK", "30m", 3, TODAY.isoformat())
        assert "Datetime,Open,High,Low,Close,Volume\n" in text and "Asia/Hong_Kong" in text


@pytest.mark.unit
class TestTheIntradayTool:
    def test_the_model_sees_only_the_bar_size_and_window(self):
        properties = tools.get_intraday_data.tool_call_schema.model_json_schema()["properties"]
        assert set(properties) == {"interval", "look_back_days"}

    def test_it_is_routed_like_the_other_price_tools(self):
        assert router.get_category_for_method("get_intraday_data") == "core_stock_apis"
        assert set(router.VENDOR_METHODS["get_intraday_data"]) == {"yfinance"}

    def test_it_serves_the_runs_instrument_and_date(self, monkeypatch):
        seen = []
        monkeypatch.setitem(router.VENDOR_METHODS["get_intraday_data"], "yfinance",
                            lambda *a: seen.append(a) or "bars")
        out = tools.get_intraday_data.func("NVDA", "15m", 2, "2026-11-11")
        assert out == "bars" and seen == [("NVDA", "15m", 2, "2026-11-11")]

    def test_a_vendor_that_does_not_serve_it_is_told_to_the_model_not_raised(self, monkeypatch):
        from tradingagents.dataflows import config

        monkeypatch.setattr(config, "get_config",
                            lambda: {"data_vendors": {"core_stock_apis": "alpha_vantage"}, "tool_vendors": {}})
        monkeypatch.setattr(router, "get_config", config.get_config)
        assert "not available" in tools.get_intraday_data.func("NVDA", "30m", 3, "2026-11-11")

    def test_the_market_analyst_is_offered_it(self):
        from tradingagents.agents.analysts import market_analyst

        assert tools.get_intraday_data in market_analyst.TOOLS


@pytest.mark.unit
class TestTheCurrentTime:
    def test_a_run_without_a_time_reads_as_before(self):
        assert "current time" not in build_instrument_context("NVDA")

    def test_a_time_is_told_with_what_it_means_for_the_daily_bar(self):
        context = build_instrument_context("NVDA", as_of_time="2026-11-11 10:35 ET, regular session")
        assert "The current time is 2026-11-11 10:35 ET, regular session." in context
        assert "still forming" in context

    def test_the_graph_passes_the_time_only_when_there_is_one(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = object.__new__(TradingAgentsGraph)
        graph.propagator = Propagator()
        seen = []
        graph.resolve_instrument_context = lambda *a, **k: seen.append((a, k)) or "ctx"

        graph.create_run_state("NVDA", TODAY.isoformat())
        graph.create_run_state("NVDA", TODAY.isoformat(), as_of_time="10:35 ET")
        assert seen[0][1] == {} and seen[1][1] == {"as_of_time": "10:35 ET"}


@pytest.mark.unit
class TestTheSessionContext:
    def test_it_starts_empty_and_is_carried_in_the_state(self):
        assert Propagator().create_initial_state("NVDA", "2026-11-11")["session_context"] == ""
        state = Propagator().create_initial_state("NVDA", "2026-11-11", session_context="Earlier today: Hold at 09:35.")
        assert state["session_context"] == "Earlier today: Hold at 09:35."

    def test_the_decision_agents_read_it_beside_the_portfolio(self):
        state = {"portfolio_context": "Portfolio at the analysis date:\n- No current position in NVDA",
                 "session_context": "Earlier today: Overweight at 09:35; bought 40 at 168.90."}
        text = get_portfolio_context_from_state(state)
        assert text.startswith("Portfolio at the analysis date:") and text.endswith("bought 40 at 168.90.")

    def test_it_is_read_even_when_no_portfolio_was_given(self):
        text = get_portfolio_context_from_state({"session_context": "Earlier today: Hold at 09:35."})
        assert "Portfolio context: not provided" in text and "Earlier today: Hold at 09:35." in text

    def test_without_it_the_portfolio_block_is_unchanged(self):
        state = {"portfolio_context": "Portfolio at the analysis date:\n- Cash available: 100.00"}
        assert get_portfolio_context_from_state(state) == state["portfolio_context"]
        assert get_portfolio_context_from_state({**state, "session_context": "  "}) == state["portfolio_context"]


def _decision(**more):
    return PortfolioDecision(rating=PortfolioRating.OVERWEIGHT, executive_summary="s", investment_thesis="t", **more)


@pytest.mark.unit
class TestTheDecisionInNumbers:
    @pytest.mark.parametrize("written, read", [
        (0.3, 0.3), ("0.3", 0.3), ("30%", 0.3), (30, 0.3), ("30", 0.3), (1, 1.0), (0, 0.0), ("100%", 1.0),
        (None, None), ("N/A", None), ("", None), ("a third", None), (-0.2, None), (250, None), (float("nan"), None),
        (True, None),
    ])
    def test_a_target_weight_is_read_as_a_fraction(self, written, read):
        assert _decision(target_weight=written).target_weight == read

    @pytest.mark.parametrize("written, read", [(151.5, 151.5), ("$151.50", 151.5), ("10%", None), ("None", None)])
    def test_a_stop_loss_is_a_price_or_nothing(self, written, read):
        assert _decision(stop_loss=written).stop_loss == read

    def test_both_are_named_in_the_rendered_decision_even_when_absent(self):
        given = render_pm_decision(_decision(target_weight=0.3, stop_loss=151.5))
        assert "**Target Weight**: 0.3" in given and "**Stop Loss**: 151.5" in given
        absent = render_pm_decision(_decision())
        assert "**Target Weight**: not provided" in absent and "**Stop Loss**: not provided" in absent

    def test_the_rating_still_parses_from_the_longer_rendering(self):
        from tradingagents.agents.rating import parse_rating

        assert parse_rating(render_pm_decision(_decision(target_weight=0.3, stop_loss=151.5))) == "Overweight"

    def test_the_portfolio_manager_keeps_the_typed_decision_in_the_state(self):
        from langchain_core.runnables import RunnableLambda

        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

        typed = _decision(target_weight=0.3, stop_loss=151.5, price_target=204.0)

        class Model:
            def with_structured_output(self, schema):
                return RunnableLambda(lambda _: typed)

        state = Propagator().create_initial_state("NVDA", "2026-11-11")
        state.update(investment_plan="plan", trader_investment_plan="proposal")
        out = create_portfolio_manager(Model())(state)
        assert out["final_rating"] == "Overweight"
        assert out["final_decision"] == {
            "rating": "Overweight", "executive_summary": "s", "investment_thesis": "t",
            "price_target": 204.0, "time_horizon": None, "target_weight": 0.3, "stop_loss": 151.5}

    def test_a_free_text_decision_has_no_typed_fields(self):
        from langchain_core.messages import AIMessage

        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

        class Model:
            def with_structured_output(self, schema):
                raise NotImplementedError

            def invoke(self, prompt):
                return AIMessage(content="**Rating**: Sell\n\nExit.")

        state = Propagator().create_initial_state("NVDA", "2026-11-11")
        state.update(investment_plan="plan", trader_investment_plan="proposal")
        out = create_portfolio_manager(Model())(state)
        assert out["final_rating"] == "Sell" and out["final_decision"] == {}
