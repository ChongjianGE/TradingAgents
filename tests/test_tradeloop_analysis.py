"""Analyses for the loop: the real graph driven by a scripted model, recorded step
by step, several tickers at once."""

from __future__ import annotations

import json
import threading
import time

import pytest

from tests.test_graph_end_to_end import ScriptedModel, _Client, offline  # noqa: F401 - offline is a fixture
from tradeloop.analysis import AnalysisRequest, Analyst, AnalystPool
from tradeloop.config import LoopConfig
from tradeloop.recorder import MAX_TEXT_CHARS, RunRecorder, _agent, _clip, _text
from tradingagents.graph import trading_graph
from tradingagents.portfolio import PortfolioContext

TRADE_DATE = "2026-01-09"


def _cfg(tmp_path, **more):
    return LoopConfig(data_dir=tmp_path / "loop", llm={
        "results_dir": str(tmp_path / "results"), "data_cache_dir": str(tmp_path / "cache"),
        "memory_log_path": str(tmp_path / "memory.md")}, **more)


@pytest.fixture
def scripted(monkeypatch):
    """Every graph built in the test talks to a scripted model; returns the models made."""
    made = []

    def client(config, tier, **kwargs):
        model = ScriptedModel(structured=True, callbacks=kwargs.get("callbacks"))
        made.append(model)
        return _Client(model)

    monkeypatch.setattr(trading_graph, "create_tier_client", client)
    return made


def _request(ticker="NVDA", run_id="r1", **more):
    return AnalysisRequest(ticker=ticker, run_id=run_id, trade_date=TRADE_DATE, **more)


@pytest.mark.unit
class TestOneAnalysis:
    def test_a_run_reaches_a_decision_and_keeps_its_typed_fields(self, tmp_path, offline, scripted):  # noqa: F811
        result = Analyst(_cfg(tmp_path)).run(_request())
        assert (result.status, result.rating, result.error) == ("ok", "Overweight", None)
        assert result.decision["rating"] == "Overweight" and result.summary == "s"
        assert result.elapsed_seconds > 0

    def test_every_step_is_named_by_the_agent_that_made_it(self, tmp_path, offline, scripted):  # noqa: F811
        result = Analyst(_cfg(tmp_path)).run(_request())
        agents = {s["agent"] for s in result.steps}
        assert {"Market Analyst", "News Analyst", "Fundamentals Analyst", "Bull Researcher", "Bear Researcher",
                "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"} <= agents
        assert "agent" not in agents and "unknown" not in agents      # the inner node name never shows

        tools = [s for s in result.steps if s["kind"] == "tool"]
        market_tools = {s["name"] for s in tools if s["agent"] == "Market Analyst"}
        assert market_tools == {"get_stock_data", "get_indicators", "get_verified_market_snapshot", "get_intraday_data"}
        # The scripted model answers structured calls outside the chat-model path,
        # so only the tool-using analysts and the debaters count as model calls here.
        assert result.tool_calls == len(tools) == 13 and result.llm_calls == 11
        assert [s["seq"] for s in result.steps] == list(range(len(result.steps)))

    def test_a_tool_step_keeps_what_the_model_asked_and_what_came_back(self, tmp_path, offline, scripted):  # noqa: F811
        result = Analyst(_cfg(tmp_path)).run(_request())
        step = next(s for s in result.steps if s["name"] == "get_indicators")
        assert step["args"] == {"indicator": "rsi", "curr_date": TRADE_DATE}     # what the model passed, no more
        assert step["output"] == "get_indicators data" and step["output_chars"] == len("get_indicators data")
        assert step["elapsed_seconds"] >= 0 and step["error"] is None

    def test_the_runs_texts_are_written_to_a_folder_of_its_own(self, tmp_path, offline, scripted):  # noqa: F811
        cfg = _cfg(tmp_path)
        book = PortfolioContext.model_validate({"cash": 1000.0, "positions": []})
        result = Analyst(cfg).run(_request(run_id="20260109T1035-NVDA", portfolio=book,
                                           as_of_time="2026-01-09 10:35 ET", session_context="Earlier: Hold."))
        folder = cfg.runs_dir / "20260109T1035-NVDA"
        assert result.artifacts_dir == str(folder)
        state = json.loads((folder / "state.json").read_text())
        assert state["final_rating"] == "Overweight" and state["market_report"].strip()
        assert "messages" not in state
        assert "The current time is 2026-01-09 10:35 ET" in state["instrument_context"]
        assert state["session_context"] == "Earlier: Hold."
        steps = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
        assert len(steps) == len(result.steps)
        request = json.loads((folder / "request.json").read_text())
        assert request["portfolio"]["cash"] == 1000.0 and request["status"] == "ok"

    def test_only_the_designated_run_goes_into_the_memory_log(self, tmp_path, offline, scripted):  # noqa: F811
        analyst = Analyst(_cfg(tmp_path))
        first = analyst.run(_request(run_id="r1"))
        assert not first.recorded_to_memory and analyst.graph.memory_log.load_entries() == []

        second = analyst.run(_request(run_id="r2", record_to_memory=True))
        assert second.recorded_to_memory
        assert [e["rating"] for e in analyst.graph.memory_log.load_entries()] == ["Overweight"]

    def test_a_graph_is_reused_and_each_run_starts_with_a_clean_record(self, tmp_path, offline, scripted):  # noqa: F811
        analyst = Analyst(_cfg(tmp_path))
        first = analyst.run(_request(run_id="r1"))
        second = analyst.run(_request(run_id="r2"))
        assert len(scripted) == 2                       # the two tiers of one graph, built once
        assert len(second.steps) == len(first.steps)    # not the two runs added together

    def test_a_failing_provider_is_an_error_result_with_the_steps_so_far(self, tmp_path, offline, monkeypatch):  # noqa: F811
        model = ScriptedModel(structured=True, fail_at=12)
        monkeypatch.setattr(trading_graph, "create_tier_client", lambda config, tier, **k: _Client(model))
        cfg = _cfg(tmp_path)
        result = Analyst(cfg).run(_request(run_id="r-fail"))
        assert result.status == "error" and "provider unavailable" in result.error and result.rating is None
        assert result.tool_calls > 0                    # the analysts' work before the failure is kept
        assert (cfg.runs_dir / "r-fail" / "steps.jsonl").is_file()

    def test_selected_analysts_are_the_only_ones_run(self, tmp_path, offline, scripted):  # noqa: F811
        result = Analyst(_cfg(tmp_path, analysts=["market"])).run(_request())
        assert result.status == "ok"
        assert {s["agent"] for s in result.steps if s["kind"] == "tool"} == {"Market Analyst"}


@pytest.mark.unit
class TestSeveralAtOnce:
    def test_five_tickers_run_at_the_same_time(self, tmp_path, offline, scripted):  # noqa: F811
        cfg = _cfg(tmp_path)
        pool = AnalystPool(cfg)
        tickers = ["NVDA", "CSCO", "HD", "DIS", "GEV"]
        results = pool.run_many([_request(t, f"r-{t}") for t in tickers])

        assert [r.ticker for r in results] == tickers and all(r.status == "ok" for r in results)
        # Each ticker's analysts run in threads of their own, so more threads served
        # tool-bound calls than one run alone would use.
        threads = set().union(*(m.threads for m in scripted))
        assert len(threads) > 4
        for ticker in tickers:
            state = json.loads((cfg.runs_dir / f"r-{ticker}" / "state.json").read_text())
            assert state["company_of_interest"] == ticker

    def test_parallel_runs_overlap_in_time(self, tmp_path):
        """With a graph that only waits, five runs take about as long as one."""
        active, peak, lock = [0], [0], threading.Lock()

        class Waiting:
            config: dict = {}
            propagator = type("P", (), {"get_graph_args": staticmethod(lambda callbacks=None: {})})()

            def create_run_state(self, *a, **k):
                return {}

            @property
            def graph(self):
                return self

            def invoke(self, state, **kwargs):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.2)
                with lock:
                    active[0] -= 1
                return {"final_trade_decision": "**Rating**: Hold", "final_rating": "Hold"}

        pool = AnalystPool(_cfg(tmp_path), graph_factory=lambda cfg, recorder: Waiting())
        started = time.monotonic()
        results = pool.run_many([_request(t, f"r-{t}") for t in ("NVDA", "CSCO", "HD", "DIS", "GEV")])
        assert all(r.status == "ok" and r.rating == "Hold" for r in results)
        assert peak[0] == 5 and time.monotonic() - started < 0.8

    def test_the_worker_limit_is_respected(self, tmp_path):
        active, peak, lock = [0], [0], threading.Lock()

        class Waiting:
            config: dict = {}
            propagator = type("P", (), {"get_graph_args": staticmethod(lambda callbacks=None: {})})()
            graph = None

            def __init__(self):
                self.graph = self

            def create_run_state(self, *a, **k):
                return {}

            def invoke(self, state, **kwargs):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.05)
                with lock:
                    active[0] -= 1
                return {"final_trade_decision": "**Rating**: Hold", "final_rating": "Hold"}

        pool = AnalystPool(_cfg(tmp_path, max_workers=2), graph_factory=lambda cfg, recorder: Waiting())
        pool.run_many([_request(t, f"r-{t}") for t in ("NVDA", "CSCO", "HD", "DIS", "GEV")])
        assert peak[0] == 2

    def test_one_tickers_failure_leaves_the_others_standing(self, tmp_path):
        class Graph:
            config: dict = {}
            propagator = type("P", (), {"get_graph_args": staticmethod(lambda callbacks=None: {})})()

            def __init__(self):
                self.graph = self

            def create_run_state(self, ticker, *a, **k):
                return {"ticker": ticker}

            def invoke(self, state, **kwargs):
                if state["ticker"] == "HD":
                    raise RuntimeError("rate limited")
                return {"final_trade_decision": "**Rating**: Buy", "final_rating": "Buy"}

        pool = AnalystPool(_cfg(tmp_path), graph_factory=lambda cfg, recorder: Graph())
        results = pool.run_many([_request(t, f"r-{t}") for t in ("NVDA", "HD", "DIS")])
        assert [(r.ticker, r.status) for r in results] == [("NVDA", "ok"), ("HD", "error"), ("DIS", "ok")]
        assert "rate limited" in results[1].error

    def test_a_graph_that_cannot_be_built_is_an_error_result(self, tmp_path):
        def broken(cfg, recorder):
            raise ValueError("ANTHROPIC_API_KEY is not set")

        (result,) = AnalystPool(_cfg(tmp_path), graph_factory=broken).run_many([_request()])
        assert result.status == "error" and "ANTHROPIC_API_KEY" in result.error

    def test_two_runs_of_one_ticker_at_once_are_refused(self, tmp_path):
        pool = AnalystPool(_cfg(tmp_path), graph_factory=lambda cfg, recorder: None)
        with pytest.raises(ValueError, match="one analysis per ticker"):
            pool.run_many([_request("NVDA", "a"), _request("NVDA", "b")])
        assert pool.run_many([]) == []


@pytest.mark.unit
class TestTheRecorder:
    @pytest.mark.parametrize("metadata, expected", [
        ({"langgraph_node": "agent", "langgraph_checkpoint_ns": "Market Analyst:abc|agent:def"}, "Market Analyst"),
        ({"langgraph_node": "Bull Researcher", "langgraph_checkpoint_ns": "Bull Researcher:abc"}, "Bull Researcher"),
        ({"langgraph_node": "Trader"}, "Trader"),
        ({}, "unknown"), (None, "unknown"),
    ])
    def test_a_call_is_named_by_its_outer_node(self, metadata, expected):
        assert _agent(metadata) == expected

    def test_typed_content_blocks_are_read_as_text(self):
        assert _text([{"type": "thinking", "thinking": "…"}, {"type": "text", "text": "Report."}]) == "Report."
        assert _text(None) == "" and _text("plain") == "plain"

    def test_a_very_long_text_is_clipped_with_a_note(self):
        clipped = _clip("x" * (MAX_TEXT_CHARS + 500))
        assert clipped.startswith("x" * MAX_TEXT_CHARS) and "500 more characters not kept" in clipped
        assert _clip("short") == "short"

    def test_usage_is_summed_per_model_and_errors_are_kept(self):
        from uuid import uuid4

        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        recorder = RunRecorder()
        for model, tokens_in, tokens_out in (("claude-sonnet-5", 1000, 200), ("claude-sonnet-5", 500, 100),
                                             ("claude-opus-5-5", 4000, 900)):
            run = uuid4()
            recorder.on_chat_model_start({}, [[AIMessage(content="prompt")]], run_id=run,
                                         metadata={"langgraph_node": "Trader"})
            message = AIMessage(content="ok", response_metadata={"model_name": model},
                                usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out,
                                                "total_tokens": tokens_in + tokens_out})
            recorder.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run)
        failed = uuid4()
        recorder.on_tool_start({"name": "get_news"}, "{}", run_id=failed, inputs={"start_date": "2026-01-02"})
        recorder.on_tool_error(RuntimeError("vendor down"), run_id=failed)
        recorder.on_llm_end(LLMResult(generations=[[]]), run_id=uuid4())      # an end with no start is ignored

        assert recorder.usage_by_model() == {
            "claude-sonnet-5": {"calls": 2, "input_tokens": 1500, "output_tokens": 300},
            "claude-opus-5-5": {"calls": 1, "input_tokens": 4000, "output_tokens": 900}}
        (tool,) = [s for s in recorder.steps() if s["kind"] == "tool"]
        assert tool["error"] == "RuntimeError: vendor down" and tool["args"] == {"start_date": "2026-01-02"}

    def test_a_tool_called_with_nothing_of_the_models_own_records_empty_arguments(self):
        from uuid import uuid4

        recorder = RunRecorder()
        run, legacy = uuid4(), uuid4()
        recorder.on_tool_start({"name": "get_insider_transactions"}, "{}", run_id=run,
                               inputs={"ticker": "NVDA", "trade_date": "2026-01-09"})
        recorder.on_tool_end("rows", run_id=run)
        recorder.on_tool_start({"name": "old_style"}, "raw input", run_id=legacy)     # no structured inputs given
        recorder.on_tool_end("rows", run_id=legacy)
        assert [s["args"] for s in recorder.steps()] == [{}, "raw input"]

        recorder.reset()
        assert recorder.steps() == []
