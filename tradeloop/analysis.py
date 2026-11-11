"""Running TradingAgents for the loop: one graph per ticker, built once and reused,
several tickers at the same time, each run recorded step by step.

The loop does not call ``propagate``: it decides itself which run of the day goes
into TradingAgents' memory log (one, not every tick), and it needs the run's steps,
not only its final state.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cli.usage import cost_of
from tradeloop.config import LoopConfig
from tradeloop.recorder import RunRecorder
from tradingagents.agents.rating import run_rating
from tradingagents.dataflows.config import run_config
from tradingagents.portfolio import PortfolioContext

# The parts of the final state kept for the dashboard; the rest is working state.
_STATE_KEYS = ("company_of_interest", "trade_date", "market_report", "sentiment_report", "news_report",
               "fundamentals_report", "investment_debate_state", "investment_plan", "trader_investment_plan",
               "risk_debate_state", "final_trade_decision", "final_rating", "final_decision",
               "past_context", "memory_note", "instrument_context", "portfolio_context", "session_context")


@dataclass(frozen=True)
class AnalysisRequest:
    ticker: str
    run_id: str
    trade_date: str
    as_of_time: str | None = None
    portfolio: PortfolioContext | None = None
    session_context: str = ""
    record_to_memory: bool = False


@dataclass
class AnalysisResult:
    run_id: str
    ticker: str
    status: str                                   # ok | error
    rating: str | None = None
    decision: dict = field(default_factory=dict)  # the Portfolio Manager's typed fields, when it gave them
    summary: str = ""
    elapsed_seconds: float = 0.0
    usage: dict = field(default_factory=dict)     # by model
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None                 # None when a model has no price on file
    llm_calls: int = 0
    tool_calls: int = 0
    recorded_to_memory: bool = False
    steps: list[dict] = field(default_factory=list)
    artifacts_dir: str = ""
    error: str | None = None


def default_graph(cfg: LoopConfig, recorder: RunRecorder):
    """A TradingAgents graph for the loop, reporting its model and tool calls to ``recorder``."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = cfg.tradingagents_config()
    config["checkpoint_enabled"] = False     # a tick is retried whole, not resumed
    return TradingAgentsGraph(cfg.analysts, config=config, callbacks=[recorder])


class Analyst:
    """One ticker's graph and recorder. Runs one analysis at a time."""

    def __init__(self, cfg: LoopConfig, graph_factory: Callable[[LoopConfig, RunRecorder], Any] = default_graph):
        self.cfg = cfg
        self.recorder = RunRecorder()
        self.graph = graph_factory(cfg, self.recorder)

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        started = time.monotonic()
        self.recorder.reset()
        result = AnalysisResult(run_id=request.run_id, ticker=request.ticker, status="error",
                                artifacts_dir=str(self.cfg.runs_dir / request.run_id))
        final: dict = {}
        try:
            graph = self.graph
            state = graph.create_run_state(request.ticker, request.trade_date, "stock", request.portfolio,
                                           as_of_time=request.as_of_time, session_context=request.session_context)
            args = graph.propagator.get_graph_args(callbacks=[self.recorder])
            # The graph's own config serves its data tools while this run is in progress.
            with run_config(graph.config):
                final = graph.graph.invoke(state, **args)
            if not final.get("final_trade_decision"):
                raise RuntimeError("the run ended without a final decision")
            result.rating = run_rating(final)
            result.decision = dict(final.get("final_decision") or {})
            result.summary = str(result.decision.get("executive_summary") or "")[:2000]
            if request.record_to_memory:
                graph.record_decision(request.ticker, request.trade_date, final)
                result.recorded_to_memory = True
            result.status = "ok"
        except Exception as exc:  # noqa: BLE001 - one ticker's failure must not end the tick
            result.error = f"{type(exc).__name__}: {str(exc)[:500]}"
        self._account(result)
        result.elapsed_seconds = round(time.monotonic() - started, 2)
        self._write_artifacts(request, result, final)
        return result

    def _account(self, result: AnalysisResult) -> None:
        """Total the recorded steps into tokens, calls and cost."""
        result.steps = self.recorder.steps()
        result.usage = self.recorder.usage_by_model()
        result.llm_calls = sum(1 for s in result.steps if s["kind"] == "llm")
        result.tool_calls = sum(1 for s in result.steps if s["kind"] == "tool")
        result.input_tokens = sum(u["input_tokens"] for u in result.usage.values())
        result.output_tokens = sum(u["output_tokens"] for u in result.usage.values())
        costs = [cost_of(model, u["input_tokens"], u["output_tokens"]) for model, u in result.usage.items()]
        # A model with no price on file leaves the cost unknown rather than understated.
        result.cost_usd = None if any(c is None for c in costs) else round(sum(costs), 6)

    def _write_artifacts(self, request: AnalysisRequest, result: AnalysisResult, final: dict) -> None:
        """The run's long texts, in a folder of its own. Losing them must not lose the run."""
        try:
            folder = Path(result.artifacts_dir)
            folder.mkdir(parents=True, exist_ok=True)
            kept = {key: final[key] for key in _STATE_KEYS if key in final}
            (folder / "state.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1, default=str),
                                               encoding="utf-8")
            with open(folder / "steps.jsonl", "w", encoding="utf-8") as f:
                for step in result.steps:
                    f.write(json.dumps(step, ensure_ascii=False, default=str) + "\n")
            (folder / "request.json").write_text(json.dumps({
                "run_id": request.run_id, "ticker": request.ticker, "trade_date": request.trade_date,
                "as_of_time": request.as_of_time, "session_context": request.session_context,
                "portfolio": request.portfolio.model_dump() if request.portfolio else None,
                "status": result.status, "error": result.error, "usage": result.usage,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:
            result.error = (result.error + "; " if result.error else "") + f"artifacts not written ({type(exc).__name__})"


class AnalystPool:
    """The analysts of every ticker, each built on first use, run side by side."""

    def __init__(self, cfg: LoopConfig, graph_factory: Callable[[LoopConfig, RunRecorder], Any] = default_graph):
        self.cfg, self._factory = cfg, graph_factory
        self._analysts: dict[str, Analyst] = {}

    def _run_one(self, request: AnalysisRequest) -> AnalysisResult:
        try:
            analyst = self._analysts.get(request.ticker)
            if analyst is None:
                # Built in the worker, so several tickers' graphs are built at once.
                analyst = self._analysts[request.ticker] = Analyst(self.cfg, self._factory)
            return analyst.run(request)
        except Exception as exc:  # noqa: BLE001 - building the graph failed (a missing key, a bad model name)
            return AnalysisResult(run_id=request.run_id, ticker=request.ticker, status="error",
                                  error=f"{type(exc).__name__}: {str(exc)[:500]}")

    def run_many(self, requests: list[AnalysisRequest]) -> list[AnalysisResult]:
        """Run ``requests`` at the same time, one per ticker; results in the order given."""
        if not requests:
            return []
        if len({r.ticker for r in requests}) != len(requests):
            raise ValueError("one analysis per ticker at a time: an analyst holds one run")
        if len(requests) == 1:
            return [self._run_one(requests[0])]
        with ThreadPoolExecutor(max_workers=min(self.cfg.workers, len(requests)),
                                thread_name_prefix="analysis") as pool:
            return list(pool.map(self._run_one, requests))
