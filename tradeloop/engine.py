"""One tick of the loop, start to finish:

    sync      prices, what became of earlier orders, the book
    gate      per ticker: call the model, or let the standing decision stand
    analyse   the tickers that need it, at the same time
    size      each standing decision as a target position
    rule      the limits, one check at a time
    execute   orders for what changed, only while the market is open
    record    every step above, for the dashboard

A ticker whose analysis fails keeps its standing decision; a tick that fails as a
whole is recorded as failed and the loop goes on to the next one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from tradeloop import context, gate, outcomes, risk, target, timeutil
from tradeloop.alpaca import Bar, Quote
from tradeloop.analysis import AnalysisRequest, AnalysisResult
from tradeloop.broker import Account, Broker
from tradeloop.config import LoopConfig
from tradeloop.executor import Executor, plan_orders
from tradeloop.schedule import Scheduler, Session, Tick
from tradeloop.store import Store

logger = logging.getLogger(__name__)


class Market(Protocol):
    def quotes(self, symbols) -> dict[str, Quote]: ...
    def news(self, symbols, start: datetime | None = None, limit: int = 50) -> list[dict]: ...
    def bars(self, symbols, timeframe: str, start: datetime, end: datetime) -> dict[str, list[Bar]]: ...


class Analysts(Protocol):
    def run_many(self, requests: list[AnalysisRequest]) -> list[AnalysisResult]: ...


@dataclass
class TickReport:
    tick_id: str
    status: str                                   # ok | error | skipped
    analysed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    orders: int = 0
    note: str = ""


class Engine:
    def __init__(self, cfg: LoopConfig, store: Store, broker: Broker, market: Market, analysts: Analysts,
                 scheduler: Scheduler, *, now: Callable[[], datetime] = timeutil.utcnow,
                 today: Callable[[], date] = date.today,
                 after_tick: Callable[[], None] | None = None):
        self.cfg, self.store, self.broker, self.market = cfg, store, broker, market
        self.analysts, self.scheduler = analysts, scheduler
        self.executor = Executor(broker, store, cfg)
        self.tz = ZoneInfo(cfg.timezone)
        self.now, self._today, self._after_tick = now, today, after_tick

    # --- the tick
    def run_tick(self, tick: Tick) -> TickReport:
        """Run ``tick`` once. A tick that already ran is skipped; one that fails is recorded and returned."""
        started = self.now()
        if not self.store.begin_tick(tick.tick_id, tick.kind, timeutil.iso(tick.at), timeutil.iso(started),
                                     self.broker.name):
            return TickReport(tick.tick_id, "skipped", note="this tick already ran")
        report = TickReport(tick.tick_id, "ok")
        try:
            self._run(tick, started, report)
        except Exception as exc:  # noqa: BLE001 - the loop outlives a failed tick
            logger.exception("tick %s failed", tick.tick_id)
            report.status, report.note = "error", f"{type(exc).__name__}: {str(exc)[:300]}"
            self._log("error", "engine", report.note, tick)
        self.store.finish_tick(tick.tick_id, report.status, timeutil.iso(self.now()), report.note)
        if self._after_tick is not None:
            try:
                self._after_tick()
            except Exception:  # noqa: BLE001 - a dashboard that cannot be built must not stop trading
                logger.exception("the after-tick step failed")
        return report

    def _run(self, tick: Tick, now: datetime, report: TickReport) -> None:
        cfg, tick_id = self.cfg, tick.tick_id
        sessions = self.scheduler.sessions(now)

        # --- sync
        quotes = self.market.quotes(cfg.tickers)
        prices = {ticker: quote.price for ticker, quote in quotes.items()}
        self.executor.settle_open_orders(prices, now, tick_id)
        account = self.broker.account(prices)
        self._snapshot(tick_id, now, account, prices)

        # --- gate
        standing = {ticker: self.store.latest_decision(ticker) for ticker in cfg.tickers}
        headlines = self._new_headlines(standing, now)
        budget_left = cfg.daily_llm_budget_usd - self.store.llm_cost_since(self._day_start(now))
        last_close = max((s.close for s in sessions if s.close <= now), default=now - timedelta(hours=24))
        decisions = {}
        for ticker in cfg.tickers:
            done = self.store.query(
                "SELECT COUNT(*) AS n FROM analyses WHERE tick_id = ? AND ticker = ? AND status = 'ok'",
                (tick_id, ticker))[0]["n"]
            watch = [a for a in self.store.analyses_between(ticker, timeutil.iso(last_close), timeutil.iso(now),
                                                             kind="offhours") if a["status"] == "ok"]
            decisions[ticker] = gate.decide(gate.GateView(
                kind=tick.kind, now=now, price=prices.get(ticker), last=standing[ticker],
                new_headlines=headlines.get(ticker, 0), budget_left=budget_left,
                done_this_tick=bool(done), offhours_done=len(watch)), cfg)
        self.store.add_observations([{
            "tick_id": tick_id, "ticker": ticker,
            "price": prices.get(ticker), "price_at": timeutil.iso(quotes[ticker].at) if ticker in quotes else None,
            "prev_close": quotes[ticker].prev_close if ticker in quotes else None,
            "position_qty": account.qty(ticker), "new_headlines": headlines.get(ticker, 0),
            "gate_action": decisions[ticker].action, "gate_reasons": list(decisions[ticker].reasons),
        } for ticker in cfg.tickers])

        # --- analyse
        to_run = [ticker for ticker in cfg.tickers if decisions[ticker].action == gate.ANALYZE]
        if to_run:
            self._analyse(tick, now, to_run, account, prices, sessions, report)
            # An analysis takes minutes: size and order on prices from after it, not before.
            prices.update({t: q.price for t, q in self.market.quotes(cfg.tickers).items()})
            account = self.broker.account(prices)

        # --- size and rule
        can_trade, window_note = self._trading_window(tick, self.now(), sessions)
        proposals = [self._proposal(ticker, prices.get(ticker), account, now) for ticker in cfg.tickers]
        rulings = risk.evaluate(proposals, risk.RiskContext(
            equity=account.equity, day_start_equity=self._day_start_equity(now, sessions),
            can_trade=can_trade, window_note=window_note), cfg)
        self.store.add_targets([r.as_row(tick_id) for r in rulings])

        # --- execute
        if can_trade:
            placed = self.executor.place(plan_orders(rulings, tick_id, cfg, cash=account.cash), prices, now, tick_id)
            report.orders = len(placed)
            stops = {p.ticker: p.stop_loss for p in proposals}
            self.executor.sync_stops(self.broker.account(prices), stops, prices, now, tick_id)

        # --- what earlier decisions came to
        try:
            outcomes.mark_due(self.store, self.market, now)
        except Exception as exc:  # noqa: BLE001 - marking can wait for the next tick
            self._log("warning", "outcomes", f"outcomes not marked ({type(exc).__name__})", tick)

        report.note = (f"{len(report.analysed)} analysed, {len(report.failed)} failed, "
                       f"{len(self.cfg.tickers) - len(to_run)} reused or skipped, {report.orders} orders")

    # --- the pieces
    def _snapshot(self, tick_id: str, now: datetime, account: Account, prices: dict[str, float]) -> None:
        positions = [{"ticker": p.ticker, "qty": p.qty, "avg_price": p.avg_price, "price": prices.get(p.ticker)}
                     for p in account.positions.values()]
        self.store.add_snapshot(tick_id, timeutil.iso(now), account.equity, account.cash, positions)

    def _new_headlines(self, standing: dict[str, dict | None], now: datetime) -> dict[str, int]:
        """Headlines per ticker since its last analysis.

        One request per ticker: asked together, a name in the news all day would
        fill the page and the quiet names would read as silent.
        """
        counts = {}
        for ticker, decision in standing.items():
            if not decision:
                continue
            since = timeutil.parse(decision["started_at"])
            try:
                news = self.market.news([ticker], start=since, limit=50)
            except Exception as exc:  # noqa: BLE001 - news is one trigger among three
                self._log("warning", "gate", f"headlines not read ({type(exc).__name__})", None, ticker=ticker)
                continue
            counts[ticker] = sum(1 for item in news if item["at"] > since)
        return counts

    def _analyse(self, tick: Tick, now: datetime, tickers: list[str], account: Account, prices: dict[str, float],
                 sessions: list[Session], report: TickReport) -> None:
        trade_date = min(now.astimezone(self.tz).date(), self._today()).isoformat()
        as_of = context.as_of_text(now, tick.kind, self.tz)
        book = context.portfolio_context(account)
        day_start, moment = self._day_start(now), timeutil.iso(now)
        requests = []
        for ticker in tickers:
            run_id = self._run_id(tick.tick_id, ticker)
            request = AnalysisRequest(
                ticker=ticker, run_id=run_id, trade_date=trade_date, as_of_time=as_of, portfolio=book,
                session_context=context.session_context(self.store, ticker, day_start, moment, self.tz),
                record_to_memory=tick.kind == self.cfg.memory_record_kind)
            self.store.begin_analysis(
                run_id=run_id, tick_id=tick.tick_id, ticker=ticker, trade_date=trade_date, kind=tick.kind,
                started_at=moment, ref_price=prices.get(ticker), artifacts_dir=str(self.cfg.runs_dir / run_id))
            requests.append(request)

        for result in self.analysts.run_many(requests):
            decision = result.decision or {}
            self.store.finish_analysis(
                result.run_id, status=result.status, finished_at=timeutil.iso(self.now()),
                elapsed_seconds=result.elapsed_seconds, rating=result.rating,
                target_weight=decision.get("target_weight"), stop_loss=decision.get("stop_loss"),
                price_target=decision.get("price_target"), time_horizon=decision.get("time_horizon"),
                decision=decision, summary=result.summary, input_tokens=result.input_tokens,
                output_tokens=result.output_tokens, cost_usd=result.cost_usd, llm_calls=result.llm_calls,
                tool_calls=result.tool_calls, recorded_to_memory=result.recorded_to_memory, error=result.error)
            self.store.add_agent_steps(result.run_id, [
                {k: step.get(k) for k in ("seq", "agent", "kind", "name", "started_at", "elapsed_seconds",
                                          "input_tokens", "output_tokens", "output_chars", "error")}
                for step in result.steps])
            if result.status == "ok":
                report.analysed.append(result.ticker)
                self.store.add_outcomes(outcomes.plan(result.run_id, now, sessions))
            else:
                report.failed.append(result.ticker)
                self._log("error", "analysis", result.error or "analysis failed", tick, ticker=result.ticker)

    def _run_id(self, tick_id: str, ticker: str) -> str:
        """An id no earlier attempt of this tick has used."""
        base = f"{tick_id}-{ticker}"
        taken = {r["run_id"] for r in self.store.query("SELECT run_id FROM analyses WHERE run_id LIKE ?", (base + "%",))}
        attempt, run_id = 1, base
        while run_id in taken:
            attempt += 1
            run_id = f"{base}-retry{attempt}"
        return run_id

    def _proposal(self, ticker: str, price: float | None, account: Account, now: datetime) -> risk.Proposal:
        recent = self.store.recent_decisions(ticker, self.cfg.confirm_rounds)
        if not recent:
            return risk.Proposal(ticker=ticker, price=price, current_qty=account.qty(ticker))
        latest = recent[0]
        stances = [target.stance(d["rating"], d["target_weight"], self.cfg) for d in recent]
        age = (now - timeutil.parse(latest["started_at"])).total_seconds() / 3600
        stopped = self.store.one(
            "SELECT 1 AS hit FROM orders WHERE ticker = ? AND filled_qty > 0 AND updated_at >= ? "
            "AND (type = 'stop' OR reason = 'exit_stop') LIMIT 1", (ticker, latest["started_at"]))
        return risk.Proposal(
            ticker=ticker, price=price, current_qty=account.qty(ticker), rating=latest["rating"],
            stance=stances[0], stop_loss=latest["stop_loss"], price_target=latest["price_target"],
            run_id=latest["run_id"], prior_stances=tuple(stances[1:]), decision_age_hours=age,
            stopped_out=stopped is not None)

    def _trading_window(self, tick: Tick, now: datetime, sessions: list[Session]) -> tuple[bool, str]:
        """Whether this tick may place orders, and the reason either way."""
        if tick.kind != "core":
            return False, f"a {tick.kind} tick analyses and records; orders wait for the session"
        session = next((s for s in sessions if s.open <= now < s.close), None)
        if session is None:
            return False, "the market is closed"
        if now < session.open + timedelta(minutes=self.cfg.no_trade_open_minutes):
            return False, f"within {self.cfg.no_trade_open_minutes} minutes of the open"
        if now > session.close - timedelta(minutes=self.cfg.no_trade_close_minutes):
            return False, f"within {self.cfg.no_trade_close_minutes} minutes of the close"
        return True, "inside the trading window"

    def _day_start(self, now: datetime) -> str:
        """Midnight at the exchange on the day of ``now``."""
        return timeutil.iso(datetime.combine(now.astimezone(self.tz).date(), time.min, self.tz))

    def _day_start_equity(self, now: datetime, sessions: list[Session]) -> float | None:
        """Equity at the last tick before today's open, or at the day's first tick when there is none before."""
        opened = next((s.open for s in sessions if s.day == now.astimezone(self.tz).date()), None)
        if opened is None:
            return None
        before = self.store.equity_at_or_before(timeutil.iso(opened))
        if before is not None:
            return before
        first = self.store.one("SELECT equity FROM snapshots WHERE at >= ? ORDER BY at LIMIT 1", (timeutil.iso(opened),))
        return None if first is None else first["equity"]

    def _log(self, level: str, source: str, message: str, tick: Tick | None, ticker: str | None = None) -> None:
        self.store.log(level, source, message, at=timeutil.iso(self.now()),
                       tick_id=tick.tick_id if tick else None, ticker=ticker)


def run_forever(engine: Engine, *, sleep: Callable[[float], None], stop: Callable[[], bool] = lambda: False,
                on_tick: Callable[[TickReport], None] | None = None) -> None:
    """Run each tick as its time comes, until ``stop`` says so.

    A tick whose time passed while the machine slept or the last tick ran long is
    skipped once it is more than its allowance late: an hourly decision made on
    two-hour-old timing is not the decision that was scheduled.
    """
    now = engine.now
    after = now()
    while not stop():
        tick = engine.scheduler.next_tick(after)
        while (wait := (tick.at - now()).total_seconds()) > 0:
            if stop():
                return
            sleep(min(wait, 30.0))
        after = tick.at
        late = (now() - tick.at).total_seconds() / 60
        allowance = {"core": engine.cfg.interval_minutes / 2, "offhours": engine.cfg.offhours_interval_minutes / 2,
                     "premarket": engine.cfg.premarket_lead_minutes, "postclose": 180.0}[tick.kind]
        if late > allowance:
            engine.store.log("warning", "scheduler", f"tick {tick.tick_id} skipped: {late:.0f} minutes late",
                             at=timeutil.iso(now()))
            continue
        report = engine.run_tick(tick)
        if on_tick is not None:
            on_tick(report)
