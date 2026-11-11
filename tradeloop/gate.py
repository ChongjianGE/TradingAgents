"""Whether to call the model for a ticker on this tick.

An analysis costs minutes and money, so a tick only asks for one when something
has changed since the last: the price, the news, or simply too much time. The
scheduled pre-market and post-close ticks always analyse; the off-hours watch
only does when the price or the news gives it a reason, and at most a set number
of times a night.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from tradeloop import timeutil
from tradeloop.config import LoopConfig

ANALYZE, REUSE, SKIP = "analyze", "reuse", "skip"


@dataclass(frozen=True)
class GateView:
    """What the gate knows about one ticker at one tick."""

    kind: str                           # the tick's kind
    now: datetime
    price: float | None
    last: dict | None                   # the latest analysis that reached a decision
    new_headlines: int = 0              # since that analysis
    budget_left: float = float("inf")   # model spend still allowed today
    done_this_tick: bool = False        # this tick already analysed the ticker (a restart)
    offhours_done: int = 0              # analyses the watch has triggered since the last close


@dataclass(frozen=True)
class GateDecision:
    action: str
    reasons: tuple[str, ...]


def _triggers(view: GateView, cfg: LoopConfig, *, with_staleness: bool) -> list[str]:
    """Reasons the standing decision no longer covers what is known now."""
    last, reasons = view.last, []
    if last.get("ref_price") and view.price:
        move = (view.price / last["ref_price"] - 1) * 100
        if abs(move) >= cfg.move_trigger_pct - 1e-9:      # a move of exactly the threshold counts
            reasons.append(f"price moved {move:+.1f}% since the last analysis")
    if view.new_headlines >= cfg.news_trigger_count:
        reasons.append(f"{view.new_headlines} new headlines since the last analysis")
    if with_staleness:
        minutes = (view.now - timeutil.parse(last["started_at"])).total_seconds() / 60
        if minutes >= cfg.max_stale_minutes:
            reasons.append(f"the last analysis is {minutes:.0f} minutes old")
    return reasons


def decide(view: GateView, cfg: LoopConfig) -> GateDecision:
    fallback = REUSE if view.last else SKIP
    if view.done_this_tick:
        return GateDecision(fallback, ("already analysed on this tick",))
    if view.budget_left <= 0:
        return GateDecision(fallback, ("the day's model budget is spent",))

    if view.kind in ("premarket", "postclose"):
        return GateDecision(ANALYZE, (f"scheduled {view.kind} analysis",))

    if view.kind == "core":
        if view.price is None:
            return GateDecision(fallback, ("no price for this ticker",))
        if view.last is None:
            return GateDecision(ANALYZE, ("no decision yet",))
        reasons = _triggers(view, cfg, with_staleness=True)
        return GateDecision(ANALYZE, tuple(reasons)) if reasons else GateDecision(REUSE, ("nothing changed enough",))

    # The off-hours watch.
    if view.last is None:
        return GateDecision(SKIP, ("no decision to revise; the pre-market analysis comes first",))
    reasons = _triggers(view, cfg, with_staleness=False)
    if not reasons:
        return GateDecision(SKIP, ("nothing changed enough",))
    if view.offhours_done >= cfg.offhours_max_analyses:
        return GateDecision(SKIP, (*reasons, "but the watch has used its analyses for tonight"))
    return GateDecision(ANALYZE, tuple(reasons))
