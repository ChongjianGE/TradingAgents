"""The rules between a decision and an order. Each proposal is checked in turn, and
every check is kept with its reason, passed or not, so the record shows why a
position did or did not change.

The rules lean one way on purpose: reducing a position needs nothing but the
decision; growing one needs agreement between analyses, room in the day's loss
limit and an expected gain worth its cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tradeloop import target
from tradeloop.config import LoopConfig


@dataclass(frozen=True)
class Proposal:
    """One ticker's standing decision and current position, as the rules see them."""

    ticker: str
    price: float | None
    current_qty: float
    rating: str | None = None
    stance: float | None = None                 # fraction of a full position asked for; None keeps
    stop_loss: float | None = None
    price_target: float | None = None
    run_id: str | None = None
    prior_stances: tuple[float | None, ...] = ()  # the analyses before this one, newest first
    decision_age_hours: float | None = None
    stopped_out: bool = False                   # this decision's stop already sold the position


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Ruling:
    ticker: str
    price: float | None
    current_qty: float
    current_weight: float
    raw_weight: float | None        # what the decision asks for, as a share of equity
    target_weight: float
    target_qty: float
    verdict: str                    # keep | increase | reduce | exit_stop | blocked
    checks: list[Check] = field(default_factory=list)
    run_id: str | None = None
    rating: str | None = None
    actionable: bool = True         # False outside the trading window: recorded, not ordered

    def as_row(self, tick_id: str) -> dict:
        return {"tick_id": tick_id, "ticker": self.ticker, "run_id": self.run_id, "rating": self.rating,
                "price": self.price, "current_qty": self.current_qty, "current_weight": self.current_weight,
                "raw_weight": self.raw_weight, "target_weight": self.target_weight, "target_qty": self.target_qty,
                "verdict": self.verdict if self.actionable else f"deferred:{self.verdict}",
                "checks": [vars(c) for c in self.checks]}


@dataclass(frozen=True)
class RiskContext:
    equity: float
    day_start_equity: float | None = None
    can_trade: bool = True
    window_note: str = "inside the trading window"


def _rule_one(p: Proposal, ctx: RiskContext, cfg: LoopConfig) -> Ruling:
    """Apply the per-ticker rules; the portfolio-wide ones follow in ``evaluate``."""
    if p.price is None or ctx.equity <= 0:
        return Ruling(p.ticker, p.price, p.current_qty, 0.0, None, 0.0, p.current_qty, "keep",
                      [Check("price", False, "no price this tick, so nothing is changed")], p.run_id, p.rating)

    current = p.current_qty * p.price / ctx.equity
    tolerance = cfg.min_trade_notional / ctx.equity
    ruling = Ruling(p.ticker, p.price, p.current_qty, current, None, current, p.current_qty, "keep",
                    [], p.run_id, p.rating)
    checks = ruling.checks

    # A position under its stop is closed, whatever the decision now says.
    under_stop = bool(p.stop_loss) and p.price <= p.stop_loss
    if p.current_qty > 0 and under_stop:
        checks.append(Check("stop_loss", False, f"price {p.price:.2f} is at or under the stop {p.stop_loss:.2f}"))
        ruling.target_weight, ruling.target_qty, ruling.verdict = 0.0, 0.0, "exit_stop"
        return ruling
    # A decision whose stop has been reached cannot open or add to a position:
    # the price it was made for is gone, and buying back in needs a new analysis.
    spent = None
    if under_stop:
        spent = f"price {p.price:.2f} is at or under the decision's stop {p.stop_loss:.2f}"
    elif p.stopped_out:
        spent = "this decision was already stopped out; buying again needs a new analysis"
    if p.stop_loss and spent is None:
        checks.append(Check("stop_loss", True, f"price {p.price:.2f} is above the stop {p.stop_loss:.2f}"))

    if p.stance is None:
        checks.append(Check("decision", True, "no decision asks for a change" if p.run_id is None
                            else f"{p.rating or 'the decision'} keeps the position as it is"))
        return ruling
    if p.decision_age_hours is not None and p.decision_age_hours > cfg.max_decision_age_hours:
        checks.append(Check("decision_age", False,
                            f"the decision is {p.decision_age_hours:.0f} hours old, past {cfg.max_decision_age_hours:.0f}"))
        return ruling

    raw = target.equity_weight(p.stance, cfg)
    ruling.raw_weight = raw
    if raw < current - tolerance:
        checks.append(Check("decision", True, f"{p.rating} asks for {raw:.1%} of equity, less than the {current:.1%} held"))
        ruling.target_weight, ruling.verdict = raw, "reduce"
    elif raw > current + tolerance:
        if spent:
            checks.append(Check("stop_loss", False, spent))
        wanted = None if spent else _confirmed_increase(p, raw, current, tolerance, ctx, cfg, checks)
        if wanted is None:
            ruling.verdict = "blocked"
        else:
            ruling.target_weight, ruling.verdict = wanted, "increase"
    else:
        checks.append(Check("decision", True, f"{p.rating} asks for {raw:.1%}, which is what is held"))
    ruling.target_qty = (p.current_qty if ruling.verdict in ("keep", "blocked")
                         else target.shares(ruling.target_weight, ctx.equity, p.price, cfg))
    return ruling


def _confirmed_increase(p: Proposal, raw: float, current: float, tolerance: float, ctx: RiskContext,
                        cfg: LoopConfig, checks: list[Check]) -> float | None:
    """The weight an increase may go to, or None when a rule stops it."""
    allowed, ok = raw, True

    needed = cfg.confirm_rounds - 1
    priors = [target.equity_weight(s, cfg) if s is not None else None for s in p.prior_stances[:needed]]
    agreeing = [w for w in priors if w is not None and w > current + tolerance]
    if len(agreeing) < needed:
        ok = False
        checks.append(Check("confirmation", False,
                            f"needs {cfg.confirm_rounds} analyses in a row asking for more; "
                            f"{len(agreeing) + 1} so far"))
    else:
        allowed = min([raw, *agreeing])   # as far as every one of them goes
        checks.append(Check("confirmation", True, f"{cfg.confirm_rounds} analyses in a row ask for at least {allowed:.1%}"))

    if ctx.day_start_equity:
        change = (ctx.equity / ctx.day_start_equity - 1) * 100
        lost = change <= -cfg.daily_loss_limit_pct
        ok = ok and not lost
        checks.append(Check("daily_loss", not lost,
                            f"the day is {change:+.2f}%, limit -{cfg.daily_loss_limit_pct:.2f}%"
                            + ("; only reductions from here" if lost else "")))

    if p.price_target:
        expected, required = p.price_target / p.price - 1, cfg.min_edge_to_cost * cfg.round_trip_cost
        worth = expected >= required
        ok = ok and worth
        checks.append(Check("edge", worth, f"the price target implies {expected:+.2%}; a round trip's cost "
                                           f"times {cfg.min_edge_to_cost:g} is {required:.2%}"))
    else:
        checks.append(Check("edge", True, "no price target given, so the gain is not compared with the cost"))
    return allowed if ok else None


def evaluate(proposals: list[Proposal], ctx: RiskContext, cfg: LoopConfig) -> list[Ruling]:
    """Rule on every proposal, then fit the increases into the exposure the book allows."""
    rulings = [_rule_one(p, ctx, cfg) for p in proposals]

    growing = [r for r in rulings if r.verdict == "increase"]
    held_after = sum(min(r.target_weight, r.current_weight) if r.verdict == "increase" else r.target_weight
                     for r in rulings)
    wanted = sum(r.target_weight - r.current_weight for r in growing)
    room = max(0.0, cfg.max_gross_exposure - held_after)
    for r in growing:
        if wanted > room + 1e-9:
            scaled = r.current_weight + (r.target_weight - r.current_weight) * room / wanted
            r.checks.append(Check("gross_exposure", False,
                                  f"all increases together exceed {cfg.max_gross_exposure:.0%} of equity; "
                                  f"scaled from {r.target_weight:.1%} to {scaled:.1%}"))
            r.target_weight = scaled
            r.target_qty = target.shares(scaled, ctx.equity, r.price, cfg)
        else:
            r.checks.append(Check("gross_exposure", True, f"fits within {cfg.max_gross_exposure:.0%} of equity"))

    for r in rulings:
        r.actionable = ctx.can_trade
        r.checks.append(Check("trading_window", ctx.can_trade, ctx.window_note))
    return rulings
