"""What the loop tells the agents that they could not otherwise know: the book as it
is now, the time of day, and what was already decided and traded this session."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from tradeloop import target, timeutil
from tradeloop.broker import Account
from tradeloop.config import LoopConfig
from tradeloop.store import Store
from tradingagents.portfolio import PortfolioContext, Position

_PHASES = {
    "premarket": "before the open",
    "core": "regular session",
    "postclose": "after the close",
    "offhours": "market closed",
}


def portfolio_context(account: Account) -> PortfolioContext:
    """The loop's book in the shape the decision agents already read."""
    return PortfolioContext(
        cash=round(account.cash, 2), currency="USD",
        positions=[Position(ticker=p.ticker, quantity=p.qty, average_price=round(p.avg_price, 4))
                   for p in account.positions.values() if p.qty])


def as_of_text(now: datetime, kind: str, tz: ZoneInfo) -> str:
    """The time of a run in the exchange's zone, with where in the trading day it falls."""
    local = now.astimezone(tz)
    return f"{local:%Y-%m-%d %H:%M} {local:%Z} ({_PHASES.get(kind, kind)})"


def _shares(count: float) -> str:
    return f"{count:g} share" + ("" if count == 1 else "s")


def sizing_context(ticker: str, account: Account, price: float | None, latest: dict | None, cfg: LoopConfig) -> str:
    """What a target weight comes to in this account, and where the position stands against the last one given.

    The decision is a fraction of "a full position". Unless it is told how large
    that is, the model guesses, and the loop then buys a different size from the
    one its text describes. Unless it is told what is held as a weight, it cannot
    tell a target it has reached from one still waiting, and repeats the number
    of the last decision while meaning "add nothing".
    """
    full = cfg.max_position_weight
    limits = ", ".join(f"{name} at most {weight:g}" if weight else f"{name} is 0"
                       for name, weight in cfg.rating_weights.items() if weight != target.KEEP)
    kept = [name for name, weight in cfg.rating_weights.items() if weight == target.KEEP]
    lines = ["How a target weight becomes a position in this account:",
             f"- A full position is {full:.0%} of the account's equity."]
    usable = price is not None and price > 0 and account.equity > 0
    held = account.qty(ticker)
    if usable:
        whole = full * account.equity
        lines[-1] = (f"- A full position is {full:.0%} of the account's equity: about ${whole:,.0f} now, "
                     f"about {_shares(target.shares(full, account.equity, price, cfg))} of {ticker} at {price:,.2f}.")
        lines.append(f"- A target weight of 0.1 is therefore about ${whole * 0.1:,.0f}, "
                     f"about {_shares(target.shares(full * 0.1, account.equity, price, cfg))}.")
    lines.append(f"- The rating limits the weight: {limits}."
                 + (f" {' and '.join(kept)} leaves the position exactly as it is, whatever weight is given." if kept else ""))
    if not usable:
        return "\n".join(lines)

    held_weight = held * price / account.equity / full
    if held:
        lines.append(f"- Held now: {_shares(held)}, about ${held * price:,.0f}, which is a target weight of {held_weight:.3f}.")
    else:
        lines.append("- Held now: nothing, which is a target weight of 0.")
    stance = target.stance(latest["rating"], latest["target_weight"], cfg) if latest else None
    if stance is not None:
        wanted = target.shares(target.equity_weight(stance, cfg), account.equity, price, cfg)
        gap = wanted - held
        said = f"- The latest decision ({latest['rating']}) asked for a target weight of {stance:g}, about {_shares(wanted)}"
        if abs(gap) * price < cfg.min_trade_notional:
            lines.append(f"{said}. That is what is held.")
        elif gap > 0:
            lines.append(f"{said}. {_shares(gap)} of that "
                         f"{'has' if gap == 1 else 'have'} not been bought.")
        else:
            lines.append(f"{said}. {_shares(-gap)} more than that {'is' if gap == -1 else 'are'} still held.")
    if held_weight > 1:
        lines.append("- The position is larger than a full one. Any weight you give asks for at least the part above "
                     "a full position to be sold; 1 keeps the rest.")
    elif held:
        lines.append(f"- The weight you give is taken as stated: above {held_weight:.3f} asks for the difference to be "
                     f"bought, below it for the difference to be sold. To keep the position as it is, give {held_weight:.3f}.")
    else:
        lines.append("- The weight you give is taken as stated: any weight above 0 asks for that much to be bought. "
                     "To stay out, give 0.")
    return "\n".join(lines)


def session_context(store: Store, ticker: str, since: str, until: str, tz: ZoneInfo) -> str:
    """What was decided and traded for ``ticker`` from ``since`` to ``until``, oldest first; empty when nothing was."""
    events: list[tuple[str, str]] = []
    for run in store.analyses_between(ticker, since, until):
        if run["status"] != "ok":
            continue
        parts = [f"{run['kind']} analysis rated it {run['rating']}"]
        if run["target_weight"] is not None:
            parts.append(f"target weight {run['target_weight']:g}")
        if run["stop_loss"] is not None:
            parts.append(f"stop {run['stop_loss']:g}")
        events.append((run["started_at"], ", ".join(parts)))
    for fill in store.query("SELECT * FROM fills WHERE ticker = ? AND at >= ? AND at < ? ORDER BY at",
                            (ticker, since, until)):
        verb = "bought" if fill["side"] == "buy" else "sold"
        events.append((fill["at"], f"{verb} {fill['qty']:g} at {fill['price']:.2f}"))
    if not events:
        return ""
    lines = [f"- {timeutil.parse(at).astimezone(tz):%H:%M} {text}" for at, text in sorted(events)]
    return ("Earlier in this session for this instrument (exchange time):\n" + "\n".join(lines)
            + "\nIf you now decide differently from the latest of these, say what has changed since it.")
