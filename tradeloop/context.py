"""What the loop tells the agents that they could not otherwise know: the book as it
is now, the time of day, and what was already decided and traded this session."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from tradeloop import timeutil
from tradeloop.broker import Account
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
