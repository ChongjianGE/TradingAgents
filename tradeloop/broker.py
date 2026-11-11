"""Where the loop's orders go: the Alpaca paper account, or a simulated book of its own.

Both answer the same four questions (what is held, send this order, cancel that
one, what became of it), so the rest of the loop does not know which it talks to.
The simulated book fills at the last price moved against the order by the
configured spread and slippage, and charges the configured fee, so a run that
sends nothing to a broker still pays to trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from tradeloop import timeutil
from tradeloop.alpaca import AlpacaClient, AlpacaError
from tradeloop.config import LoopConfig
from tradeloop.store import Store

_BOOK_KEY = "sim_book"


@dataclass(frozen=True)
class Position:
    ticker: str
    qty: float
    avg_price: float


@dataclass(frozen=True)
class Account:
    equity: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def qty(self, ticker: str) -> float:
        position = self.positions.get(ticker)
        return position.qty if position else 0.0


@dataclass(frozen=True)
class OrderRequest:
    ticker: str
    side: str                      # buy | sell
    qty: float
    type: str                      # market | limit | stop
    client_order_id: str
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "day"
    reason: str = ""


@dataclass(frozen=True)
class OrderState:
    status: str                    # new | partially_filled | filled | canceled | rejected | expired
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    broker_order_id: str | None = None
    fee_usd: float = 0.0
    error: str | None = None
    filled_at: datetime | None = None   # when the broker says it filled, when it says

    @property
    def is_open(self) -> bool:
        return self.status in ("new", "accepted", "pending_new", "partially_filled", "held")


class Broker(Protocol):
    name: str

    def account(self, prices: dict[str, float]) -> Account: ...
    def submit(self, order: OrderRequest, price: float, at: datetime) -> OrderState: ...
    def cancel(self, order: dict) -> None: ...
    def refresh(self, order: dict, price: float | None, at: datetime) -> OrderState: ...


# ---------------------------------------------------------------------------
# Alpaca paper
# ---------------------------------------------------------------------------


def _state(payload: dict) -> OrderState:
    price, filled_at = payload.get("filled_avg_price"), payload.get("filled_at")
    return OrderState(status=payload.get("status", "new"), filled_qty=float(payload.get("filled_qty") or 0),
                      filled_avg_price=float(price) if price else None, broker_order_id=payload.get("id"),
                      filled_at=timeutil.parse(filled_at) if filled_at else None)


class AlpacaBroker:
    name = "alpaca_paper"

    def __init__(self, client: AlpacaClient):
        self.client = client

    def account(self, prices: dict[str, float]) -> Account:
        """The paper account as Alpaca holds it; its own marks, not ``prices``."""
        account = self.client.account()
        positions = {
            p["symbol"]: Position(p["symbol"], float(p["qty"]), float(p["avg_entry_price"]))
            for p in self.client.positions()
        }
        return Account(equity=float(account["equity"]), cash=float(account["cash"]), positions=positions)

    def submit(self, order: OrderRequest, price: float, at: datetime) -> OrderState:
        # The same client order id is never sent twice: after a crash between
        # sending and recording, the order is found rather than repeated.
        existing = self.client.order_by_client_id(order.client_order_id)
        if existing is not None:
            return _state(existing)
        body = {"symbol": order.ticker, "side": order.side, "qty": _number(order.qty), "type": order.type,
                "time_in_force": order.time_in_force, "client_order_id": order.client_order_id}
        if order.limit_price is not None:
            body["limit_price"] = _number(round(order.limit_price, 2))
        if order.stop_price is not None:
            body["stop_price"] = _number(round(order.stop_price, 2))
        try:
            return _state(self.client.submit_order(**body))
        except AlpacaError as exc:
            if exc.status in (403, 422):     # not enough buying power or shares; parameters refused
                return OrderState(status="rejected", error=str(exc))
            raise

    def cancel(self, order: dict) -> None:
        if order.get("broker_order_id"):
            self.client.cancel_order(order["broker_order_id"])

    def refresh(self, order: dict, price: float | None, at: datetime) -> OrderState:
        payload = self.client.order_by_client_id(order["client_order_id"])
        if payload is None:
            return OrderState(status="canceled", error="the broker has no such order")
        return _state(payload)


def _number(value: float) -> str:
    """A quantity or price as Alpaca takes it: a plain decimal string."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Simulated book
# ---------------------------------------------------------------------------


class SimBroker:
    """A long-only cash book kept in the store. Nothing here reaches a broker."""

    name = "simulate"

    def __init__(self, store: Store, cfg: LoopConfig):
        self.store, self.cfg = store, cfg
        if store.get(_BOOK_KEY) is None:
            store.put(_BOOK_KEY, {"cash": cfg.sim_starting_cash, "positions": {}})

    def _book(self) -> dict:
        return self.store.get(_BOOK_KEY)

    def account(self, prices: dict[str, float]) -> Account:
        book = self._book()
        positions = {t: Position(t, p["qty"], p["avg_price"]) for t, p in book["positions"].items() if p["qty"]}
        # A position with no price this tick is carried at what was paid for it.
        value = sum(p.qty * prices.get(t, p.avg_price) for t, p in positions.items())
        return Account(equity=book["cash"] + value, cash=book["cash"], positions=positions)

    def submit(self, order: OrderRequest, price: float, at: datetime) -> OrderState:
        book = self._book()
        held = book["positions"].get(order.ticker, {"qty": 0.0})["qty"]
        if order.qty <= 0:
            return OrderState(status="rejected", error="quantity must be positive")
        if order.side == "sell" and order.qty > held + 1e-9:
            return OrderState(status="rejected", error=f"cannot sell {order.qty:g}: {held:g} held (no short selling)")
        if order.type == "stop":
            return OrderState(status="new", broker_order_id=f"sim-{order.client_order_id}")
        return self._try_fill(order.ticker, order.side, order.qty, order.type, order.limit_price,
                              price, f"sim-{order.client_order_id}")

    def cancel(self, order: dict) -> None:
        """Nothing to tell anyone: an open simulated order exists only as its row."""

    def refresh(self, order: dict, price: float | None, at: datetime) -> OrderState:
        """Whether an open simulated order fills at ``price``."""
        waiting = OrderState(status=order["status"], filled_qty=order["filled_qty"],
                             filled_avg_price=order["filled_avg_price"], broker_order_id=order["broker_order_id"])
        if price is None:
            return waiting
        if order["type"] == "stop":
            if price > order["stop_price"]:
                return waiting
            held = self._book()["positions"].get(order["ticker"], {"qty": 0.0})["qty"]
            qty = min(order["qty"], held)
            if qty <= 0:
                return OrderState(status="canceled", broker_order_id=order["broker_order_id"],
                                  error="nothing left to sell")
            return self._try_fill(order["ticker"], "sell", qty, "market", None, price, order["broker_order_id"])
        return self._try_fill(order["ticker"], order["side"], order["qty"], order["type"], order["limit_price"],
                              price, order["broker_order_id"])

    def _try_fill(self, ticker: str, side: str, qty: float, kind: str, limit: float | None,
                  price: float, order_id: str) -> OrderState:
        """Fill the whole order at ``price`` moved against it, when its limit allows."""
        drift = (self.cfg.half_spread_bps + self.cfg.slippage_bps) / 10_000
        fill_price = price * (1 + drift) if side == "buy" else price * (1 - drift)
        beyond = (side == "buy" and fill_price > (limit or 0)) or (side == "sell" and fill_price < (limit or 0))
        if kind == "limit" and limit is not None and beyond:
            return OrderState(status="new", broker_order_id=order_id)
        book = self._book()
        position = book["positions"].get(ticker, {"qty": 0.0, "avg_price": 0.0})
        fee = 0.0
        if side == "buy":
            cost = qty * fill_price
            if cost > book["cash"] + 1e-6:
                return OrderState(status="rejected", broker_order_id=order_id,
                                  error=f"needs {cost:,.2f}, cash is {book['cash']:,.2f} (no margin)")
            total = position["qty"] + qty
            position = {"qty": total, "avg_price": (position["qty"] * position["avg_price"] + cost) / total}
            book["cash"] -= cost
        else:
            fee = qty * self.cfg.sell_fee_per_share
            book["cash"] += qty * fill_price - fee
            left = position["qty"] - qty
            position = {"qty": 0.0 if left < 1e-9 else left, "avg_price": position["avg_price"] if left >= 1e-9 else 0.0}
        if position["qty"]:
            book["positions"][ticker] = position
        else:
            book["positions"].pop(ticker, None)
        self.store.put(_BOOK_KEY, book)
        return OrderState(status="filled", filled_qty=qty, filled_avg_price=fill_price,
                          broker_order_id=order_id, fee_usd=fee)


def make_broker(cfg: LoopConfig, store: Store, client: AlpacaClient | None) -> Broker:
    if cfg.execution == "alpaca_paper":
        if client is None:
            raise ValueError("execution is alpaca_paper but there is no Alpaca client")
        return AlpacaBroker(client)
    return SimBroker(store, cfg)

