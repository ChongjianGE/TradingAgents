"""From rulings to orders, and from orders to the record of what filled.

Each tick starts from the positions actually held, so an order that never filled
or filled in part needs no special handling: the next tick sees the book as it
is and orders the difference again. Client order ids name the tick, the ticker
and the side, so running a tick twice cannot place its orders twice.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

from tradeloop import costs, timeutil
from tradeloop.broker import Account, Broker, OrderRequest, OrderState
from tradeloop.config import LoopConfig
from tradeloop.risk import Ruling
from tradeloop.store import Store


def plan_orders(rulings: list[Ruling], tick_id: str, cfg: LoopConfig,
                cash: float | None = None) -> list[OrderRequest]:
    """The orders that move each position to its target: sells first, to free the cash buys need.

    With ``cash`` given, buys are cut to what the cash on hand and this tick's
    sells pay for at their limit prices. Targets add up to the whole book, but
    trading costs something, so without this the last buy would ask for a little
    more than is left and be refused outright.
    """
    orders = []
    for r in rulings:
        if not r.actionable or r.price is None:
            continue
        delta = r.target_qty - r.current_qty
        closing = r.target_qty == 0 and r.current_qty > 0
        if delta == 0 or (abs(delta) * r.price < cfg.min_trade_notional and not closing):
            continue    # too small to be worth its cost; a full exit is always worth it
        side = "buy" if delta > 0 else "sell"
        qty = abs(delta) if cfg.allow_fractional else float(math.floor(abs(delta) + 1e-9))
        if qty <= 0:
            continue
        common = {"ticker": r.ticker, "side": side, "qty": qty, "reason": r.verdict,
                  "client_order_id": f"tl-{tick_id}-{r.ticker}-{side}"}
        if r.verdict == "exit_stop":
            orders.append(OrderRequest(type="market", **common))    # out now, at whatever the price is
        else:
            reach = cfg.limit_offset_bps / 10_000
            limit = r.price * (1 + reach) if side == "buy" else r.price * (1 - reach)
            orders.append(OrderRequest(type="limit", limit_price=round(limit, 2), **common))
    orders.sort(key=lambda o: o.side != "sell")
    return orders if cash is None else _fit_to_cash(orders, {r.ticker: r.price for r in rulings}, cash, cfg)


def _fit_to_cash(orders: list[OrderRequest], prices: dict, cash: float, cfg: LoopConfig) -> list[OrderRequest]:
    """Cut each buy, in order, to what is left to spend; drop one that becomes too small."""
    available = cash + sum((o.limit_price or prices[o.ticker] * (1 - cfg.limit_offset_bps / 10_000)) * o.qty
                           for o in orders if o.side == "sell")
    fitted = []
    for order in orders:
        if order.side == "sell":
            fitted.append(order)
            continue
        worst = order.limit_price or prices[order.ticker]
        affordable = max(0.0, available / worst)
        qty = min(order.qty, affordable if cfg.allow_fractional else float(math.floor(affordable + 1e-9)))
        if qty <= 0 or qty * worst < cfg.min_trade_notional:
            continue
        available -= qty * worst
        fitted.append(order if qty == order.qty else replace(order, qty=qty))
    return fitted


class Executor:
    def __init__(self, broker: Broker, store: Store, cfg: LoopConfig):
        self.broker, self.store, self.cfg = broker, store, cfg

    # --- what became of earlier orders
    def settle_open_orders(self, prices: dict[str, float], at: datetime, tick_id: str) -> int:
        """Bring every open order up to date; cancel plain orders left over from earlier ticks.

        Returns the number of orders that filled, in part or in full.
        """
        filled = 0
        for row in self.store.open_orders(self.broker.name):
            state = self.broker.refresh(row, prices.get(row["ticker"]), at)
            filled += self._record(row, state, at)
            if state.is_open and row["type"] != "stop" and row["tick_id"] != tick_id:
                self._cancel(row, at)
        return filled

    # --- new orders
    def place(self, orders: list[OrderRequest], prices: dict[str, float], at: datetime, tick_id: str) -> list[dict]:
        """Send ``orders``; each is recorded before it is sent, so a crash leaves a trace."""
        placed = []
        for order in orders:
            existing = self.store.order(order.client_order_id)
            if existing and existing["status"] != "pending_new":
                placed.append(existing)      # this tick already placed it
                continue
            price = prices[order.ticker]
            if order.type != "stop":
                # A resting stop holds the shares a sell needs, and a broker refuses a
                # buy that could trade against one's own stop. It is put back after.
                self._cancel_stops(order.ticker, at)
            if order.side == "sell":
                # The book may have changed since it was read: a stop can fire while
                # an analysis runs. Selling more than is held would open a short.
                held = self.broker.account(prices).qty(order.ticker)
                if held < order.qty:
                    if held <= 0 or (order.type == "stop" and held < order.qty):
                        self.store.log("warning", "executor",
                                       f"sell of {order.qty:g} {order.ticker} dropped: {held:g} held now",
                                       at=timeutil.iso(at), tick_id=tick_id, ticker=order.ticker)
                        continue
                    order = replace(order, qty=held if self.cfg.allow_fractional else float(math.floor(held + 1e-9)))
            if existing is None:
                moment = timeutil.iso(at)
                self.store.add_order(
                    client_order_id=order.client_order_id, tick_id=tick_id, ticker=order.ticker, side=order.side,
                    qty=order.qty, type=order.type, limit_price=order.limit_price, stop_price=order.stop_price,
                    time_in_force=order.time_in_force, reason=order.reason, status="pending_new",
                    broker=self.broker.name, broker_order_id=None, submitted_at=moment, updated_at=moment,
                    est_cost_usd=costs.trade_cost(order.qty, price, order.side, self.cfg))
            state = self.broker.submit(order, price, at)
            self._record(self.store.order(order.client_order_id), state, at)
            placed.append(self.store.order(order.client_order_id))
        return placed

    # --- protective stops
    def sync_stops(self, account: Account, stops: dict[str, float | None], prices: dict[str, float],
                   at: datetime, tick_id: str) -> None:
        """Keep one stop order per held position, matching its size and the decision's level."""
        open_stops: dict[str, list[dict]] = {}
        for row in self.store.open_orders(self.broker.name):
            if row["type"] == "stop":
                open_stops.setdefault(row["ticker"], []).append(row)
        for ticker in set(account.positions) | set(open_stops):
            held = account.qty(ticker)
            qty = held if self.cfg.allow_fractional else float(math.floor(held + 1e-9))
            level = stops.get(ticker)
            existing = open_stops.get(ticker, [])
            wanted = qty > 0 and level is not None and ticker in prices and level < prices[ticker]
            if wanted and len(existing) == 1 and existing[0]["qty"] == qty \
                    and abs(existing[0]["stop_price"] - level) < 0.005:
                continue
            for row in existing:
                self._cancel(row, at)
            if wanted:
                stop = OrderRequest(ticker=ticker, side="sell", qty=qty, type="stop", stop_price=round(level, 2),
                                    time_in_force="gtc", reason="protective stop",
                                    client_order_id=f"tl-{tick_id}-{ticker}-stop")
                if self.store.order(stop.client_order_id) is None:
                    self.place([stop], prices, at, tick_id)

    # --- bookkeeping
    def _cancel_stops(self, ticker: str, at: datetime) -> None:
        for row in self.store.open_orders(self.broker.name):
            if row["ticker"] == ticker and row["type"] == "stop":
                self._cancel(row, at)

    def _cancel(self, row: dict, at: datetime) -> None:
        self.broker.cancel(row)
        self.store.update_order(row["client_order_id"], status="canceled", updated_at=timeutil.iso(at))

    def _record(self, row: dict, state: OrderState, at: datetime) -> int:
        """Store an order's new state and any quantity filled since the last look; 1 when something filled."""
        moment = timeutil.iso(at)
        newly = state.filled_qty - (row["filled_qty"] or 0)
        self.store.update_order(
            row["client_order_id"], status=state.status, updated_at=moment,
            broker_order_id=state.broker_order_id or row["broker_order_id"],
            filled_qty=state.filled_qty, filled_avg_price=state.filled_avg_price, error=state.error)
        if newly <= 1e-9 or state.filled_avg_price is None:
            return 0
        # The price of just this part, from the two running averages.
        before = (row["filled_qty"] or 0) * (row["filled_avg_price"] or 0)
        price = (state.filled_qty * state.filled_avg_price - before) / newly
        if state.filled_at is not None:
            moment = timeutil.iso(state.filled_at)
        self.store.add_fills([{
            "fill_id": f"{row['client_order_id']}:{state.filled_qty:g}",
            "broker_order_id": state.broker_order_id or row["broker_order_id"],
            "client_order_id": row["client_order_id"], "ticker": row["ticker"], "side": row["side"],
            "qty": newly, "price": price, "at": moment, "fee_usd": state.fee_usd,
            "cost_usd": costs.trade_cost(newly, price, row["side"], self.cfg),
        }])
        return 1
