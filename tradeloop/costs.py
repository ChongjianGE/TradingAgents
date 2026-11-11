"""What a trade costs: half the spread and the slippage on the amount traded, and a
per-share fee on sells. The same model sizes orders, prices the simulated book's
fills, and is recorded with every real fill, where the broker does not report it."""

from __future__ import annotations

from tradeloop.config import LoopConfig


def trade_cost(qty: float, price: float, side: str, cfg: LoopConfig) -> float:
    """The estimated cost in USD of trading ``qty`` at ``price``."""
    qty = abs(qty)
    cost = qty * price * (cfg.half_spread_bps + cfg.slippage_bps) / 10_000
    if side == "sell":
        cost += qty * cfg.sell_fee_per_share
    return cost
