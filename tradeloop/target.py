"""From a decision to a size. A rating stands for a fraction of a full position; the
Portfolio Manager's own figure may make that smaller, never larger. A full position
is a fixed share of equity, so the number of shares follows from the price."""

from __future__ import annotations

import math

from tradeloop.config import KEEP, LoopConfig


def stance(rating: str | None, pm_weight: float | None, cfg: LoopConfig) -> float | None:
    """The fraction of a full position a decision asks for, 0 to 1; None leaves the position alone.

    A decision with no readable rating, or a rating mapped to "keep", asks for
    nothing. The model's own target only ever lowers what its rating allows.
    """
    weight = cfg.rating_weights.get(rating) if rating else None
    if weight is None or weight == KEEP:
        return None
    if pm_weight is not None and not math.isnan(pm_weight):
        weight = min(weight, max(0.0, min(1.0, pm_weight)))
    return float(weight)


def equity_weight(stance_value: float, cfg: LoopConfig) -> float:
    """A stance as a share of equity."""
    return stance_value * cfg.max_position_weight


def shares(weight: float, equity: float, price: float, cfg: LoopConfig) -> float:
    """How many shares make up ``weight`` of ``equity`` at ``price``, rounded down."""
    if price <= 0 or weight <= 0 or equity <= 0:
        return 0.0
    exact = weight * equity / price
    if cfg.allow_fractional:
        return math.floor(exact * 10_000 + 1e-6) / 10_000
    return float(math.floor(exact + 1e-9))
