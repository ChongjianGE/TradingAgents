"""What an analysis is told about the size of a position: a target weight is only as
good as the model's idea of what it buys."""

from __future__ import annotations

import re

import pytest

from tradeloop import context
from tradeloop.broker import Account, Position
from tradeloop.config import LoopConfig
from tradeloop.executor import plan_orders
from tradeloop.risk import Proposal, RiskContext, evaluate

CFG = LoopConfig(max_position_weight=0.5)


def book(qty=0.0, ticker="NVDA", equity=100_000.0):
    positions = {ticker: Position(ticker, qty, 174.71)} if qty else {}
    return Account(equity=equity, cash=equity - qty * 174.71, positions=positions)


def said(rating="Overweight", weight=0.25):
    return {"rating": rating, "target_weight": weight}


def told(qty=65.0, price=175.0, latest=None, cfg=CFG, equity=100_000.0):
    return context.sizing_context("NVDA", book(qty, equity=equity), price, latest, cfg)


@pytest.mark.unit
class TestTheSizeOfAFullPosition:
    def test_it_is_given_in_dollars_and_in_shares_of_this_ticker(self):
        text = told(qty=0.0, price=200.0)
        assert "- A full position is 50% of the account's equity: about $50,000 now, about 250 shares of NVDA at 200.00." in text
        assert "- A target weight of 0.1 is therefore about $5,000, about 25 shares." in text

    def test_it_follows_the_config_and_the_account(self):
        text = told(qty=0.0, price=200.0, cfg=LoopConfig(max_position_weight=0.2), equity=50_000.0)
        assert "20% of the account's equity: about $10,000 now, about 50 shares" in text

    def test_the_ratings_limits_are_the_configs(self):
        text = told()
        assert ("- The rating limits the weight: Buy at most 1, Overweight at most 0.6, Underweight at most 0.3, "
                "Sell is 0. Hold leaves the position exactly as it is, whatever weight is given.") in text
        custom = LoopConfig(rating_weights={"Buy": 0.8, "Overweight": 0.5, "Hold": "keep", "Underweight": "keep", "Sell": 0.0})
        assert "Buy at most 0.8, Overweight at most 0.5, Sell is 0. Hold and Underweight leaves" in told(cfg=custom)

    def test_without_a_price_only_what_needs_none_is_said(self):
        text = context.sizing_context("NVDA", book(65.0), None, said(), CFG)
        assert text.splitlines() == [
            "How a target weight becomes a position in this account:",
            "- A full position is 50% of the account's equity.",
            "- The rating limits the weight: Buy at most 1, Overweight at most 0.6, Underweight at most 0.3, "
            "Sell is 0. Hold leaves the position exactly as it is, whatever weight is given.",
        ]

    def test_fractional_shares_are_said_as_they_would_be_bought(self):
        text = told(qty=0.0, price=3000.0, cfg=LoopConfig(max_position_weight=0.5, allow_fractional=True))
        assert "about 16.6666 shares of NVDA" in text and "about 1.6666 shares." in text


@pytest.mark.unit
class TestWhereThePositionStands:
    def test_what_is_held_is_given_as_a_weight(self):
        text = told(qty=65.0, price=175.0)                    # 11,375 of a 50,000 full position
        assert "- Held now: 65 shares, about $11,375, which is a target weight of 0.228." in text
        assert "above 0.228 asks for the difference to be bought, below it for the difference to be sold" in text
        assert "To keep the position as it is, give 0.228." in text

    def test_with_nothing_held_staying_out_is_zero(self):
        text = told(qty=0.0)
        assert "- Held now: nothing, which is a target weight of 0." in text
        assert "any weight above 0 asks for that much to be bought. To stay out, give 0." in text

    def test_a_target_that_was_not_bought_is_said_to_be_waiting(self):
        """The first live day: 0.23 was asked for and held back, and the next analysis took 57 shares for 0.23."""
        text = told(qty=57.0, price=176.75, latest=said("Overweight", 0.23))
        assert ("- The latest decision (Overweight) asked for a target weight of 0.23, about 65 shares. "
                "8 shares of that have not been bought.") in text

    def test_one_share_short_is_singular(self):
        text = told(qty=64.0, price=100.0, latest=said("Overweight", 0.13), cfg=LoopConfig(max_position_weight=0.5, min_trade_notional=50.0))
        assert "about 65 shares. 1 share of that has not been bought." in text

    def test_a_target_that_is_in_place_is_said_to_be(self):
        text = told(qty=65.0, price=176.75, latest=said("Overweight", 0.23))
        assert "asked for a target weight of 0.23, about 65 shares. That is what is held." in text

    def test_more_held_than_the_last_target_is_said_too(self):
        text = told(qty=65.0, price=175.0, latest=said("Underweight", 0.1))
        assert "asked for a target weight of 0.1, about 28 shares. 37 shares more than that are still held." in text

    def test_the_weight_quoted_is_the_one_the_rating_allowed(self):
        text = told(qty=0.0, price=100.0, latest=said("Overweight", 0.9))
        assert "(Overweight) asked for a target weight of 0.6, about 300 shares" in text

    def test_a_position_grown_past_a_full_one_cannot_be_kept_whole(self):
        text = told(qty=300.0, price=175.0)                   # 52,500 against a full position of 50,000
        assert "which is a target weight of 1.050." in text
        assert ("- The position is larger than a full one. Any weight you give asks for at least the part above "
                "a full position to be sold; 1 keeps the rest.") in text
        assert "To keep the position as it is" not in text

    def test_a_hold_or_no_decision_yet_says_nothing_of_a_last_target(self):
        assert "latest decision" not in told(latest=said("Hold", 0.0))
        assert "latest decision" not in told(latest=None)

    @pytest.mark.parametrize(("qty", "price", "equity"), [
        (65.0, 173.49, 100_004.10), (126.0, 119.72, 100_004.10), (15.0, 809.88, 100_004.10),
        (1.0, 17.31, 250_000.0), (733.0, 61.07, 98_211.55), (3.0, 4411.2, 100_000.0),
    ])
    def test_giving_back_the_weight_quoted_trades_nothing(self, qty, price, equity):
        """ "To keep the position as it is, give X" has to be true of the rules that read X."""
        text = context.sizing_context("NVDA", book(qty, equity=equity), price, None, CFG)
        quoted = float(re.search(r"To keep the position as it is, give ([0-9.]+)\.$", text, re.M).group(1))
        for prior in ((), (quoted,), (1.0,)):
            proposal = Proposal(ticker="NVDA", price=price, current_qty=qty, rating="Overweight", stance=quoted,
                                price_target=price * 1.3, prior_stances=prior, decision_age_hours=0.1)
            rulings = evaluate([proposal], RiskContext(equity=equity, day_start_equity=equity), CFG)
            assert plan_orders(rulings, "20261112T0935-core", CFG, cash=equity) == []
