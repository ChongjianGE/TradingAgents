"""The rules between a decision and an order: when to call the model, how much a
rating is worth, what a trade costs, and what the limits allow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradeloop import costs, gate, target, timeutil
from tradeloop.config import LoopConfig
from tradeloop.gate import ANALYZE, REUSE, SKIP, GateView
from tradeloop.risk import Proposal, RiskContext, evaluate

CFG = LoopConfig(news_trigger_count=3)
NOW = datetime(2026, 11, 11, 15, 35, tzinfo=UTC)


def last(minutes_ago=30, ref_price=100.0):
    return {"run_id": "r0", "started_at": timeutil.iso(NOW - timedelta(minutes=minutes_ago)), "ref_price": ref_price}


def view(kind="core", price=100.0, **more):
    more.setdefault("last", last())
    return GateView(kind=kind, now=NOW, price=price, **more)


@pytest.mark.unit
class TestCosts:
    def test_a_buy_costs_the_spread_and_slippage_on_the_amount(self):
        cfg = LoopConfig(half_spread_bps=2, slippage_bps=1)
        assert costs.trade_cost(10, 200.0, "buy", cfg) == pytest.approx(0.60)

    def test_a_sell_adds_the_per_share_fee(self):
        cfg = LoopConfig(half_spread_bps=2, slippage_bps=1, sell_fee_per_share=0.01)
        assert costs.trade_cost(-10, 200.0, "sell", cfg) == pytest.approx(0.70)


@pytest.mark.unit
class TestFromRatingToSize:
    @pytest.mark.parametrize("rating, pm_weight, expected", [
        ("Buy", None, 1.0),
        ("Overweight", None, 0.6),
        ("Underweight", None, 0.3),
        ("Sell", None, 0.0),
        ("Hold", None, None),              # keeps the position
        ("Hold", 0.9, None),               # whatever number comes with it
        ("Buy", 0.3, 0.3),                 # the model's own figure lowers it
        ("Underweight", 0.8, 0.3),         # and never raises it
        ("Buy", 1.7, 1.0),
        ("Buy", -0.2, 0.0),
        ("Buy", float("nan"), 1.0),
        ("REVIEW", None, None),            # an unreadable decision asks for nothing
        (None, None, None),
    ])
    def test_the_stance_a_decision_asks_for(self, rating, pm_weight, expected):
        assert target.stance(rating, pm_weight, CFG) == expected

    def test_a_stance_is_a_share_of_the_full_position(self):
        assert target.equity_weight(0.6, LoopConfig(max_position_weight=0.2)) == pytest.approx(0.12)

    def test_shares_are_whole_and_rounded_down(self):
        assert target.shares(0.12, 100_000, 169.0, CFG) == 71.0
        assert target.shares(0.12, 100_000, 169.0, LoopConfig(allow_fractional=True)) == 71.0059
        assert target.shares(0.0, 100_000, 169.0, CFG) == 0.0
        assert target.shares(0.12, 100_000, 0.0, CFG) == 0.0


@pytest.mark.unit
class TestTheGate:
    @pytest.mark.parametrize("kind", ["premarket", "postclose"])
    def test_the_scheduled_analyses_always_run(self, kind):
        decision = gate.decide(view(kind, last=None), CFG)
        assert decision.action == ANALYZE and kind in decision.reasons[0]

    def test_a_quiet_hour_reuses_the_standing_decision(self):
        assert gate.decide(view(), CFG) == gate.GateDecision(REUSE, ("nothing changed enough",))

    def test_the_first_tick_with_no_decision_analyses(self):
        assert gate.decide(view(last=None), CFG).action == ANALYZE

    @pytest.mark.parametrize("price, triggers", [(101.4, False), (101.5, True), (98.5, True), (98.6, False)])
    def test_a_price_move_past_the_threshold_triggers(self, price, triggers):
        decision = gate.decide(view(price=price), CFG)
        assert (decision.action == ANALYZE) is triggers
        if triggers:
            assert "price moved" in decision.reasons[0]

    def test_enough_new_headlines_trigger(self):
        assert gate.decide(view(new_headlines=2), CFG).action == REUSE
        decision = gate.decide(view(new_headlines=3), CFG)
        assert decision.action == ANALYZE and decision.reasons == ("3 new headlines since the last analysis",)

    def test_a_decision_goes_stale(self):
        assert gate.decide(view(last=last(minutes_ago=179)), CFG).action == REUSE
        assert "180 minutes old" in gate.decide(view(last=last(minutes_ago=180)), CFG).reasons[0]

    def test_every_reason_is_reported(self):
        decision = gate.decide(view(price=103.0, new_headlines=5, last=last(minutes_ago=200)), CFG)
        assert len(decision.reasons) == 3

    def test_a_spent_budget_stops_every_analysis(self):
        for kind in ("premarket", "core", "postclose", "offhours"):
            decision = gate.decide(view(kind, price=110.0, budget_left=0.0), CFG)
            assert decision.action == REUSE and "budget" in decision.reasons[0]
        assert gate.decide(view("premarket", last=None, budget_left=-1.0), CFG).action == SKIP

    def test_a_restarted_tick_does_not_analyse_twice(self):
        assert gate.decide(view("premarket", done_this_tick=True), CFG).action == REUSE

    def test_no_price_means_no_intraday_analysis(self):
        assert gate.decide(view(price=None), CFG).action == REUSE
        assert gate.decide(view(price=None, last=None), CFG).action == SKIP

    def test_the_watch_only_acts_on_a_reason(self):
        assert gate.decide(view("offhours"), CFG).action == SKIP
        assert gate.decide(view("offhours", last=last(minutes_ago=600)), CFG).action == SKIP   # age alone is no reason
        assert gate.decide(view("offhours", price=104.0), CFG).action == ANALYZE
        assert gate.decide(view("offhours", new_headlines=4), CFG).action == ANALYZE

    def test_the_watch_has_a_nightly_allowance(self):
        decision = gate.decide(view("offhours", price=104.0, offhours_done=1), CFG)
        assert decision.action == SKIP and "used its analyses" in decision.reasons[-1]

    def test_the_watch_never_makes_the_first_decision(self):
        assert gate.decide(view("offhours", last=None, price=104.0), CFG).action == SKIP


def proposal(**more):
    fields = {"ticker": "NVDA", "price": 100.0, "current_qty": 0.0, "rating": "Buy", "stance": 1.0,
              "run_id": "r2", "prior_stances": (1.0,), "decision_age_hours": 1.0}
    return Proposal(**{**fields, **more})


def rule(*proposals, cfg=CFG, **ctx):
    ctx.setdefault("equity", 100_000.0)
    return evaluate(list(proposals), RiskContext(**ctx), cfg)


def check(ruling, name):
    return next(c for c in ruling.checks if c.name == name)


@pytest.mark.unit
class TestTheRiskRules:
    def test_a_confirmed_buy_goes_to_a_full_position(self):
        (r,) = rule(proposal())
        assert (r.verdict, r.target_weight, r.target_qty) == ("increase", pytest.approx(0.2), 200.0)
        assert all(c.passed for c in r.checks)

    def test_a_first_buy_waits_for_a_second_analysis(self):
        (r,) = rule(proposal(prior_stances=()))
        assert r.verdict == "blocked" and r.target_qty == 0.0
        assert "1 so far" in check(r, "confirmation").detail

    @pytest.mark.parametrize("prior", [None, 0.0])
    def test_a_buy_after_a_hold_or_a_sell_is_not_confirmed(self, prior):
        (r,) = rule(proposal(prior_stances=(prior,)))
        assert r.verdict == "blocked" and not check(r, "confirmation").passed

    def test_two_analyses_that_disagree_on_size_get_the_smaller(self):
        (r,) = rule(proposal(stance=1.0, prior_stances=(0.6,)))
        assert r.verdict == "increase" and r.target_weight == pytest.approx(0.12)

    def test_more_rounds_need_more_agreement(self):
        cfg = LoopConfig(confirm_rounds=3)
        assert rule(proposal(prior_stances=(1.0,)), cfg=cfg)[0].verdict == "blocked"
        assert rule(proposal(prior_stances=(1.0, 1.0)), cfg=cfg)[0].verdict == "increase"

    def test_one_round_needs_no_earlier_analysis(self):
        assert rule(proposal(prior_stances=()), cfg=LoopConfig(confirm_rounds=1))[0].verdict == "increase"

    def test_a_reduction_needs_no_confirmation(self):
        (r,) = rule(proposal(current_qty=200.0, rating="Underweight", stance=0.3, prior_stances=()))
        assert (r.verdict, r.target_weight, r.target_qty) == ("reduce", pytest.approx(0.06), 60.0)

    def test_a_sell_closes_the_position(self):
        (r,) = rule(proposal(current_qty=200.0, rating="Sell", stance=0.0, prior_stances=()))
        assert (r.verdict, r.target_qty) == ("reduce", 0.0)

    def test_a_hold_changes_nothing(self):
        (r,) = rule(proposal(current_qty=57.0, rating="Hold", stance=None))
        assert (r.verdict, r.target_qty, r.raw_weight) == ("keep", 57.0, None)
        assert "keeps the position" in check(r, "decision").detail

    def test_no_decision_changes_nothing(self):
        (r,) = rule(Proposal(ticker="NVDA", price=100.0, current_qty=10.0))
        assert r.verdict == "keep" and r.target_qty == 10.0

    def test_a_difference_too_small_to_trade_is_not_a_change(self):
        # 0.2 of 100,000 is 20,000; 199 shares at 100 is 19,900, 100 short of it.
        (r,) = rule(proposal(current_qty=199.0))
        assert r.verdict == "keep" and r.target_qty == 199.0

    def test_a_price_under_the_stop_closes_the_position_whatever_the_rating(self):
        (r,) = rule(proposal(current_qty=100.0, price=91.9, stop_loss=92.0))
        assert (r.verdict, r.target_qty, r.target_weight) == ("exit_stop", 0.0, 0.0)
        assert not check(r, "stop_loss").passed

    def test_a_decision_whose_stop_is_reached_opens_nothing(self):
        (r,) = rule(proposal(price=91.9, stop_loss=92.0, prior_stances=(1.0,)))
        assert r.verdict == "blocked" and r.target_qty == 0.0
        assert "decision's stop" in check(r, "stop_loss").detail

    def test_a_decision_that_was_stopped_out_does_not_buy_back_when_the_price_recovers(self):
        (r,) = rule(proposal(price=96.0, stop_loss=92.0, stopped_out=True))
        assert r.verdict == "blocked" and "needs a new analysis" in check(r, "stop_loss").detail
        # It may still reduce: a later Sell on the same decision is not blocked by its own stop.
        (cut,) = rule(proposal(price=96.0, current_qty=50.0, rating="Sell", stance=0.0, stop_loss=92.0,
                               stopped_out=True))
        assert cut.verdict == "reduce"

    def test_above_its_stop_a_decision_is_only_noted_as_such(self):
        (r,) = rule(proposal(price=100.0, stop_loss=92.0))
        assert r.verdict == "increase" and check(r, "stop_loss").passed

    def test_an_old_decision_no_longer_moves_the_book(self):
        (r,) = rule(proposal(decision_age_hours=73.0))
        assert r.verdict == "keep" and not check(r, "decision_age").passed

    def test_past_the_days_loss_limit_only_reductions_go_through(self):
        (grow,) = rule(proposal(), equity=97_900.0, day_start_equity=100_000.0)
        assert grow.verdict == "blocked" and "only reductions" in check(grow, "daily_loss").detail
        (cut,) = rule(proposal(current_qty=200.0, rating="Sell", stance=0.0), equity=97_900.0,
                      day_start_equity=100_000.0)
        assert cut.verdict == "reduce"

    def test_inside_the_loss_limit_the_day_goes_on(self):
        (r,) = rule(proposal(), equity=98_100.0, day_start_equity=100_000.0)
        assert r.verdict == "increase" and check(r, "daily_loss").passed

    def test_a_target_too_close_to_cover_the_cost_blocks_the_buy(self):
        # A round trip costs 6 bps; three times that is 0.18%.
        (near,) = rule(proposal(price_target=100.1))
        assert near.verdict == "blocked" and not check(near, "edge").passed
        (far,) = rule(proposal(price_target=110.0))
        assert far.verdict == "increase"
        (unknown,) = rule(proposal(price_target=None))
        assert unknown.verdict == "increase" and "no price target" in check(unknown, "edge").detail

    def test_no_price_changes_nothing(self):
        (r,) = rule(proposal(price=None, current_qty=10.0))
        assert r.verdict == "keep" and r.target_qty == 10.0 and not check(r, "price").passed

    def test_increases_are_scaled_to_fit_the_book(self):
        cfg = LoopConfig(max_position_weight=0.5, max_gross_exposure=0.8)
        held = Proposal(ticker="HD", price=100.0, current_qty=200.0)       # 20% already held, no decision
        a, b, h = rule(proposal(ticker="NVDA"), proposal(ticker="CSCO"), held, cfg=cfg)
        # 60% of room for two increases wanting 50% each.
        assert a.target_weight == pytest.approx(0.3) and b.target_weight == pytest.approx(0.3)
        assert not check(a, "gross_exposure").passed and a.target_qty == 300.0
        assert h.verdict == "keep"

    def test_increases_that_fit_are_left_alone(self):
        rulings = rule(*[proposal(ticker=t) for t in ("NVDA", "CSCO", "HD", "DIS", "GEV")])
        assert sum(r.target_weight for r in rulings) == pytest.approx(1.0)
        assert all(check(r, "gross_exposure").passed for r in rulings)

    def test_outside_the_trading_window_the_ruling_is_recorded_but_not_actionable(self):
        (r,) = rule(proposal(), can_trade=False, window_note="the market is closed")
        assert r.verdict == "increase" and not r.actionable
        row = r.as_row("20261111T0845-premarket")
        assert row["verdict"] == "deferred:increase" and row["checks"][-1]["detail"] == "the market is closed"

    def test_a_ruling_becomes_a_row_with_every_check(self):
        (r,) = rule(proposal())
        row = r.as_row("t1")
        assert row["verdict"] == "increase" and row["run_id"] == "r2" and row["rating"] == "Buy"
        assert [c["name"] for c in row["checks"]] == ["confirmation", "edge", "gross_exposure", "trading_window"]
