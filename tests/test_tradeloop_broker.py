"""Where orders go: the simulated book pays to trade and keeps to cash; the Alpaca
broker sends each order once and reads back what became of it."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tradeloop.alpaca import AlpacaError
from tradeloop.broker import AlpacaBroker, OrderRequest, SimBroker, make_broker
from tradeloop.config import LoopConfig
from tradeloop.store import Store

NOW = datetime(2026, 11, 11, 15, 35, tzinfo=UTC)
CFG = LoopConfig(half_spread_bps=2.0, slippage_bps=1.0, sell_fee_per_share=0.01, sim_starting_cash=10_000.0)
DRIFT = 0.0003


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "loop.db") as s:
        yield s


def order(side="buy", qty=10, kind="market", **more):
    return OrderRequest(ticker="NVDA", side=side, qty=qty, type=kind,
                        client_order_id=f"tl-{side}-{qty}-{kind}-{len(more)}", **more)


def stored(request: OrderRequest, state) -> dict:
    """The row the executor would keep for an order, as ``refresh`` and ``cancel`` take it."""
    return {"client_order_id": request.client_order_id, "ticker": request.ticker, "side": request.side,
            "qty": request.qty, "type": request.type, "limit_price": request.limit_price,
            "stop_price": request.stop_price, "status": state.status, "filled_qty": state.filled_qty,
            "filled_avg_price": state.filled_avg_price, "broker_order_id": state.broker_order_id}


@pytest.mark.unit
class TestTheSimulatedBook:
    def test_it_starts_with_cash_and_nothing_else(self, store):
        account = SimBroker(store, CFG).account({})
        assert (account.equity, account.cash, account.positions) == (10_000.0, 10_000.0, {})

    def test_a_buy_fills_above_the_last_price_and_a_sell_below_it(self, store):
        broker = SimBroker(store, CFG)
        bought = broker.submit(order("buy", 10), 100.0, NOW)
        assert bought.status == "filled" and bought.filled_avg_price == pytest.approx(100.0 * (1 + DRIFT))
        sold = broker.submit(order("sell", 10), 100.0, NOW)
        assert sold.filled_avg_price == pytest.approx(100.0 * (1 - DRIFT)) and sold.fee_usd == pytest.approx(0.10)

        # A round trip at an unchanged price loses the spread, the slippage and the fee.
        account = broker.account({"NVDA": 100.0})
        assert account.positions == {}
        assert account.cash == pytest.approx(10_000.0 - 10 * 100.0 * 2 * DRIFT - 0.10)

    def test_buying_twice_averages_the_price_paid(self, store):
        broker = SimBroker(store, CFG)
        broker.submit(order("buy", 10), 100.0, NOW)
        broker.submit(order("buy", 30), 120.0, NOW)
        position = broker.account({"NVDA": 120.0}).positions["NVDA"]
        assert position.qty == 40
        assert position.avg_price == pytest.approx((10 * 100.0 + 30 * 120.0) * (1 + DRIFT) / 40)

    def test_equity_marks_positions_at_the_given_prices(self, store):
        broker = SimBroker(store, CFG)
        broker.submit(order("buy", 10), 100.0, NOW)
        cash = broker.account({}).cash
        assert broker.account({"NVDA": 110.0}).equity == pytest.approx(cash + 1100.0)
        # With no price this tick the position is carried at cost, not at zero.
        assert broker.account({}).equity == pytest.approx(cash + 10 * 100.0 * (1 + DRIFT))

    def test_it_cannot_spend_more_than_its_cash(self, store):
        broker = SimBroker(store, CFG)
        state = broker.submit(order("buy", 101), 100.0, NOW)
        assert state.status == "rejected" and "no margin" in state.error
        assert broker.account({}).cash == 10_000.0

    def test_it_cannot_sell_what_it_does_not_hold(self, store):
        broker = SimBroker(store, CFG)
        broker.submit(order("buy", 5), 100.0, NOW)
        state = broker.submit(order("sell", 6), 100.0, NOW)
        assert state.status == "rejected" and "no short selling" in state.error
        assert broker.account({}).qty("NVDA") == 5

    def test_a_limit_order_waits_until_the_price_reaches_it(self, store):
        broker = SimBroker(store, CFG)
        request = order("buy", 10, "limit", limit_price=99.0)
        state = broker.submit(request, 100.0, NOW)
        assert state.status == "new" and state.is_open and broker.account({}).qty("NVDA") == 0

        row = stored(request, state)
        assert broker.refresh(row, 99.5, NOW).status == "new"
        assert broker.refresh(row, None, NOW).status == "new"      # no price, no change
        filled = broker.refresh(row, 98.9, NOW)
        assert filled.status == "filled" and filled.filled_avg_price <= 99.0
        assert broker.account({}).qty("NVDA") == 10

    def test_a_marketable_limit_fills_at_once(self, store):
        state = SimBroker(store, CFG).submit(order("buy", 10, "limit", limit_price=100.1), 100.0, NOW)
        assert state.status == "filled" and state.filled_avg_price == pytest.approx(100.03)

    def test_a_stop_sells_once_the_price_falls_to_it(self, store):
        broker = SimBroker(store, CFG)
        broker.submit(order("buy", 10), 100.0, NOW)
        request = order("sell", 10, "stop", stop_price=92.0, time_in_force="gtc")
        state = broker.submit(request, 100.0, NOW)
        assert state.status == "new"

        row = stored(request, state)
        assert broker.refresh(row, 92.5, NOW).status == "new"
        triggered = broker.refresh(row, 91.0, NOW)
        assert triggered.status == "filled" and triggered.filled_avg_price == pytest.approx(91.0 * (1 - DRIFT))
        assert broker.account({}).positions == {}

    def test_a_stop_for_shares_already_sold_is_cancelled_not_shorted(self, store):
        broker = SimBroker(store, CFG)
        broker.submit(order("buy", 10), 100.0, NOW)
        request = order("sell", 10, "stop", stop_price=92.0)
        row = stored(request, broker.submit(request, 100.0, NOW))
        broker.submit(order("sell", 10), 100.0, NOW)
        assert broker.refresh(row, 90.0, NOW).status == "canceled"
        assert broker.account({}).positions == {}

    def test_the_book_survives_a_restart(self, tmp_path):
        with Store(tmp_path / "loop.db") as s:
            SimBroker(s, CFG).submit(order("buy", 10), 100.0, NOW)
        with Store(tmp_path / "loop.db") as s:
            account = SimBroker(s, CFG).account({"NVDA": 100.0})
            assert account.qty("NVDA") == 10 and account.cash < 10_000.0


class _FakeClient:
    """Stands in for AlpacaClient; keeps what was sent."""

    def __init__(self, existing=None, submit_error=None):
        self.existing, self.submit_error, self.sent, self.cancelled = existing or {}, submit_error, [], []

    def account(self):
        return {"equity": "100250.50", "cash": "83300.00"}

    def positions(self):
        return [{"symbol": "NVDA", "qty": "100", "avg_entry_price": "168.50"}]

    def order_by_client_id(self, client_order_id):
        return self.existing.get(client_order_id)

    def submit_order(self, **body):
        if self.submit_error:
            raise self.submit_error
        self.sent.append(body)
        return {"id": "o-1", "status": "accepted", "filled_qty": "0", "filled_avg_price": None}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)


@pytest.mark.unit
class TestTheAlpacaBroker:
    def test_the_account_is_read_into_numbers(self):
        account = AlpacaBroker(_FakeClient()).account({"NVDA": 1.0})
        assert (account.equity, account.cash) == (100_250.50, 83_300.0)
        assert account.positions["NVDA"].qty == 100 and account.positions["NVDA"].avg_price == 168.5

    def test_an_order_is_sent_with_plain_decimal_strings(self):
        client = _FakeClient()
        request = OrderRequest("NVDA", "buy", 12, "limit", "tl-1-NVDA-buy", limit_price=169.204999)
        state = AlpacaBroker(client).submit(request, 169.0, NOW)
        assert state.status == "accepted" and state.is_open and state.broker_order_id == "o-1"
        assert client.sent == [{"symbol": "NVDA", "side": "buy", "qty": "12", "type": "limit",
                                "time_in_force": "day", "client_order_id": "tl-1-NVDA-buy", "limit_price": "169.2"}]

    def test_an_order_already_at_the_broker_is_not_sent_again(self):
        found = {"id": "o-9", "status": "filled", "filled_qty": "12", "filled_avg_price": "169.11"}
        client = _FakeClient(existing={"tl-1-NVDA-buy": found})
        state = AlpacaBroker(client).submit(OrderRequest("NVDA", "buy", 12, "market", "tl-1-NVDA-buy"), 169.0, NOW)
        assert client.sent == []
        assert (state.status, state.filled_qty, state.filled_avg_price) == ("filled", 12.0, 169.11)

    @pytest.mark.parametrize("status", [403, 422])
    def test_a_refused_order_is_a_rejection_not_a_crash(self, status):
        client = _FakeClient(submit_error=AlpacaError(status, f"{status} insufficient buying power"))
        state = AlpacaBroker(client).submit(OrderRequest("NVDA", "buy", 1, "market", "c1"), 169.0, NOW)
        assert state.status == "rejected" and "insufficient buying power" in state.error

    def test_an_outage_while_sending_is_raised(self):
        client = _FakeClient(submit_error=AlpacaError(503, "HTTP 503"))
        with pytest.raises(AlpacaError):
            AlpacaBroker(client).submit(OrderRequest("NVDA", "buy", 1, "market", "c1"), 169.0, NOW)

    def test_refresh_reads_the_orders_state_and_a_vanished_order_is_closed(self):
        found = {"id": "o-9", "status": "partially_filled", "filled_qty": "5", "filled_avg_price": "169.0"}
        broker = AlpacaBroker(_FakeClient(existing={"c1": found}))
        state = broker.refresh({"client_order_id": "c1"}, None, NOW)
        assert state.status == "partially_filled" and state.filled_qty == 5.0 and state.is_open
        assert broker.refresh({"client_order_id": "gone"}, None, NOW).status == "canceled"

    def test_cancel_names_the_brokers_order(self):
        client = _FakeClient()
        broker = AlpacaBroker(client)
        broker.cancel({"broker_order_id": "o-1"})
        broker.cancel({"broker_order_id": None})          # never reached the broker: nothing to cancel
        assert client.cancelled == ["o-1"]


@pytest.mark.unit
class TestChoosingABroker:
    def test_simulate_is_the_default_and_needs_no_client(self, store):
        assert make_broker(LoopConfig(), store, None).name == "simulate"

    def test_alpaca_paper_needs_a_client(self, store):
        cfg = LoopConfig(execution="alpaca_paper")
        assert make_broker(cfg, store, _FakeClient()).name == "alpaca_paper"
        with pytest.raises(ValueError, match="no Alpaca client"):
            make_broker(cfg, store, None)
