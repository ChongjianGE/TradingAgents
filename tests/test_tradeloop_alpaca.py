"""The Alpaca client: paper only, reads retried, orders sent once, responses read into plain values."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import requests

from tradeloop.alpaca import PAPER_URL, AlpacaClient, AlpacaError, load_credentials


class _Response:
    def __init__(self, status=200, payload=None, content=True):
        self.status_code, self._payload = status, payload
        self.content = b"x" if content and payload is not None else b""

    def json(self):
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


class _Session:
    """Answers each request with the next scripted response; keeps what was asked."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params, "json": json, "headers": headers})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _client(*responses, **kwargs):
    session = _Session(*responses)
    return AlpacaClient("PKTESTKEY", "s3cr3t-value", session=session, sleep=lambda _: None, **kwargs), session


@pytest.mark.unit
class TestPaperOnly:
    @pytest.mark.parametrize("url", ["https://api.alpaca.markets", "https://api.alpaca.markets/v2",
                                     "https://paper-api.alpaca.markets.evil.example"])
    def test_any_other_trading_host_is_refused(self, url):
        with pytest.raises(ValueError, match="only talks to the paper account"):
            AlpacaClient("k", "s", trading_url=url)

    def test_the_paper_endpoint_is_accepted_with_or_without_its_version_suffix(self):
        assert AlpacaClient("k", "s", trading_url=PAPER_URL + "/v2").trading_url == PAPER_URL
        assert AlpacaClient("k", "s").trading_url == PAPER_URL

    def test_every_request_carries_the_keys_as_headers_not_in_the_url(self):
        client, session = _client(_Response(payload={"is_open": False}))
        client.clock()
        call = session.calls[0]
        assert call["headers"]["APCA-API-KEY-ID"] == "PKTESTKEY"
        assert call["headers"]["APCA-API-SECRET-KEY"] == "s3cr3t-value"
        assert "s3cr3t" not in call["url"] and call["url"] == PAPER_URL + "/v2/clock"


@pytest.mark.unit
class TestCredentials:
    def test_the_standard_names_are_read_from_the_environment(self):
        assert load_credentials(environ={"APCA_API_KEY_ID": " PK1 ", "APCA_API_SECRET_KEY": "sec"}) == ("PK1", "sec")

    def test_a_file_is_read_when_the_environment_has_none(self, tmp_path):
        env = tmp_path / "tokens.env"
        env.write_text("ALPCA_PAPER_TRADING_API_KEY=PK2\nALPCA_PAPER_TRADING_SECRET_KEY=sec2\n")
        assert load_credentials(env, environ={}) == ("PK2", "sec2")

    def test_the_environment_wins_over_the_file(self, tmp_path):
        env = tmp_path / "tokens.env"
        env.write_text("APCA_API_KEY_ID=PKFILE\nAPCA_API_SECRET_KEY=file\n")
        assert load_credentials(env, environ={"APCA_API_KEY_ID": "PKENV", "APCA_API_SECRET_KEY": "env"}) == ("PKENV", "env")

    def test_half_a_pair_is_not_a_pair(self):
        with pytest.raises(AlpacaError, match="no Alpaca paper keys found"):
            load_credentials(environ={"APCA_API_KEY_ID": "PK1"})


@pytest.mark.unit
class TestTransport:
    def test_a_busy_read_is_tried_again(self):
        client, session = _client(_Response(429, {"message": "too many requests"}), _Response(503, None),
                                  _Response(payload={"is_open": True}))
        assert client.clock() == {"is_open": True}
        assert len(session.calls) == 3

    def test_a_read_that_stays_busy_gives_up_with_the_brokers_reason(self):
        client, session = _client(*[_Response(429, {"message": "too many requests"})] * 3)
        with pytest.raises(AlpacaError, match="429 too many requests") as caught:
            client.account()
        assert caught.value.status == 429 and len(session.calls) == 3

    def test_an_order_is_never_sent_twice(self):
        client, session = _client(_Response(503, None), _Response(payload={"id": "o1"}))
        with pytest.raises(AlpacaError) as caught:
            client.submit_order(symbol="NVDA", side="buy", qty="1", type="market", time_in_force="day")
        assert caught.value.status == 503 and len(session.calls) == 1

    def test_a_dropped_connection_is_retried_and_reported_without_the_url(self):
        boom = requests.ConnectionError("https://paper-api.alpaca.markets/v2/account?secret=s3cr3t-value failed")
        client, session = _client(boom, boom, boom)
        with pytest.raises(AlpacaError) as caught:
            client.account()
        assert caught.value.status == 0 and len(session.calls) == 3
        assert "s3cr3t" not in str(caught.value) and "ConnectionError" in str(caught.value)

    def test_a_refusal_is_not_retried(self):
        client, session = _client(_Response(403, {"message": "forbidden"}))
        with pytest.raises(AlpacaError, match="403 forbidden"):
            client.account()
        assert len(session.calls) == 1


@pytest.mark.unit
class TestTrading:
    def test_an_unknown_client_order_id_is_none_rather_than_an_error(self):
        client, session = _client(_Response(404, {"code": 40410000, "message": "order not found"}))
        assert client.order_by_client_id("tl-none") is None
        assert session.calls[0]["params"] == {"client_order_id": "tl-none"}

    def test_an_order_is_posted_as_given(self):
        client, session = _client(_Response(payload={"id": "o1", "status": "accepted"}))
        order = {"symbol": "NVDA", "side": "buy", "qty": "3", "type": "limit", "limit_price": "169.2",
                 "time_in_force": "day", "client_order_id": "tl-1-NVDA-buy"}
        assert client.submit_order(**order)["id"] == "o1"
        assert session.calls[0]["method"] == "POST" and session.calls[0]["json"] == order

    @pytest.mark.parametrize("status", [404, 422])
    def test_cancelling_an_order_that_is_already_done_is_not_an_error(self, status):
        client, _ = _client(_Response(status, {"message": "order is not cancelable"}))
        client.cancel_order("o1")

    def test_a_cancel_that_fails_for_another_reason_is_raised(self):
        client, _ = _client(_Response(403, {"message": "forbidden"}))
        with pytest.raises(AlpacaError):
            client.cancel_order("o1")

    def test_the_calendar_is_asked_for_a_date_range(self):
        rows = [{"date": "2026-11-27", "open": "09:30", "close": "13:00"}]
        client, session = _client(_Response(payload=rows))
        assert client.calendar("2026-11-27", "2026-11-27") == rows
        assert session.calls[0]["params"] == {"start": "2026-11-27", "end": "2026-11-27"}


@pytest.mark.unit
class TestMarketData:
    def test_quotes_for_every_symbol_come_from_one_request(self):
        payload = {
            "NVDA": {"latestTrade": {"p": 169, "t": "2026-11-10T21:17:52.661781656Z"}, "prevDailyBar": {"c": 178.64}},
            "CSCO": {"latestTrade": {"p": 98.15, "t": "2026-11-10T20:59:59.901302182Z"}, "prevDailyBar": None},
            "HALT": {"latestTrade": None},
        }
        client, session = _client(_Response(payload=payload))
        quotes = client.quotes(["NVDA", "CSCO", "HALT", "GONE"])
        assert len(session.calls) == 1
        assert session.calls[0]["params"] == {"symbols": "NVDA,CSCO,HALT,GONE", "feed": "iex"}
        assert set(quotes) == {"NVDA", "CSCO"}          # no trade, no quote
        assert quotes["NVDA"].price == 169.0 and quotes["NVDA"].prev_close == 178.64
        assert quotes["NVDA"].at == datetime(2026, 11, 10, 21, 17, 52, 661781, tzinfo=UTC)
        assert quotes["CSCO"].prev_close is None

    def test_no_symbols_asks_nothing(self):
        client, session = _client()
        assert client.quotes([]) == {} and client.news([]) == []
        assert client.bars([], "1Hour", datetime(2026, 11, 10, tzinfo=UTC), datetime(2026, 11, 11, tzinfo=UTC)) == {}
        assert session.calls == []

    def test_bars_follow_every_page(self):
        bar = lambda t, c: {"t": t, "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 1000}  # noqa: E731
        client, session = _client(
            _Response(payload={"bars": {"NVDA": [bar("2026-11-10T15:00:00Z", 168.6)]}, "next_page_token": "p2"}),
            _Response(payload={"bars": {"NVDA": [bar("2026-11-10T16:00:00Z", 169.4)],
                                        "CSCO": [bar("2026-11-10T15:00:00Z", 97.67)]}, "next_page_token": None}))
        start, end = datetime(2026, 11, 10, 14, 30, tzinfo=UTC), datetime(2026, 11, 10, 21, 0, tzinfo=UTC)
        bars = client.bars(["NVDA", "CSCO", "HD"], "1Hour", start, end)
        assert [b.close for b in bars["NVDA"]] == [168.6, 169.4]
        assert bars["CSCO"][0].at == datetime(2026, 11, 10, 15, tzinfo=UTC) and bars["HD"] == []
        assert session.calls[0]["params"]["start"] == "2026-11-10T14:30:00Z"
        assert "page_token" not in session.calls[0]["params"] and session.calls[1]["params"]["page_token"] == "p2"

    def test_headlines_are_read_newest_first_from_a_start_time(self):
        payload = {"news": [
            {"id": 2, "created_at": "2026-11-11T07:46:44Z", "headline": "B", "symbols": ["NVDA"]},
            {"id": 1, "created_at": "2026-11-11T06:56:59Z", "headline": "A", "symbols": ["NVDA", "SPY"]},
            {"id": 0, "headline": "undated"},
        ]}
        client, session = _client(_Response(payload=payload))
        news = client.news(["NVDA", "CSCO"], start=datetime(2026, 11, 11, 6, tzinfo=UTC), limit=200)
        assert [n["headline"] for n in news] == ["B", "A"]
        assert news[0]["at"] == datetime(2026, 11, 11, 7, 46, 44, tzinfo=UTC)
        assert session.calls[0]["params"] == {"symbols": "NVDA,CSCO", "limit": 50, "sort": "desc",
                                               "start": "2026-11-11T06:00:00Z"}
