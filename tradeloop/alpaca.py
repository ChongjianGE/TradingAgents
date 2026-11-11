"""A thin client for Alpaca's Trading and Market Data APIs. Paper accounts only.

The trading host is checked when the client is built: this code has no path to
a live account. Reads are retried when the service is busy; an order is sent
once, and its client order id is what makes sending it again safe.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

from tradeloop import timeutil

PAPER_HOST = "paper-api.alpaca.markets"
PAPER_URL = f"https://{PAPER_HOST}"
DATA_URL = "https://data.alpaca.markets"

# The key pair under each name it is commonly stored as; the first complete pair wins.
_KEY_NAMES = (
    ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
    ("ALPACA_PAPER_TRADING_API_KEY", "ALPACA_PAPER_TRADING_SECRET_KEY"),
    ("ALPCA_PAPER_TRADING_API_KEY", "ALPCA_PAPER_TRADING_SECRET_KEY"),
)
_BUSY = (429, 500, 502, 503, 504)


class AlpacaError(RuntimeError):
    """A request Alpaca refused, or that never reached it. ``status`` is 0 for the latter."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def load_credentials(env_file: str | Path | None = None, environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    """The paper account's key id and secret, from the environment, else from ``env_file``."""
    sources: list[Mapping[str, str | None]] = [os.environ if environ is None else environ]
    if env_file is not None:
        from dotenv import dotenv_values

        sources.append(dotenv_values(Path(env_file).expanduser()))
    for source in sources:
        for key_name, secret_name in _KEY_NAMES:
            key, secret = (source.get(key_name) or "").strip(), (source.get(secret_name) or "").strip()
            if key and secret:
                return key, secret
    raise AlpacaError(0, "no Alpaca paper keys found: set APCA_API_KEY_ID and APCA_API_SECRET_KEY, "
                         "or point alpaca_env_file at a file that holds them")


@dataclass(frozen=True)
class Quote:
    """The last trade of a symbol and the close before it."""

    ticker: str
    price: float
    at: datetime
    prev_close: float | None


@dataclass(frozen=True)
class Bar:
    at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class AlpacaClient:
    def __init__(self, key: str, secret: str, *, trading_url: str = PAPER_URL, data_url: str = DATA_URL,
                 feed: str = "iex", session: requests.Session | None = None, timeout: float = 15.0,
                 retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        trading_url = trading_url.rstrip("/").removesuffix("/v2")
        if urlparse(trading_url).netloc != PAPER_HOST:
            raise ValueError(f"refusing {trading_url}: this client only talks to the paper account at {PAPER_HOST}")
        self.trading_url, self.data_url, self.feed = trading_url, data_url.rstrip("/"), feed
        self.session = session or requests.Session()
        self.timeout, self.retries, self._sleep = timeout, max(1, retries), sleep
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "Accept": "application/json"}

    # --- transport
    def _request(self, method: str, base: str, path: str, *, params: dict | None = None,
                 body: dict | None = None, missing_ok: bool = False):
        attempts = self.retries if method == "GET" else 1   # an order is never sent twice
        for attempt in range(attempts):
            last = attempt + 1 == attempts
            try:
                response = self.session.request(method, base + path, params=params, json=body,
                                                headers=self._headers, timeout=self.timeout)
            except requests.RequestException as exc:
                if last:
                    # The exception's text can carry the URL; its type is enough.
                    raise AlpacaError(0, f"{method} {path} did not reach Alpaca ({type(exc).__name__})") from None
            else:
                if response.status_code in _BUSY and not last:
                    pass
                elif response.status_code == 404 and missing_ok:
                    return None
                elif response.status_code >= 400:
                    raise AlpacaError(response.status_code, f"{method} {path}: {_message(response)}")
                else:
                    return response.json() if response.content else None
            self._sleep(0.5 * 2 ** attempt)
        raise AssertionError("unreachable")   # the loop returns or raises on its last attempt

    def _trading(self, method: str, path: str, **kwargs):
        return self._request(method, self.trading_url, path, **kwargs)

    def _data(self, path: str, **params):
        return self._request("GET", self.data_url, path, params={k: v for k, v in params.items() if v is not None})

    # --- the account
    def clock(self) -> dict:
        return self._trading("GET", "/v2/clock")

    def calendar(self, start: str, end: str) -> list[dict]:
        """Trading days from ``start`` to ``end`` (YYYY-MM-DD): date, open and close, in exchange time."""
        return self._trading("GET", "/v2/calendar", params={"start": start, "end": end})

    def account(self) -> dict:
        return self._trading("GET", "/v2/account")

    def positions(self) -> list[dict]:
        return self._trading("GET", "/v2/positions")

    def open_orders(self) -> list[dict]:
        return self._trading("GET", "/v2/orders", params={"status": "open", "limit": 500})

    def order_by_client_id(self, client_order_id: str) -> dict | None:
        """The order sent under ``client_order_id``, or None when there is none."""
        return self._trading("GET", "/v2/orders:by_client_order_id",
                             params={"client_order_id": client_order_id}, missing_ok=True)

    def submit_order(self, **order) -> dict:
        return self._trading("POST", "/v2/orders", body=order)

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order. One that is already gone or can no longer be cancelled is left as it is."""
        try:
            self._trading("DELETE", f"/v2/orders/{order_id}", missing_ok=True)
        except AlpacaError as exc:
            if exc.status != 422:    # 422: filled or cancelled in the meantime
                raise

    # --- market data
    def quotes(self, symbols: Iterable[str]) -> dict[str, Quote]:
        """Last trade and previous close for every symbol, in one request."""
        symbols = list(symbols)
        if not symbols:
            return {}
        payload = self._data("/v2/stocks/snapshots", symbols=",".join(symbols), feed=self.feed) or {}
        quotes = {}
        for symbol in symbols:
            trade = (payload.get(symbol) or {}).get("latestTrade") or {}
            if trade.get("p"):
                previous = ((payload[symbol].get("prevDailyBar") or {}).get("c")) or None
                quotes[symbol] = Quote(symbol, float(trade["p"]), timeutil.parse(trade["t"]),
                                       float(previous) if previous else None)
        return quotes

    def bars(self, symbols: Iterable[str], timeframe: str, start: datetime, end: datetime,
             feed: str | None = None) -> dict[str, list[Bar]]:
        """Bars per symbol from ``start`` to ``end``, oldest first, following every page.

        ``feed`` names another feed than the client's own for this one request:
        "sip" is every exchange, which a free plan may read once it is 15 minutes old.
        """
        symbols = list(symbols)
        out: dict[str, list[Bar]] = {symbol: [] for symbol in symbols}
        if not symbols:
            return out
        token = None
        while True:
            payload = self._data("/v2/stocks/bars", symbols=",".join(symbols), timeframe=timeframe,
                                 start=timeutil.iso(start), end=timeutil.iso(end), feed=feed or self.feed,
                                 limit=10000, adjustment="raw", sort="asc", page_token=token) or {}
            for symbol, rows in (payload.get("bars") or {}).items():
                out.setdefault(symbol, []).extend(
                    Bar(timeutil.parse(r["t"]), float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]),
                        float(r["v"])) for r in rows)
            token = payload.get("next_page_token")
            if not token:
                return out

    def news(self, symbols: Iterable[str], start: datetime | None = None, limit: int = 50) -> list[dict]:
        """The newest headlines naming any of ``symbols``, newest first, at most ``limit``."""
        symbols = list(symbols)
        if not symbols:
            return []
        payload = self._data("/v1beta1/news", symbols=",".join(symbols), limit=min(limit, 50), sort="desc",
                             start=timeutil.iso(start) if start else None) or {}
        return [{"id": n.get("id"), "at": timeutil.parse(n["created_at"]), "headline": n.get("headline", ""),
                 "symbols": n.get("symbols") or []} for n in payload.get("news") or [] if n.get("created_at")]


def _message(response: requests.Response) -> str:
    """Alpaca's own reason for a refusal, else the status alone."""
    try:
        reason = response.json().get("message")
    except (ValueError, AttributeError):
        reason = None
    return f"{response.status_code} {reason}" if reason else f"HTTP {response.status_code}"
