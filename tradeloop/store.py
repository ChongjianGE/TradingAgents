"""The loop's record: every tick, what it saw, what was decided, what was ordered and filled.

One SQLite file holds the structured rows; the long texts of an analysis (each
agent's output, each tool's data) live in a folder per run, named by the row.
Everything the dashboard shows is read from here.

A single connection serves every thread behind one lock: analyses run in
parallel, but their writes are short and SQLite takes one writer at a time anyway.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ticks (
    tick_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,                 -- premarket | core | postclose | offhours
    scheduled_at TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,               -- running | ok | error
    execution TEXT NOT NULL,            -- simulate | alpaca_paper
    note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS snapshots (  -- the book at each tick
    tick_id TEXT PRIMARY KEY REFERENCES ticks,
    at TEXT NOT NULL, equity REAL NOT NULL, cash REAL NOT NULL,
    positions TEXT NOT NULL             -- JSON: [{ticker, qty, avg_price, price}]
);
CREATE INDEX IF NOT EXISTS snapshots_at ON snapshots (at);
CREATE TABLE IF NOT EXISTS observations (  -- per ticker per tick: what was seen, and whether to analyse
    tick_id TEXT NOT NULL REFERENCES ticks, ticker TEXT NOT NULL,
    price REAL, price_at TEXT, prev_close REAL,
    position_qty REAL NOT NULL DEFAULT 0,
    new_headlines INTEGER NOT NULL DEFAULT 0,
    gate_action TEXT NOT NULL,          -- analyze | reuse | skip
    gate_reasons TEXT NOT NULL,         -- JSON list
    PRIMARY KEY (tick_id, ticker)
);
CREATE TABLE IF NOT EXISTS analyses (
    run_id TEXT PRIMARY KEY,
    tick_id TEXT NOT NULL REFERENCES ticks, ticker TEXT NOT NULL,
    trade_date TEXT NOT NULL, kind TEXT NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT, elapsed_seconds REAL,
    status TEXT NOT NULL,               -- running | ok | error
    ref_price REAL,                     -- the price when the analysis started
    rating TEXT, target_weight REAL, stop_loss REAL, price_target REAL, time_horizon TEXT,
    decision TEXT,                      -- JSON: the structured decision
    summary TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL, llm_calls INTEGER NOT NULL DEFAULT 0, tool_calls INTEGER NOT NULL DEFAULT 0,
    recorded_to_memory INTEGER NOT NULL DEFAULT 0,
    artifacts_dir TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS analyses_ticker_time ON analyses (ticker, started_at);
CREATE INDEX IF NOT EXISTS analyses_tick ON analyses (tick_id);
CREATE TABLE IF NOT EXISTS agent_steps (  -- each model call and tool call of an analysis
    run_id TEXT NOT NULL REFERENCES analyses, seq INTEGER NOT NULL,
    agent TEXT NOT NULL, kind TEXT NOT NULL,   -- llm | tool
    name TEXT NOT NULL,                        -- the model, or the tool
    started_at TEXT NOT NULL, elapsed_seconds REAL,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    output_chars INTEGER NOT NULL DEFAULT 0, error TEXT,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS targets (    -- what each tick wanted to hold, and what the rules allowed
    tick_id TEXT NOT NULL REFERENCES ticks, ticker TEXT NOT NULL,
    run_id TEXT,                        -- the decision behind it, when there is one
    rating TEXT, price REAL,
    current_qty REAL NOT NULL, current_weight REAL NOT NULL,
    raw_weight REAL,                    -- what the decision asks for; NULL keeps the position
    target_weight REAL NOT NULL, target_qty REAL NOT NULL,
    verdict TEXT NOT NULL,
    checks TEXT NOT NULL,               -- JSON: [{name, passed, detail}]
    PRIMARY KEY (tick_id, ticker)
);
CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    tick_id TEXT NOT NULL REFERENCES ticks, ticker TEXT NOT NULL,
    side TEXT NOT NULL, qty REAL NOT NULL, type TEXT NOT NULL,
    limit_price REAL, stop_price REAL, time_in_force TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL, broker TEXT NOT NULL, broker_order_id TEXT,
    submitted_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    filled_qty REAL NOT NULL DEFAULT 0, filled_avg_price REAL,
    est_cost_usd REAL NOT NULL DEFAULT 0, error TEXT
);
CREATE INDEX IF NOT EXISTS orders_ticker_time ON orders (ticker, submitted_at);
CREATE INDEX IF NOT EXISTS orders_tick ON orders (tick_id);
CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    broker_order_id TEXT, client_order_id TEXT, ticker TEXT NOT NULL,
    side TEXT NOT NULL, qty REAL NOT NULL, price REAL NOT NULL, at TEXT NOT NULL,
    fee_usd REAL NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS fills_time ON fills (at);
CREATE TABLE IF NOT EXISTS price_bars (  -- one-minute and daily prices, kept for the charts
    ticker TEXT NOT NULL, timeframe TEXT NOT NULL, at TEXT NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (ticker, timeframe, at)
);
CREATE INDEX IF NOT EXISTS price_bars_time ON price_bars (timeframe, at);
CREATE TABLE IF NOT EXISTS outcomes (   -- how the price moved after each decision
    run_id TEXT NOT NULL REFERENCES analyses, horizon TEXT NOT NULL,
    due_at TEXT NOT NULL, price REAL, ret REAL, marked_at TEXT,
    PRIMARY KEY (run_id, horizon)
);
CREATE INDEX IF NOT EXISTS outcomes_due ON outcomes (marked_at, due_at);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL, tick_id TEXT, ticker TEXT,
    level TEXT NOT NULL, source TEXT NOT NULL, message TEXT NOT NULL, data TEXT
);
CREATE INDEX IF NOT EXISTS events_tick ON events (tick_id);
"""

# Columns holding JSON, decoded on the way out.
_JSON_COLUMNS = {"positions", "gate_reasons", "decision", "checks", "data"}
# Orders a broker may still fill.
OPEN_ORDER_STATUSES = ("new", "accepted", "pending_new", "partially_filled", "held")


def _encode(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, bool):
        return int(value)
    return value


def _decode(row: sqlite3.Row) -> dict:
    out = dict(row)
    for key in _JSON_COLUMNS & out.keys():
        if isinstance(out[key], str):
            out[key] = json.loads(out[key])
    return out


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            # WAL lets the dashboard read while the loop writes.
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(_SCHEMA)
        found = self.get("schema_version")
        if found is None:
            self.put("schema_version", SCHEMA_VERSION)
        elif found != SCHEMA_VERSION:
            raise RuntimeError(f"{self.path} has schema version {found}; this code reads {SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- plumbing
    def execute(self, sql: str, params: Iterable = ()) -> int:
        """Run one statement; the number of rows it changed."""
        with self._lock:
            return self._db.execute(sql, tuple(params)).rowcount

    def query(self, sql: str, params: Iterable = ()) -> list[dict]:
        with self._lock:
            return [_decode(row) for row in self._db.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def _insert(self, table: str, rows: list[dict], conflict: str = "") -> int:
        """Insert ``rows`` (all with the same keys) in one transaction."""
        if not rows:
            return 0
        columns = list(rows[0])
        sql = (f"INSERT {conflict} INTO {table} ({', '.join(columns)}) "
               f"VALUES ({', '.join('?' for _ in columns)})")
        values = [tuple(_encode(row[c]) for c in columns) for row in rows]
        with self._lock:
            self._db.execute("BEGIN")
            try:
                changed = self._db.executemany(sql, values).rowcount
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return changed

    def _update(self, table: str, where: dict, fields: dict) -> int:
        if not fields:
            return 0
        sql = (f"UPDATE {table} SET {', '.join(f'{k} = ?' for k in fields)} "
               f"WHERE {' AND '.join(f'{k} = ?' for k in where)}")
        return self.execute(sql, [_encode(v) for v in fields.values()] + list(where.values()))

    # --- small values that outlive a tick (the simulated book, cursors)
    def get(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value FROM kv WHERE key = ?", (key,))
        return default if row is None else json.loads(row["value"])

    def put(self, key: str, value: Any) -> None:
        self.execute("INSERT INTO kv (key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     (key, json.dumps(value, ensure_ascii=False, default=str)))

    # --- ticks
    def begin_tick(self, tick_id: str, kind: str, scheduled_at: str, started_at: str, execution: str) -> bool:
        """Open a tick; False when it already ran to the end, so a restart cannot run it twice.

        A tick that was left running (a crash) is taken over and run again.
        """
        existing = self.tick(tick_id)
        if existing and existing["status"] != "running":
            return False
        self.execute(
            "INSERT INTO ticks (tick_id, kind, scheduled_at, started_at, status, execution) "
            "VALUES (?, ?, ?, ?, 'running', ?) "
            "ON CONFLICT(tick_id) DO UPDATE SET started_at = excluded.started_at, execution = excluded.execution",
            (tick_id, kind, scheduled_at, started_at, execution))
        return True

    def finish_tick(self, tick_id: str, status: str, finished_at: str, note: str = "") -> None:
        self._update("ticks", {"tick_id": tick_id}, {"status": status, "finished_at": finished_at, "note": note})

    def tick(self, tick_id: str) -> dict | None:
        return self.one("SELECT * FROM ticks WHERE tick_id = ?", (tick_id,))

    # --- the book
    def add_snapshot(self, tick_id: str, at: str, equity: float, cash: float, positions: list[dict]) -> None:
        self._insert("snapshots", [{"tick_id": tick_id, "at": at, "equity": equity, "cash": cash,
                                    "positions": positions}], conflict="OR REPLACE")

    def equity_at_or_before(self, at: str) -> float | None:
        """The last equity recorded at or before ``at``: the day's starting point."""
        row = self.one("SELECT equity FROM snapshots WHERE at <= ? ORDER BY at DESC LIMIT 1", (at,))
        return None if row is None else row["equity"]

    def add_observations(self, rows: list[dict]) -> None:
        self._insert("observations", rows, conflict="OR REPLACE")

    # --- analyses
    def begin_analysis(self, **fields) -> None:
        self._insert("analyses", [{**fields, "status": "running"}])

    def finish_analysis(self, run_id: str, **fields) -> None:
        self._update("analyses", {"run_id": run_id}, fields)

    def add_agent_steps(self, run_id: str, steps: list[dict]) -> None:
        self._insert("agent_steps", [{"run_id": run_id, **step} for step in steps], conflict="OR REPLACE")

    def latest_decision(self, ticker: str, since: str | None = None) -> dict | None:
        """The newest analysis of ``ticker`` that reached a rating, optionally no older than ``since``."""
        return self.one(
            "SELECT * FROM analyses WHERE ticker = ? AND status = 'ok' AND started_at >= ? "
            "ORDER BY started_at DESC LIMIT 1", (ticker, since or ""))

    def recent_decisions(self, ticker: str, limit: int) -> list[dict]:
        """The last ``limit`` rated analyses of ``ticker``, newest first."""
        return self.query(
            "SELECT * FROM analyses WHERE ticker = ? AND status = 'ok' "
            "ORDER BY started_at DESC LIMIT ?", (ticker, limit))

    def analyses_between(self, ticker: str, start: str, end: str, kind: str | None = None) -> list[dict]:
        """Every analysis of ``ticker`` started in ``[start, end)``, oldest first, whatever its outcome."""
        sql = "SELECT * FROM analyses WHERE ticker = ? AND started_at >= ? AND started_at < ?"
        params: list = [ticker, start, end]
        if kind:
            sql, params = sql + " AND kind = ?", [*params, kind]
        return self.query(sql + " ORDER BY started_at", params)

    def llm_cost_since(self, since: str) -> float:
        row = self.one("SELECT COALESCE(SUM(cost_usd), 0) AS cost FROM analyses WHERE started_at >= ?", (since,))
        return float(row["cost"])

    # --- targets, orders, fills
    def add_targets(self, rows: list[dict]) -> None:
        self._insert("targets", rows, conflict="OR REPLACE")

    def add_order(self, **fields) -> None:
        self._insert("orders", [fields])

    def update_order(self, client_order_id: str, **fields) -> None:
        self._update("orders", {"client_order_id": client_order_id}, fields)

    def order(self, client_order_id: str) -> dict | None:
        return self.one("SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,))

    def open_orders(self, broker: str) -> list[dict]:
        marks = ", ".join("?" for _ in OPEN_ORDER_STATUSES)
        return self.query(f"SELECT * FROM orders WHERE broker = ? AND status IN ({marks}) ORDER BY submitted_at",
                          (broker, *OPEN_ORDER_STATUSES))

    def add_fills(self, fills: list[dict]) -> int:
        """Record fills, skipping any already known; the number that were new."""
        return self._insert("fills", fills, conflict="OR IGNORE")

    # --- prices for the charts
    def add_bars(self, rows: list[dict]) -> None:
        """Keep bars; one already held is replaced, since a bar fetched early may not have been final."""
        self._insert("price_bars", rows, conflict="OR REPLACE")

    def latest_bar_at(self, timeframe: str) -> str | None:
        """The time of the newest bar of ``timeframe`` held for any ticker."""
        return self.one("SELECT MAX(at) AS at FROM price_bars WHERE timeframe = ?", (timeframe,))["at"]

    def bars(self, timeframe: str, since: str = "") -> list[dict]:
        """Every bar of ``timeframe`` from ``since`` on, by ticker and then by time."""
        return self.query("SELECT ticker, at, open, high, low, close, volume FROM price_bars "
                          "WHERE timeframe = ? AND at >= ? ORDER BY ticker, at", (timeframe, since))

    # --- outcomes
    def add_outcomes(self, rows: list[dict]) -> None:
        self._insert("outcomes", rows, conflict="OR IGNORE")

    def due_outcomes(self, now: str) -> list[dict]:
        """Outcomes whose time has come and that have no price yet, with what is needed to mark them."""
        return self.query(
            "SELECT o.run_id, o.horizon, o.due_at, a.ticker, a.ref_price, a.started_at "
            "FROM outcomes o JOIN analyses a USING (run_id) "
            "WHERE o.marked_at IS NULL AND o.due_at <= ? AND a.ref_price IS NOT NULL "
            "ORDER BY o.due_at", (now,))

    def mark_outcome(self, run_id: str, horizon: str, price: float, ret: float, marked_at: str) -> None:
        self._update("outcomes", {"run_id": run_id, "horizon": horizon},
                     {"price": price, "ret": ret, "marked_at": marked_at})

    # --- the loop's own log
    def log(self, level: str, source: str, message: str, *, at: str, tick_id: str | None = None,
            ticker: str | None = None, data: Any = None) -> None:
        self._insert("events", [{"at": at, "tick_id": tick_id, "ticker": ticker, "level": level,
                                 "source": source, "message": message,
                                 "data": None if data is None else json.dumps(data, ensure_ascii=False, default=str)}])
