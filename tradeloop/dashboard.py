"""The loop's record as pages to read: an overview across days, and one page per day
that shows every tick, what each decision was based on, and what was traded.

The pages are static files built from the store and the per-run folders. They
load nothing and run nothing: charts are drawn as SVG when the page is built,
and the long texts sit in collapsed sections. Text written by a model is
rendered as Markdown with raw HTML and images off.
"""

from __future__ import annotations

import contextlib
import json
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, time, timedelta
from html import escape
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from markdown_it import MarkdownIt

from tradeloop import timeutil
from tradeloop.config import LoopConfig
from tradeloop.store import Store

_TONE = {"Buy": "up", "Overweight": "up", "Hold": "level", "Underweight": "down", "Sell": "down"}
_DIRECTION = {"Buy": 1, "Overweight": 1, "Underweight": -1, "Sell": -1}
_HORIZONS = ("60m", "close", "1d", "5d")
_SESSION = (time(9, 30), time(16, 0))   # the regular session in exchange time: what the price charts show
_CHART_DAYS = 10                        # sessions on an overview chart
_CHART_POINTS = 800                     # beyond this a chart's bars are thinned
_LABELLED_TRADES = 10                   # beyond this the marks keep their hover text but lose their labels
# The agents' outputs in the order they were written: (label, path into the saved state).
_AGENT_TEXTS = (
    ("Market Analyst", ("market_report",)), ("Sentiment Analyst", ("sentiment_report",)),
    ("News Analyst", ("news_report",)), ("Fundamentals Analyst", ("fundamentals_report",)),
    ("Bull Researcher", ("investment_debate_state", "bull_history")),
    ("Bear Researcher", ("investment_debate_state", "bear_history")),
    ("Research Manager", ("investment_plan",)), ("Trader", ("trader_investment_plan",)),
    ("Aggressive Analyst", ("risk_debate_state", "aggressive_history")),
    ("Conservative Analyst", ("risk_debate_state", "conservative_history")),
    ("Neutral Analyst", ("risk_debate_state", "neutral_history")),
    ("Portfolio Manager", ("final_trade_decision",)),
)

_CSS = """
:root { color-scheme: light dark;
  --bg:#f2f4f3; --panel:#e4e9e6; --surface:#fbfcfb; --ink:#16201c; --muted:#56645e; --rule:#c9d2ce;
  --up:#0f7a5c; --down:#a33a2b; --level:#56645e; --warn:#9a6400; --accent:#1f6f9c;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif; }
@media (prefers-color-scheme: dark) { :root {
  --bg:#101513; --panel:#19201d; --surface:#212a26; --ink:#e5ebe8; --muted:#9aa8a2; --rule:#34413b;
  --up:#4fc3a1; --down:#f0917f; --level:#9aa8a2; --warn:#e0b25a; --accent:#74b9e0; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14.5px/1.55 var(--sans);
  padding:28px max(16px, env(safe-area-inset-right)) 64px max(16px, env(safe-area-inset-left)); }
.page { max-width:1180px; margin:0 auto; display:flex; flex-direction:column; gap:26px; }
h1 { font-size:26px; line-height:1.2; margin:0; } h2 { font-size:18px; margin:0; } h3 { font-size:15px; margin:0; }
a { color:var(--accent); } .muted { color:var(--muted); } .mono { font-family:var(--mono); font-size:.92em; }
.eyebrow { font:12px var(--mono); color:var(--muted); letter-spacing:.06em; }
section { display:flex; flex-direction:column; gap:12px; min-width:0; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; }
.tile { background:var(--surface); border-radius:10px; padding:10px 14px; }
.tile b { display:block; font-size:19px; font-variant-numeric:tabular-nums; } .tile span { color:var(--muted); font-size:12.5px; }
.scroll { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:left; vertical-align:top; padding:6px 10px; border-bottom:1px solid var(--rule); }
th { font-size:12px; color:var(--muted); font-weight:600; white-space:nowrap; } td.num, th.num { text-align:right; }
.up { color:var(--up); } .down { color:var(--down); } .level { color:var(--level); } .review { color:var(--muted); font-style:italic; }
.pill { display:inline-block; padding:0 7px; border-radius:9px; border:1px solid currentColor; font-size:12px; white-space:nowrap; }
.bad { color:var(--down); } .warnc { color:var(--warn); }
details { background:var(--surface); border-radius:10px; padding:0 14px; }
details > summary { cursor:pointer; padding:10px 0; list-style-position:outside; }
details[open] { padding-bottom:12px; } details details { background:var(--panel); margin-top:8px; }
.md { max-width:80ch; overflow-wrap:anywhere; } .md table { font-size:13px; } .md h4, .md h5, .md h6 { margin:14px 0 4px; font-size:14px; }
pre { margin:0; padding:10px 12px; background:var(--panel); border-radius:8px; overflow-x:auto; font:12px/1.5 var(--mono);
  white-space:pre-wrap; overflow-wrap:anywhere; max-height:420px; }
svg { display:block; width:100%; height:auto; } .chart { background:var(--surface); border-radius:10px; padding:8px; }
.strip { display:flex; flex-wrap:wrap; gap:3px; } .strip i { width:13px; height:13px; border-radius:3px; background:currentColor; display:block; }
ul.checks { margin:0; padding-left:18px; } nav.tickers { display:flex; flex-wrap:wrap; gap:8px 16px; }
a.out { font-size:.8em; font-weight:400; margin-left:.5em; white-space:nowrap; }
.key { margin:0; font-size:12.5px; color:var(--muted); } .key b { font-weight:400; margin-right:12px; white-space:nowrap; }
"""


def _md() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": False}).enable("table").disable("image")
    md.add_render_rule("table_open", lambda *a: '<div class="scroll"><table>\n')
    md.add_render_rule("table_close", lambda *a: "</table></div>\n")
    return md


def _render_md(md: MarkdownIt, text: str) -> str:
    """An agent's text as HTML, its headings pushed below the page's own."""
    tokens = md.parse(text or "")
    for token in tokens:
        if token.type in ("heading_open", "heading_close"):
            token.tag = f"h{min(int(token.tag[1]) + 3, 6)}"
    return f'<div class="md">{md.renderer.render(tokens, md.options, {})}</div>'


def _money(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"${value:,.{digits}f}"


def _pct(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value * 100:+.{digits}f}%"


def _signed(value: float | None) -> str:
    """A change with its direction as a class, so gains and losses read at a glance."""
    if value is None:
        return "–"
    tone = "up" if value > 0 else "down" if value < 0 else "level"
    return f'<span class="{tone}">{"+" if value > 0 else "−" if value < 0 else ""}${abs(value):,.2f}</span>'


def _rating(rating: str | None) -> str:
    if not rating:
        return '<span class="muted">–</span>'
    return f'<span class="pill {_TONE.get(rating, "review")}">{escape(rating)}</span>'


def _line_chart(points: list[tuple[datetime, float]], marks: list[tuple[datetime, str, str]] = (),
                height: int = 150, label: str = "") -> str:
    """A line over time with labelled marks on it, as an inline SVG. Empty with fewer than two points."""
    if len(points) < 2:
        return '<p class="muted">Not enough points for a chart yet.</p>'
    width, left, right, top, bottom = 900, 84, 14, 12, 22      # room on the left for a six-figure equity
    t0, t1 = points[0][0].timestamp(), points[-1][0].timestamp()
    low, high = min(v for _, v in points), max(v for _, v in points)
    pad = (high - low) * 0.08 or max(abs(high) * 0.002, 0.01)
    low, high = low - pad, high + pad

    def x(moment: datetime) -> float:
        return left + (moment.timestamp() - t0) / ((t1 - t0) or 1) * (width - left - right)

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * (height - top - bottom)

    path = " ".join(f"{'M' if i == 0 else 'L'}{x(t):.1f},{y(v):.1f}" for i, (t, v) in enumerate(points))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}">']
    for value in (low + pad, (low + high) / 2, high - pad):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(value):.1f}" y2="{y(value):.1f}" '
                     'stroke="currentColor" stroke-opacity=".12"/>'
                     f'<text x="{left - 6}" y="{y(value) + 4:.1f}" text-anchor="end" font-size="11" '
                     f'fill="currentColor" fill-opacity=".65">{value:,.2f}</text>')
    parts.append(f'<path d="{path}" fill="none" stroke="currentColor" stroke-width="1.6"/>')
    for moment, text, tone in marks:
        if not points[0][0] <= moment <= points[-1][0]:
            continue
        value = min(points, key=lambda p: abs((p[0] - moment).total_seconds()))[1]
        parts.append(f'<g class="{tone}"><circle cx="{x(moment):.1f}" cy="{y(value):.1f}" r="4.5" fill="currentColor"/>'
                     f'<title>{escape(text)}</title></g>')
    for moment in (points[0][0], points[-1][0]):
        anchor = "start" if moment is points[0][0] else "end"
        parts.append(f'<text x="{x(moment):.1f}" y="{height - 6}" text-anchor="{anchor}" font-size="11" '
                     f'fill="currentColor" fill-opacity=".65">{moment:%m-%d %H:%M}</text>')
    return '<div class="chart">' + "".join(parts) + "</svg></div>"


def _price_chart(bars: list[tuple[datetime, float]], trades: list[tuple[datetime, str, float, float]] = (),
                 analyses: list[tuple[datetime, str, str]] = (), height: int = 230, label: str = "") -> str:
    """A ticker's price through the regular session, with each buy and sell marked where it filled.

    ``bars`` are (exchange time, close), oldest first. Time runs by bar rather
    than by the clock, so nights and weekends take no room; each day is its own
    line, and what it opened at shows as the step between two lines. ``trades``
    are (time, side, shares, price); fills of one side inside the same bar are
    one mark, a triangle whose tip is at the price paid. ``analyses`` are (time,
    text, tone), drawn as ticks along the bottom. Empty with fewer than two bars.
    """
    if len(bars) < 2:
        return '<p class="muted">Not enough prices for a chart yet.</p>'
    if len(bars) > _CHART_POINTS:                       # thin a long record, keeping each day's last bar
        step = -(-len(bars) // _CHART_POINTS)
        bars = [b for i, b in enumerate(bars)
                if i % step == 0 or i + 1 == len(bars) or bars[i + 1][0].date() != b[0].date()]
    width, left, right, top, bottom = 900, 64, 16, 24, 34
    floor = height - bottom
    times = [t for t, _ in bars]
    slots, slot, day_starts = [], 0, [0]
    for i, moment in enumerate(times):
        if i and moment.date() != times[i - 1].date():
            slot += 2                                   # a little air between two days
            day_starts.append(i)
        slots.append(slot)
        slot += 1
    span = slots[-1] or 1

    def x(index: int) -> float:
        return left + slots[index] / span * (width - left - right)

    def bar_of(moment: datetime) -> int | None:
        """The bar a moment falls in, or None when no bar is near it."""
        index = max(bisect_right(times, moment) - 1, 0)
        return index if abs(moment - times[index]) <= timedelta(minutes=30) else None

    # Fills of one side within one bar are one trade to the eye: an order often fills in parts.
    merged: dict[tuple[int, str], list[float]] = {}
    for moment, side, qty, price in trades:
        index = bar_of(moment)
        if index is not None:
            total = merged.setdefault((index, side), [0.0, 0.0])
            total[0] += qty
            total[1] += qty * price
    marks = [(index, side, qty, paid / qty) for (index, side), (qty, paid) in sorted(merged.items()) if qty]

    values = [v for _, v in bars] + [price for *_, price in marks]
    low, high = min(values), max(values)
    pad = (high - low) * 0.12 or max(abs(high) * 0.002, 0.01)
    low, high = low - pad, high + pad

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * (floor - top)

    faint = 'fill="currentColor" fill-opacity=".65" font-size="11"'
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}">']
    for value in (low + pad, (low + high) / 2, high - pad):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(value):.1f}" y2="{y(value):.1f}" '
                     f'stroke="currentColor" stroke-opacity=".12"/>'
                     f'<text x="{left - 6}" y="{y(value) + 4:.1f}" text-anchor="end" {faint}>{value:,.2f}</text>')

    several = len(day_starts) > 1
    for n, first in enumerate(day_starts):
        last = (day_starts[n + 1] if n + 1 < len(day_starts) else len(bars)) - 1
        if several:
            parts.append(f'<text x="{(x(first) + x(last)) / 2:.1f}" y="{height - 6}" text-anchor="middle" {faint}>'
                         f"{times[first]:%m-%d}</text>")
        else:
            ends = {first: "start", last: "end"}
            hours = {i: "middle" for i in range(first + 1, last) if times[i].minute == 0 and times[i].hour % 2 == 0
                     and min(i - first, last - i) >= 6}
            for i, anchor in {**hours, **ends}.items():
                parts.append(f'<text x="{x(i):.1f}" y="{height - 6}" text-anchor="{anchor}" {faint}>{times[i]:%H:%M}</text>')
        path = " ".join(f"{'M' if i == first else 'L'}{x(i):.1f},{y(bars[i][1]):.1f}" for i in range(first, last + 1))
        parts.append(f'<path d="{path}" fill="none" stroke="currentColor" stroke-width="2" '
                     'stroke-linejoin="round" stroke-linecap="round"/>')

    for moment, text, tone in analyses:
        index = bar_of(moment)
        if index is not None:
            parts.append(f'<g class="{tone}"><rect x="{x(index) - 5:.1f}" y="{floor}" width="10" height="12" fill="transparent"/>'
                         f'<line x1="{x(index):.1f}" x2="{x(index):.1f}" y1="{floor + 2}" y2="{floor + 10}" '
                         f'stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/><title>{escape(text)}</title></g>')

    labelled = len(marks) <= _LABELLED_TRADES
    for index, side, qty, price in marks:
        cx, tip, buy = x(index), y(price), side == "buy"
        away = 1 if buy else -1                         # a buy sits under the price it paid, a sell above
        base = tip + away * 11
        shape = f"{cx:.1f},{tip:.1f} {cx - 6.5:.1f},{base:.1f} {cx + 6.5:.1f},{base:.1f}"
        said = (f"{times[index]:%m-%d %H:%M} {'bought' if buy else 'sold'} {qty:g} at {_money(price)}")
        parts.append(f'<g class="{"up" if buy else "down"}"><circle cx="{cx:.1f}" cy="{tip + away * 6:.1f}" r="12" fill="transparent"/>'
                     f'<polygon points="{shape}" fill="currentColor" stroke="var(--surface)" stroke-width="2" '
                     f'stroke-linejoin="round" paint-order="stroke"/><title>{escape(said)}</title></g>')
        if labelled:
            anchor = "end" if cx > width - right - 30 else "start" if cx < left + 30 else "middle"
            parts.append(f'<text x="{cx:.1f}" y="{base + (13 if buy else -5):.1f}" text-anchor="{anchor}" font-size="11.5" '
                         f'fill="var(--ink)" stroke="var(--surface)" stroke-width="3" paint-order="stroke">'
                         f'{"+" if buy else "−"}{qty:g}</text>')
    key = []
    if any(side == "buy" for _, side, *_ in marks):
        key.append('<b><span class="up">▲</span> bought</b>')
    if any(side == "sell" for _, side, *_ in marks):
        key.append('<b><span class="down">▼</span> sold</b>')
    if key:
        key.append("<b>the number is the shares; the tip is the price paid</b>")
    if analyses:
        key.append("<b>ticks along the bottom: analyses</b>")
    if key:
        key.append("<b>hover a mark for its detail</b>")
    return ('<div class="chart">' + "".join(parts) + "</svg>"
            + (f'<p class="key">{"".join(key)}</p>' if key else "") + "</div>")


def _page(title: str, eyebrow: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{escape(title)}</title><style>{_CSS}</style></head><body><div class=\"page\">"
            f'<header><div class="eyebrow">{escape(eyebrow)}</div><h1>{escape(title)}</h1></header>'
            f"{body}</div></body></html>")


def _table(headers: list[str], rows: list[list[str]], numeric: tuple[int, ...] = ()) -> str:
    if not rows:
        return '<p class="muted">Nothing here yet.</p>'
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{escape(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{cell}</td>'
                                    for i, cell in enumerate(row)) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _tiles(items: list[tuple[str, str]]) -> str:
    return '<div class="tiles">' + "".join(f'<div class="tile"><b>{value}</b><span>{escape(label)}</span></div>'
                                           for label, value in items) + "</div>"


class _Data:
    """Everything the pages need, read once and grouped by day and by ticker."""

    def __init__(self, cfg: LoopConfig, store: Store):
        self.cfg, self.tz = cfg, ZoneInfo(cfg.timezone)
        q = store.query
        self.ticks = q("SELECT * FROM ticks ORDER BY scheduled_at")
        self.snapshots = q("SELECT * FROM snapshots ORDER BY at")
        self.analyses = q("SELECT * FROM analyses ORDER BY started_at")
        self.observations = q("SELECT * FROM observations")
        self.targets = q("SELECT * FROM targets")
        self.orders = q("SELECT * FROM orders ORDER BY submitted_at")
        self.fills = q("SELECT * FROM fills ORDER BY at")
        self.outcomes = q("SELECT * FROM outcomes")
        self.events = q("SELECT * FROM events ORDER BY id")
        self.bars: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        for bar in q("SELECT ticker, at, close FROM bars ORDER BY at"):
            moment = self.local(bar["at"])
            if _SESSION[0] <= moment.time() < _SESSION[1]:
                self.bars[bar["ticker"]].append((moment, bar["close"]))
        self.tick_day = {t["tick_id"]: self.day_of(t["scheduled_at"]) for t in self.ticks}
        self.days = sorted(set(self.tick_day.values()))

    def local(self, at: str) -> datetime:
        return timeutil.parse(at).astimezone(self.tz)

    def day_of(self, at: str) -> str:
        return self.local(at).date().isoformat()

    def clock(self, at: str | None) -> str:
        return "–" if not at else f"{self.local(at):%H:%M}"


def _day_stats(data: _Data, day: str) -> dict:
    ticks = [t for t in data.ticks if data.tick_day[t["tick_id"]] == day]
    ids = {t["tick_id"] for t in ticks}
    analyses = [a for a in data.analyses if a["tick_id"] in ids]
    fills = [f for f in data.fills if data.day_of(f["at"]) == day]
    snaps = [s for s in data.snapshots if s["tick_id"] in ids]
    earlier = [s for s in data.snapshots if s["at"] < snaps[0]["at"]] if snaps else []
    start = earlier[-1]["equity"] if earlier else (snaps[0]["equity"] if snaps else None)
    end = snaps[-1]["equity"] if snaps else None
    return {
        "day": day, "ticks": ticks, "ids": ids, "analyses": analyses, "fills": fills,
        "orders": [o for o in data.orders if o["tick_id"] in ids],
        "failed": sum(1 for a in analyses if a["status"] == "error"),
        "llm_cost": sum(a["cost_usd"] or 0 for a in analyses),
        "trade_cost": sum(f["cost_usd"] + f["fee_usd"] for f in fills),
        "equity_start": start, "equity_end": end,
        "pnl": None if start is None or end is None else end - start,
    }


# ---------------------------------------------------------------------------
# The overview
# ---------------------------------------------------------------------------


def _outcome_table(data: _Data) -> str:
    """Per rating and horizon: how many decisions, their mean move, and how often the move went the rating's way."""
    rating_of = {a["run_id"]: a["rating"] for a in data.analyses}
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for o in data.outcomes:
        if o["ret"] is not None and rating_of.get(o["run_id"]):
            cells[(rating_of[o["run_id"]], o["horizon"])].append(o["ret"])
    rows = []
    for rating in _TONE:
        if not any((rating, h) in cells for h in _HORIZONS):
            continue
        row = [_rating(rating)]
        for horizon in _HORIZONS:
            moves = cells.get((rating, horizon), [])
            if not moves:
                row.append('<span class="muted">–</span>')
                continue
            text = f"{_pct(sum(moves) / len(moves))} <span class=\"muted\">n={len(moves)}</span>"
            direction = _DIRECTION.get(rating)
            if direction:
                hits = sum(1 for m in moves if m * direction > 0)
                text += f' <span class="muted">· right way {hits}/{len(moves)}</span>'
            row.append(text)
        rows.append(row)
    return _table(["Rating", "After 60 min", "At the next close", "1 trading day on", "5 trading days on"], rows)


def _chart_link(cfg: LoopConfig, ticker: str) -> str:
    """A link out to a price chart of ``ticker``, when the config names a site for one."""
    if not cfg.chart_url:
        return ""
    url = cfg.chart_url.replace("{ticker}", quote(ticker, safe=""))
    return (f'<a class="out" href="{escape(url)}" target="_blank" rel="noopener noreferrer" '
            f'title="Price chart of {escape(ticker)}, on another site">chart ↗</a>')


def _index(data: _Data) -> str:
    cfg, parts = data.cfg, []
    stats = [_day_stats(data, day) for day in data.days]
    first = data.snapshots[0]["equity"] if data.snapshots else None
    last = data.snapshots[-1] if data.snapshots else None
    llm = sum(a["cost_usd"] or 0 for a in data.analyses)
    trade = sum(f["cost_usd"] + f["fee_usd"] for f in data.fills)
    change = None if not last else last["equity"] - first
    parts.append(_tiles([
        ("Equity now", _money(last["equity"]) if last else "–"),
        ("Change since the start", _signed(change)),
        ("Model spend", _money(llm)),
        ("Estimated trading cost", _money(trade)),
        ("After model spend", _signed(None if change is None else change - llm)),
        ("Analyses", f"{len(data.analyses):,}"),
        ("Fills", f"{len(data.fills):,}"),
    ]))
    parts.append(f'<p class="muted">Book: {escape(data.ticks[-1]["execution"]) if data.ticks else "–"}. '
                 "Equity already includes trading costs; model spend is outside the book and shown apart.</p>")

    parts.append("<section><h2>Equity</h2>"
                 + _line_chart([(data.local(s["at"]), s["equity"]) for s in data.snapshots], label="Equity over time")
                 + "</section>")

    if last:
        rows = [[escape(p["ticker"]) + _chart_link(cfg, p["ticker"]), f"{p['qty']:g}", _money(p["avg_price"]),
                 _money(p.get("price")),
                 _signed(None if p.get("price") is None else (p["price"] - p["avg_price"]) * p["qty"])]
                for p in last["positions"]]
        parts.append("<section><h2>Positions now</h2>"
                     + (_table(["Ticker", "Shares", "Average price", "Last price", "Unrealised"], rows, (1, 2, 3, 4))
                        if rows else '<p class="muted">No positions.</p>')
                     + f'<p class="muted">Cash {_money(last["cash"])}.</p></section>')

    charts = []
    for ticker in cfg.tickers:
        recent = sorted({moment.date() for moment, _ in data.bars[ticker]})[-_CHART_DAYS:]
        bars = [b for b in data.bars[ticker] if b[0].date() >= recent[0]] if recent else []
        if len(bars) < 2:
            continue
        trades = [(data.local(f["at"]), f["side"], f["qty"], f["price"]) for f in data.fills if f["ticker"] == ticker]
        marks = [(data.local(a["started_at"]), f'{data.local(a["started_at"]):%m-%d %H:%M} {a["rating"]}',
                  _TONE.get(a["rating"], "review")) for a in data.analyses
                 if a["ticker"] == ticker and a["status"] == "ok"]
        held = next((p for p in (last["positions"] if last else []) if p["ticker"] == ticker), None)
        charts.append(f'<h3><a href="day-{data.days[-1]}.html#{escape(ticker)}">{escape(ticker)}</a>'
                      + _chart_link(cfg, ticker)
                      + (f' <span class="muted">· {held["qty"]:g} held at {_money(held["avg_price"])} on average</span>'
                         if held else ' <span class="muted">· none held</span>') + "</h3>"
                      + _price_chart(bars, trades, marks, label=f"{ticker} price by five-minute bar, with each "
                                                                "purchase and sale marked where it filled"))
    if charts:
        parts.append("<section><h2>Each ticker's price, with what was bought and sold</h2>" + "".join(charts) + "</section>")

    rows = []
    for ticker in cfg.tickers:
        runs = [a for a in data.analyses if a["ticker"] == ticker and a["status"] == "ok"]
        strip = "".join(f'<i class="{_TONE.get(a["rating"], "review")}" '
                        f'title="{escape(data.local(a["started_at"]).strftime("%m-%d %H:%M"))} {escape(str(a["rating"]))}"></i>'
                        for a in runs[-60:])
        latest = runs[-1] if runs else None
        flips = sum(1 for a, b in zip(runs, runs[1:], strict=False) if a["rating"] != b["rating"])
        # The name opens the ticker's own record of the latest day; the chart is someone else's page.
        name = (f'<a href="day-{data.days[-1]}.html#{escape(ticker)}"><b>{escape(ticker)}</b></a>' if data.days
                else f"<b>{escape(ticker)}</b>")
        rows.append([name + _chart_link(cfg, ticker), _rating(latest["rating"]) if latest else "–",
                     "–" if not latest or latest["target_weight"] is None else f"{latest['target_weight']:g}",
                     "–" if not latest or latest["stop_loss"] is None else f"{latest['stop_loss']:g}",
                     str(len(runs)), str(flips), f'<div class="strip">{strip}</div>'])
    parts.append("<section><h2>Each ticker's decisions</h2>"
                 + _table(["Ticker", "Latest", "Target weight", "Stop", "Analyses", "Rating changes",
                           "Rating over time (oldest first)"], rows, (2, 3, 4, 5)) + "</section>")

    parts.append("<section><h2>What the price did after each rating</h2>" + _outcome_table(data)
                 + '<p class="muted">The mean price move from the analysis to each mark. "Right way" counts the moves '
                   "in the direction the rating called; Hold calls no direction. Few decisions make these figures "
                   "noise rather than evidence.</p></section>")

    rows = [[f'<a href="day-{s["day"]}.html">{s["day"]}</a>', str(len(s["ticks"])),
             f"{len(s['analyses'])}" + (f' <span class="bad">({s["failed"]} failed)</span>' if s["failed"] else ""),
             str(len(s["orders"])), str(len(s["fills"])), _money(s["llm_cost"]), _money(s["trade_cost"]),
             _money(s["equity_end"]), _signed(s["pnl"])] for s in reversed(stats)]
    parts.append("<section><h2>Days</h2>"
                 + _table(["Day", "Ticks", "Analyses", "Orders", "Fills", "Model spend", "Trading cost",
                           "Equity at the end", "Change"], rows, (1, 2, 3, 4, 5, 6, 7, 8)) + "</section>")
    return _page("Trading loop", f"tradeloop · {', '.join(cfg.tickers)} · built {timeutil.iso(timeutil.utcnow())}",
                 "".join(parts))


# ---------------------------------------------------------------------------
# One day
# ---------------------------------------------------------------------------


def _read_artifacts(folder: str | None) -> tuple[dict, list[dict]]:
    """A run's saved state and steps; empty when its folder is gone."""
    state, steps = {}, []
    if not folder:
        return state, steps
    with contextlib.suppress(OSError, ValueError):
        state = json.loads((Path(folder) / "state.json").read_text(encoding="utf-8"))
    with contextlib.suppress(OSError, ValueError), open(Path(folder) / "steps.jsonl", encoding="utf-8") as f:
        steps = [json.loads(line) for line in f if line.strip()]
    return state, steps


def _checks(checks: list[dict]) -> str:
    items = "".join(f'<li class="{"" if c["passed"] else "bad"}">{"✓" if c["passed"] else "✗"} '
                    f'<b>{escape(c["name"])}</b>: {escape(c["detail"])}</li>' for c in checks)
    return f'<ul class="checks">{items}</ul>'


def _analysis(data: _Data, md: MarkdownIt, run: dict, ruling: dict | None, outcomes: list[dict]) -> str:
    """One analysis, collapsed: what it decided, what it was told, what each agent wrote, and each step."""
    cost = "no price on file" if run["cost_usd"] is None and run["status"] == "ok" else _money(run["cost_usd"])
    head = (f'<b>{data.clock(run["started_at"])}</b> {escape(run["kind"])} · '
            + (_rating(run["rating"]) if run["status"] == "ok" else f'<span class="pill bad">{escape(run["status"])}</span>')
            + (f' · target {run["target_weight"]:g}' if run["target_weight"] is not None else "")
            + (f' · stop {run["stop_loss"]:g}' if run["stop_loss"] is not None else "")
            + f' <span class="muted">· {run["elapsed_seconds"] or 0:.0f}s · {cost} · '
              f'{run["llm_calls"]} model calls, {run["tool_calls"]} tool calls</span>')
    body = []
    if run["error"]:
        body.append(f'<p class="bad">{escape(run["error"])}</p>')
    if run["summary"]:
        body.append(f"<p>{escape(run['summary'])}</p>")
    if ruling:
        body.append(f"<h3>What the rules did with it on this tick: {escape(ruling['verdict'])}</h3>"
                    + _checks(ruling["checks"]))
    marks = [o for o in outcomes if o["price"] is not None]
    if marks:
        order = {h: i for i, h in enumerate(_HORIZONS)}
        body.append("<h3>What the price did next</h3>" + _table(
            ["Mark", "Price", "Move from the analysis"],
            [[escape(o["horizon"]), _money(o["price"]), _pct(o["ret"])]
             for o in sorted(marks, key=lambda o: order.get(o["horizon"], 9))], (1, 2)))

    state, steps = _read_artifacts(run["artifacts_dir"])
    told = [(label, state.get(key)) for label, key in (("The instrument and the time", "instrument_context"),
                                                       ("The book", "portfolio_context"),
                                                       ("Position size and this session so far", "session_context"),
                                                       ("Lessons from past decisions", "past_context"))]
    if any(text for _, text in told):
        body.append("<details><summary>What it was told</summary>"
                    + "".join(f"<h3>{escape(label)}</h3><pre>{escape(str(text))}</pre>" for label, text in told if text)
                    + "</details>")
    for label, path in _AGENT_TEXTS:
        text = state
        for key in path:
            text = text.get(key, "") if isinstance(text, dict) else ""
        if isinstance(text, str) and text.strip():
            body.append(f"<details><summary>{escape(label)}</summary>{_render_md(md, text)}</details>")
    if steps:
        rows = []
        for s in steps:
            detail = ""
            if s["kind"] == "tool":
                detail = (f'<details><summary class="mono">{escape(json.dumps(s.get("args"), ensure_ascii=False, default=str))}'
                          f'</summary><pre>{escape(s.get("output") or s.get("error") or "")}</pre></details>')
            elif s.get("tool_calls"):
                detail = '<span class="muted">calls ' + escape(", ".join(c["name"] or "?" for c in s["tool_calls"])) + "</span>"
            rows.append([escape(s["agent"]), escape(s["kind"]), f'<span class="mono">{escape(str(s["name"]))}</span>',
                         f'{s.get("elapsed_seconds") or 0:.1f}s', f'{s.get("input_tokens") or 0:,}',
                         f'{s.get("output_tokens") or 0:,}',
                         (f'<span class="bad">{escape(s["error"])}</span>' if s.get("error") else detail)])
        body.append("<details><summary>Every step: " + f"{len(steps)} model and tool calls</summary>"
                    + _table(["Agent", "Kind", "Model or tool", "Took", "Tokens in", "Tokens out", "Detail"],
                             rows, (3, 4, 5)) + "</details>")
    elif run["status"] == "ok":
        body.append('<p class="muted">The folder with this run\'s texts is no longer there.</p>')
    return f"<details><summary>{head}</summary>{''.join(body)}</details>"


def _day(data: _Data, day: str) -> str:
    cfg, md, s = data.cfg, _md(), _day_stats(data, day)
    ids, parts = s["ids"], []
    parts.append('<p><a href="index.html">← All days</a></p>')
    parts.append(_tiles([
        ("Equity at the end", _money(s["equity_end"])), ("Change on the day", _signed(s["pnl"])),
        ("Model spend", _money(s["llm_cost"])), ("Estimated trading cost", _money(s["trade_cost"])),
        ("Analyses", f"{len(s['analyses'])}" + (f" ({s['failed']} failed)" if s["failed"] else "")),
        ("Orders / fills", f"{len(s['orders'])} / {len(s['fills'])}"),
    ]))
    parts.append('<nav class="tickers">' + "".join(f'<a href="#{escape(t)}">{escape(t)}</a>' for t in cfg.tickers)
                 + '<a href="#orders">Orders</a><a href="#log">Log</a></nav>')

    rows = [[data.clock(t["scheduled_at"]), escape(t["kind"]),
             f'<span class="{"bad" if t["status"] == "error" else ""}">{escape(t["status"])}</span>',
             escape(t["note"])] for t in s["ticks"]]
    parts.append("<section><h2>Ticks</h2>" + _table(["Time", "Kind", "Status", "What happened"], rows) + "</section>")

    obs = defaultdict(dict)
    for o in data.observations:
        if o["tick_id"] in ids:
            obs[o["ticker"]][o["tick_id"]] = o
    rulings = {(t["tick_id"], t["ticker"]): t for t in data.targets if t["tick_id"] in ids}
    outcomes = defaultdict(list)
    for o in data.outcomes:
        outcomes[o["run_id"]].append(o)

    for ticker in cfg.tickers:
        runs = [a for a in s["analyses"] if a["ticker"] == ticker]
        fills = [f for f in s["fills"] if f["ticker"] == ticker]
        points = [(data.local(t["scheduled_at"]), obs[ticker][t["tick_id"]]["price"]) for t in s["ticks"]
                  if obs[ticker].get(t["tick_id"]) and obs[ticker][t["tick_id"]]["price"] is not None]
        marks = [(data.local(a["started_at"]), f"{data.clock(a['started_at'])} {a['rating']}",
                  _TONE.get(a["rating"], "review")) for a in runs if a["status"] == "ok"]
        rows = []
        for t in s["ticks"]:
            o, r = obs[ticker].get(t["tick_id"]), rulings.get((t["tick_id"], ticker))
            if not o:
                continue
            failed = [c for c in (r["checks"] if r else []) if not c["passed"]]
            orders = [x for x in s["orders"] if x["tick_id"] == t["tick_id"] and x["ticker"] == ticker]
            rows.append([
                data.clock(t["scheduled_at"]), escape(t["kind"]), _money(o["price"]), f"{o['position_qty']:g}",
                f'<b>{escape(o["gate_action"])}</b><br><span class="muted">{escape("; ".join(o["gate_reasons"]))}</span>',
                "–" if not r else (f'<b>{escape(r["verdict"])}</b>'
                                   + (f' → {r["target_weight"]:.1%}' if r["target_weight"] != r["current_weight"] else "")
                                   + "".join(f'<br><span class="warnc">✗ {escape(c["name"])}: {escape(c["detail"])}</span>'
                                             for c in failed)),
                "<br>".join(f'{escape(x["side"])} {x["qty"]:g} {escape(x["type"])} · {escape(x["status"])}'
                            for x in orders) or "–",
            ])
        bars = [b for b in data.bars[ticker] if b[0].date().isoformat() == day]
        trades = [(data.local(f["at"]), f["side"], f["qty"], f["price"]) for f in fills]
        chart = (_price_chart(bars, trades, marks, label=f"{ticker} price by five-minute bar on {day}, with each "
                                                         "purchase and sale marked where it filled")
                 if len(bars) >= 2 else
                 _line_chart(points, marks, label=f"{ticker} price at each tick, with the day's analyses marked"))
        section = [f'<section id="{escape(ticker)}"><h2>{escape(ticker)}{_chart_link(cfg, ticker)}</h2>', chart,
                   _table(["Time", "Tick", "Price", "Shares held", "Gate", "Ruling", "Orders"], rows, (2, 3))]
        if fills:
            section.append("<p>" + " · ".join(
                f'{data.clock(f["at"])} {"bought" if f["side"] == "buy" else "sold"} {f["qty"]:g} at {_money(f["price"])}'
                for f in fills) + "</p>")
        section.append(f"<h3>Analyses ({len(runs)})</h3>")
        section.extend(_analysis(data, md, run, rulings.get((run["tick_id"], ticker)), outcomes[run["run_id"]])
                       for run in runs)
        parts.append("".join(section) + "</section>")

    rows = [[data.clock(o["submitted_at"]), escape(o["ticker"]), escape(o["side"]), f"{o['qty']:g}", escape(o["type"]),
             _money(o["limit_price"] or o["stop_price"]), escape(o["status"]), f"{o['filled_qty']:g}",
             _money(o["filled_avg_price"]), _money(o["est_cost_usd"]), escape(o["reason"]),
             f'<span class="bad">{escape(o["error"])}</span>' if o["error"] else ""] for o in s["orders"]]
    parts.append('<section id="orders"><h2>Orders</h2>'
                 + _table(["Time", "Ticker", "Side", "Shares", "Type", "Limit or stop", "Status", "Filled",
                           "Fill price", "Est. cost", "Why", "Error"], rows, (3, 5, 7, 8, 9)) + "</section>")

    events = [e for e in data.events if (e["tick_id"] in ids) or (not e["tick_id"] and data.day_of(e["at"]) == day)]
    rows = [[data.clock(e["at"]), f'<span class="{"bad" if e["level"] == "error" else "warnc"}">{escape(e["level"])}</span>',
             escape(e["source"]), escape(e["ticker"] or ""), escape(e["message"])] for e in events]
    parts.append('<section id="log"><h2>Log</h2>' + _table(["Time", "Level", "From", "Ticker", "Message"], rows)
                 + "</section>")
    return _page(f"{day}", f"tradeloop · one day · times are {cfg.timezone}", "".join(parts))


def build(cfg: LoopConfig, store: Store, days: int | None = 1) -> Path:
    """Write the overview and the pages of the ``days`` most recent days (None: every day); the overview's path."""
    data = _Data(cfg, store)
    out = cfg.dashboard_dir
    out.mkdir(parents=True, exist_ok=True)
    for day in data.days if days is None else data.days[-days:]:
        (out / f"day-{day}.html").write_text(_day(data, day), encoding="utf-8")
    index = out / "index.html"
    index.write_text(_index(data), encoding="utf-8")
    return index
