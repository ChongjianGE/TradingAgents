"""The dashboard: built from a day of the loop's own record, it shows every tick,
each decision with what it was based on, and what was traded."""

from __future__ import annotations

import json
import re
from datetime import timedelta

import pytest

from tests.test_tradeloop_engine import ET, TICKERS, Loop, et
from tradeloop import dashboard, timeutil
from tradeloop.alpaca import Bar


def _a_day(tmp_path, **settings) -> Loop:
    """A pre-market Buy on NVDA, confirmed by a second analysis after a move, bought, then the close."""
    loop = Loop(tmp_path, **settings)
    fields = {"target_weight": 0.5, "stop_loss": 95.0, "price_target": 120.0}
    loop.analysts.say("NVDA", ("Buy", fields), ("Buy", fields))
    loop.analysts.say("HD", RuntimeError("rate limited"))
    loop.tick("premarket", 11, "08:45")
    loop.tick("core", 11, "09:35")
    loop.market.prices["NVDA"] = 102.0
    loop.market.series["NVDA"] = [Bar(et(11, "09:30"), 100.0, 101.0, 99.5, 100.8, 1000)]
    loop.tick("core", 11, "10:35")
    loop.tick("postclose", 11, "16:30")
    return loop


def _write_artifacts(loop, run_id, report="## Trend\n\n| Level | Price |\n|---|---|\n| Support | 95 |"):
    folder = loop.cfg.runs_dir / run_id
    folder.mkdir(parents=True)
    (folder / "state.json").write_text(json.dumps({
        "market_report": report, "final_trade_decision": "**Rating**: Buy\n\nBuy in thirds.",
        "investment_debate_state": {"bull_history": "Bull: cheap at 10x.", "bear_history": "Bear: momentum is broken."},
        "risk_debate_state": {"neutral_history": "Neutral: size it down."},
        "instrument_context": "The instrument to analyze is `NVDA`. The current time is 2026-11-11 10:35 EST.",
        "portfolio_context": "Portfolio at the analysis date:\n- No current position in NVDA",
        "session_context": "Earlier in this session: 08:45 premarket analysis rated it Buy",
    }), encoding="utf-8")
    steps = [
        {"seq": 0, "agent": "Market Analyst", "kind": "llm", "name": "claude-sonnet-5", "elapsed_seconds": 3.2,
         "input_tokens": 3383, "output_tokens": 60, "output": "", "tool_calls": [{"name": "get_stock_data", "args": {}}],
         "error": None},
        {"seq": 1, "agent": "Market Analyst", "kind": "tool", "name": "get_stock_data", "elapsed_seconds": 0.4,
         "input_tokens": 0, "output_tokens": 0, "args": {"start_date": "2026-09-01", "end_date": "2026-11-11"},
         "output": "Date,Open,High,Low,Close\n2026-11-10,173.15,174.25,167.69,168.89", "error": None},
    ]
    (folder / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps), encoding="utf-8")
    loop.store.finish_analysis(run_id, artifacts_dir=str(folder))


@pytest.fixture
def day(tmp_path):
    loop = _a_day(tmp_path)
    _write_artifacts(loop, "20261111T1035-core-NVDA")
    yield loop
    loop.store.close()


@pytest.mark.unit
class TestTheDayPage:
    def test_the_day_and_the_overview_are_written(self, day):
        index = dashboard.build(day.cfg, day.store)
        assert index == day.cfg.dashboard_dir / "index.html" and index.is_file()
        assert (day.cfg.dashboard_dir / "day-2026-11-11.html").is_file()

    def test_every_tick_is_listed_with_what_happened(self, day):
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "day-2026-11-11.html").read_text()
        for clock, kind in (("08:45", "premarket"), ("09:35", "core"), ("10:35", "core"), ("16:30", "postclose")):
            assert f"<td class=\"\">{clock}</td><td class=\"\">{kind}</td>" in page
        assert "4 analysed, 1 failed" in page and "1 orders" in page

    def test_each_ticker_shows_the_gate_the_ruling_and_the_order(self, day):
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "day-2026-11-11.html").read_text()
        for ticker in TICKERS:
            assert f'<section id="{ticker}">' in page
        assert "price moved +2.0% since the last analysis" in page        # why the model was called
        assert "nothing changed enough" in page                            # and why it was not
        assert "✗ confirmation: needs 2 analyses in a row asking for more; 1 so far" in page
        assert "buy 98 limit · filled" in page                             # half of a full position at 102
        assert "bought 98 at $102.03" in page

    def test_an_analysis_opens_to_what_it_was_told_what_each_agent_wrote_and_every_step(self, day):
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "day-2026-11-11.html").read_text()
        assert "target 0.5 · stop 95" in page and "380s · $1.00 · 17 model calls, 28 tool calls" in page
        assert "<summary>What it was told</summary>" in page and "The current time is 2026-11-11 10:35 EST." in page
        assert "Earlier in this session: 08:45 premarket analysis rated it Buy" in page
        for agent in ("Market Analyst", "Bull Researcher", "Bear Researcher", "Neutral Analyst", "Portfolio Manager"):
            assert f"<summary>{agent}</summary>" in page
        assert "<summary>Sentiment Analyst</summary>" not in page          # it wrote nothing in this run
        assert "<td>Support</td>" in page                                  # the report's table is a table
        assert "Every step: 2 model and tool calls" in page
        assert "&quot;start_date&quot;: &quot;2026-09-01&quot;" in page    # what the tool was asked
        assert "2026-11-10,173.15,174.25,167.69,168.89" in page            # and what came back
        assert "calls get_stock_data" in page

    def test_the_rules_verdict_and_the_price_afterwards_sit_with_the_analysis(self, day):
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "day-2026-11-11.html").read_text()
        assert "What the rules did with it on this tick: increase" in page
        assert "What the rules did with it on this tick: deferred:blocked" in page
        assert "What the price did next" in page and "+0.80%" in page      # the pre-market run's 60-minute mark

    def test_a_failed_analysis_and_its_log_line_are_shown(self, day):
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "day-2026-11-11.html").read_text()
        assert '<span class="pill bad">error</span>' in page and "RuntimeError: rate limited" in page
        assert '<section id="log">' in page and "analysis" in page

    def test_a_run_whose_folder_is_gone_says_so(self, day):
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "day-2026-11-11.html").read_text()
        assert "The folder with this run's texts is no longer there." in page

    def test_text_a_model_wrote_cannot_inject_markup(self, tmp_path):
        loop = _a_day(tmp_path)
        _write_artifacts(loop, "20261111T1035-core-NVDA",
                         report='<script>alert(1)</script>\n\n![x](https://evil.example/p.png)\n\n**bold**')
        loop.store.finish_analysis("20261111T1035-core-NVDA", summary="<img src=x onerror=alert(2)>")
        dashboard.build(loop.cfg, loop.store)
        page = (loop.cfg.dashboard_dir / "day-2026-11-11.html").read_text()
        # No tag of the model's reaches the page, and nothing is loaded: the image is left as a plain link.
        assert "<script>" not in page and "<img" not in page
        assert "&lt;img src=x onerror=alert(2)&gt;" in page               # shown as the text it is
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page and "<strong>bold</strong>" in page
        loop.store.close()


@pytest.mark.unit
class TestTheOverview:
    def test_it_totals_the_book_the_spend_and_the_days(self, day):
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "index.html").read_text()
        # Eleven analyses at a dollar: 4 + HD's retry + NVDA's second + 5 after the close. The failed one cost nothing.
        assert "<b>$11.00</b><span>Model spend</span>" in page
        assert "<b>$3.00</b><span>Estimated trading cost</span>" in page    # 3 bps on 98 shares at 102
        assert "<span>Estimated trading cost</span>" in page and "<span>After model spend</span>" in page
        assert '<a href="day-2026-11-11.html">2026-11-11</a>' in page
        assert '<span class="bad">(1 failed)</span>' in page
        assert "Book: simulate." in page

    def test_positions_and_each_tickers_latest_decision_are_listed(self, day):
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "index.html").read_text()
        assert '>chart ↗</a></td><td class="num">98</td>' in page and '<td class="">NVDA<a class="out"' in page
        assert '<span class="pill level">Hold</span>' in page             # NVDA's post-close analysis
        assert 'title="11-11 08:45 Buy"' in page and "<svg" in page       # the rating strip and the equity chart

    def test_each_ticker_opens_its_record_of_the_latest_day_and_links_out_to_a_chart(self, day):
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "index.html").read_text()
        assert '<a href="day-2026-11-11.html#NVDA"><b>NVDA</b></a>' in page
        out = ('<a class="out" href="https://finance.yahoo.com/quote/NVDA" target="_blank" rel="noopener noreferrer" '
               'title="Price chart of NVDA, on another site">chart ↗</a>')
        assert page.count(out) == 2                     # beside the position and beside the decisions
        assert page.count('class="out"') == 1 + len(day.cfg.tickers)
        day_page = (day.cfg.dashboard_dir / "day-2026-11-11.html").read_text()
        assert f'<section id="NVDA"><h2>NVDA{out}</h2>' in day_page

    def test_the_chart_site_is_the_configs_and_the_symbol_is_made_safe_for_an_address(self, tmp_path):
        loop = _a_day(tmp_path, chart_url="https://charts.example/s?sym={ticker}&range=1d")
        loop.cfg.tickers.append("BRK/B")                  # no real ticker has a slash; an address must survive one
        dashboard.build(loop.cfg, loop.store)
        page = (loop.cfg.dashboard_dir / "index.html").read_text()
        assert 'href="https://charts.example/s?sym=NVDA&amp;range=1d"' in page
        assert 'href="https://charts.example/s?sym=BRK%2FB&amp;range=1d"' in page
        loop.store.close()

    def test_no_chart_site_means_no_link_out(self, tmp_path):
        loop = _a_day(tmp_path, chart_url="")
        dashboard.build(loop.cfg, loop.store)
        for name in ("index.html", "day-2026-11-11.html"):
            assert 'class="out"' not in (loop.cfg.dashboard_dir / name).read_text().split("</style>")[1]
        assert '<a href="day-2026-11-11.html#NVDA"><b>NVDA</b></a>' in (loop.cfg.dashboard_dir / "index.html").read_text()
        loop.store.close()

    def test_outcomes_are_grouped_by_rating_with_how_often_the_move_went_its_way(self, day):
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "index.html").read_text()
        assert "What the price did after each rating" in page
        # NVDA's two Buys: +0.80% from the pre-market one, -1.18% from the one made at 102.
        assert "-0.19% <span class=\"muted\">n=2</span>" in page and "right way 1/2" in page

    def test_an_empty_record_still_builds(self, tmp_path):
        loop = Loop(tmp_path)
        index = dashboard.build(loop.cfg, loop.store)
        text = index.read_text()
        assert "Not enough points for a chart yet." in text and "Nothing here yet." in text
        loop.store.close()

    def test_only_the_most_recent_days_are_rebuilt_unless_all_are_asked_for(self, day):
        day.tick("premarket", 12, "08:45")
        dashboard.build(day.cfg, day.store, days=1)
        assert not (day.cfg.dashboard_dir / "day-2026-11-11.html").exists()
        assert (day.cfg.dashboard_dir / "day-2026-11-12.html").is_file()
        dashboard.build(day.cfg, day.store, days=None)
        assert (day.cfg.dashboard_dir / "day-2026-11-11.html").is_file()


@pytest.mark.unit
class TestTheChart:
    def test_a_line_with_marks_is_drawn_inside_its_box(self):
        points = [(et(11, "09:35"), 100.0), (et(11, "10:35"), 102.0), (et(11, "11:35"), 101.0)]
        svg = dashboard._line_chart(points, [(et(11, "10:35"), "10:35 Buy", "up"), (et(12, "10:35"), "outside", "up")],
                                    label="NVDA price")
        assert svg.count("<circle") == 1 and "<title>10:35 Buy</title>" in svg
        assert 'aria-label="NVDA price"' in svg and "M84.0," in svg

    def test_a_flat_line_does_not_divide_by_zero(self):
        svg = dashboard._line_chart([(et(11, "09:35"), 100.0), (et(11, "10:35"), 100.0)])
        assert "<path" in svg and "nan" not in svg.lower()


def _bars(day, start="09:30", count=78, base=100.0, step=0.1):
    first = et(day, start)
    return [(first + timedelta(minutes=5 * i), base + step * i) for i in range(count)]


@pytest.mark.unit
class TestThePriceChart:
    """The price by five-minute bar, with a triangle where each buy and sell filled."""

    def test_fewer_than_two_bars_says_so(self):
        assert "Not enough prices" in dashboard._price_chart(_bars(11, count=2))     # an open and one close

    def test_a_buy_is_a_triangle_under_the_price_paid_and_a_sell_one_above_it(self):
        bars = _bars(11)
        svg = dashboard._price_chart(bars, [(et(11, "09:41"), "buy", 57.0, 100.2), (et(11, "14:02"), "sell", 12.0, 105.3)])
        buy, sell = re.findall(r'<g class="(up|down)">.*?<polygon points="([^"]+)".*?<title>([^<]+)</title>', svg)
        assert (buy[0], buy[2]) == ("up", "11-11 09:41 bought 57 at $100.20")
        assert (sell[0], sell[2]) == ("down", "11-11 14:02 sold 12 at $105.30")
        tip, left, right = (tuple(map(float, point.split(","))) for point in buy[1].split())
        assert left[1] == right[1] > tip[1] and right[0] - left[0] >= 8      # under its tip, and wide enough to see
        tip, left, right = (tuple(map(float, point.split(","))) for point in sell[1].split())
        assert left[1] == right[1] < tip[1]
        assert ">+57</text>" in svg and ">−12</text>" in svg
        assert '<span class="up">▲</span> bought' in svg and '<span class="down">▼</span> sold' in svg

    def test_the_tip_is_at_the_price_paid_not_at_the_bars_close(self):
        bars = [(et(11, "09:30"), 100.0), (et(11, "09:35"), 100.0), (et(11, "09:40"), 100.0)]

        def tip_and_line(price):
            svg = dashboard._price_chart(bars, [(et(11, "09:36"), "buy", 1.0, price)])
            return (float(re.search(r'<polygon points="[\d.]+,([\d.]+) ', svg).group(1)),
                    float(re.search(r'<path d="M[\d.]+,([\d.]+) ', svg).group(1)))

        tip, line = tip_and_line(98.0)
        assert tip > line                                # paid less than the bars show: the tip is below the line
        tip, line = tip_and_line(102.0)
        assert tip < line

    def test_a_mark_sits_at_the_moment_of_its_fill_between_the_two_points_around_it(self):
        points = [(et(11, "09:30"), 100.0), (et(11, "09:35"), 100.0), (et(11, "09:40"), 100.0), (et(11, "09:45"), 100.0)]
        xs = [float(v) for v in re.findall(r"[ML]([\d.]+),", re.search(r'<path d="([^"]+)"', dashboard._price_chart(points)).group(1))]

        def tip_x(moment):
            svg = dashboard._price_chart(points, [(moment, "buy", 1.0, 100.0)])
            return float(re.search(r'<polygon points="([\d.]+),', svg).group(1))

        assert tip_x(et(11, "09:40")) == pytest.approx(xs[2], abs=0.1)
        assert tip_x(et(11, "09:42") + timedelta(seconds=30)) == pytest.approx((xs[2] + xs[3]) / 2, abs=0.1)
        assert xs[2] < tip_x(et(11, "09:40") + timedelta(seconds=6)) < xs[2] + (xs[3] - xs[2]) * 0.05

    def test_an_order_filled_in_parts_within_one_bar_is_one_mark_at_its_average(self):
        svg = dashboard._price_chart(_bars(11), [(et(11, "09:40") + timedelta(seconds=6), "buy", 19.0, 174.41),
                                                 (et(11, "09:40") + timedelta(seconds=12), "buy", 38.0, 174.38)])
        assert svg.count("<polygon") == 1 and "bought 57 at $174.39" in svg

    def test_a_buy_and_a_sell_in_one_bar_stay_two_marks(self):
        svg = dashboard._price_chart(_bars(11), [(et(11, "10:01"), "buy", 5.0, 100.7), (et(11, "10:03"), "sell", 5.0, 100.8)])
        assert svg.count("<polygon") == 2

    def test_a_fill_with_no_bar_near_it_is_left_off(self):
        svg = dashboard._price_chart(_bars(11), [(et(11, "18:20"), "buy", 5.0, 100.7)])
        assert "<polygon" not in svg and "bought" not in svg

    def test_analyses_are_ticks_along_the_bottom_with_their_rating_on_hover(self):
        svg = dashboard._price_chart(_bars(11), analyses=[(et(11, "10:35"), "10:35 Overweight", "up")])
        assert '<g class="up"><rect' in svg and "<title>10:35 Overweight</title>" in svg
        assert "ticks along the bottom: analyses" in svg and "<polygon" not in svg

    def test_a_single_day_is_labelled_by_the_clock_and_several_by_the_date(self):
        one = dashboard._price_chart(_bars(11))
        assert ">09:30</text>" in one and ">12:00</text>" in one and ">15:55</text>" in one and one.count("<path") == 1
        three = dashboard._price_chart(_bars(11) + _bars(12, base=103.0) + _bars(13, base=101.0))
        assert all(f">11-{d}</text>" in three for d in (11, 12, 13)) and ">09:30</text>" not in three
        assert three.count("<path") == 3                 # each day its own line: the night is a step, not a slope

    def test_time_runs_by_bar_so_a_night_takes_almost_no_room(self):
        svg = dashboard._price_chart(_bars(11) + _bars(12))
        first, second = re.findall(r'<path d="M([\d.]+),[\d.]+ .*? L([\d.]+),[\d.]+"', svg)
        day_width, night = float(first[1]) - float(first[0]), float(second[0]) - float(first[1])
        assert 0 < night < day_width / 20

    def test_a_long_record_is_thinned_but_every_day_keeps_its_last_bar(self):
        many = [bar for day in (10, 11, 12, 13, 16, 17, 18, 19, 20) for bar in _bars(day, count=140, step=0.01)]
        svg = dashboard._price_chart(many)
        points = sum(path.count("L") + 1 for path in re.findall(r'<path d="([^"]+)"', svg))
        assert points <= 800 and svg.count("<path") == 9

    def test_many_trades_keep_their_hover_text_but_lose_their_labels(self):
        trades = [(et(11, "09:30") + timedelta(minutes=30 * i), "buy", 1.0 + i, 100.0 + i * 0.5) for i in range(12)]
        svg = dashboard._price_chart(_bars(11), trades)
        assert svg.count("<polygon") == 12 and ">+1</text>" not in svg and "bought 1 at $100.00" in svg

    def test_a_flat_price_does_not_divide_by_zero(self):
        assert "<path" in dashboard._price_chart(_bars(11, step=0.0))


@pytest.mark.unit
class TestThePriceLineFromBars:
    """A bar is five minutes of trading: its close belongs at its end, or the line runs ahead of the trades."""

    ROWS = [   # NVDA on the first live day, as Alpaca gave it
        {"at": "2026-11-11T13:00:00Z", "open": 175.30, "close": 175.35},      # 08:00, before the open
        {"at": "2026-11-11T14:30:00Z", "open": 172.71, "close": 173.86},
        {"at": "2026-11-11T14:35:00Z", "open": 173.86, "close": 174.57},
        {"at": "2026-11-11T14:40:00Z", "open": 174.57, "close": 171.83},
        {"at": "2026-11-11T14:45:00Z", "open": 171.865, "close": 172.09},
        {"at": "2026-11-11T21:00:00Z", "open": 173.60, "close": 173.55},      # 16:00, after the close
    ]

    def _points(self, rows=None):
        return dashboard._price_points(rows or self.ROWS, lambda at: timeutil.parse(at).astimezone(ET))

    def test_each_close_is_put_at_the_end_of_its_bar_and_the_day_starts_at_its_open(self):
        assert self._points() == [
            (et(11, "09:30"), 172.71), (et(11, "09:35"), 173.86), (et(11, "09:40"), 174.57),
            (et(11, "09:45"), 171.83), (et(11, "09:50"), 172.09)]

    def test_every_day_starts_at_its_own_open(self):
        rows = self.ROWS[1:3] + [{"at": "2026-11-12T14:30:00Z", "open": 180.0, "close": 181.0}]
        assert self._points(rows)[-2:] == [(et(12, "09:30"), 180.0), (et(12, "09:35"), 181.0)]

    def test_a_purchase_at_the_start_of_a_falling_bar_touches_the_line(self):
        """57 shares filled at 09:40:06 for 174.39, six seconds into a bar that then fell to 171.83.
        Drawn with each close at its bar's start, the line stood at 171.83 where the purchase was marked."""
        svg = dashboard._price_chart(self._points(), [(et(11, "09:40") + timedelta(seconds=6), "buy", 57.0, 174.39)])
        line = [(float(x), float(y)) for x, y in re.findall(r"[ML]([\d.]+),([\d.]+)", re.search(r'<path d="([^"]+)"', svg).group(1))]
        tip_x, tip_y = map(float, re.search(r'<polygon points="([\d.]+),([\d.]+) ', svg).groups())
        (x0, y0), (x1, y1) = line[2], line[3]                 # the points at 09:40 and 09:45
        on_the_line = y0 + (tip_x - x0) / (x1 - x0) * (y1 - y0)
        full_drop = y1 - y0
        assert x0 < tip_x < x1 and abs(tip_y - on_the_line) < 0.1 * full_drop


@pytest.mark.unit
class TestTheChartsOnThePages:
    def _with_bars(self, loop):
        rows = [{"ticker": "NVDA", "at": timeutil.iso(moment), "open": close, "high": close, "low": close,
                 "close": close, "volume": 100.0} for moment, close in _bars(11, base=100.0, step=0.05)]
        rows.append({"ticker": "NVDA", "at": timeutil.iso(et(11, "08:10")), "open": 90.0, "high": 90.0, "low": 90.0,
                     "close": 90.0, "volume": 5.0})                  # before the open: not part of the session's line
        loop.store.add_bars(rows)

    def test_the_day_page_draws_the_session_with_the_days_purchase_on_it(self, day):
        self._with_bars(day)
        dashboard.build(day.cfg, day.store)
        section = (day.cfg.dashboard_dir / "day-2026-11-11.html").read_text().split('<section id="NVDA">')[1].split("</section>")[0]
        assert 'aria-label="NVDA price every five minutes on 2026-11-11' in section
        assert section.count("<polygon") == 1 and "11-11 10:35 bought 98 at" in section
        assert "90.00" not in section.split("</svg>")[0]              # the pre-market print is not on the axis

    def test_without_bars_the_day_page_keeps_the_line_through_the_ticks(self, day):
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "day-2026-11-11.html").read_text()
        assert 'aria-label="NVDA price at each tick' in page and "<polygon" not in page

    def test_the_overview_has_a_chart_for_each_ticker_with_bars_and_says_what_is_held(self, day):
        dashboard.build(day.cfg, day.store)
        assert "with what was bought and sold" not in (day.cfg.dashboard_dir / "index.html").read_text()
        self._with_bars(day)
        dashboard.build(day.cfg, day.store)
        page = (day.cfg.dashboard_dir / "index.html").read_text()
        section = page.split("Each ticker's price, with what was bought and sold</h2>")[1].split("</section>")[0]
        assert section.count("<svg") == 1 and '<a href="day-2026-11-11.html#NVDA">NVDA</a>' in section
        assert "98 held at $" in section and section.count("<polygon") == 1
