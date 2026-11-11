"""The dashboard: built from a day of the loop's own record, it shows every tick,
each decision with what it was based on, and what was traded."""

from __future__ import annotations

import json

import pytest

from tests.test_tradeloop_engine import TICKERS, Loop, et
from tradeloop import dashboard
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
