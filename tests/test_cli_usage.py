"""What a run used and cost: counted per model, priced, and kept in the usage log."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from typer.testing import CliRunner

from cli import usage
from cli.stats_handler import StatsCallbackHandler


def _answer(model, tokens_in, tokens_out, cache_read=0, named_in="response_metadata"):
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": tokens_in, "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
            "input_token_details": {"cache_read": cache_read},
        },
        response_metadata={"model_name": model} if named_in == "response_metadata" else {},
    )
    llm_output = {"model_name": model} if named_in == "llm_output" else None
    return LLMResult(generations=[[ChatGeneration(message=message)]], llm_output=llm_output)


def _record(usage_by_model, **overrides):
    fields = {"ticker": "NVDA", "trade_date": "2026-11-10", "analysts": ["market", "news"],
              "elapsed_seconds": 382.04, "completed": True}
    return usage.build_record({"llm_calls": 17, "tool_calls": 28}, usage_by_model, **{**fields, **overrides})


@pytest.mark.unit
class TestCountingPerModel:
    def test_each_model_keeps_its_own_tokens(self):
        handler = StatsCallbackHandler()
        handler.on_llm_end(_answer("claude-sonnet-5", 1000, 200))
        handler.on_llm_end(_answer("claude-sonnet-5", 500, 100, cache_read=300))
        handler.on_llm_end(_answer("claude-opus-5-5", 4000, 900))

        assert handler.get_usage() == {
            "claude-sonnet-5": {"calls": 2, "input_tokens": 1500, "output_tokens": 300,
                                "cache_read_tokens": 300, "cache_creation_tokens": 0},
            "claude-opus-5-5": {"calls": 1, "input_tokens": 4000, "output_tokens": 900,
                                "cache_read_tokens": 0, "cache_creation_tokens": 0},
        }
        # The live view's totals are unchanged by the per-model count.
        assert handler.get_stats()["tokens_in"] == 5500
        assert handler.get_stats()["tokens_out"] == 1200

    def test_a_model_named_only_in_the_llm_output_is_still_found(self):
        handler = StatsCallbackHandler()
        handler.on_llm_end(_answer("gpt-6-luna", 10, 5, named_in="llm_output"))
        assert list(handler.get_usage()) == ["gpt-6-luna"]

    def test_an_answer_that_names_no_model_is_counted_as_unknown(self):
        handler = StatsCallbackHandler()
        handler.on_llm_end(_answer("", 10, 5, named_in="nowhere"))
        assert handler.get_usage()["unknown"]["input_tokens"] == 10

    def test_an_answer_without_token_counts_adds_nothing(self):
        handler = StatsCallbackHandler()
        handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]]))
        assert handler.get_usage() == {}


@pytest.mark.unit
class TestPricing:
    @pytest.mark.parametrize("model, expected", [
        ("claude-sonnet-5", (2.00, 10.00)),
        ("claude-opus-5-5", (4.00, 20.00)),
        ("claude-opus-5", (5.00, 25.00)),
        ("claude-sonnet-5-20260801", (2.00, 10.00)),     # a dated snapshot of the same model
        ("claude-opus-5-5-20260901", (4.00, 20.00)),
    ])
    def test_a_known_model_has_its_price(self, model, expected):
        assert usage.price_for(model) == expected

    @pytest.mark.parametrize("model", [
        "claude-sonnet-5-7",                 # a later model is not an earlier one's snapshot
        "us.anthropic.claude-opus-5-5",      # a reseller sets its own prices
        "gpt-6-sol",
        "unknown",
    ])
    def test_a_model_with_no_price_on_file_is_not_guessed_at(self, model):
        assert usage.price_for(model) is None
        assert usage.cost_of(model, 1_000_000, 1_000_000) is None

    def test_cost_is_tokens_at_the_per_million_price(self):
        assert usage.cost_of("claude-sonnet-5", 171_726, 29_052) == pytest.approx(0.343452 + 0.29052)


@pytest.mark.unit
class TestTheRecord:
    def test_it_totals_the_models_and_prices_each(self):
        record = _record({
            "claude-sonnet-5": {"calls": 15, "input_tokens": 171_726, "output_tokens": 29_052,
                                "cache_read_tokens": 0, "cache_creation_tokens": 0},
            "claude-opus-5-5": {"calls": 2, "input_tokens": 19_434, "output_tokens": 2_869,
                                "cache_read_tokens": 0, "cache_creation_tokens": 0},
        })
        assert record["input_tokens"] == 191_160
        assert record["output_tokens"] == 31_921
        assert record["models"]["claude-opus-5-5"]["cost_usd"] == pytest.approx(0.135116)
        assert record["cost_usd"] == pytest.approx(0.769088)
        assert record["unpriced_models"] == []
        assert (record["ticker"], record["trade_date"], record["completed"]) == ("NVDA", "2026-11-10", True)
        assert record["elapsed_seconds"] == 382.0
        assert (record["llm_calls"], record["tool_calls"]) == (17, 28)

    def test_a_model_without_a_price_is_named_rather_than_counted_as_free(self):
        record = _record({
            "claude-sonnet-5": {"calls": 1, "input_tokens": 1_000_000, "output_tokens": 0,
                                "cache_read_tokens": 0, "cache_creation_tokens": 0},
            "gpt-6-sol": {"calls": 1, "input_tokens": 5000, "output_tokens": 700,
                          "cache_read_tokens": 0, "cache_creation_tokens": 0},
        })
        assert record["cost_usd"] == pytest.approx(2.0)
        assert record["unpriced_models"] == ["gpt-6-sol"]
        assert record["models"]["gpt-6-sol"]["cost_usd"] is None
        summary = usage.format_summary(record)
        assert "$2.00" in summary and "no price on file for gpt-6-sol" in summary

    def test_the_summary_line_reads_tokens_and_cost(self):
        record = _record({"claude-sonnet-5": {"calls": 1, "input_tokens": 171_726, "output_tokens": 29_052,
                                              "cache_read_tokens": 0, "cache_creation_tokens": 0}})
        assert usage.format_summary(record) == (
            "Usage: 171.7k tokens in, 29.1k out | $0.63 at list prices (claude-sonnet-5 $0.63)"
        )

    def test_a_run_whose_provider_reported_nothing_says_so(self):
        assert "reported no token counts" in usage.format_summary(_record({}))


@pytest.mark.unit
class TestTheLog:
    def test_each_run_adds_one_line_and_earlier_lines_stay(self, tmp_path):
        row = {"calls": 1, "input_tokens": 1000, "output_tokens": 100,
               "cache_read_tokens": 0, "cache_creation_tokens": 0}
        first = _record({"claude-sonnet-5": row})
        second = _record({"claude-sonnet-5": row}, ticker="AAPL", completed=False)

        path = usage.append_record(tmp_path / "logs", first)
        assert usage.append_record(tmp_path / "logs", second) == path
        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["ticker"] for line in lines] == ["NVDA", "AAPL"]
        assert usage.read_log(tmp_path / "logs") == [first, second]

    def test_a_log_that_cannot_be_written_does_not_end_the_run(self, tmp_path):
        blocker = tmp_path / "logs"
        blocker.write_text("a file where the directory should be")
        assert usage.append_record(blocker, _record({})) is None

    def test_a_damaged_line_is_skipped(self, tmp_path):
        good = _record({})
        (tmp_path / usage.USAGE_LOG).write_text(
            "not json\n" + json.dumps(good) + "\n[1, 2]\n", encoding="utf-8"
        )
        assert usage.read_log(tmp_path) == [good]

    def test_no_log_yet_reads_as_no_runs(self, tmp_path):
        assert usage.read_log(tmp_path) == []

    def test_days_are_totalled_and_failed_runs_counted(self):
        def run(finished_at, cost, completed=True, unpriced=()):
            return {"finished_at": finished_at, "completed": completed, "input_tokens": 1000,
                    "output_tokens": 100, "cost_usd": cost, "unpriced_models": list(unpriced)}

        totals = usage.daily_totals([
            run("2026-11-10T19:32:16-08:00", 0.77),
            run("2026-11-10T20:05:00-08:00", 0.30, completed=False),
            run("2026-11-11T09:30:00-08:00", 0.59, unpriced=["gpt-6-sol"]),
        ])
        assert totals == [
            {"day": "2026-11-10", "runs": 2, "failed": 1, "input_tokens": 2000, "output_tokens": 200,
             "cost_usd": pytest.approx(1.07), "unpriced": False},
            {"day": "2026-11-11", "runs": 1, "failed": 0, "input_tokens": 1000, "output_tokens": 100,
             "cost_usd": pytest.approx(0.59), "unpriced": True},
        ]


@pytest.mark.unit
class TestARunIsLogged:
    def test_a_finished_run_adds_its_line_and_says_what_it_used(self, tmp_path, monkeypatch):
        from tests.test_cli_memory_log import _FakeGraph, _run_cli

        buffer = _run_cli(monkeypatch, tmp_path, _FakeGraph())

        (record,) = usage.read_log(tmp_path / "results")
        assert (record["ticker"], record["trade_date"], record["completed"]) == ("NVDA", "2026-01-10", True)
        assert record["analysts"] == ["market"]
        assert any(kind == "System" and text.startswith("Usage:") for _, kind, text in buffer.messages)

    def test_a_run_that_fails_halfway_is_logged_as_not_completed(self, tmp_path, monkeypatch):
        from tests.test_cli_memory_log import _FakeGraph, _run_cli

        class _FailsMidway(_FakeGraph):
            def stream_run(self, graph_input, **kwargs):
                yield [], {"messages": [], "market_report": "M"}
                raise RuntimeError("the provider went away")

        with pytest.raises(RuntimeError, match="went away"):
            _run_cli(monkeypatch, tmp_path, _FailsMidway())

        (record,) = usage.read_log(tmp_path / "results")
        assert record["completed"] is False


@pytest.mark.unit
class TestTheUsageCommand:
    def test_it_lists_each_day(self, tmp_path, monkeypatch):
        from cli import main

        row = {"calls": 1, "input_tokens": 171_726, "output_tokens": 29_052,
               "cache_read_tokens": 0, "cache_creation_tokens": 0}
        record = {**_record({"claude-sonnet-5": row}), "finished_at": "2026-11-10T19:32:16-08:00"}
        usage.append_record(tmp_path, record)
        monkeypatch.setitem(main.DEFAULT_CONFIG, "results_dir", str(tmp_path))

        result = CliRunner().invoke(main.app, ["usage"])
        assert result.exit_code == 0
        assert "2026-11-10" in result.output
        assert "171,726" in result.output and "$0.63" in result.output

    def test_it_says_so_when_nothing_has_run(self, tmp_path, monkeypatch):
        from cli import main

        monkeypatch.setitem(main.DEFAULT_CONFIG, "results_dir", str(tmp_path))
        result = CliRunner().invoke(main.app, ["usage"])
        assert result.exit_code == 0
        assert "No runs" in result.output
