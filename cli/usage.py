"""What a run used and cost: tokens per model, priced, and kept in a log.

Every CLI run appends one line to ``usage_log.jsonl`` under ``results_dir``,
whether it finished or not: a run that failed halfway was still billed.

Token counts are the provider's own, so reasoning tokens are in the output
count. Cost is those counts at the list prices below; a model with no price on
file is counted but not priced. All input is priced at the plain input rate, so
a provider that bills cached input differently will differ from this figure.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

USAGE_LOG = "usage_log.jsonl"

# USD per million tokens (input, output): Anthropic's first-party API list
# prices as of 2026-11-03. Resellers (Bedrock, Vertex, OpenRouter) set their own.
PRICES_AS_OF = "2026-11-03"
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.00, 40.00),
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def price_for(model: str) -> tuple[float, float] | None:
    """The (input, output) price of ``model``, or None when there is none on file.

    A dated snapshot (``claude-sonnet-5-20260801``) takes its model's price. Any
    other suffix is a different model, and is not guessed at.
    """
    if model in PRICES:
        return PRICES[model]
    for name, price in PRICES.items():
        if re.fullmatch(re.escape(name) + r"-\d{8}", model):
            return price
    return None


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """USD for these tokens on ``model``, or None when it has no price on file."""
    price = price_for(model)
    if price is None:
        return None
    return input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1]


def build_record(stats: dict, usage: dict, *, ticker: str, trade_date: str, analysts: list[str],
                 elapsed_seconds: float, completed: bool) -> dict:
    """One run's usage: what ran, the tokens per model, and their cost.

    ``cost_usd`` sums the models that have a price; ``unpriced_models`` names
    the rest, so a partial figure is never read as the whole bill.
    """
    models, unpriced, total = {}, [], 0.0
    for model, row in sorted(usage.items()):
        cost = cost_of(model, row["input_tokens"], row["output_tokens"])
        models[model] = {**row, "cost_usd": None if cost is None else round(cost, 6)}
        if cost is None:
            unpriced.append(model)
        else:
            total += cost
    return {
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ticker": ticker,
        "trade_date": trade_date,
        "completed": completed,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "analysts": list(analysts),
        "llm_calls": stats.get("llm_calls", 0),
        "tool_calls": stats.get("tool_calls", 0),
        "input_tokens": sum(row["input_tokens"] for row in usage.values()),
        "output_tokens": sum(row["output_tokens"] for row in usage.values()),
        "cost_usd": round(total, 6),
        "unpriced_models": unpriced,
        "models": models,
    }


def append_record(results_dir, record: dict) -> Path | None:
    """Add ``record`` to the usage log; the log's path, or None when it could not be written.

    Never raises: the log is bookkeeping, and losing a line must not lose a run.
    """
    path = Path(results_dir) / USAGE_LOG
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return None
    return path


def _tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def format_summary(record: dict) -> str:
    """One line on what the run used, for the end of a run and the run log."""
    if not record["models"]:
        return "Usage: the provider reported no token counts for this run"
    line = f"Usage: {_tokens(record['input_tokens'])} tokens in, {_tokens(record['output_tokens'])} out"
    priced = [(m, row["cost_usd"]) for m, row in record["models"].items() if row["cost_usd"] is not None]
    if priced:
        parts = ", ".join(f"{model} ${cost:.2f}" for model, cost in priced)
        line += f" | ${record['cost_usd']:.2f} at list prices ({parts})"
    if record["unpriced_models"]:
        line += f" | no price on file for {', '.join(record['unpriced_models'])}"
    return line


def read_log(results_dir) -> list[dict]:
    """The usage log's records, oldest first; a line that is not a record is skipped."""
    path = Path(results_dir) / USAGE_LOG
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "finished_at" in record:
            records.append(record)
    return records


def daily_totals(records: list[dict]) -> list[dict]:
    """Runs, tokens and cost per calendar day (the day each run finished, local time)."""
    days: dict[str, dict] = defaultdict(lambda: {
        "runs": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "unpriced": False,
    })
    for record in records:
        day = days[str(record["finished_at"])[:10]]
        day["runs"] += 1
        day["failed"] += 0 if record.get("completed") else 1
        day["input_tokens"] += record.get("input_tokens", 0)
        day["output_tokens"] += record.get("output_tokens", 0)
        day["cost_usd"] += record.get("cost_usd") or 0.0
        day["unpriced"] = day["unpriced"] or bool(record.get("unpriced_models"))
    return [{"day": name, **days[name]} for name in sorted(days)]
