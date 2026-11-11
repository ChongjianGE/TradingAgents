"""What happened inside one analysis: every model call and every tool call, with the
agent that made it, how long it took and what it used.

TradingAgents' own reports keep each agent's final text. This keeps the steps in
between, which is what shows how a decision was reached: which tools an analyst
called and what came back, how many turns it took, where the time and the tokens went.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from tradeloop import timeutil

# Tool results and model outputs are kept whole up to this size.
MAX_TEXT_CHARS = 60_000


def _agent(metadata: dict | None) -> str:
    """The graph node a call belongs to. An analyst runs in a graph of its own, so
    its calls are named by the outer node (``Market Analyst``), not the inner one (``agent``)."""
    metadata = metadata or {}
    namespace = str(metadata.get("langgraph_checkpoint_ns") or "")
    outer = namespace.split("|", 1)[0].split(":", 1)[0]
    return outer or str(metadata.get("langgraph_node") or "unknown")


def _text(content: Any) -> str:
    """Message content as plain text: a string as is, typed blocks joined by their text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text", "") if isinstance(block, dict) else str(block) for block in content]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _clip(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + f"\n… ({len(text) - MAX_TEXT_CHARS:,} more characters not kept)"


class RunRecorder(BaseCallbackHandler):
    """Collects the steps of the analysis in progress. ``reset`` starts the next one."""

    raise_error = False    # a recording failure must never fail the analysis

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._open: dict[UUID, dict] = {}
            self._steps: list[dict] = []

    # --- model calls
    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or (metadata or {}).get("ls_model_name") or "model"
        chars = sum(len(_text(getattr(m, "content", m))) for batch in messages for m in batch)
        self._begin(run_id, "llm", _agent(metadata), str(model), input_chars=chars)

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs) -> None:
        self._begin(run_id, "llm", _agent(metadata), "model", input_chars=sum(len(p) for p in prompts))

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        fields: dict = {}
        try:
            message = response.generations[0][0].message
        except (IndexError, AttributeError):
            message = None
        if message is not None:
            usage = getattr(message, "usage_metadata", None) or {}
            meta = getattr(message, "response_metadata", None) or {}
            calls = [{"name": c.get("name"), "args": c.get("args")} for c in getattr(message, "tool_calls", None) or []]
            fields = {"input_tokens": usage.get("input_tokens", 0) or 0,
                      "output_tokens": usage.get("output_tokens", 0) or 0,
                      "output": _clip(_text(message.content)), "tool_calls": calls}
            if meta.get("model_name") or meta.get("model"):
                fields["name"] = meta.get("model_name") or meta.get("model")
        self._end(run_id, **fields)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._end(run_id, error=f"{type(error).__name__}: {str(error)[:300]}")

    # --- tool calls
    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, inputs=None, **kwargs) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        # The run's ticker and date are injected by the graph, not chosen by the model.
        args = {k: v for k, v in (inputs or {}).items() if k not in ("state", "trade_date", "symbol", "ticker")}
        # A tool called with no arguments of the model's own is recorded as such, not as the raw input.
        self._begin(run_id, "tool", _agent(metadata), str(name), args=args if inputs is not None else input_str)

    def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        self._end(run_id, output=_clip(_text(getattr(output, "content", output))))

    def on_tool_error(self, error, *, run_id, **kwargs) -> None:
        self._end(run_id, error=f"{type(error).__name__}: {str(error)[:300]}")

    # --- bookkeeping
    def _begin(self, run_id: UUID, kind: str, agent: str, name: str, **fields) -> None:
        with self._lock:
            self._open[run_id] = {"agent": agent, "kind": kind, "name": name, "started_at": timeutil.iso(timeutil.utcnow()),
                                  "_t0": time.monotonic(), "input_tokens": 0, "output_tokens": 0,
                                  "output": "", "error": None, **fields}

    def _end(self, run_id: UUID, **fields) -> None:
        with self._lock:
            step = self._open.pop(run_id, None)
            if step is None:
                return
            step.update(fields)
            step["elapsed_seconds"] = round(time.monotonic() - step.pop("_t0"), 3)
            step["output_chars"] = len(step.get("output") or "")
            step["seq"] = len(self._steps)
            self._steps.append(step)

    # --- what was recorded
    def steps(self) -> list[dict]:
        """The finished steps in the order they ended; a call still running is not among them."""
        with self._lock:
            return [dict(step) for step in self._steps]

    def usage_by_model(self) -> dict[str, dict[str, int]]:
        usage: dict[str, dict[str, int]] = {}
        for step in self.steps():
            if step["kind"] != "llm":
                continue
            row = usage.setdefault(step["name"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            row["calls"] += 1
            row["input_tokens"] += step["input_tokens"]
            row["output_tokens"] += step["output_tokens"]
        return usage
