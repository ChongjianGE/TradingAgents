import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        # Tokens per model as the provider named it, for pricing a run: the
        # quick and deep tiers bill at different rates.
        self.by_model: dict[str, dict[str, int]] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from LLM response."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        if usage_metadata:
            tokens_in = usage_metadata.get("input_tokens", 0) or 0
            tokens_out = usage_metadata.get("output_tokens", 0) or 0
            cached = usage_metadata.get("input_token_details") or {}
            model = _model_name(generation.message, response)
            with self._lock:
                self.tokens_in += tokens_in
                self.tokens_out += tokens_out
                row = self.by_model.setdefault(model, {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cache_creation_tokens": 0,
                })
                row["calls"] += 1
                row["input_tokens"] += tokens_in
                row["output_tokens"] += tokens_out
                row["cache_read_tokens"] += cached.get("cache_read", 0) or 0
                row["cache_creation_tokens"] += cached.get("cache_creation", 0) or 0

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            }

    def get_usage(self) -> dict[str, dict[str, int]]:
        """Tokens per model, for the calls whose provider reported a count."""
        with self._lock:
            return {model: dict(row) for model, row in self.by_model.items()}


def _model_name(message: AIMessage, response: LLMResult) -> str:
    """The model that answered, as the provider reports it."""
    metadata = getattr(message, "response_metadata", None) or {}
    output = getattr(response, "llm_output", None) or {}
    return (metadata.get("model_name") or metadata.get("model")
            or output.get("model_name") or output.get("model") or "unknown")
