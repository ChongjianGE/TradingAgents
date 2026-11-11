"""Settings of the trading loop: what to trade, when to run, and the limits it keeps to."""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingagents.dataflows.symbols import safe_ticker_component

RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
ANALYSTS = ("market", "social", "news", "fundamentals")
KEEP = "keep"   # a rating that leaves the position as it is


def default_data_dir() -> Path:
    return Path(os.getenv("TRADELOOP_DATA_DIR") or Path.home() / ".tradingagents" / "tradeloop")


class LoopConfig(BaseModel):
    """Every knob of the loop. Unknown keys are refused, so a typo cannot pass as a setting."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # --- what to analyse
    tickers: list[str] = ["NVDA", "CSCO", "HD", "DIS", "GEV"]
    analysts: list[str] = list(ANALYSTS)
    # Overrides for the TradingAgents config (models, provider, rounds, language).
    llm: dict = Field(default_factory=dict)
    # Analyses run at once; None runs every ticker at the same time.
    max_workers: int | None = None

    # --- when to run (times are the exchange's)
    timezone: str = "America/New_York"
    interval_minutes: int = 60            # between ticks while the market is open
    open_offset_minutes: int = 5          # first tick this long after the open
    last_tick_before_close_minutes: int = 20
    premarket_lead_minutes: int = 45      # the pre-market analysis, this long before the open
    postclose_delay_minutes: int = 30     # the post-close analysis, this long after the close
    offhours_interval_minutes: int = 60   # the rule-only watch while the market is closed

    # --- when to call the model
    move_trigger_pct: float = 1.5         # price move since the last analysis
    news_trigger_count: int = 8           # new headlines since the last analysis: a burst, not the daily flow
    max_stale_minutes: int = 180          # longest a decision stands during the session
    offhours_max_analyses: int = 1        # per ticker, per night
    daily_llm_budget_usd: float = 60.0

    # --- from decision to position
    # The fraction of a full position each rating stands for; "keep" changes nothing.
    rating_weights: dict[str, float | Literal["keep"]] = {
        "Buy": 1.0, "Overweight": 0.6, "Hold": KEEP, "Underweight": 0.3, "Sell": 0.0,
    }
    max_position_weight: float = 0.2      # a full position, as a share of equity
    max_gross_exposure: float = 1.0       # all positions together; 1.0 keeps to cash
    allow_fractional: bool = False

    # --- limits
    confirm_rounds: int = 2               # analyses that must agree before a position grows
    daily_loss_limit_pct: float = 2.0     # past this the day only reduces
    min_trade_notional: float = 200.0     # smaller changes are not worth their cost
    min_edge_to_cost: float = 3.0         # expected gain over a round trip's cost
    no_trade_open_minutes: int = 5
    no_trade_close_minutes: int = 10
    max_decision_age_hours: float = 72.0  # an older decision no longer moves the book

    # --- cost model, used for sizing checks and for the simulated book
    half_spread_bps: float = 2.0
    slippage_bps: float = 1.0
    sell_fee_per_share: float = 0.0       # regulatory fees; set from the broker's schedule
    limit_offset_bps: float = 10.0        # how far past the last price a limit order reaches

    # --- where orders go
    # "simulate" keeps a book of its own and sends nothing; "alpaca_paper" sends
    # orders to the Alpaca paper account.
    execution: Literal["simulate", "alpaca_paper"] = "simulate"
    sim_starting_cash: float = 100_000.0
    # The run recorded in TradingAgents' own memory log: one a day.
    memory_record_kind: Literal["premarket", "postclose", "none"] = "postclose"

    # --- files
    data_dir: Path = Field(default_factory=default_data_dir)
    # A dotenv file holding keys that are not in the environment: the Alpaca paper
    # keys, and any key the analysis reads (a model provider's, FRED, TypeSafe).
    env_file: Path | None = None
    # A command run after the dashboard is built, as a list of arguments with
    # "{dir}" standing for the dashboard folder: to copy the pages somewhere.
    after_dashboard: list[str] = Field(default_factory=list)

    @field_validator("tickers")
    @classmethod
    def _tickers(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(safe_ticker_component(t.strip().upper()) for t in value))
        if not cleaned:
            raise ValueError("name at least one ticker")
        return cleaned

    @field_validator("analysts")
    @classmethod
    def _analysts(cls, value: list[str]) -> list[str]:
        unknown = [a for a in value if a not in ANALYSTS]
        if unknown or not value:
            raise ValueError(f"analysts must be chosen from {', '.join(ANALYSTS)}")
        return [a for a in ANALYSTS if a in value]

    @field_validator("data_dir", "env_file")
    @classmethod
    def _expand(cls, value: Path | None) -> Path | None:
        return None if value is None else Path(value).expanduser()

    @model_validator(mode="after")
    def _consistent(self) -> LoopConfig:
        if set(self.rating_weights) != set(RATINGS):
            raise ValueError(f"rating_weights needs exactly these ratings: {', '.join(RATINGS)}")
        for rating, weight in self.rating_weights.items():
            if weight != KEEP and not 0.0 <= weight <= 1.0:
                raise ValueError(f"rating_weights[{rating}] must be between 0 and 1, or \"{KEEP}\"")
        positive = ("interval_minutes", "offhours_interval_minutes", "confirm_rounds",
                    "max_position_weight", "max_gross_exposure", "sim_starting_cash")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.interval_minutes < 5:
            raise ValueError("interval_minutes below 5 leaves no time for an analysis")
        if self.max_position_weight > 1 or self.max_gross_exposure > 1:
            raise ValueError("max_position_weight and max_gross_exposure cannot exceed 1: the loop does not use margin")
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        return self

    # --- derived
    @property
    def db_path(self) -> Path:
        return self.data_dir / "tradeloop.db"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def dashboard_dir(self) -> Path:
        return self.data_dir / "dashboard"

    @property
    def workers(self) -> int:
        return self.max_workers or len(self.tickers)

    @property
    def round_trip_cost(self) -> float:
        """What buying and selling again costs, as a fraction of the amount traded."""
        return 2 * (self.half_spread_bps + self.slippage_bps) / 10_000

    def tradingagents_config(self) -> dict:
        """The TradingAgents config for this loop's analyses: its defaults, then ``llm``."""
        from tradingagents.default_config import build_default_config

        config = build_default_config()
        config.update(self.llm)
        return config

    @classmethod
    def load(cls, path: str | Path | None = None) -> LoopConfig:
        """Read a config file (TOML or JSON); with no path, ``config.toml`` in the data
        directory when it exists, else the defaults."""
        if path is None:
            candidate = default_data_dir() / "config.toml"
            if not candidate.is_file():
                return cls()
            path = candidate
        path = Path(path).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text) if path.suffix == ".json" else tomllib.loads(text)
        except (OSError, ValueError) as exc:
            raise ValueError(f"config file {path} is not usable: {exc}") from exc
        return cls.model_validate(data)


CONFIG_TEMPLATE = '''\
# tradeloop settings. Every key is optional; what is left out keeps its default.

tickers = ["NVDA", "CSCO", "HD", "DIS", "GEV"]
analysts = ["market", "social", "news", "fundamentals"]

# "simulate" keeps a book of its own and sends nothing to a broker.
# "alpaca_paper" sends orders to the Alpaca paper account.
execution = "simulate"
sim_starting_cash = 100000.0

# Minutes between ticks while the market is open.
interval_minutes = 60

# A full position as a share of equity, and all positions together (1.0 = cash only).
max_position_weight = 0.2
max_gross_exposure = 1.0

# Model spend allowed per day, in USD. Past it the loop stops calling the model.
daily_llm_budget_usd = 60.0

# A dotenv file with keys that are not in the environment: the Alpaca paper keys,
# and any key the analysis reads (a model provider's, FRED, TypeSafe).
# env_file = "~/.config/secrets/tokens.env"

# A command run after the dashboard is built; "{dir}" is the dashboard folder.
# after_dashboard = ["rclone", "copy", "{dir}", "remote:bucket/tradeloop"]

# The fraction of a full position each rating stands for; "keep" changes nothing.
[rating_weights]
Buy = 1.0
Overweight = 0.6
Hold = "keep"
Underweight = 0.3
Sell = 0.0

# Overrides for the TradingAgents config.
[llm]
# llm_provider = "anthropic"
# quick_think_llm = "claude-sonnet-5"
# deep_think_llm = "claude-opus-5-5"
# output_language = "Chinese"
'''
