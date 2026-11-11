"""Keys from a dotenv file, for the loop and for the analysis it runs.

TradingAgents reads its keys from the environment. Someone who keeps theirs in a
file of their own, under names of their own, points ``env_file`` at it: the keys
the pipeline reads are copied into the environment under the names it reads them
by. Only those are taken; whatever else the file holds stays in the file.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

# Each name the loop or TradingAgents reads, and other names the same key is stored under.
_ALIASES: dict[str, tuple[str, ...]] = {
    "APCA_API_KEY_ID": ("ALPACA_PAPER_TRADING_API_KEY", "ALPCA_PAPER_TRADING_API_KEY"),
    "APCA_API_SECRET_KEY": ("ALPACA_PAPER_TRADING_SECRET_KEY", "ALPCA_PAPER_TRADING_SECRET_KEY"),
    "TYPESAFE_API_KEY": ("TYPESAFE_TOKEN",),
    "TYPESAFE_DEFAULT_MODEL": (),
    "TYPESAFE_BASE_URL": (),
    "ALPHA_VANTAGE_API_KEY": ("ALPHAVANTAGE_API_KEY",),
    "FRED_API_KEY": (),
    "SEC_EDGAR_USER_AGENT": (),
}
# Data keys the analysis can use but does not need, for the preflight check to name.
OPTIONAL_DATA_KEYS = {
    "FRED_API_KEY": "macro data for the News Analyst",
    "TYPESAFE_API_KEY": "screening of social posts for the Sentiment Analyst",
    "ALPHA_VANTAGE_API_KEY": "Alpha Vantage as a data vendor",
}


def known_names() -> dict[str, tuple[str, ...]]:
    """Every environment name taken from an env file: the data and broker keys above, and each model provider's key."""
    from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV

    names = dict(_ALIASES)
    for name in PROVIDER_API_KEY_ENV.values():
        if name:
            names.setdefault(name, ())
    return names


def load_env_file(path: str | Path | None, environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Copy the keys the pipeline reads from the dotenv file at ``path`` into the environment.

    A name already set in the environment is left alone. Returns the names set,
    never their values. A missing file is an error: a path that was configured
    and is not there would otherwise look like keys that were never given.
    """
    if path is None:
        return []
    from dotenv import dotenv_values

    file = Path(path).expanduser()
    if not file.is_file():
        raise ValueError(f"env_file {file} does not exist")
    environ = os.environ if environ is None else environ
    values = dotenv_values(file)
    taken = []
    for name, aliases in known_names().items():
        if environ.get(name):
            continue
        for candidate in (name, *aliases):
            value = (values.get(candidate) or "").strip()
            if value:
                environ[name] = value
                taken.append(name)
                break
    return taken
