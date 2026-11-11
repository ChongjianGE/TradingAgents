"""The loop's command line: set it up, check it, run one tick or run it for good, and
build the pages that show what it did."""

from __future__ import annotations

import os
import subprocess
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import typer
from rich.console import Console

from tradeloop import dashboard, secrets, timeutil
from tradeloop.alpaca import AlpacaClient, AlpacaError, load_credentials
from tradeloop.analysis import AnalystPool
from tradeloop.broker import make_broker
from tradeloop.config import CONFIG_TEMPLATE, LoopConfig, default_data_dir
from tradeloop.engine import Engine, TickReport, run_forever
from tradeloop.schedule import KINDS, Scheduler, Tick
from tradeloop.store import Store

# Lines are not wrapped: a wrapped line cannot be copied or searched as one.
console = Console(soft_wrap=True)
app = typer.Typer(name="tradeloop", help="A trading loop around TradingAgents, on a paper account.",
                  no_args_is_help=True, add_completion=False)

ConfigOption = Annotated[Path | None, typer.Option(
    "--config", "-c", help="Config file (TOML or JSON). Default: config.toml in the data directory, when it exists.")]


@dataclass
class Loop:
    """The loop's parts, wired once."""

    cfg: LoopConfig
    store: Store
    client: AlpacaClient
    engine: Engine


def _config(path: Path | None) -> LoopConfig:
    """The config, with the keys of its env file put where the pipeline reads them."""
    try:
        cfg = LoopConfig.load(path)
        secrets.load_env_file(cfg.env_file)
        return cfg
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None


def publish_dashboard(cfg: LoopConfig, store: Store, days: int | None = 1) -> Path:
    """Build the pages, then run the configured command on them. A command that fails is logged, not raised."""
    index = dashboard.build(cfg, store, days=days)
    if cfg.after_dashboard:
        command = [part.replace("{dir}", str(cfg.dashboard_dir)) for part in cfg.after_dashboard]
        problem = None
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if done.returncode != 0:
                problem = f"exit code {done.returncode}: {(done.stderr or done.stdout).strip()[-300:]}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            problem = type(exc).__name__
        if problem:
            store.log("warning", "dashboard", f"after_dashboard command failed ({problem})",
                      at=timeutil.iso(timeutil.utcnow()))
    return index


def _client(cfg: LoopConfig) -> AlpacaClient:
    try:
        key, secret = load_credentials()
    except AlpacaError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    return AlpacaClient(key, secret)


def build_loop(cfg: LoopConfig, client: AlpacaClient | None = None, analysts=None) -> Loop:
    """Wire the store, broker, market data, scheduler, analysts and engine for ``cfg``."""
    client = client or _client(cfg)
    store = Store(cfg.db_path)
    engine = Engine(cfg, store, make_broker(cfg, store, client), client, analysts or AnalystPool(cfg),
                    Scheduler(client.calendar, cfg), now=lambda: timeutil.utcnow(),
                    after_tick=lambda: publish_dashboard(cfg, store))
    return Loop(cfg, store, client, engine)


def _print_report(report: TickReport) -> None:
    colour = {"ok": "green", "error": "red", "skipped": "yellow"}[report.status]
    console.print(f"[{colour}]{report.tick_id}  {report.status}[/{colour}]  {report.note}", highlight=False)


@app.command()
def init(force: Annotated[bool, typer.Option("--force", help="Replace an existing config file.")] = False):
    """Write a config file with the defaults to the data directory."""
    path = default_data_dir() / "config.toml"
    if path.exists() and not force:
        console.print(f"[yellow]{path} already exists; use --force to replace it.[/yellow]")
        raise typer.Exit(code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    console.print(f"Wrote {path}. Edit it, then run: tradeloop check")


@app.command()
def check(config: ConfigOption = None,
          llm: Annotated[bool, typer.Option("--llm", help="Also send each model one tiny request.")] = False):
    """Check that everything the loop needs is in place. Reads only; places no order."""
    cfg = _config(config)
    failures = 0

    def line(ok: bool, what: str, detail: str = "") -> None:
        nonlocal failures
        failures += 0 if ok else 1
        mark = "[green]ok  [/green]" if ok else "[red]FAIL[/red]"
        console.print(f"{mark} {what}" + (f"  [dim]{detail}[/dim]" if detail else ""), highlight=False)

    line(True, "config", f"{len(cfg.tickers)} tickers, every {cfg.interval_minutes} min, execution={cfg.execution}, "
                         f"data in {cfg.data_dir}")
    try:
        client = AlpacaClient(*load_credentials())
        account, clock = client.account(), client.clock()
        line(account.get("status") == "ACTIVE" and not account.get("trading_blocked"), "Alpaca paper account",
             f"status {account.get('status')}, equity {float(account.get('equity', 0)):,.2f}, "
             f"market {'open' if clock.get('is_open') else 'closed'}")
        quotes = client.quotes(cfg.tickers)
        missing = [t for t in cfg.tickers if t not in quotes]
        line(not missing, "prices for every ticker",
             f"no price for {', '.join(missing)}" if missing else ", ".join(f"{t} {q.price:g}" for t, q in quotes.items()))
        today = timeutil.utcnow().astimezone(ZoneInfo(cfg.timezone)).date().isoformat()
        session = client.calendar(today, today)
        line(True, "exchange calendar", f"today {session[0]['open']} to {session[0]['close']}" if session
             else "no session today")
    except (AlpacaError, ValueError) as exc:
        line(False, "Alpaca", str(exc))

    from tradingagents.llm_clients import tier_provider
    from tradingagents.llm_clients.api_key_env import get_api_key_env

    ta = cfg.tradingagents_config()
    for tier in ("quick", "deep"):
        provider, model = tier_provider(ta, tier), ta[f"{tier}_think_llm"]
        env = get_api_key_env(provider)
        line(env is None or bool(os.environ.get(env)), f"{tier} model {provider}/{model}",
             "no key needed" if env is None else f"{env} {'is set' if os.environ.get(env) else 'is not set'}")
        if llm and (env is None or os.environ.get(env)):
            try:
                from tradingagents.llm_clients import create_tier_client

                create_tier_client(ta, tier, max_tokens=64).get_llm().invoke("Reply with the single word: ok")
                line(True, f"{tier} model answers")
            except Exception as exc:  # noqa: BLE001 - whatever the provider raises is the finding
                line(False, f"{tier} model answers", f"{type(exc).__name__}: {str(exc)[:160]}")
    for name, purpose in secrets.OPTIONAL_DATA_KEYS.items():
        line(True, f"optional: {name}", f"set ({purpose})" if os.environ.get(name) else f"not set; no {purpose}")
    try:
        with Store(cfg.db_path):
            line(True, "store", str(cfg.db_path))
    except Exception as exc:  # noqa: BLE001
        line(False, "store", f"{type(exc).__name__}: {exc}")
    if failures:
        raise typer.Exit(code=1)


@app.command()
def status(config: ConfigOption = None):
    """Show the book, each ticker's standing decision, and when the next tick is."""
    loop = build_loop(_config(config))
    cfg, store, engine = loop.cfg, loop.store, loop.engine
    tz, now = ZoneInfo(cfg.timezone), timeutil.utcnow()
    upcoming = engine.scheduler.next_tick(now)
    console.print(f"Now {now.astimezone(tz):%Y-%m-%d %H:%M %Z}; market "
                  f"{'open' if engine.scheduler.is_open(now) else 'closed'}; next tick "
                  f"{upcoming.at:%m-%d %H:%M} ({upcoming.kind})", highlight=False)
    quotes = loop.client.quotes(cfg.tickers)
    prices = {t: q.price for t, q in quotes.items()}
    account = engine.broker.account(prices)
    spent = store.llm_cost_since(engine._day_start(now))
    console.print(f"Book ({engine.broker.name}): equity {account.equity:,.2f}, cash {account.cash:,.2f}; "
                  f"model spend today {spent:,.2f} of {cfg.daily_llm_budget_usd:,.2f}", highlight=False)
    for ticker in cfg.tickers:
        decision = store.latest_decision(ticker)
        held = account.qty(ticker)
        stance = "no decision yet" if decision is None else (
            f"{decision['rating']} at {timeutil.parse(decision['started_at']).astimezone(tz):%m-%d %H:%M}"
            + (f", target {decision['target_weight']:g}" if decision["target_weight"] is not None else "")
            + (f", stop {decision['stop_loss']:g}" if decision["stop_loss"] is not None else ""))
        console.print(f"  {ticker:<6} {prices.get(ticker, float('nan')):>10.2f}  held {held:>8g}  {stance}",
                      highlight=False)
    store.close()


@app.command()
def tick(config: ConfigOption = None,
         kind: Annotated[str, typer.Option(help=f"The kind of tick to run: {', '.join(KINDS)}.")] = "core"):
    """Run one tick now. Orders are placed only by a core tick while the market is open."""
    if kind not in KINDS:
        console.print(f"[red]kind must be one of {', '.join(KINDS)}[/red]")
        raise typer.Exit(code=1)
    loop = build_loop(_config(config))
    now = timeutil.utcnow().astimezone(ZoneInfo(loop.cfg.timezone)).replace(second=0, microsecond=0)
    report = loop.engine.run_tick(Tick(now, kind))
    _print_report(report)
    console.print(f"Dashboard: {loop.cfg.dashboard_dir / 'index.html'}", highlight=False)
    loop.store.close()
    if report.status == "error":
        raise typer.Exit(code=1)


@app.command()
def run(config: ConfigOption = None):
    """Run the loop until interrupted: every tick as its time comes."""
    loop = build_loop(_config(config))
    cfg = loop.cfg
    upcoming = loop.engine.scheduler.next_tick(timeutil.utcnow())
    console.print(f"tradeloop: {', '.join(cfg.tickers)}; execution={cfg.execution}; every {cfg.interval_minutes} min "
                  f"while the market is open. Next tick {upcoming.at:%m-%d %H:%M %Z} ({upcoming.kind}). "
                  "Ctrl-C stops it.", highlight=False)
    try:
        run_forever(loop.engine, sleep=time.sleep, on_tick=_print_report)
    except KeyboardInterrupt:
        console.print("Stopped.")
    finally:
        loop.store.close()


@app.command(name="dashboard")
def dashboard_command(config: ConfigOption = None,
                      open_page: Annotated[bool, typer.Option("--open", help="Open it in the browser.")] = False):
    """Build the pages for every day from what the loop has recorded."""
    cfg = _config(config)
    with Store(cfg.db_path) as store:
        index = publish_dashboard(cfg, store, days=None)
    console.print(f"Dashboard: {index}", highlight=False)
    if open_page:
        webbrowser.open(index.as_uri())


if __name__ == "__main__":
    app()
