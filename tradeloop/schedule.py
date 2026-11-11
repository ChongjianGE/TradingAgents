"""When the loop runs. The exchange's calendar gives each day's open and close; from
them come the day's ticks:

- ``premarket``: one analysis before the open, so the day starts with a plan;
- ``core``: every interval while the market is open, the only ticks that trade;
- ``postclose``: one analysis after the close, when results and filings land;
- ``offhours``: a rule-only watch on a fixed grid while the market is closed.

Everything here is arithmetic on times, so a half day or a holiday needs no
special case: the calendar already says when that day opens and closes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from tradeloop.config import LoopConfig

KINDS = ("premarket", "core", "postclose", "offhours")
# An off-hours tick keeps this far from the session's own first and last ticks.
_QUIET_MARGIN = timedelta(minutes=30)


@dataclass(frozen=True)
class Session:
    """One trading day: when the market opens and closes, in exchange time."""

    day: date
    open: datetime
    close: datetime


@dataclass(frozen=True, order=True)
class Tick:
    at: datetime
    kind: str

    @property
    def tick_id(self) -> str:
        return f"{self.at:%Y%m%dT%H%M}-{self.kind}"


def sessions_from_calendar(rows: list[dict], tz: ZoneInfo) -> list[Session]:
    """Sessions from Alpaca calendar rows (``date``, ``open`` and ``close`` as HH:MM), oldest first."""
    sessions = []
    for row in rows:
        day = date.fromisoformat(row["date"])
        sessions.append(Session(day, datetime.combine(day, time.fromisoformat(row["open"]), tz),
                                datetime.combine(day, time.fromisoformat(row["close"]), tz)))
    return sorted(sessions, key=lambda s: s.day)


def session_ticks(session: Session, cfg: LoopConfig) -> list[Tick]:
    """The ticks of one trading day, in order."""
    ticks = [Tick(session.open - timedelta(minutes=cfg.premarket_lead_minutes), "premarket")]
    moment = session.open + timedelta(minutes=cfg.open_offset_minutes)
    last = session.close - timedelta(minutes=cfg.last_tick_before_close_minutes)
    while moment <= last:
        ticks.append(Tick(moment, "core"))
        moment += timedelta(minutes=cfg.interval_minutes)
    ticks.append(Tick(session.close + timedelta(minutes=cfg.postclose_delay_minutes), "postclose"))
    return ticks


def _busy(moment: datetime, sessions: list[Session], cfg: LoopConfig) -> bool:
    """Whether ``moment`` falls in a session's own span, from before its first tick to after its last."""
    for session in sessions:
        first = session.open - timedelta(minutes=cfg.premarket_lead_minutes) - _QUIET_MARGIN
        last = session.close + timedelta(minutes=cfg.postclose_delay_minutes) + _QUIET_MARGIN
        if first <= moment <= last:
            return True
    return False


def next_offhours(after: datetime, sessions: list[Session], cfg: LoopConfig) -> Tick:
    """The first watch tick strictly after ``after``: on the interval grid, outside every session."""
    step = timedelta(minutes=cfg.offhours_interval_minutes)
    midnight = after.replace(hour=0, minute=0, second=0, microsecond=0)
    moment = midnight + step * (int((after - midnight) / step) + 1)
    while _busy(moment, sessions, cfg):
        moment += step
    return Tick(moment, "offhours")


def next_tick(after: datetime, sessions: list[Session], cfg: LoopConfig) -> Tick:
    """The first tick of any kind strictly after ``after``."""
    tz = ZoneInfo(cfg.timezone)
    after = after.astimezone(tz)
    upcoming = [tick for session in sessions for tick in session_ticks(session, cfg) if tick.at > after]
    return min([*upcoming, next_offhours(after, sessions, cfg)])


def session_on(moment: datetime, sessions: list[Session]) -> Session | None:
    """The trading day ``moment`` falls on, when it is one."""
    return next((s for s in sessions if s.day == moment.astimezone(s.open.tzinfo).date()), None)


def is_open(moment: datetime, sessions: list[Session]) -> bool:
    session = session_on(moment, sessions)
    return session is not None and session.open <= moment < session.close


class Scheduler:
    """The calendar, fetched once a day, and the ticks that follow from it."""

    def __init__(self, fetch_calendar: Callable[[str, str], list[dict]], cfg: LoopConfig):
        self.cfg, self.tz = cfg, ZoneInfo(cfg.timezone)
        self._fetch = fetch_calendar
        self._sessions: list[Session] = []
        self._fetched_for: date | None = None

    def sessions(self, now: datetime) -> list[Session]:
        """Trading days around ``now``: a week back, for marking outcomes, and two ahead."""
        today = now.astimezone(self.tz).date()
        if self._fetched_for != today:
            rows = self._fetch((today - timedelta(days=10)).isoformat(), (today + timedelta(days=14)).isoformat())
            self._sessions, self._fetched_for = sessions_from_calendar(rows, self.tz), today
        return self._sessions

    def next_tick(self, after: datetime) -> Tick:
        return next_tick(after, self.sessions(after), self.cfg)

    def is_open(self, moment: datetime) -> bool:
        return is_open(moment, self.sessions(moment))

    def session_on(self, moment: datetime) -> Session | None:
        return session_on(moment, self.sessions(moment))
