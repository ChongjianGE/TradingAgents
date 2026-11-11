"""When the loop runs: each day's ticks follow from the exchange calendar."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tradeloop.config import LoopConfig
from tradeloop.schedule import (
    Scheduler,
    Tick,
    is_open,
    next_offhours,
    next_tick,
    session_on,
    session_ticks,
    sessions_from_calendar,
)

ET = ZoneInfo("America/New_York")
CALENDAR = [
    {"date": "2026-11-11", "open": "09:30", "close": "16:00"},      # Wednesday
    {"date": "2026-11-12", "open": "09:30", "close": "16:00"},
    {"date": "2026-11-13", "open": "09:30", "close": "16:00"},      # Friday
    {"date": "2026-11-16", "open": "09:30", "close": "16:00"},      # Monday
    {"date": "2026-11-27", "open": "09:30", "close": "13:00"},      # the half day after Thanksgiving
]
CFG = LoopConfig()
SESSIONS = sessions_from_calendar(CALENDAR, ET)


def et(day, hhmm):
    return datetime.fromisoformat(f"2026-11-{day:02d}T{hhmm}").replace(tzinfo=ET)


def times(ticks, kind=None):
    return [f"{t.at:%H:%M}" for t in ticks if kind is None or t.kind == kind]


@pytest.mark.unit
class TestADaysTicks:
    def test_a_full_day_has_a_premarket_seven_hourly_and_a_postclose_tick(self):
        ticks = session_ticks(SESSIONS[0], CFG)
        assert times(ticks, "premarket") == ["08:45"]
        assert times(ticks, "core") == ["09:35", "10:35", "11:35", "12:35", "13:35", "14:35", "15:35"]
        assert times(ticks, "postclose") == ["16:30"]
        assert ticks == sorted(ticks)

    def test_a_half_day_stops_with_its_close(self):
        ticks = session_ticks(SESSIONS[-1], CFG)
        assert times(ticks, "core") == ["09:35", "10:35", "11:35", "12:35"]
        assert times(ticks, "postclose") == ["13:30"]

    def test_the_interval_is_a_setting(self):
        ticks = session_ticks(SESSIONS[0], LoopConfig(interval_minutes=30))
        assert len(times(ticks, "core")) == 13 and times(ticks, "core")[:3] == ["09:35", "10:05", "10:35"]

    def test_no_tick_falls_in_the_last_minutes_before_the_close(self):
        ticks = session_ticks(SESSIONS[0], LoopConfig(interval_minutes=25, last_tick_before_close_minutes=30))
        assert times(ticks, "core")[-1] <= "15:30"

    def test_a_ticks_id_names_its_exchange_time_and_kind(self):
        assert session_ticks(SESSIONS[0], CFG)[1].tick_id == "20261111T0935-core"
        assert Tick(et(11, "18:00"), "offhours").tick_id == "20261111T1800-offhours"


@pytest.mark.unit
class TestWhatComesNext:
    @pytest.mark.parametrize("after, expected", [
        ((11, "07:30"), ("offhours", (11, "08:00"))),     # the watch runs up to the pre-market analysis
        ((11, "08:00"), ("premarket", (11, "08:45"))),
        ((11, "08:45"), ("core", (11, "09:35"))),         # strictly after: the same tick is not returned
        ((11, "09:35"), ("core", (11, "10:35"))),
        ((11, "15:35"), ("postclose", (11, "16:30"))),
        ((11, "16:30"), ("offhours", (11, "18:00"))),     # the watch resumes once the day's own ticks are done
        ((11, "23:10"), ("offhours", (12, "00:00"))),
        ((13, "16:30"), ("offhours", (13, "18:00"))),     # Friday evening
        ((14, "11:59"), ("offhours", (14, "12:00"))),     # Saturday: the watch runs all day
        ((16, "07:59"), ("offhours", (16, "08:00"))),     # Monday morning, then the pre-market analysis
        ((16, "08:00"), ("premarket", (16, "08:45"))),
    ])
    def test_the_next_tick(self, after, expected):
        tick = next_tick(et(*after), SESSIONS, CFG)
        assert (tick.kind, tick.at) == (expected[0], et(*expected[1]))

    def test_walking_a_trading_day_visits_every_tick_once_in_order(self):
        moment, seen = et(11, "00:30"), []
        while moment < et(12, "00:00"):
            tick = next_tick(moment, SESSIONS, CFG)
            seen.append(tick)
            moment = tick.at
        kinds = [t.kind for t in seen if t.at < et(12, "00:00")]
        assert kinds == ["offhours"] * 8 + ["premarket"] + ["core"] * 7 + ["postclose"] + ["offhours"] * 6
        assert [t.at for t in seen] == sorted({t.at for t in seen})

    def test_the_watch_keeps_clear_of_the_sessions_own_ticks(self):
        assert next_offhours(et(11, "08:10"), SESSIONS, CFG).at == et(11, "18:00")
        assert next_offhours(et(11, "16:59"), SESSIONS, CFG).at == et(11, "18:00")

    def test_a_time_in_another_zone_is_read_as_the_same_instant(self):
        utc = et(11, "09:35").astimezone(ZoneInfo("UTC"))
        assert next_tick(utc, SESSIONS, CFG).at == et(11, "10:35")


@pytest.mark.unit
class TestSessions:
    def test_the_market_is_open_from_the_open_to_just_before_the_close(self):
        assert is_open(et(11, "09:30"), SESSIONS) and is_open(et(11, "15:59"), SESSIONS)
        assert not is_open(et(11, "09:29"), SESSIONS) and not is_open(et(11, "16:00"), SESSIONS)
        assert not is_open(et(14, "12:00"), SESSIONS)                 # Saturday
        assert not is_open(et(27, "13:30"), SESSIONS)                 # after the half day's close

    def test_a_day_is_found_by_its_exchange_date(self):
        assert session_on(et(11, "23:59"), SESSIONS).day.isoformat() == "2026-11-11"
        assert session_on(et(14, "10:00"), SESSIONS) is None
        late_utc = et(11, "23:00").astimezone(ZoneInfo("UTC"))        # already the 12th in UTC
        assert session_on(late_utc, SESSIONS).day.isoformat() == "2026-11-11"

    def test_calendar_rows_come_back_sorted(self):
        sessions = sessions_from_calendar(list(reversed(CALENDAR)), ET)
        assert [s.day.day for s in sessions] == [11, 12, 13, 16, 27]
        assert sessions[-1].close == et(27, "13:00")


@pytest.mark.unit
class TestTheScheduler:
    def test_the_calendar_is_fetched_once_a_day(self):
        asked = []
        scheduler = Scheduler(lambda start, end: asked.append((start, end)) or CALENDAR, CFG)

        assert scheduler.next_tick(et(11, "08:00")).kind == "premarket"
        assert scheduler.is_open(et(11, "10:00")) and scheduler.session_on(et(11, "10:00")).close == et(11, "16:00")
        assert asked == [("2026-11-01", "2026-11-25")]                # ten days back, two weeks ahead

        scheduler.next_tick(et(12, "08:00"))
        assert len(asked) == 2

    def test_a_calendar_that_cannot_be_fetched_is_raised_not_guessed(self):
        def fail(start, end):
            raise RuntimeError("calendar unavailable")

        with pytest.raises(RuntimeError, match="calendar unavailable"):
            Scheduler(fail, CFG).next_tick(et(11, "08:00") + timedelta(minutes=1))
