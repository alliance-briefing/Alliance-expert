"""Calendrier de travail et heures en Europe/Paris (changement d'heure compris)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
MONTREAL = ZoneInfo("America/Toronto")  # même fuseau que Montréal, disponible partout

COLLECT_TIME = time(8, 30, 2)  # heure de la collecte du matin (cf. exemple du contrat A)

# Jours fériés français : suffisant pour 2026-2027, étendre si besoin.
FRENCH_HOLIDAYS = {
    date(2026, 1, 1),
    date(2026, 4, 6),
    date(2026, 5, 1),
    date(2026, 5, 8),
    date(2026, 5, 14),
    date(2026, 5, 25),
    date(2026, 7, 14),
    date(2026, 8, 15),
    date(2026, 11, 1),
    date(2026, 11, 11),
    date(2026, 12, 25),
    date(2027, 1, 1),
    date(2027, 3, 29),
    date(2027, 5, 1),
    date(2027, 5, 6),
    date(2027, 5, 8),
    date(2027, 5, 17),
    date(2027, 7, 14),
    date(2027, 8, 15),
    date(2027, 11, 1),
    date(2027, 11, 11),
    date(2027, 12, 25),
}


def at(day: date, hour: int, minute: int = 0, tz: ZoneInfo = PARIS) -> datetime:
    """Heure locale → datetime avec le bon décalage (+02:00 l'été, +01:00 l'hiver)."""
    return datetime.combine(day, time(hour, minute), tzinfo=tz)


def to_paris(moment: datetime) -> datetime:
    return moment.astimezone(PARIS)


def collect_time(day: date) -> datetime:
    return datetime.combine(day, COLLECT_TIME, tzinfo=PARIS)


def is_working_day(day: date) -> bool:
    return day.weekday() < 5 and day not in FRENCH_HOLIDAYS


def working_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if is_working_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def add_working_days(day: date, count: int) -> date:
    step = 1 if count >= 0 else -1
    current, remaining = day, abs(count)
    while remaining:
        current += timedelta(days=step)
        if is_working_day(current):
            remaining -= 1
    return current


def next_monday(day: date) -> date:
    return day + timedelta(days=(7 - day.weekday()) % 7)
