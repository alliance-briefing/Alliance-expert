"""Heures de Paris, sans dépendance externe (guide L1.4 : décalage explicite +01:00 / +02:00).

Règle européenne : heure d'été du dernier dimanche de mars au dernier dimanche
d'octobre, à 01:00 UTC. En 2026, le passage à l'heure d'hiver a lieu le dimanche 25/10.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone


def _last_sunday(year: int, month: int) -> date:
    last = date(year, month, 31)  # mars et octobre ont 31 jours
    return last - timedelta(days=(last.weekday() + 1) % 7)


def to_paris(moment: datetime) -> datetime:
    """Le même instant, écrit avec le décalage de Paris."""
    utc = moment.astimezone(UTC)
    summer_start = datetime.combine(_last_sunday(utc.year, 3), time(1), UTC)
    summer_end = datetime.combine(_last_sunday(utc.year, 10), time(1), UTC)
    hours = 2 if summer_start <= utc < summer_end else 1
    return utc.astimezone(timezone(timedelta(hours=hours)))


def paris_time(day: date, hour: int = 0, minute: int = 0) -> datetime:
    """Heure locale de Paris (ex. 8h30 le 19/10) avec le bon décalage."""
    for hours in (2, 1):
        candidate = datetime.combine(day, time(hour, minute), timezone(timedelta(hours=hours)))
        if to_paris(candidate).utcoffset() == timedelta(hours=hours):
            return candidate
    raise ValueError(f"heure inexistante à Paris : {day} {hour}h{minute:02d}")


def paris_midnight(day: date) -> datetime:
    """Minuit à Paris le jour `day` (début d'un événement « toute la journée »)."""
    return paris_time(day)


def parse_graph(value: str) -> datetime:
    """Date Microsoft Graph en UTC : '2026-10-26T07:15:00.0000000' ou '…00.5191677Z'."""
    head = value.rstrip("Z")[:26]  # Python accepte au plus 6 décimales
    return datetime.fromisoformat(head).replace(tzinfo=UTC)
