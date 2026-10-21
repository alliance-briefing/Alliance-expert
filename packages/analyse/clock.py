"""Heure de Paris pour tout le lot 2 (guide §6 : toutes les heures en heure de Paris).

Le contrat A n'exige qu'une date avec décalage : un item peut arriver en UTC (Microsoft
Graph) comme en +02:00. Avant d'en tirer un jour ou une heure, on lit donc l'instant en
heure de Paris. Seules les journées entières gardent leurs dates telles quelles.
"""

from __future__ import annotations

from datetime import datetime
from datetime import time as Time
from zoneinfo import ZoneInfo

from contracts.models import Item

PARIS = ZoneInfo("Europe/Paris")
MIDNIGHT = Time(0)


def to_paris(value: datetime) -> datetime:
    """Le même instant, exprimé en heure de Paris."""
    return value.astimezone(PARIS)


def is_all_day(event: Item) -> bool:
    """Événement sur des journées entières, dont les dates se lisent telles quelles.

    Contrat A 1.1 : le lot 1 le dit dans `attributes.calendar.is_all_day`, et on le croit.
    Sans ce bloc (items 1.0), on le déduit : début et fin à minuit dans son propre
    décalage. Graph envoie une journée entière de minuit UTC à minuit UTC ; lue en heure
    de Paris, elle irait de 02:00 à 02:00 et déborderait sur le lendemain.
    """
    calendar = event.attributes.calendar if event.attributes else None
    if calendar is not None and calendar.is_all_day is not None:
        return calendar.is_all_day
    start, end = event.start_at, event.end_at
    return (
        start is not None
        and end is not None
        and end > start
        and start.time() == MIDNIGHT
        and end.time() == MIDNIGHT
    )
