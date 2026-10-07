"""Règles de sélection du brief à base de règles (stub L2.0).

Trois fonctions pures : mêmes entrées, même sortie. Aucune ne lit l'horloge système ;
le « maintenant » est toujours passé en paramètre. Chaque tri se termine par item_id
pour qu'un ex aequo donne toujours le même ordre.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta

from contracts.models import Item

UNREAD_MAILS_LIMIT = 5

_ONE_MICROSECOND = timedelta(microseconds=1)


def unread_mails(items: Sequence[Item], limit: int = UNREAD_MAILS_LIMIT) -> list[Item]:
    """Les `limit` mails non lus les plus récents, du plus récent au plus ancien."""
    unread = [item for item in items if item.source == "mail" and item.is_read is False]
    unread.sort(key=lambda item: (-item.created_at.timestamp(), item.item_id))
    return unread[:limit]


def meetings_of_day(items: Sequence[Item], day: date) -> list[Item]:
    """Les événements qui occupent au moins une partie de `day`, par heure de début.

    Les dates sont lues dans le décalage horaire de chaque instant, que le lot 1
    garantit en Europe/Paris (L1.4). La fin est exclue : un événement qui se termine
    à 00:00 n'occupe pas la journée qui commence à cette heure-là.
    """
    meetings = [item for item in items if item.source == "calendar" and _occupies(item, day)]
    return sorted(meetings, key=lambda item: (item.start_at, item.item_id))


def overdue_tasks(items: Sequence[Item], now: datetime) -> list[Item]:
    """Les tâches en retard à `now`, de l'échéance la plus ancienne à la plus récente."""
    overdue = [item for item in items if item.is_overdue(now)]
    return sorted(overdue, key=lambda item: (item.due_at, item.item_id))


def _occupies(event: Item, day: date) -> bool:
    start, end = event.start_at, event.end_at
    if start is None or end is None:  # impossible pour "calendar" (contrat A), utile au typage
        return False
    first_day = start.date()
    last_day = (end - _ONE_MICROSECOND).date() if end > start else first_day
    return first_day <= day <= last_day
