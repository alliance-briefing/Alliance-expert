"""Connecteur calendrier : 48 prochaines heures + événement en cours (guide L1.4).

La source renvoie les événements bruts de Graph entre deux instants, comme l'appel
MCP `get-calendar-view`. Deux sources au choix (variable CALENDAR_SOURCE) :
- « files » (par défaut) : les fichiers JSON du jeu alliance-graph-synth (tests), servis
  uniquement à l'utilisateur de démonstration DEMO_USER (isolation L1.7) ;
- « graph » : le vrai compte, via GET /me/calendarView et le jeton de get_access_token.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from contracts.models import Item
from packages.collect.connectors.calendar_graph.mapping import event_to_item
from packages.collect.core.auth import get_access_token
from packages.collect.core.connector import Connector, Health, Window
from packages.collect.core.graph_http import Send, get_all, urllib_send
from packages.collect.core.normalize import normalize_all
from packages.collect.core.paris_time import to_paris

Source = Callable[[str, datetime, datetime], list[dict]]  # (utilisateur, début, fin)
HORIZON = timedelta(hours=48)
DEMO_USER = "u_demo_graph"  # persona fictive d'alliance-graph-synth (Hélène MARCHAND)
DEFAULT_DIR = Path(
    os.environ.get(
        "CALENDAR_GRAPH_DIR", Path(__file__).parents[2] / "tests/fixtures/graph/calendar"
    )
)


def file_source(folder: Path = DEFAULT_DIR, owner: str = DEMO_USER) -> Source:
    """Lit les fichiers AAAA-MM-JJ.json (une journée de Paris chacun) qui couvrent la période.

    Ces données appartiennent à `owner` : tout autre utilisateur ne reçoit rien.
    """

    def fetch(user: str, start: datetime, end: datetime) -> list[dict]:
        if user != owner:
            return []
        events, day = [], to_paris(start).date()
        while day <= to_paris(end).date():
            path = Path(folder) / f"{day.isoformat()}.json"
            if path.exists():
                events.extend(json.loads(path.read_text("utf-8"))["value"])
            day += timedelta(days=1)
        return events

    return fetch


def graph_source(
    token_for: Callable[[str], str] = get_access_token, send: Send = urllib_send
) -> Source:
    """Le vrai agenda : calendarView déplie les séries ; heures demandées en UTC, comme le MCP."""

    def fetch(user: str, start: datetime, end: datetime) -> list[dict]:
        params = {
            "startDateTime": start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endDateTime": end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "$top": "100",
            "$orderby": "start/dateTime",
        }
        prefer = {"Prefer": 'outlook.timezone="UTC"'}
        return get_all("/me/calendarView", token_for(user), params, prefer, send=send)

    return fetch


class GraphCalendarConnector(Connector):
    name = "calendar_graph"
    version = "0.2.0"

    def __init__(self, source: Source | None = None) -> None:
        if source is None:
            source = (
                graph_source() if os.environ.get("CALENDAR_SOURCE") == "graph" else file_source()
            )
        self.source = source

    def required_scopes(self) -> list[str]:
        return ["Calendars.Read"]  # lecture seule (guide L1.2)

    def collect(self, user: str, window: Window) -> Iterator[Item]:
        """Événements qui ne sont pas finis à `window.end` et commencent dans les 48 h qui suivent."""
        now = window.end
        events = [
            e
            for e in self.source(user, now - timedelta(days=1), now + HORIZON)
            if e.get("type") != "seriesMaster"
        ]  # une série n'apparaît que par ses occurrences
        result = normalize_all(
            events, lambda e: event_to_item(dict(e), user, now), connector=self.name
        )
        seen: set[str] = set()
        for item in result.items:  # un événement invalide est rejeté seul (L1.6), pas tout l'agenda
            if item.item_id not in seen and item.end_at > now and item.start_at < now + HORIZON:
                seen.add(item.item_id)
                yield item

    def health(self) -> Health:
        try:
            self.source("health", datetime.now(UTC), datetime.now(UTC))
        except Exception as exc:  # noqa: BLE001
            return Health(False, type(exc).__name__)
        return Health(True, "source joignable")
