"""Traduction d'un événement Microsoft Graph (get-calendar-view) en Item du contrat A 1.1.

Les détails de la réunion (lieu, visio, toute la journée, annulation, réponse, série)
vont dans `attributes.calendar` ; la même réunion reçue en invitation par mail partage le
`correlation_id` (calculé par core/correlation.py, L1.6).
Événement privé : titre remplacé par « Événement privé », ni texte, ni lieu, ni participants,
ni correlation_id (il contiendrait le titre) — décision validée par le lot 2.
"""

from __future__ import annotations

import re
from datetime import datetime

from contracts.models import SNIPPET_MAX_CHARS, Item
from packages.collect.core.correlation import graph_event_correlation_id
from packages.collect.core.paris_time import paris_midnight, parse_graph, to_paris

CONNECTOR = {"name": "calendar_graph", "version": "0.2.0"}
PRIVATE_TITLE = "Événement privé"
# Réponse Graph → valeurs du contrat A 1.1
RESPONSES = {
    "organizer": "organizer",
    "accepted": "accepted",
    "tentativelyAccepted": "tentative",
    "declined": "declined",
    "notResponded": "none",
    "none": "none",
}
_SPACES = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Texte brut sur une ligne : retours à la ligne et caractères de contrôle retirés."""
    return _SPACES.sub(" ", "".join(c if c.isprintable() else " " for c in text)).strip()


def _start_end(event: dict) -> tuple[datetime, datetime]:
    start, end = event["start"], event["end"]
    if start["timeZone"] != "UTC" or end["timeZone"] != "UTC":
        raise ValueError(f"fuseau non géré : {start['timeZone']} (attendu : UTC, comme le MCP)")
    if event["isAllDay"]:  # une date, pas une heure : minuit à Paris, fin exclue
        return (
            paris_midnight(parse_graph(start["dateTime"]).date()),
            paris_midnight(parse_graph(end["dateTime"]).date()),
        )
    return to_paris(parse_graph(start["dateTime"])), to_paris(parse_graph(end["dateTime"]))


def _attributes(event: dict, private: bool) -> dict:
    online = (event.get("onlineMeeting") or {}).get("joinUrl")
    place = (event.get("location") or {}).get("displayName") or None
    return {
        "calendar": {
            "location": None if private else place,
            "online_url": online
            if online and online.startswith("https://") and not private
            else None,
            "is_all_day": event["isAllDay"],
            "is_cancelled": event["isCancelled"],
            "response": RESPONSES.get((event.get("responseStatus") or {}).get("response", "none")),
            "series_id": event.get("seriesMasterId"),
        }
    }


def _participants(event: dict) -> list[dict]:
    organizer = event["organizer"]["emailAddress"]
    people = [{"name": organizer["name"], "email": organizer["address"], "role": "organizer"}]
    for attendee in event["attendees"]:
        who = attendee["emailAddress"]
        if who["address"] != organizer["address"]:
            people.append({"name": who["name"], "email": who["address"], "role": "attendee"})
    return people


def _snippet(event: dict, private: bool) -> str:
    if private:
        return ""
    body = _clean(event.get("bodyPreview") or "")
    origin = event.get("originalStartTimeZone")
    if origin and origin not in ("Romance Standard Time", "UTC"):
        body = f"Fuseau d'origine : {origin}" + (f" — {body}" if body else "")
    return body[:SNIPPET_MAX_CHARS]


def event_to_item(event: dict, owner: str, collected_at: datetime) -> Item:
    private = event["sensitivity"] == "private"
    start, end = _start_end(event)
    created = to_paris(parse_graph(event["createdDateTime"]))
    # Exchange peut dater la modification une fraction de seconde avant la création
    updated = max(created, to_paris(parse_graph(event["lastModifiedDateTime"])))
    return Item(
        item_id=f"calendar:{event['id']}",
        source="calendar",
        provider="microsoft",
        owner_user_id=owner,
        title=PRIVATE_TITLE if private else _clean(event["subject"]),
        snippet=_snippet(event, private),
        created_at=created,
        updated_at=updated,
        start_at=start,
        end_at=end,
        participants=[] if private else _participants(event),
        native_importance=event["importance"],
        url=event["webLink"],
        acl={"visible_to": [owner]},
        collected_at=to_paris(collected_at),
        connector=CONNECTOR,
        correlation_id=None if private else graph_event_correlation_id(event),
        attributes=_attributes(event, private),
    )
