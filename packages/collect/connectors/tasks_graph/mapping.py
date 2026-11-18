"""Traduction d'une tâche Microsoft To Do (Graph `todoTask`) en Item du contrat A 1.1.

To Do ne stocke qu'une **date** d'échéance, qu'il renvoie comme le minuit local de l'utilisateur
exprimé en UTC : une tâche due le 18/11 à Paris arrive en « 2026-11-17T23:00:00, UTC ». On
convertit donc en heure de Paris **avant** de prendre la date, puis on la place à 23h59, heure
de Paris, avec le bon décalage (+01:00 / +02:00). Une tâche due le 25/10 reste donc « à faire »
jusqu'à la fin de cette journée.
Pas de bloc `attributes` : le contrat 1.1 n'en définit pas pour les tâches.
"""

from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import quote

from contracts.models import Item
from packages.collect.core.paris_time import paris_time, parse_graph, to_paris
from packages.collect.core.text import make_snippet, truncate

CONNECTOR = {"name": "tasks_graph", "version": "0.1.0"}
TODO_URL = "https://to-do.office.com/tasks/id/{id}/details"
_SPACES = re.compile(r"\s+")


def _one_line(text: str) -> str:
    """Texte brut sur une ligne : retours à la ligne et caractères de contrôle retirés."""
    return _SPACES.sub(" ", "".join(c if c.isprintable() else " " for c in text)).strip()


def due_at(task: dict) -> datetime | None:
    """Fin de la journée d'échéance à Paris ; `None` si la tâche n'a pas d'échéance."""
    due = task.get("dueDateTime")
    if not due:
        return None
    moment = parse_graph(due["dateTime"])
    if due.get("timeZone", "UTC") == "UTC":  # minuit de Paris écrit en UTC : la veille à 22h/23h
        day = to_paris(moment).date()
    else:  # autre fuseau : la date écrite est déjà la date locale
        day = moment.date()
    return paris_time(day, 23, 59)


def _snippet(task: dict) -> str:
    body = task.get("body") or {}
    content = body.get("content") or ""
    if body.get("contentType") == "html":
        return _one_line(make_snippet(content))
    return truncate(_one_line(content))


def task_to_item(task: dict, owner: str, collected_at: datetime) -> Item:
    created = to_paris(parse_graph(task["createdDateTime"]))
    updated = max(created, to_paris(parse_graph(task["lastModifiedDateTime"])))
    return Item(
        item_id=f"task:{task['id']}",
        source="task",
        provider="microsoft",
        owner_user_id=owner,
        title=_one_line(task["title"]),
        snippet=_snippet(task),
        created_at=created,
        updated_at=updated,
        due_at=due_at(task),
        completed=task["status"] == "completed",
        native_importance=task.get("importance"),
        url=TODO_URL.format(id=quote(task["id"], safe="")),
        acl={"visible_to": [owner]},
        collected_at=to_paris(collected_at),
        connector=CONNECTOR,
    )
