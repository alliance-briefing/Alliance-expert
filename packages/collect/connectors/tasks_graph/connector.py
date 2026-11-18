"""Connecteur tâches : en retard (plafond 50) + dues dans les 24 h (guide L1.4).

La source renvoie les tâches To Do brutes de Graph (toutes listes confondues). Deux sources
au choix (variable TASKS_SOURCE) :
- « files » (par défaut) : fichiers JSON fabriqués (tests), servis uniquement à
  l'utilisateur de démonstration DEMO_USER (isolation L1.7) ;
- « graph » : le vrai compte, via GET /me/todo/lists puis /me/todo/lists/{id}/tasks
  (permission Tasks.Read) et le jeton de get_access_token.

Règles : en retard = échéance dépassée ET non terminée ; sans échéance, jamais en retard
(et jamais collectée : elle n'est ni en retard ni due dans les 24 h).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

from contracts.models import Item
from packages.collect.connectors.tasks_graph.mapping import task_to_item
from packages.collect.core.auth import get_access_token
from packages.collect.core.connector import Connector, Health, Window
from packages.collect.core.graph_http import Send, get_all, urllib_send
from packages.collect.core.normalize import normalize_all

logger = logging.getLogger("collect.tasks_graph")

Source = Callable[[str], list[dict]]  # utilisateur → tâches non terminées de toutes ses listes
SOON = timedelta(hours=24)
MAX_OVERDUE = 50
DEMO_USER = "u_demo_graph"  # persona fictive d'alliance-graph-synth (Hélène MARCHAND)
DEFAULT_DIR = Path(
    os.environ.get("TASKS_GRAPH_DIR", Path(__file__).parents[2] / "tests/fixtures/graph/tasks")
)


def file_source(folder: Path = DEFAULT_DIR, owner: str = DEMO_USER) -> Source:
    """Lit lists.json puis <id de liste>.json. Ces données appartiennent à `owner` seul."""

    def fetch(user: str) -> list[dict]:
        if user != owner:
            return []
        lists = json.loads((Path(folder) / "lists.json").read_text("utf-8"))["value"]
        return [
            task
            for todo_list in lists
            for task in json.loads((Path(folder) / f"{todo_list['id']}.json").read_text("utf-8"))[
                "value"
            ]
        ]

    return fetch


def graph_source(
    token_for: Callable[[str], str] = get_access_token, send: Send = urllib_send
) -> Source:
    """Le vrai compte : les listes, puis les tâches non terminées de chaque liste."""

    def fetch(user: str) -> list[dict]:
        token = token_for(user)
        tasks: list[dict] = []
        for todo_list in get_all("/me/todo/lists", token, send=send):
            path = f"/me/todo/lists/{quote(todo_list['id'], safe='')}/tasks"
            params = {"$filter": "status ne 'completed'", "$top": "100"}
            tasks.extend(get_all(path, token, params, send=send))
        return tasks

    return fetch


class GraphTasksConnector(Connector):
    name = "tasks_graph"
    version = "0.1.0"

    def __init__(self, source: Source | None = None) -> None:
        if source is None:
            source = graph_source() if os.environ.get("TASKS_SOURCE") == "graph" else file_source()
        self.source = source
        # tâches en retard écartées par le plafond, par utilisateur : « signalées à part »
        self.omitted_overdue: dict[str, int] = {}

    def required_scopes(self) -> list[str]:
        return ["Tasks.Read"]  # lecture seule (guide L1.2)

    def collect(self, user: str, window: Window) -> Iterator[Item]:
        """Tâches en retard (les 50 dont l'échéance est la plus récente) et dues dans les 24 h."""
        now = window.end
        result = normalize_all(
            self.source(user), lambda t: task_to_item(dict(t), user, now), connector=self.name
        )  # une tâche invalide est rejetée seule (L1.6), pas toute la liste
        tasks = {item.item_id: item for item in result.items if item.completed is False}
        overdue = sorted(
            (t for t in tasks.values() if t.is_overdue(now)), key=lambda t: t.due_at, reverse=True
        )
        kept = overdue[:MAX_OVERDUE]
        omitted = self.omitted_overdue[user] = len(overdue) - len(kept)
        if omitted:
            logger.warning("%s : %d tâche(s) en retard au-delà du plafond", self.name, omitted)
        soon = [t for t in tasks.values() if t.due_at is not None and now <= t.due_at <= now + SOON]
        yield from sorted(kept + soon, key=lambda t: t.due_at)

    def health(self) -> Health:
        try:
            self.source("health")
        except Exception as exc:  # noqa: BLE001
            return Health(False, type(exc).__name__)
        return Health(True, "source joignable")
