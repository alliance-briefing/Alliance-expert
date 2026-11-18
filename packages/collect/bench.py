"""Mesure de performance de la collecte (guide L1.9).

  uv run python -m packages.collect.bench                         # 10 collectes, p50 et p95
  uv run python -m packages.collect.bench --runs 3 --latency 1.0  # + 1 s par requête Graph

Volume du guide : 500 mails, 20 événements, 30 tâches, 100 fichiers, en moins de 30 s (p95).
Microsoft est simulé (pas de réseau) : la mesure couvre le chemin complet du lot 1 — requêtes
Graph et pagination, lecture JSON, sélection, nettoyage du HTML, contrat A, runner. Le temps
de réponse de Microsoft s'ajoute avec --latency (mesuré sur le compte de démo : 0,4 à 2 s par
requête), pour estimer une collecte réelle à ce volume.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from datetime import UTC, date, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from packages.collect.connectors.calendar_graph import GraphCalendarConnector
from packages.collect.connectors.calendar_graph import connector as calendar
from packages.collect.connectors.files_graph import GraphFilesConnector
from packages.collect.connectors.files_graph import connector as files
from packages.collect.connectors.mail_graph import GraphMailConnector
from packages.collect.connectors.mail_graph import connector as mail
from packages.collect.connectors.tasks_graph import GraphTasksConnector
from packages.collect.connectors.tasks_graph import connector as tasks
from packages.collect.core.connector import collection_window
from packages.collect.core.runner import CollectResult, run

VOLUME = {"mail": 500, "calendar": 20, "task": 30, "file": 100}
BUDGET_S = 30.0
DAY = date(2026, 10, 21)  # collecte à 8h30, Paris (06:30 UTC)
PAGE = 100
FIXTURES = Path(__file__).parent / "tests/fixtures/graph"
PERSONA = "helene.marchand@verdier-expertises.example"


def _utc(moment) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(relative: str) -> dict:
    data = json.loads((FIXTURES / relative).read_text("utf-8"))
    return data["value"][0] if isinstance(data, dict) else data[0]


def build_world() -> dict[str, list[dict]]:
    """Données au format Graph, toutes dans la fenêtre de collecte."""
    end = collection_window(DAY).end
    mail_template = _load("mail/2026-10-20.json")
    event_template = _load("calendar/2026-10-21.json")
    task_template = _load("tasks/list_travail.json")
    world: dict[str, list[dict]] = {"mail": [], "calendar": [], "task": [], "file": []}
    for n in range(VOLUME["mail"]):
        message = copy.deepcopy(mail_template)
        moment = _utc(end - timedelta(minutes=2 * n + 1))  # tous reçus dans les 24 h
        message.update(
            id=f"mail-{n}",
            conversationId=f"conv-{n}",
            receivedDateTime=moment,
            sentDateTime=moment,
            createdDateTime=moment,
            lastModifiedDateTime=moment,
        )
        world["mail"].append(message)
    for n in range(VOLUME["calendar"]):
        event = copy.deepcopy(event_template)
        start = end + timedelta(hours=2 * n + 1)
        event.update(
            id=f"event-{n}",
            iCalUId=f"uid-{n}",
            subject=f"Réunion {n}",
            start={"dateTime": _utc(start)[:-1], "timeZone": "UTC"},
            end={"dateTime": _utc(start + timedelta(minutes=30))[:-1], "timeZone": "UTC"},
        )
        world["calendar"].append(event)
    for n in range(VOLUME["task"]):
        task = copy.deepcopy(task_template)
        due = (end - timedelta(days=n % 20)).astimezone(UTC).strftime("%Y-%m-%dT00:00:00.0000000")
        task.update(id=f"task-{n}", dueDateTime={"dateTime": due, "timeZone": "UTC"})
        world["task"].append(task)
    for n in range(VOLUME["file"]):
        moment = _utc(end - timedelta(minutes=20 * n + 1))
        world["file"].append(
            {
                "id": f"file-{n}",
                "name": f"Note {n}.txt" if n % 2 else f"Rapport {n}.pdf",
                "size": 2048,
                "file": {},
                "createdDateTime": moment,
                "lastModifiedDateTime": moment,
                "webUrl": f"https://onedrive.live.com/?id=file-{n}",
                "parentReference": {"path": "/drive/root:/Dossiers"},
            }
        )
    return world


class FakeGraph:
    """Répond comme Graph, par pages de 100, à partir du monde simulé."""

    def __init__(self, world: dict[str, list[dict]], latency: float = 0.0) -> None:
        self.world = world
        self.latency = latency
        self.calls = 0

    def _page(self, url: str, values: list[dict]) -> bytes:
        page = int(parse_qs(urlparse(url).query).get("page", ["0"])[0])
        body: dict = {"value": values[page * PAGE : (page + 1) * PAGE]}
        if (page + 1) * PAGE < len(values):
            body["@odata.nextLink"] = f"{url.split('?')[0]}?page={page + 1}"
        return json.dumps(body).encode()

    def __call__(self, url: str, headers: dict) -> tuple[int, dict, bytes]:
        self.calls += 1
        if self.latency:
            time.sleep(self.latency)
        path = urlparse(url).path.removeprefix("/v1.0")
        if urlparse(url).netloc == "download.example":
            return 206, {}, "Compte rendu fictif : décisions et suites à donner.".encode()
        if path == "/me":
            return 200, {}, json.dumps({"mail": PERSONA}).encode()
        routes = {
            "/me/mailFolders/inbox/messages": self.world["mail"],
            "/me/mailFolders/sentitems/messages": [],
            "/me/calendarView": self.world["calendar"],
            "/me/todo/lists": [{"id": "liste"}],
            "/me/todo/lists/liste/tasks": self.world["task"],
            "/me/drive/root/delta": self.world["file"],
        }
        if path in routes:
            return 200, {}, self._page(url, routes[path])
        if path.startswith("/me/drive/items/"):
            file_id = path.rsplit("/", 1)[-1]
            link = {"@microsoft.graph.downloadUrl": f"https://download.example/{file_id}"}
            return 200, {}, json.dumps(link).encode()
        return 404, {}, b"{}"


def collect_once(world: dict[str, list[dict]], latency: float = 0.0) -> tuple[float, CollectResult]:
    send = FakeGraph(world, latency)
    token = lambda user: "jeton-factice"
    connectors = [
        GraphMailConnector(source=mail.graph_source(token_for=token, send=send)),
        GraphCalendarConnector(source=calendar.graph_source(token_for=token, send=send)),
        GraphTasksConnector(source=tasks.graph_source(token_for=token, send=send)),
        GraphFilesConnector(source=files.graph_source(token_for=token, send=send)),
    ]
    started = time.perf_counter()
    result = run("u_bench", collection_window(DAY), connectors)
    return time.perf_counter() - started, result


def measure(runs: int = 10, latency: float = 0.0) -> dict:
    world = build_world()
    durations, result = [], None
    for _ in range(runs):
        seconds, result = collect_once(world, latency)
        durations.append(seconds)
    ordered = sorted(durations)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    counts = {s: sum(1 for i in result.items if i.source == s) for s in VOLUME} if result else {}
    return {"runs": runs, "p50": statistics.median(durations), "p95": p95, "items": counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mesure de performance de la collecte (L1.9)")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--latency", type=float, default=0.0, help="secondes par requête Graph")
    args = parser.parse_args(argv)
    report = measure(args.runs, args.latency)
    print(
        f"{report['runs']} collectes de {report['items']}, latence {args.latency} s : "
        f"p50 {report['p50']:.2f} s, p95 {report['p95']:.2f} s (budget {BUDGET_S:.0f} s)"
    )
    return 0 if report["p95"] < BUDGET_S else 1


if __name__ == "__main__":
    raise SystemExit(main())
