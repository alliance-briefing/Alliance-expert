"""Pannes Microsoft Graph, pour les 4 connecteurs (guide L1.9).

Chaque panne doit finir en warning du runner : pas de plantage, pas de journée partielle
silencieuse, et les autres sources de l'utilisateur continuent. Tableau dans le README.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import pytest

from packages.collect.connectors.calendar_graph import GraphCalendarConnector
from packages.collect.connectors.calendar_graph import connector as calendar
from packages.collect.connectors.files_graph import GraphFilesConnector
from packages.collect.connectors.files_graph import connector as files
from packages.collect.connectors.mail_graph import GraphMailConnector
from packages.collect.connectors.mail_graph import connector as mail
from packages.collect.connectors.tasks_graph import GraphTasksConnector
from packages.collect.connectors.tasks_graph import connector as tasks
from packages.collect.core import graph_http
from packages.collect.core.auth import AuthError
from packages.collect.core.connector import collection_window
from packages.collect.core.runner import run
from packages.collect.tests.test_graph_http import TOKEN

USER = "u_test"
WINDOW = collection_window(date(2026, 10, 21))
NEXT = "https://graph.microsoft.com/v1.0/page-suivante"
# Réponse vide mais valide pour tous les appels : /me, listes, pages, métadonnées.
EMPTY = {"value": [], "mail": "persona@example.com", "userPrincipalName": "persona@example.com"}

CONNECTORS = {
    "mail": (GraphMailConnector, mail.graph_source),
    "calendar": (GraphCalendarConnector, calendar.graph_source),
    "tasks": (GraphTasksConnector, tasks.graph_source),
    "files": (GraphFilesConnector, files.graph_source),
}


class Script:
    """Graph simulé : rejoue une suite de réponses, puis répète la dernière."""

    def __init__(self, *responses: tuple[int, dict, bytes]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, url: str, headers: dict) -> tuple[int, dict, bytes]:
        self.calls += 1
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def _json(status: int, body: dict, headers: dict | None = None) -> tuple[int, dict, bytes]:
    return status, headers or {}, json.dumps(body).encode()


def _connector(name: str, send, token_for=lambda user: TOKEN):
    cls, graph_source = CONNECTORS[name]
    return cls(source=graph_source(token_for=token_for, send=send))


@pytest.fixture(autouse=True)
def no_wait(monkeypatch) -> None:
    monkeypatch.setattr(graph_http, "DELAYS", (0, 0, 0, 0))  # réessais sans attendre


FAILURES = {
    "réseau coupé": [(599, {}, b"")],
    "401 jeton refusé": [_json(401, {"error": {"code": "InvalidAuthenticationToken"}})],
    "403 accès interdit": [_json(403, {"error": {"code": "ErrorAccessDenied"}})],
    "404 introuvable": [_json(404, {"error": {"code": "ResourceNotFound"}})],
    "429 trop de requêtes": [_json(429, {}, {"Retry-After": "0"})],
    "500 panne": [_json(500, {})],
    "503 indisponible": [_json(503, {})],
    "JSON tronqué": [(200, {}, b'{"value": [{"id": "1", "subj')],
    "page suivante manquante": [
        _json(200, EMPTY | {"@odata.nextLink": NEXT}),
        _json(404, {}),
    ],
}


@pytest.mark.parametrize("failure", FAILURES)
@pytest.mark.parametrize("name", CONNECTORS)
def test_every_failure_becomes_a_warning_never_a_crash(name, failure, caplog) -> None:
    send = Script(*FAILURES[failure])
    with caplog.at_level(logging.ERROR):
        result = run(USER, WINDOW, [_connector(name, send)])
    assert result.items == []  # rien de partiel
    assert len(result.warnings) == 1 and result.warnings[0].startswith(
        f"{CONNECTORS[name][0].name}:"
    )
    assert TOKEN not in caplog.text + " ".join(result.warnings)


@pytest.mark.parametrize("name", CONNECTORS)
def test_retries_then_succeeds_after_429_and_503(name) -> None:
    send = Script(_json(429, {}, {"Retry-After": "0"}), _json(503, {}), _json(200, EMPTY))
    result = run(USER, WINDOW, [_connector(name, send)])
    assert result.warnings == [] and send.calls >= 3


@pytest.mark.parametrize("name", CONNECTORS)
def test_client_errors_are_not_retried(name) -> None:
    send = Script(_json(403, {}))
    run(USER, WINDOW, [_connector(name, send)])
    assert send.calls == 1  # 4xx : réessayer ne changerait rien


@pytest.mark.parametrize("name", CONNECTORS)
def test_server_errors_give_up_after_five_attempts(name) -> None:
    send = Script(_json(500, {}))
    run(USER, WINDOW, [_connector(name, send)])
    assert send.calls == 5


@pytest.mark.parametrize("name", CONNECTORS)
def test_revoked_connection_is_reauth_required(name) -> None:
    def revoked(user: str) -> str:
        raise AuthError

    result = run(USER, WINDOW, [_connector(name, Script(_json(200, EMPTY)), token_for=revoked)])
    assert result.warnings == [f"{CONNECTORS[name][0].name}: échec (AuthError)"]


def test_one_failing_source_does_not_stop_the_others() -> None:
    down = [_connector(name, Script(_json(503, {}))) for name in ("mail", "calendar", "tasks")]
    working = GraphFilesConnector(source=files.file_source())
    result = run(files.DEMO_USER, WINDOW, [*down, working])
    assert len(result.warnings) == 3 and result.items  # les fichiers sont bien là
    assert {i.source for i in result.items} == {"file"}


def test_invalid_file_is_rejected_alone(caplog: pytest.LogCaptureFixture) -> None:
    good = {
        "id": "ok",
        "name": "ok.pdf",
        "size": 10,
        "file": {},
        "createdDateTime": "2026-10-20T10:00:00Z",
        "lastModifiedDateTime": "2026-10-20T10:00:00Z",
        "webUrl": "https://onedrive.live.com/?id=ok",
    }
    broken = good | {"id": "ko", "name": "ko.pdf", "webUrl": "http://pas-https.example"}
    source = files.DriveSource(lambda user: [good, broken], lambda user, item, limit: b"x")
    with caplog.at_level(logging.WARNING):
        items = list(GraphFilesConnector(source=source).collect(USER, WINDOW))
    assert [i.title for i in items] == ["ok.pdf"] and "objet rejeté" in caplog.text
