"""Tests du connecteur tâches Graph / To Do (guide L1.4, critères d'acceptation)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from packages.collect.connectors.tasks_graph import GraphTasksConnector
from packages.collect.connectors.tasks_graph.connector import (
    DEMO_USER,
    MAX_OVERDUE,
    graph_source,
)
from packages.collect.connectors.tasks_graph.mapping import task_to_item
from packages.collect.core.connector import Window
from packages.collect.core.runner import run
from packages.collect.tests.test_graph_http import TOKEN, FakeGraph, _ok

FIXTURES = Path(__file__).parent / "fixtures" / "graph" / "tasks"
NOW = "2026-10-26T06:30:00+00:00"  # 07h30 à Paris, lendemain du passage à l'heure d'hiver


def _task(task_id: str, day: str | None, status: str = "notStarted", **extra) -> dict:
    task = {
        "id": task_id,
        "title": f"Tâche {task_id}",
        "status": status,
        "importance": "normal",
        "createdDateTime": "2026-09-01T08:00:00.0000000Z",
        "lastModifiedDateTime": "2026-09-01T08:00:00.0000000Z",
        "body": {"content": "", "contentType": "text"},
    }
    if day:
        task["dueDateTime"] = {"dateTime": f"{day}T00:00:00.0000000", "timeZone": "UTC"}
    return task | extra


def _collect(now: str = NOW, user: str = DEMO_USER, connector=None) -> list:
    moment = datetime.fromisoformat(now)
    return list((connector or GraphTasksConnector()).collect(user, Window(moment, moment)))


def _fixture_tasks() -> list[dict]:
    lists = json.loads((FIXTURES / "lists.json").read_text("utf-8"))["value"]
    return [
        task
        for todo_list in lists
        for task in json.loads((FIXTURES / f"{todo_list['id']}.json").read_text("utf-8"))["value"]
    ]


def test_every_fixture_task_becomes_a_valid_item() -> None:
    now = datetime.fromisoformat(NOW)
    tasks = _fixture_tasks()
    assert len(tasks) == 8
    for task in tasks:
        item = task_to_item(task, DEMO_USER, now)
        assert item.source == "task" and item.item_id == f"task:{task['id']}"


def test_keeps_overdue_and_due_within_24h_only() -> None:
    titles = [item.title for item in _collect()]
    assert titles == [  # triés par échéance
        "Renouveler l'assurance du local",  # 20/10, en retard
        "Relancer le client Delmas pour le devis",  # 24/10, en retard
        "Valider le devis Lefranc",  # 25/10, en retard (jour du changement d'heure)
        "Préparer le comité du 27/10",  # 26/10 23h59, dans les 24 h
    ]


def test_overdue_rule_and_no_due_date_is_never_overdue() -> None:
    now = datetime.fromisoformat(NOW)
    items = {i.item_id: i for i in _collect()}
    assert items["task:task_relance_delmas"].is_overdue(now)
    assert not items["task:task_comite_27"].is_overdue(now)  # due ce soir : pas encore en retard
    no_due = task_to_item(_task("sans_echeance", None), DEMO_USER, now)
    assert no_due.due_at is None and not no_due.is_overdue(now)
    assert "task:task_tableau_suivi" not in items and "task:task_plombier" not in items


def test_completed_task_is_excluded_even_when_overdue() -> None:
    now = "2026-10-26T06:30:00+00:00"
    done = _task("finie", "2026-10-20", status="completed")
    connector = GraphTasksConnector(source=lambda user: [done])
    assert _collect(now, connector=connector) == []
    assert "task:task_fournitures" not in {i.item_id for i in _collect()}  # jeu de test


def test_due_date_is_end_of_day_in_paris_around_the_time_change() -> None:
    items = {i.item_id: i for i in _collect()}
    assert items["task:task_relance_delmas"].due_at.isoformat() == "2026-10-24T23:59:00+02:00"
    assert items["task:task_valider_lefranc"].due_at.isoformat() == "2026-10-25T23:59:00+01:00"
    assert items["task:task_comite_27"].due_at.isoformat() == "2026-10-26T23:59:00+01:00"
    assert items["task:task_comite_27"].collected_at.isoformat() == "2026-10-26T07:30:00+01:00"


def test_24h_window_includes_both_ends_and_excludes_beyond() -> None:
    now = datetime.fromisoformat("2026-10-26T22:59:00+00:00")  # 23h59 à Paris, le jour J
    tasks = [
        _task(n, d)
        for n, d in (("jour_j", "2026-10-26"), ("j1", "2026-10-27"), ("j2", "2026-10-28"))
    ]
    connector = GraphTasksConnector(source=lambda user: tasks)
    items = list(connector.collect("u", Window(now, now)))
    assert [i.item_id for i in items] == ["task:jour_j", "task:j1"]  # now et now + 24 h inclus


def test_overdue_is_capped_at_50_and_the_oldest_are_reported_apart(
    caplog: pytest.LogCaptureFixture,
) -> None:
    start = datetime.fromisoformat("2026-08-01T00:00:00+00:00")
    tasks = [_task(f"t{n:02d}", (start + timedelta(days=n)).date().isoformat()) for n in range(60)]
    connector = GraphTasksConnector(source=lambda user: tasks)
    items = _collect(connector=connector)
    assert len(items) == MAX_OVERDUE == 50
    ids = {i.item_id for i in items}
    assert "task:t59" in ids and "task:t10" in ids  # les plus récentes sont gardées
    assert "task:t09" not in ids and "task:t00" not in ids  # les 10 plus anciennes sont écartées
    assert connector.omitted_overdue[DEMO_USER] == 10
    assert "10 tâche(s) en retard" in caplog.text and "Tâche t00" not in caplog.text


def test_html_body_becomes_a_short_plain_snippet() -> None:
    item = {i.item_id: i for i in _collect()}["task:task_valider_lefranc"]
    assert item.snippet == "Vérifier la ligne transport."  # sans balise ni signature
    assert item.native_importance == "normal" and item.participants == []
    now = datetime.fromisoformat(NOW)
    long_task = _task("long", None, body={"content": "mot " * 400, "contentType": "text"})
    assert len(task_to_item(long_task, DEMO_USER, now).snippet) <= 500


def test_invalid_task_is_rejected_alone(caplog: pytest.LogCaptureFixture) -> None:
    broken = _task("cassee", "2026-10-20")
    del broken["status"]
    good = _task("bonne", "2026-10-20")
    connector = GraphTasksConnector(source=lambda user: [broken, good])
    assert [i.item_id for i in _collect(connector=connector)] == ["task:bonne"]
    assert "objet rejeté" in caplog.text and "Tâche cassee" not in caplog.text


def test_read_only_scope_and_test_data_isolated() -> None:
    connector = GraphTasksConnector()
    assert connector.required_scopes() == ["Tasks.Read"]
    assert _collect()  # l'utilisateur de démo reçoit ses tâches
    assert _collect(user="u_inconnu") == []  # isolation L1.7 : personne d'autre
    assert connector.health().ok


def test_tasks_ask_graph_the_right_questions() -> None:
    lists = [{"id": "AAMk=liste1"}, {"id": "AAMk=liste2"}]
    fake = FakeGraph(
        _ok(lists),
        _ok([_task("a1", "2026-10-20")]),
        _ok([_task("b1", "2026-10-26"), _task("b2", None)]),
    )
    connector = GraphTasksConnector(source=graph_source(lambda user: TOKEN, send=fake))
    items = _collect(user="u_demo_002", connector=connector)
    assert [i.item_id for i in items] == ["task:a1", "task:b1"]
    paths = [urlparse(url).path for url, _ in fake.requests]
    assert paths == [
        "/v1.0/me/todo/lists",
        "/v1.0/me/todo/lists/AAMk%3Dliste1/tasks",
        "/v1.0/me/todo/lists/AAMk%3Dliste2/tasks",
    ]
    query = parse_qs(urlparse(fake.requests[1][0]).query)
    assert query["$filter"] == ["status ne 'completed'"] and query["$top"] == ["100"]
    assert all(h["Authorization"] == f"Bearer {TOKEN}" for _, h in fake.requests)


def test_without_token_the_tasks_become_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPH_ACCESS_TOKEN", raising=False)
    now = datetime.fromisoformat(NOW)
    result = run(DEMO_USER, Window(now, now), [GraphTasksConnector(source=graph_source())])
    assert result.warnings == ["tasks_graph: échec (AuthError)"]
