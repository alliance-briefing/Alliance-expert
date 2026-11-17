"""Tests du journal d'audit de la collecte (guide L1.8, critères d'acceptation)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest

from contracts.demo_data import available_days, load_items
from contracts.models import AuditEvent
from packages.collect.cli import main
from packages.collect.connectors.demo_fixtures.connector import DemoFixturesConnector
from packages.collect.core.audit import AuditLog
from packages.collect.core.connector import Connector, Health, Window, collection_window
from packages.collect.core.runner import run

USER = "u_demo_001"
DAY = date(2026, 10, 21)
AT = datetime(2026, 10, 21, 6, 30, 5, tzinfo=UTC)
SECRET = "eyJ0eXAi.jeton-secret"
EMAIL = "camille.laurent@reseau-expertis.test"


class Broken(Connector):
    """Une source qui plante avec un message contenant un jeton et une adresse."""

    name = "broken"
    version = "0.0.1"

    def required_scopes(self) -> list[str]:
        return []

    def collect(self, user: str, window: Window) -> Iterator:
        raise PermissionError(f"401 pour {EMAIL} avec Bearer {SECRET}")

    def health(self) -> Health:
        return Health(False, "en panne")


class Empty(Broken):
    name = "empty"

    def collect(self, user: str, window: Window) -> Iterator:
        return iter(())


def _run(tmp_path, connectors, user: str = USER):
    log = AuditLog(tmp_path)
    result = run(user, collection_window(DAY), connectors, audit=log.write, clock=lambda: AT)
    return result, log.read(user)


def test_one_event_per_source_with_ids_and_counts(tmp_path) -> None:
    result, events = _run(tmp_path, [DemoFixturesConnector()])
    assert [e.source for e in events] == ["calendar", "file", "mail", "task"]
    assert sum(e.count for e in events) == len(result.items) == 199
    for event in events:
        assert event.actor == "connector:demo" and event.action == "collect"
        assert event.status == "ok" and event.user_id == USER and event.ts == AT
        assert event.count == len(event.item_ids)
        assert all(i.startswith(f"{event.source}:") for i in event.item_ids)


def test_empty_source_still_leaves_a_trace(tmp_path) -> None:
    _, events = _run(tmp_path, [Empty()])
    assert [(e.actor, e.count, e.status, e.source) for e in events] == [
        ("connector:empty", 0, "ok", None)
    ]


def test_failure_is_logged_with_its_type_only(tmp_path) -> None:
    result, events = _run(tmp_path, [Broken(), DemoFixturesConnector()])
    assert result.items  # la source en panne n'empêche pas les autres
    failure = events[0]
    assert failure.status == "error" and failure.error_code == "PermissionError"
    assert failure.count == 0 and failure.item_ids == []


def test_no_content_token_or_address_in_the_file(tmp_path) -> None:
    result, _ = _run(tmp_path, [Broken(), DemoFixturesConnector()])
    text = AuditLog(tmp_path).path(USER).read_text("utf-8")
    assert SECRET not in text and "Bearer" not in text and "@" not in text
    for item in result.items:  # aucun titre ni extrait recopié
        if len(item.title) > 12:
            assert item.title not in text
        if len(item.snippet) > 12:
            assert item.snippet not in text


def test_the_file_is_only_appended_to(tmp_path) -> None:
    _run(tmp_path, [Empty()])
    path = AuditLog(tmp_path).path(USER)
    first = path.read_text("utf-8")
    _run(tmp_path, [Broken()])
    second = path.read_text("utf-8")
    assert second.startswith(first) and len(second.splitlines()) == 2


def test_every_item_can_be_traced_to_its_collection(tmp_path) -> None:
    """Pour n'importe quel Item : quand, par quel connecteur et quelle version."""
    result, events = _run(tmp_path, [DemoFixturesConnector()])
    by_id = {item_id: e for e in events for item_id in e.item_ids}
    for item in result.items:
        event = by_id[item.item_id]
        assert event.actor == f"connector:{item.connector.name}"
        assert event.ts == AT and item.connector.version


def test_users_have_separate_journals(tmp_path) -> None:
    _run(tmp_path, [DemoFixturesConnector()])
    _run(tmp_path, [Empty()], user="u_demo_002")
    log = AuditLog(tmp_path)
    assert {e.user_id for e in log.read(USER)} == {USER}
    assert {e.user_id for e in log.read("u_demo_002")} == {"u_demo_002"}
    assert log.read("u_inconnu") == []
    with pytest.raises(ValueError):
        log.path("../u_demo_001")  # pas de sortie du dossier de données


def test_without_audit_nothing_is_written(tmp_path) -> None:
    run(USER, collection_window(DAY), [Empty()])
    assert not any(tmp_path.iterdir())


def test_cli_write_also_writes_the_journal_without_breaking_the_api(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert main(["--user", USER, "--day", "2026-10-21", "--source", "demo", "--write"]) == 0
    events = AuditLog(tmp_path).read(USER)
    assert events and all(isinstance(e, AuditEvent) for e in events)
    # l'API ne voit que les journées, pas le journal posé à côté
    assert available_days(USER, tmp_path) == ["2026-10-21"]
    assert len(load_items("2026-10-21", USER, tmp_path)) == sum(e.count for e in events)
