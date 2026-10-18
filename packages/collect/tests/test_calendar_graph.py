"""Tests du connecteur calendrier Graph (guide L1.4, critères d'acceptation)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from packages.collect.connectors.calendar_graph import GraphCalendarConnector
from packages.collect.connectors.calendar_graph.connector import DEMO_USER, graph_source
from packages.collect.connectors.calendar_graph.mapping import event_to_item
from packages.collect.core.connector import Window
from packages.collect.core.runner import run
from packages.collect.tests.test_graph_http import TOKEN, FakeGraph, _ok

FIXTURES = Path(__file__).parent / "fixtures" / "graph" / "calendar"
USER = DEMO_USER


def _collect(now: str) -> list:
    moment = datetime.fromisoformat(now)
    return list(GraphCalendarConnector().collect(USER, Window(moment, moment)))


def _find(items: list, title: str) -> list:
    return [item for item in items if item.title.startswith(title)]


def test_every_event_of_the_dataset_becomes_a_valid_item() -> None:
    now = datetime.fromisoformat("2026-10-05T06:30:00+00:00")
    events = [
        e for p in sorted(FIXTURES.glob("*.json")) for e in json.loads(p.read_text())["value"]
    ]
    assert len(events) == 200
    for event in events:
        assert event_to_item(event, USER, now).source == "calendar"


def test_winter_time_change_keeps_9h_in_paris() -> None:
    before = _find(_collect("2026-10-23T06:00:00+00:00"), "Point flash opérations")[0]
    after = _find(_collect("2026-10-26T06:00:00+00:00"), "Point flash opérations")[0]
    assert before.start_at.isoformat() == "2026-10-23T09:00:00+02:00"
    assert after.start_at.isoformat() == "2026-10-26T09:00:00+01:00"


def test_weekly_series_appears_once_with_the_right_date() -> None:
    items = _collect("2026-11-01T08:00:00+00:00")
    weekly = _find(items, "Point hebdo équipe projet")
    assert [item.start_at.isoformat() for item in weekly] == ["2026-11-02T09:30:00+01:00"]
    assert len({item.item_id for item in items}) == len(items)


def test_all_day_event_is_a_date_not_an_hour() -> None:
    salon = _find(_collect("2026-11-02T07:00:00+00:00"), "Salon des professionnels")[0]
    assert salon.start_at.isoformat() == "2026-11-03T00:00:00+01:00"
    assert salon.end_at.isoformat() == "2026-11-06T00:00:00+01:00"
    assert salon.attributes.calendar.is_all_day


def test_foreign_timezone_is_converted_and_reported() -> None:
    webinar = _find(_collect("2026-10-27T07:00:00+00:00"), "Webinaire partenaires")[0]
    assert webinar.start_at.isoformat() == "2026-10-28T15:00:00+01:00"
    assert "Fuseau d'origine : Eastern Standard Time" in webinar.snippet


def test_cancelled_event_is_flagged() -> None:
    items = _collect("2026-10-26T06:00:00+00:00")
    assert any(item.attributes.calendar.is_cancelled for item in items)


def test_calendar_attributes_and_correlation_follow_contract_1_1() -> None:
    items = _collect("2026-10-26T06:00:00+00:00")
    assert all(item.schema_version == "1.1" and item.attributes.calendar for item in items)
    responses = {item.attributes.calendar.response for item in items}
    assert responses <= {"none", "organizer", "accepted", "tentative", "declined"}
    visio = [i for i in items if i.attributes.calendar.online_url]
    assert visio and all(
        str(i.attributes.calendar.online_url).startswith("https://") for i in visio
    )
    for item in items:
        if item.title != "Événement privé":
            assert item.correlation_id.startswith("meet:") and item.correlation_id.endswith("Z")


def test_private_event_shows_nothing() -> None:
    private = _find(_collect("2026-10-22T06:00:00+00:00"), "Événement privé")[0]
    assert private.participants == [] and private.snippet == ""
    assert private.correlation_id is None and private.attributes.calendar.location is None


def test_window_is_48h_plus_the_current_event() -> None:
    now = datetime.fromisoformat("2026-10-23T07:10:00+00:00")
    items = list(GraphCalendarConnector().collect(USER, Window(now, now)))
    assert all(item.end_at > now for item in items)  # rien de déjà terminé
    assert any(item.start_at <= now < item.end_at for item in items)  # l'événement en cours
    assert all((item.start_at - now).total_seconds() < 48 * 3600 for item in items)


def test_invalid_event_is_rejected_alone(caplog: pytest.LogCaptureFixture) -> None:
    event = json.loads((FIXTURES / "2026-10-23.json").read_text())["value"][0]
    event["start"]["timeZone"] = "Pacific Standard Time"
    connector = GraphCalendarConnector(source=lambda user, start, end: [event])
    now = datetime.fromisoformat("2026-10-23T06:00:00+00:00")
    assert list(connector.collect(USER, Window(now, now))) == []  # rejeté, sans planter
    assert "objet rejeté" in caplog.text and event["subject"] not in caplog.text


def test_read_only_scope_and_test_data_isolated() -> None:
    assert GraphCalendarConnector().required_scopes() == ["Calendars.Read"]
    assert _collect("2026-10-26T06:00:00+00:00")  # l'utilisateur de démo reçoit ses événements
    moment = datetime.fromisoformat("2026-10-26T06:00:00+00:00")
    other = list(GraphCalendarConnector().collect("u_inconnu", Window(moment, moment)))
    assert other == []  # isolation L1.7 : personne d'autre
    assert GraphCalendarConnector().health().ok


def test_event_seen_in_two_day_files_is_kept_once() -> None:
    """L'événement du 11/11 (toute la journée) est dans les fichiers du 11 et du 12."""
    pont = _find(_collect("2026-11-10T07:00:00+00:00"), "Pont de l'Armistice")
    assert len(pont) == 1 and pont[0].start_at.isoformat() == "2026-11-11T00:00:00+01:00"


def test_calendar_asks_graph_the_right_question() -> None:
    event = json.loads((FIXTURES / "2026-10-23.json").read_text("utf-8"))["value"][0]
    fake = FakeGraph(_ok([event]))
    connector = GraphCalendarConnector(source=graph_source(lambda user: TOKEN, send=fake))
    now = datetime.fromisoformat("2026-10-23T06:00:00+00:00")
    items = list(connector.collect("u_demo_002", Window(now, now)))
    url, headers = fake.requests[0]
    query = parse_qs(urlparse(url).query)
    assert urlparse(url).path == "/v1.0/me/calendarView"
    assert query["startDateTime"] == ["2026-10-22T06:00:00Z"]
    assert query["endDateTime"] == ["2026-10-25T06:00:00Z"]
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert headers["Prefer"] == 'outlook.timezone="UTC"'
    assert [item.item_id for item in items] == [f"calendar:{event['id']}"]


def test_without_token_the_calendar_becomes_a_warning(monkeypatch) -> None:
    monkeypatch.delenv("GRAPH_ACCESS_TOKEN", raising=False)
    now = datetime.fromisoformat("2026-10-23T06:00:00+00:00")
    result = run(USER, Window(now, now), [GraphCalendarConnector(source=graph_source())])
    assert result.warnings == ["calendar_graph: échec (AuthError)"]
