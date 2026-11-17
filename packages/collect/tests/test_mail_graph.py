"""Tests du connecteur e-mails Graph (guide L1.3, critères d'acceptation).

Fixtures : `fixtures/graph/mail/` (format alliance-graph-synth), collecte du 21/10 à 8h30.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from packages.collect.connectors.calendar_graph import GraphCalendarConnector
from packages.collect.connectors.mail_graph import GraphMailConnector
from packages.collect.connectors.mail_graph.connector import (
    DEMO_USER,
    Mailbox,
    graph_source,
)
from packages.collect.connectors.mail_graph.mapping import message_to_item
from packages.collect.core.connector import collection_window
from packages.collect.core.runner import run
from packages.collect.tests.test_graph_http import TOKEN, FakeGraph, _ok

FIXTURES = Path(__file__).parent / "fixtures" / "graph"
DAY = date(2026, 10, 21)
ME = "helene.marchand@verdier-expertises.example"


def _collect(day=DAY, user: str = DEMO_USER) -> dict[str, object]:
    items = GraphMailConnector().collect(user, collection_window(day))
    return {item.item_id.removeprefix("mail:AAMkADmail-"): item for item in items}


def _message(n: int, when: datetime, to: str = ME, conv: str | None = None) -> dict:
    return {
        "id": f"m{n:04d}",
        "subject": f"Sujet {n}",
        "body": {"contentType": "text", "content": "Bonjour"},
        "from": {"emailAddress": {"name": "Marc", "address": "marc@verdier.example"}},
        "toRecipients": [{"emailAddress": {"name": "H", "address": to}}],
        "receivedDateTime": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "conversationId": conv or f"c{n}",
        "webLink": f"https://outlook.office365.com/owa/?ItemID=m{n:04d}",
    }


def test_every_fixture_message_becomes_a_valid_item() -> None:
    now = datetime.fromisoformat("2026-10-21T06:30:00+00:00")
    messages = [
        m
        for p in sorted((FIXTURES / "mail").glob("*.json"))
        for m in json.loads(p.read_text())["value"]
    ]
    assert len(messages) == 12
    for message in messages:
        assert message_to_item(message, DEMO_USER, now).source == "mail"


def test_window_is_24h_plus_unanswered_direct_mails_of_the_last_7_days() -> None:
    items = _collect()
    # 0001-0003 : reçus dans les 24 h (même en copie) ; 0004 : direct, sans réponse ;
    # 0009 : direct, la réponse est postérieure à la collecte.
    assert sorted(items) == ["0001", "0002", "0003", "0004", "0009"]
    # écartés : 0005 (déjà répondu), 0006 (en copie), 0007 (plus de 7 jours),
    # 0008 (reçu après la collecte), 0105 et 0109 (envoyés par l'utilisatrice)


def test_snippet_is_clean_text_and_body_is_not_kept() -> None:
    devis = _collect()["0001"]
    assert devis.snippet.startswith("Pouvez-vous valider le devis 2026-3025 avant vendredi ?")
    assert "12 400 €" in devis.snippet
    for noise in ("<", "color:red", "Cordialement", "Je reviens vers vous", "Envoyé :"):
        assert noise not in devis.snippet
    assert devis.content_text is None  # rétention par défaut (L1.7)
    assert devis.native_importance == "high" and devis.is_read is False
    assert devis.thread_id == "AAQkADconv-devis"
    assert str(devis.url).startswith("https://outlook.office365.com/owa/")


def test_mail_without_subject_and_text_signature() -> None:
    note = _collect()["0003"]
    assert note.title == ""
    assert note.snippet == "Pour info : le planning S43 est en ligne."
    assert [(p.role, p.email) for p in note.participants][-1] == ("cc", ME)


def test_invitation_shares_the_correlation_id_of_its_calendar_event() -> None:
    invitation = _collect()["0002"]
    events = GraphCalendarConnector().collect("u_demo_graph", collection_window(DAY))
    event = next(e for e in events if e.title == "Visio Cabinet Noviel")
    assert (
        invitation.correlation_id
        == event.correlation_id
        == ("meet:visio-cabinet-noviel:2026-10-22T10:15Z")
    )


def test_dates_are_in_paris_after_the_winter_time_change() -> None:
    lundi = _collect(day=date(2026, 10, 26))["0011"]
    assert lundi.created_at.isoformat() == "2026-10-25T10:00:00+01:00"
    assert lundi.collected_at.isoformat() == "2026-10-26T08:30:00+01:00"


def test_cap_keeps_the_500_most_recent_without_duplicate() -> None:
    now = datetime.fromisoformat("2026-10-21T06:30:00+00:00")
    messages = [_message(n, now - timedelta(minutes=n)) for n in range(1, 601)]
    box = Mailbox(ME, received=messages + messages[:50])  # 50 doublons
    connector = GraphMailConnector(source=lambda user, start, end: box)
    items = list(connector.collect(DEMO_USER, collection_window(DAY)))
    assert len(items) == 500 and len({i.item_id for i in items}) == 500
    assert items[0].item_id == "mail:m0001" and items[-1].item_id == "mail:m0500"


def test_invalid_message_is_rejected_alone(caplog: pytest.LogCaptureFixture) -> None:
    now = datetime.fromisoformat("2026-10-21T06:00:00+00:00")
    bad = _message(1, now) | {"webLink": "http://pas-en-https.example/secret-subject"}
    good = _message(2, now)
    connector = GraphMailConnector(source=lambda user, start, end: Mailbox(ME, [bad, good]))
    items = list(connector.collect(DEMO_USER, collection_window(DAY)))
    assert [i.item_id for i in items] == ["mail:m0002"]
    assert "objet rejeté" in caplog.text and "secret-subject" not in caplog.text


def test_read_only_scope_and_test_data_isolated() -> None:
    connector = GraphMailConnector()
    assert connector.required_scopes() == ["Mail.Read"]
    assert _collect(user="u_inconnu") == {}  # isolation L1.7 : personne d'autre
    assert connector.health().ok


def test_graph_source_asks_the_right_questions_and_follows_pages() -> None:
    now = datetime.fromisoformat("2026-10-21T06:30:00+00:00")
    first, second = _message(1, now - timedelta(hours=1)), _message(2, now - timedelta(hours=2))
    fake = FakeGraph(
        (200, {}, {"mail": ME}),
        _ok([first], "https://graph.microsoft.com/v1.0/next?p=2"),
        _ok([second]),
        _ok([]),
    )
    connector = GraphMailConnector(source=graph_source(lambda user: TOKEN, send=fake))
    items = list(connector.collect("u_demo_002", collection_window(DAY)))
    assert [i.item_id for i in items] == ["mail:m0001", "mail:m0002"]
    paths = [urlparse(url).path for url, _ in fake.requests]
    assert paths == [
        "/v1.0/me",
        "/v1.0/me/mailFolders/inbox/messages",
        "/v1.0/next",
        "/v1.0/me/mailFolders/sentitems/messages",
    ]
    inbox_query = parse_qs(urlparse(fake.requests[1][0]).query)
    assert inbox_query["$filter"] == [
        "receivedDateTime ge 2026-10-14T06:30:00Z and receivedDateTime le 2026-10-21T06:30:00Z"
    ]
    assert fake.requests[1][1]["Prefer"] == 'outlook.timezone="UTC"'
    assert all(headers["Authorization"] == f"Bearer {TOKEN}" for _, headers in fake.requests)


def test_without_token_the_mails_become_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPH_ACCESS_TOKEN", raising=False)
    result = run(DEMO_USER, collection_window(DAY), [GraphMailConnector(source=graph_source())])
    assert result.warnings == ["mail_graph: échec (AuthError)"]
    assert GraphMailConnector(source=graph_source()).health().ok is False


def test_drafts_and_mails_sent_by_the_user_are_not_collected() -> None:
    now = datetime.fromisoformat("2026-10-21T06:00:00+00:00")
    draft = _message(1, now) | {"isDraft": True}
    to_self = _message(2, now) | {"from": {"emailAddress": {"name": "H", "address": ME.upper()}}}
    connector = GraphMailConnector(source=lambda user, start, end: Mailbox(ME, [draft, to_self]))
    assert list(connector.collect(DEMO_USER, collection_window(DAY))) == []
