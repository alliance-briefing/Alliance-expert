"""Contrat A et C : ce qui est conforme passe, tout le reste est rejeté (guide §2.5, L3.1)."""

from __future__ import annotations

import copy
from datetime import datetime

import pytest
from pydantic import ValidationError

from contracts.export_schemas import export
from contracts.models import AuditEvent, Item

VALID_MAIL = {
    "schema_version": "1.0",
    "item_id": "mail:AAMkAD-1",
    "source": "mail",
    "provider": "demo",
    "owner_user_id": "u_123",
    "title": "Re: Devis sinistre 2026-0412",
    "snippet": "Merci de valider le devis avant demain 12h.",
    "content_text": None,
    "created_at": "2026-09-29T14:03:00+02:00",
    "updated_at": "2026-09-30T08:12:00+02:00",
    "participants": [{"name": "Julie Marchand", "email": "julie@horizon-sud.test", "role": "from"}],
    "thread_id": "conv:1",
    "native_importance": "high",
    "is_read": False,
    "url": "https://outlook.reseau-expertis.test/mail/id/1",
    "acl": {"visible_to": ["u_123"]},
    "collected_at": "2026-09-30T08:30:02+02:00",
    "connector": {"name": "demo", "version": "0.1.0"},
}
VALID_EVENT = VALID_MAIL | {
    "item_id": "calendar:evt-1",
    "source": "calendar",
    "is_read": None,
    "thread_id": None,
    "start_at": "2026-10-01T09:00:00+02:00",
    "end_at": "2026-10-01T10:00:00+02:00",
}
VALID_TASK = VALID_MAIL | {
    "item_id": "task:t-1",
    "source": "task",
    "is_read": None,
    "due_at": "2026-10-01T12:00:00+02:00",
    "completed": False,
}


def _with(base: dict, **changes: object) -> dict:
    data = copy.deepcopy(base)
    for key, value in changes.items():
        if value is ...:
            data.pop(key)
        else:
            data[key] = value
    return data


@pytest.mark.parametrize(
    "payload", [VALID_MAIL, VALID_EVENT, VALID_TASK], ids=["mail", "calendar", "task"]
)
def test_valid_items(payload: dict) -> None:
    item = Item.model_validate(payload)
    assert item.created_at.utcoffset() is not None


INVALID = {
    "url absente": _with(VALID_MAIL, url=...),
    "url en http": _with(VALID_MAIL, url="http://exemple.test/1"),
    "date sans fuseau": _with(VALID_MAIL, created_at="2026-09-29T14:03:00"),
    "extrait > 500": _with(VALID_MAIL, snippet="x" * 501),
    "préfixe incohérent": _with(VALID_MAIL, item_id="task:1"),
    "propriétaire hors acl": _with(VALID_MAIL, acl={"visible_to": ["u_999"]}),
    "acl vide": _with(VALID_MAIL, acl={"visible_to": []}),
    "champ inconnu": _with(VALID_MAIL, couleur="rouge"),
    "source inconnue": _with(VALID_MAIL, source="sms", item_id="sms:1"),
    "fournisseur inconnu": _with(VALID_MAIL, provider="yahoo"),
    "version de schéma": _with(VALID_MAIL, schema_version="2.0"),
    "updated avant created": _with(VALID_MAIL, updated_at="2026-09-28T00:00:00+02:00"),
    "email invalide": _with(
        VALID_MAIL, participants=[{"name": "X", "email": "pas-un-mail", "role": "from"}]
    ),
    "rôle inconnu": _with(
        VALID_MAIL, participants=[{"name": "X", "email": "x@y.test", "role": "boss"}]
    ),
    "version connecteur": _with(VALID_MAIL, connector={"name": "demo", "version": "v1"}),
    "due_at sur un mail": _with(VALID_MAIL, due_at="2026-10-01T12:00:00+02:00"),
    "événement sans début": _with(VALID_EVENT, start_at=None),
    "fin avant début": _with(VALID_EVENT, end_at="2026-10-01T08:00:00+02:00"),
    "tâche sans completed": _with(VALID_TASK, completed=None),
    "is_read sur une tâche": _with(VALID_TASK, is_read=False),
    "caractère de contrôle": _with(VALID_MAIL, title="Devis\x00"),
}


@pytest.mark.parametrize("payload", INVALID.values(), ids=INVALID.keys())
def test_invalid_items_are_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        Item.model_validate(payload)


def test_items_are_immutable() -> None:
    item = Item.model_validate(VALID_MAIL)
    with pytest.raises(ValidationError):
        item.title = "modifié"  # type: ignore[misc]


def test_overdue_rule() -> None:
    now = datetime.fromisoformat("2026-10-02T08:30:00+02:00")
    assert Item.model_validate(VALID_TASK).is_overdue(now)
    assert not Item.model_validate(_with(VALID_TASK, due_at=None)).is_overdue(
        now
    )  # sans échéance : jamais
    assert not Item.model_validate(_with(VALID_TASK, completed=True)).is_overdue(now)


def test_json_schema_export(tmp_path) -> None:
    paths = export(tmp_path)
    assert {p.name for p in paths} == {
        "item.schema.json",
        "brief.schema.json",
        "audit_event.schema.json",
    }
    schema = paths[0].read_text("utf-8")
    assert '"item_id"' in schema and '"required"' in schema


def test_audit_event_refuses_personal_data() -> None:
    base = {
        "ts": "2026-09-30T08:30:02+02:00",
        "actor": "connector:demo",
        "user_id": "u_123",
        "action": "collect",
        "item_ids": ["mail:1"],
        "count": 1,
        "status": "ok",
    }
    AuditEvent.model_validate(base)
    with pytest.raises(ValidationError):
        AuditEvent.model_validate(base | {"user_id": "camille@reseau-expertis.test"})
    with pytest.raises(ValidationError):
        AuditEvent.model_validate(base | {"status": "error"})  # error_code obligatoire
    with pytest.raises(ValidationError):
        AuditEvent.model_validate(base | {"actor": "moi"})
