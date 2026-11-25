"""Chaîne d'un brief pour l'administrateur : GET /admin/chain (L3.7, étape B)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import pytest
from alliance_api.audit import COLLECT_FILE, JsonlAuditLog, LogAuditSink, api_event
from alliance_api.auth import DEMO_HEADER
from alliance_api.chain import brief_chain
from alliance_api.main import create_app
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A, USER_B, make_item, write_day
from contracts.models import AuditEvent, Brief, Item

ADMIN = {DEMO_HEADER: "u_admin_001"}
AS_A = {DEMO_HEADER: USER_A}
VEILLE = "2026-09-29"
EMPTY_BRIEF = (
    Path(__file__).parents[3] / "contracts/fixtures/brief/valid/brief-vide-avec-alerte.json"
)
SECRET = "Texte du brief que l'admin ne lit pas"


def _generator(items: Sequence[Item], user: str, day: date | str, **kw: object) -> Brief:
    """Un brief qui ne cite que le premier item du jour."""
    entry = {"rank": 1, "priority_score": 80, "priority_reasons": ["raison"], "text": SECRET,
             "claims": [], "recommended_action": None, "source_item_ids": [items[0].item_id]}  # fmt: skip
    return Brief.model_validate(
        {
            "brief_id": f"b_{day}_{user}",
            "owner_user_id": user,
            "date": str(day),
            "generated_at": f"{day}T08:31:10+02:00",
            "generator": {"llm": "modele-local", "llm_version": "1", "prompt_version": "p7"},
            "sections": [{"kind": "actions", "entries": [entry]}],
            "stats": {
                "items_in": len(items),
                "items_out": 1,
                "tokens_in": 1800,
                "tokens_out": 62,
                "latency_ms": 2100,
                "cost_eur": 0.0,
            },
            "warnings": ["1 élément ignoré : source indisponible"],
        }
    )


def _collect(user: str, ts: str, ids: list[str], error: str | None = None) -> AuditEvent:
    return AuditEvent(
        ts=datetime.fromisoformat(ts),
        actor="connector:mail-graph",
        user_id=user,
        action="collect",
        source=None if error else "mail",
        item_ids=ids,
        count=len(ids),
        status="error" if error else "ok",
        error_code=error,
    )


@pytest.fixture
def audit() -> LogAuditSink:
    return LogAuditSink()


@pytest.fixture
def api(data_root: Path, audit: LogAuditSink) -> TestClient:
    write_day(data_root, USER_A, VEILLE, [make_item(USER_A, 3)])
    collect = JsonlAuditLog(data_root, COLLECT_FILE)
    collect.write(
        _collect(USER_A, f"{DAY}T06:30:00+02:00", [f"mail:{USER_A}-1", f"mail:{USER_A}-2"])
    )
    collect.write(_collect(USER_A, f"{DAY}T06:31:00+02:00", [], error="GraphTimeout"))
    collect.write(_collect(USER_A, f"{VEILLE}T06:30:00+02:00", [f"mail:{USER_A}-3"]))
    collect.write(_collect(USER_B, f"{DAY}T06:30:00+02:00", [f"mail:{USER_B}-1"]))
    # Le lendemain, le même mail non lu est recollecté : sans rapport avec le brief du jour.
    collect.write(_collect(USER_A, "2026-10-01T06:30:00+02:00", [f"mail:{USER_A}-1"]))
    app = create_app(
        item_store=DemoItemStore(data_root),
        generator=_generator,
        audit=audit,
        collect_audit=collect,
        auth_mode="demo",
    )
    client = TestClient(app)
    for day in (VEILLE, DAY):
        assert client.post("/briefs", json={"date": day}, headers=AS_A).is_success
    assert client.get(f"/briefs/{DAY}", headers=AS_A).is_success
    assert client.get("/briefs", params={"end": DAY}, headers=AS_A).is_success
    return client


def _chain(client: TestClient, user: str = USER_A, day: str = DAY) -> dict:
    response = client.get("/admin/chain", params={"user": user, "date": day}, headers=ADMIN)
    assert response.status_code == 200, response.text
    return response.json()


def test_reserve_a_la_permission_audit_read(api: TestClient) -> None:
    params = {"user": USER_A, "date": DAY}
    assert api.get("/admin/chain", params=params).status_code == 401
    assert api.get("/admin/chain", params=params, headers=AS_A).status_code == 403


def test_collecte_du_jour_et_items_cites(api: TestClient) -> None:
    steps = [
        (s["connector"], s["count"], s["cited"], s["status"]) for s in _chain(api)["collection"]
    ]
    # Le jour du brief : la collecte réussie (1 item cité sur 2) et l'échec ; ni la veille
    # (aucun item cité), ni le lendemain (postérieur au brief), ni l'utilisateur B.
    assert steps == [("mail-graph", 2, 1, "ok"), ("mail-graph", 0, 0, "error")]


def test_generation_lue_dans_le_brief(api: TestClient) -> None:
    generation = _chain(api)["generation"]
    assert generation["brief_id"] == f"b_{DAY}_{USER_A}"
    assert (generation["llm"], generation["prompt_version"]) == ("modele-local", "p7")
    assert (generation["tokens_in"], generation["tokens_out"]) == (1800, 62)
    assert (generation["warnings"], generation["cited_items"]) == (1, 1)
    assert len(generation["api_calls"]) == 1  # la génération de la veille n'est pas rattachée


def test_consultations_du_brief_et_de_l_historique(api: TestClient) -> None:
    kinds = [(c["user_id"], c["kind"]) for c in _chain(api)["consultations"]]
    assert kinds == [(USER_A, "brief"), (USER_A, "historique")]


def test_ni_texte_du_brief_ni_consultation_journalisee(
    api: TestClient, audit: LogAuditSink
) -> None:
    before = len(audit.events)
    body = api.get("/admin/chain", params={"user": USER_A, "date": DAY}, headers=ADMIN).text
    assert SECRET not in body and "source indisponible" not in body and "@" not in body
    assert len(audit.events) == before  # l'admin ne « consulte » pas le brief du manager


def test_brief_non_genere_et_utilisateur_invalide(api: TestClient) -> None:
    response = api.get("/admin/chain", params={"user": USER_B, "date": DAY}, headers=ADMIN)
    assert (response.status_code, response.json()["code"]) == (404, "brief_not_found")
    response = api.get("/admin/chain", params={"user": "../x", "date": DAY}, headers=ADMIN)
    assert (response.status_code, response.json()["code"]) == (422, "invalid_request")


def test_brief_sans_item_cite_sans_rapprochement() -> None:
    brief = Brief.model_validate_json(EMPTY_BRIEF.read_text(encoding="utf-8"))
    owner = brief.owner_user_id
    chain = brief_chain(
        brief, [api_event(owner, "generate", [], 0), api_event(owner, "view", [], 0)], []
    )
    assert chain.generation.api_calls == [] and chain.consultations == []
