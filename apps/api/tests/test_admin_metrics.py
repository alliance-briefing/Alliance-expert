"""Métriques techniques : calcul, GET /admin/metrics et page /admin (L3.7, étape C)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from alliance_api.audit import COLLECT_FILE, JsonlAuditLog, api_event
from alliance_api.auth import DEMO_HEADER
from alliance_api.main import create_app
from alliance_api.metrics import PARIS, MetricsRecorder, _durations, compute
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A, USER_B, make_item
from apps.api.tests.test_admin_chain import _collect, _generator
from contracts.models import AuditEvent, Brief, Item

ADMIN = {DEMO_HEADER: "u_admin_001"}
END = date(2026, 9, 30)


def _at(event: AuditEvent, ts: str) -> AuditEvent:
    return event.model_copy(update={"ts": datetime.fromisoformat(ts)})


# --- Calcul ------------------------------------------------------------------------------


def test_taux_d_erreur_par_action_et_par_connecteur() -> None:
    events = [
        _collect(USER_A, "2026-09-30T06:30:00+02:00", ["mail:a-1", "mail:a-2"]),
        _collect(USER_A, "2026-09-29T06:30:00+02:00", [], error="GraphTimeout"),
        _collect(USER_B, "2026-09-30T06:31:00+02:00", ["mail:b-1"]),
        _collect(USER_A, "2026-09-20T06:30:00+02:00", [], error="Ancien"),  # hors période
        _at(api_event(USER_A, "view", [], 0, "brief_not_found"), "2026-09-30T09:00:00+02:00"),
        _at(api_event(USER_A, "view", [], 0, "brief_not_found"), "2026-09-30T09:01:00+02:00"),
        _at(api_event(USER_A, "view", ["mail:a-1"], 1), "2026-09-30T09:02:00+02:00"),
    ]
    metrics = compute(events, MetricsRecorder(), END, days=7)
    assert (metrics.start, metrics.end) == (date(2026, 9, 24), END)
    collect, view, generate = (metrics.actions[a] for a in ("collect", "view", "generate"))
    assert (collect.total, collect.errors, collect.error_rate) == (3, 1, 0.3333)
    assert (view.total, view.errors, view.error_rate) == (3, 2, 0.6667)
    assert (generate.total, generate.error_rate) == (0, None)
    [mail] = metrics.connectors
    assert (mail.connector, mail.total, mail.errors, mail.items) == ("mail-graph", 3, 1, 3)
    assert mail.last_run == datetime.fromisoformat("2026-09-30T06:31:00+02:00")
    assert list(metrics.error_codes.items()) == [("brief_not_found", 2), ("GraphTimeout", 1)]


def test_jour_selon_paris() -> None:
    """22 h 30 UTC le 30 = 0 h 30 le 1er octobre à Paris : hors d'une période qui finit le 30."""
    late = _collect(USER_A, "2026-09-30T22:30:00+00:00", [])
    assert compute([late], MetricsRecorder(), END, days=1).actions["collect"].total == 0


def test_percentiles_au_rang_le_plus_proche() -> None:
    d = _durations(list(range(20, 0, -1)))
    assert (d.count, d.p50_ms, d.p95_ms, d.max_ms) == (20, 10, 19, 20)
    assert _durations([7]).p95_ms == 7
    assert _durations([]).model_dump() == {
        "count": 0,
        "p50_ms": None,
        "p95_ms": None,
        "max_ms": None,
    }


def test_enregistreur_de_generations() -> None:
    start = datetime(2026, 11, 25, 8, 0, tzinfo=UTC)
    recorder = MetricsRecorder(clock=lambda: start)
    brief = _generator([make_item(USER_A, 1)], USER_A, DAY)
    recorder.generation(120, brief)
    recorder.generation(40, None)
    g = compute([], recorder, END, days=1).generation
    assert (g.since, g.total, g.failures) == (start, 2, 1)
    assert (g.measured.count, g.measured.max_ms) == (2, 120)
    assert (g.declared.count, g.declared.p50_ms) == (1, 2100)  # latence annoncée par le brief
    assert (g.tokens_in, g.tokens_out) == (1800, 62)  # l'échec ne compte pas


# --- API et page -------------------------------------------------------------------------


def _failing(items: Sequence[Item], user: str, day: date | str, **kw: object) -> Brief:
    raise RuntimeError("panne du générateur")


@pytest.fixture
def api(data_root: Path) -> TestClient:
    now = datetime.now(UTC).isoformat()
    collect = JsonlAuditLog(data_root, COLLECT_FILE)
    collect.write(_collect(USER_A, now, [f"mail:{USER_A}-1"]))
    collect.write(_collect(USER_A, now, [], error="GraphTimeout"))
    app = create_app(
        item_store=DemoItemStore(data_root),
        generator=_generator,
        collect_audit=collect,
        auth_mode="demo",
    )
    client = TestClient(app)
    assert client.post("/briefs", json={"date": DAY}, headers={DEMO_HEADER: USER_A}).is_success
    return client


def test_reserve_a_la_permission_audit_read(api: TestClient) -> None:
    assert api.get("/admin/metrics").status_code == 401
    assert api.get("/admin/metrics", headers={DEMO_HEADER: USER_A}).status_code == 403
    for days in (0, 91):
        assert api.get("/admin/metrics", params={"days": days}, headers=ADMIN).status_code == 422


def test_metriques_de_l_api(api: TestClient) -> None:
    body = api.get("/admin/metrics", params={"days": 1}, headers=ADMIN).json()
    assert body["actions"]["collect"] == {"total": 2, "errors": 1, "error_rate": 0.5}
    assert body["actions"]["generate"]["total"] == 1
    assert body["connectors"][0]["connector"] == "mail-graph"
    generation = body["generation"]
    assert (generation["total"], generation["failures"]) == (1, 0)
    assert generation["measured"]["count"] == 1 and generation["declared"]["p50_ms"] == 2100


def test_echec_de_generation_compte(data_root: Path) -> None:
    client = TestClient(
        create_app(item_store=DemoItemStore(data_root), generator=_failing, auth_mode="demo")
    )
    assert (
        client.post("/briefs", json={"date": DAY}, headers={DEMO_HEADER: USER_A}).status_code == 500
    )
    generation = client.get("/admin/metrics", headers=ADMIN).json()["generation"]
    assert (generation["total"], generation["failures"], generation["declared"]["count"]) == (
        1,
        1,
        0,
    )


def test_page_affiche_les_metriques(api: TestClient) -> None:
    assert api.post("/login", data={"user_id": "u_admin_001"}, follow_redirects=False).is_redirect
    page = api.get("/admin").text
    assert f"au {datetime.now(PARIS).strftime('%d/%m/%Y')}" in page
    assert "50.0 %" in page and "GraphTimeout (1)" in page
    assert "1, dont 0 en échec" in page and "non mesurées" in page
