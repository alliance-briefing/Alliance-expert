"""Journal d'audit persistant et GET /admin/audit (L3.7, étape A)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from alliance_api.audit import API_FILE, COLLECT_FILE, JsonlAuditLog, LogAuditSink, api_event
from alliance_api.auth import DEMO_HEADER
from alliance_api.main import create_app
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A, USER_B, XSS
from contracts.models import AuditEvent

ADMIN = "u_admin_001"
AS_ADMIN = {DEMO_HEADER: ADMIN}


def _collect(user: str, ts: str, count: int = 2, error: str | None = None) -> AuditEvent:
    return AuditEvent(
        ts=datetime.fromisoformat(ts),
        actor="connector:mail-graph",
        user_id=user,
        action="collect",
        source=None if error else "mail",
        item_ids=[] if error else [f"mail:{user}-{n}" for n in range(1, count + 1)],
        count=0 if error else count,
        status="error" if error else "ok",
        error_code=error,
    )


@pytest.fixture
def audit_dir(tmp_path: Path) -> Path:
    return tmp_path / "audit"


@pytest.fixture
def api(data_root: Path, audit_dir: Path) -> TestClient:
    """Le lot 1 a collecté pour A (le matin du jour, puis en échec la veille) et pour B."""
    collect = JsonlAuditLog(data_root, COLLECT_FILE)
    collect.write(_collect(USER_A, f"{DAY}T06:30:00+02:00"))
    collect.write(_collect(USER_A, "2026-09-29T06:30:00+02:00", error="GraphTimeout"))
    collect.write(_collect(USER_B, f"{DAY}T06:31:00+02:00", count=1))
    app = create_app(
        item_store=DemoItemStore(data_root),
        audit=JsonlAuditLog(audit_dir),
        collect_audit=collect,
        auth_mode="demo",
    )
    client = TestClient(app)
    for user in (USER_A, USER_B):
        assert client.post("/briefs", json={"date": DAY}, headers={DEMO_HEADER: user}).is_success
    assert client.get(f"/briefs/{DAY}", headers={DEMO_HEADER: USER_A}).is_success
    return client


def _audit(client: TestClient, **params: str | int) -> dict:
    response = client.get("/admin/audit", params=params, headers=AS_ADMIN)
    assert response.status_code == 200, response.text
    return response.json()


# --- Droits ------------------------------------------------------------------------------


def test_reserve_a_la_permission_audit_read(api: TestClient) -> None:
    assert api.get("/admin/audit").status_code == 401
    response = api.get("/admin/audit", headers={DEMO_HEADER: USER_A})
    assert (response.status_code, response.json()["code"]) == (403, "forbidden")
    assert api.get("/admin/audit", headers=AS_ADMIN).status_code == 200


def test_identifiant_utilisateur_invalide_refuse(api: TestClient) -> None:
    response = api.get("/admin/audit", params={"user": "../u_test_a"}, headers=AS_ADMIN)
    assert (response.status_code, response.json()["code"]) == (422, "invalid_request")


# --- Contenu et filtres ------------------------------------------------------------------


def test_reunit_collecte_et_api_du_plus_recent_au_plus_ancien(api: TestClient) -> None:
    page = _audit(api)
    actions = [(e["user_id"], e["action"], e["actor"]) for e in page["events"]]
    assert page["total"] == len(actions) == 6
    assert actions[0] == (USER_A, "view", "api")
    assert actions[-1] == (USER_A, "collect", "connector:mail-graph")
    stamps = [e["ts"] for e in page["events"]]
    assert stamps == sorted(stamps, reverse=True)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"user": USER_B}, [(USER_B, "generate"), (USER_B, "collect")]),
        ({"user": USER_A, "action": "collect"}, [(USER_A, "collect"), (USER_A, "collect")]),
        ({"status": "error"}, [(USER_A, "collect")]),
        ({"date": "2026-09-29"}, [(USER_A, "collect")]),
        ({"user": USER_A, "action": "view"}, [(USER_A, "view")]),
    ],
)
def test_filtres(api: TestClient, params: dict[str, str], expected: list[tuple[str, str]]) -> None:
    assert [(e["user_id"], e["action"]) for e in _audit(api, **params)["events"]] == expected


def test_limite_garde_le_total(api: TestClient) -> None:
    page = _audit(api, limit=2)
    assert (page["total"], len(page["events"])) == (6, 2)
    assert api.get("/admin/audit", params={"limit": 0}, headers=AS_ADMIN).status_code == 422


def test_aucun_contenu_de_message(api: TestClient) -> None:
    body = api.get("/admin/audit", headers=AS_ADMIN).text
    assert XSS not in body and "snippet" not in body and "@" not in body


def test_jour_selon_paris(data_root: Path) -> None:
    """23 h 30 UTC le 29 = 1 h 30 le 30 à Paris : l'événement appartient au 30."""
    collect = JsonlAuditLog(data_root, COLLECT_FILE)
    collect.write(_collect(USER_A, "2026-09-29T23:30:00+00:00"))
    client = TestClient(
        create_app(item_store=DemoItemStore(data_root), collect_audit=collect, auth_mode="demo")
    )
    page = client.get("/admin/audit", params={"date": DAY}, headers=AS_ADMIN).json()
    assert page["total"] == 1


# --- Stockage ----------------------------------------------------------------------------


def test_journal_api_persistant_et_journal_du_lot_1_intact(
    api: TestClient, data_root: Path, audit_dir: Path
) -> None:
    lot1 = (data_root / USER_A / COLLECT_FILE).read_text(encoding="utf-8")
    api.get(f"/briefs/{DAY}", headers={DEMO_HEADER: USER_A})
    assert (data_root / USER_A / COLLECT_FILE).read_text(encoding="utf-8") == lot1
    # Après « redémarrage » : un nouveau journal sur le même dossier relit tout.
    relu = JsonlAuditLog(audit_dir).read(USER_A)
    assert [e.action for e in relu] == ["generate", "view", "view"]
    assert (audit_dir / USER_A / API_FILE).is_file()


def test_ligne_illisible_ignoree(audit_dir: Path) -> None:
    log = JsonlAuditLog(audit_dir)
    log.write(api_event(USER_A, "view", [], 0))
    with (audit_dir / USER_A / API_FILE).open("a", encoding="utf-8") as out:
        out.write('{"ts": "coupé\n')
    log.write(api_event(USER_A, "generate", [], 0))
    assert [e.action for e in log.read()] == ["view", "generate"]


def test_dossier_absent_et_utilisateur_invalide(tmp_path: Path) -> None:
    log = JsonlAuditLog(tmp_path / "absent")
    assert log.read() == [] and log.read(USER_A) == []
    with pytest.raises(ValueError, match="invalide"):
        log.read("../etc")


def test_journal_en_memoire_filtre_par_utilisateur() -> None:
    sink = LogAuditSink()
    sink.write(api_event(USER_A, "view", [], 0))
    sink.write(api_event(USER_B, "view", [], 0))
    assert [e.user_id for e in sink.read(USER_B)] == [USER_B] and len(sink.read()) == 2


def test_journaux_par_defaut_selon_les_variables(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("DATA_DIR", str(data_root))
    app = create_app(auth_mode="demo")
    assert isinstance(app.state.audit, JsonlAuditLog)
    assert app.state.audit.root == tmp_path / "audit"
    assert app.state.collect_audit.root == data_root
    assert app.state.collect_audit.file_name == COLLECT_FILE
    monkeypatch.delenv("AUDIT_DIR")
    assert isinstance(create_app(auth_mode="demo").state.audit, LogAuditSink)
