"""GET /briefs?days= (#95) : historique des briefs, fenêtre, statuts, isolation."""

from collections.abc import Callable
from pathlib import Path

import pytest
from alliance_api.audit import LogAuditSink
from alliance_api.main import create_app
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A, USER_B, make_item, write_day

Headers = Callable[[str], dict[str, str]]
# Jours de USER_A : DAY (2026-09-30, fixture) plus ceux-ci ; USER_B n'a que DAY.
OTHER_DAYS = ["2026-08-31", "2026-09-01", "2026-09-15", "2026-10-01"]


@pytest.fixture
def api(data_root: Path, audit: LogAuditSink) -> TestClient:
    for day in OTHER_DAYS:
        write_day(data_root, USER_A, day, [make_item(USER_A, 1)])
    write_day(data_root, USER_A, "pas-un-jour", [make_item(USER_A, 1)])  # ignoré
    return TestClient(
        create_app(item_store=DemoItemStore(data_root), audit=audit, auth_mode="demo")
    )


def _history(client: TestClient, user: str, **params: object) -> list[dict]:
    response = client.get("/briefs", params=params, headers={"X-Demo-User": user})
    assert response.status_code == 200, response.text
    return response.json()


def test_fenetre_de_30_jours_du_plus_recent_au_plus_ancien(api: TestClient) -> None:
    days = _history(api, USER_A, days=30, end=DAY)
    # 2026-08-31 est hors fenêtre (31 jours avant), 2026-10-01 est après `end`.
    assert [d["date"] for d in days] == [DAY, "2026-09-15", "2026-09-01"]
    assert all(d == {"date": d["date"], "status": "not_generated"} for d in days)


def test_brief_genere_resume_et_lien_vers_le_brief(api: TestClient, as_user: Headers) -> None:
    api.post("/briefs", json={"date": DAY}, headers=as_user(USER_A))
    done, *_ = _history(api, USER_A, end=DAY)
    assert done["status"] == "done"
    assert done["brief_id"] == f"b_{DAY}_{USER_A}"
    assert {"generated_at", "entries"} <= done.keys()
    assert "sections" not in done
    full = api.get(f"/briefs/{done['date']}", headers=as_user(USER_A)).json()
    assert done["entries"] == sum(len(s["entries"]) for s in full["sections"])


def test_un_utilisateur_ne_voit_que_ses_jours(api: TestClient, as_user: Headers) -> None:
    api.post("/briefs", json={"date": DAY}, headers=as_user(USER_A))
    assert _history(api, USER_B, end=DAY) == [{"date": DAY, "status": "not_generated"}]


def test_aucun_jour_liste_vide(api: TestClient) -> None:
    assert _history(api, USER_A, end="2026-01-01") == []
    assert _history(api, "u_sans_donnees", end=DAY) == []


def test_un_seul_jour(api: TestClient) -> None:
    assert [d["date"] for d in _history(api, USER_A, days=1, end=DAY)] == [DAY]


@pytest.mark.parametrize("params", [{"days": 0}, {"days": 91}, {"days": "x"}, {"end": "hier"}])
def test_parametres_invalides_422(api: TestClient, as_user: Headers, params: dict) -> None:
    response = api.get("/briefs", params=params, headers=as_user(USER_A))
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_sans_utilisateur_401(api: TestClient) -> None:
    assert api.get("/briefs").status_code == 401


def test_consultation_journalisee(api: TestClient, as_user: Headers, audit: LogAuditSink) -> None:
    api.post("/briefs", json={"date": DAY}, headers=as_user(USER_A))
    _history(api, USER_A, end=DAY)
    event = audit.events[-1]
    assert (event.action, event.user_id, event.status) == ("view", USER_A, "ok")
    assert event.count == len(event.item_ids) > 0
