import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pytest
from alliance_api.audit import LogAuditSink
from alliance_api.main import create_app
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A, USER_B, XSS, make_item, write_day
from contracts.demo_data import DEFAULT_USER
from contracts.models import Brief, Item
from packages.analyse import generate_brief

Headers = Callable[[str], dict[str, str]]


def _client(root: Path, audit: LogAuditSink | None = None, **deps: object) -> TestClient:
    store = DemoItemStore(root)
    return TestClient(create_app(item_store=store, audit=audit, auth_mode="demo", **deps))


def _generate(client: TestClient, user: str, day: str = DAY) -> dict:
    response = client.post("/briefs", json={"date": day}, headers={"X-Demo-User": user})
    assert response.status_code == 201, response.text
    return response.json()


def test_generer_puis_lire_un_brief_valide(client: TestClient, as_user: Headers) -> None:
    status = _generate(client, USER_A)
    assert status == {"status": "done", "brief_id": f"b_{DAY}_{USER_A}", "date": DAY}
    response = client.get(f"/briefs/{DAY}", headers=as_user(USER_A))
    assert response.status_code == 200
    brief = Brief.model_validate(response.json())
    assert brief.owner_user_id == USER_A


def test_sans_utilisateur_401(client: TestClient) -> None:
    assert client.post("/briefs", json={"date": DAY}).status_code == 401
    assert client.get(f"/briefs/{DAY}").status_code == 401


def test_le_generateur_ne_recoit_que_les_items_de_l_utilisateur(tmp_path: Path) -> None:
    # Un item de B rangé par erreur dans le dossier de A ne doit pas être transmis.
    write_day(tmp_path, USER_A, DAY, [make_item(USER_A, 1), make_item(USER_B, 9)])
    received: list[Item] = []

    def spy(items: Sequence[Item], user: str, day: date | str, **kw: object) -> Brief:
        received.extend(items)
        return generate_brief(items, user, day)

    _generate(_client(tmp_path, generator=spy), USER_A)
    assert {item.owner_user_id for item in received} == {USER_A}


def test_un_utilisateur_ne_lit_pas_le_brief_d_un_autre(
    client: TestClient, as_user: Headers
) -> None:
    _generate(client, USER_A)
    response = client.get(f"/briefs/{DAY}", headers=as_user(USER_B))
    assert response.status_code == 404
    assert USER_A not in response.text


def test_jour_sans_collecte_404_et_audit_en_erreur(
    client: TestClient, as_user: Headers, audit: LogAuditSink
) -> None:
    response = client.post("/briefs", json={"date": "2026-01-01"}, headers=as_user(USER_A))
    assert response.status_code == 404
    assert response.json()["code"] == "no_items"
    assert (audit.events[-1].action, audit.events[-1].error_code) == ("generate", "no_items")


def test_brief_non_genere_404(client: TestClient, as_user: Headers) -> None:
    response = client.get(f"/briefs/{DAY}", headers=as_user(USER_A))
    assert response.status_code == 404
    assert response.json()["code"] == "brief_not_found"


def test_date_invalide_422(client: TestClient, as_user: Headers) -> None:
    assert client.get("/briefs/demain", headers=as_user(USER_A)).status_code == 422
    response = client.post("/briefs", json={"date": "07/10/2026"}, headers=as_user(USER_A))
    assert response.status_code == 422


def test_audit_generate_puis_view_avec_les_items_cites(
    client: TestClient, as_user: Headers, audit: LogAuditSink
) -> None:
    _generate(client, USER_A)
    client.get(f"/briefs/{DAY}", headers=as_user(USER_A))
    generate, view = audit.events[-2], audit.events[-1]
    assert (generate.action, view.action) == ("generate", "view")
    assert generate.item_ids == view.item_ids == [f"mail:{USER_A}-1", f"mail:{USER_A}-2"]
    assert generate.status == view.status == "ok"


def _failing(items: Sequence[Item], user: str, day: date | str, **kw: object) -> Brief:
    raise RuntimeError("LLM indisponible")


def _citing_unknown(items: Sequence[Item], user: str, day: date | str, **kw: object) -> Brief:
    data = generate_brief(items, user, day).model_dump(mode="json")
    data["sections"][0]["entries"][0]["source_item_ids"] = ["mail:inconnu"]
    return Brief.model_validate(data)


@pytest.mark.parametrize("generator", [_failing, _citing_unknown])
def test_generation_en_echec_500_sans_cache(
    data_root: Path, as_user: Headers, generator: Callable[..., Brief]
) -> None:
    audit = LogAuditSink()
    client = _client(data_root, audit, generator=generator)
    response = client.post("/briefs", json={"date": DAY}, headers=as_user(USER_A))
    assert response.status_code == 500
    assert response.json()["code"] == "generation_failed"
    assert "LLM" not in response.text
    assert audit.events[-1].error_code == "generation_failed"
    assert client.get(f"/briefs/{DAY}", headers=as_user(USER_A)).status_code == 404


def test_l_api_ne_depend_pas_du_generateur(data_root: Path, as_user: Headers) -> None:
    def llm(items: Sequence[Item], user: str, day: date | str, **kw: object) -> Brief:
        data = generate_brief(items, user, day).model_dump(mode="json")
        data["generator"] = {"llm": "modele-local", "llm_version": "1", "temperature": 0}
        data["sections"] = [s for s in data["sections"] if s["kind"] == "actions"]
        return Brief.model_validate(data)

    client = _client(data_root, generator=llm)
    _generate(client, USER_A)
    body = client.get(f"/briefs/{DAY}", headers=as_user(USER_A)).json()
    assert body["generator"]["llm"] == "modele-local"


def test_textes_bruts_renvoyes_tels_quels(tmp_path: Path, as_user: Headers) -> None:
    write_day(tmp_path, USER_A, DAY, [make_item(USER_A, 1, title=XSS, snippet=XSS)])
    client = _client(tmp_path)
    _generate(client, USER_A)
    brief = client.get(f"/briefs/{DAY}", headers=as_user(USER_A)).json()
    assert XSS in brief["sections"][0]["entries"][0]["text"]


def test_demo_chaque_item_cite_a_un_lien_source_et_lecture_rapide(as_user: Headers) -> None:
    client = TestClient(create_app(auth_mode="demo"))
    _generate(client, DEFAULT_USER, "2026-10-07")
    started = time.perf_counter()
    response = client.get("/briefs/2026-10-07", headers=as_user(DEFAULT_USER))
    assert time.perf_counter() - started < 2
    brief = Brief.model_validate(response.json())
    assert brief.cited_item_ids()
    for item_id in brief.cited_item_ids():
        item = client.get(f"/items/{item_id}", headers=as_user(DEFAULT_USER))
        assert item.status_code == 200
        assert item.json()["url"].startswith("https://")
