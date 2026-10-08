from collections.abc import Callable

from alliance_api.audit import LogAuditSink
from alliance_api.main import create_app
from fastapi.testclient import TestClient

from apps.api.tests.factories import USER_A, USER_B, XSS
from contracts.demo_data import DEFAULT_USER, available_days, load_items

Headers = Callable[[str], dict[str, str]]


def test_item_de_l_utilisateur_avec_son_lien(client: TestClient, as_user: Headers) -> None:
    response = client.get(f"/items/mail:{USER_A}-2", headers=as_user(USER_A))
    assert response.status_code == 200
    body = response.json()
    assert body["url"] == f"https://outlook.reseau-expertis.test/mail/id/{USER_A}-2"
    assert body["connector"] == "demo/0.1.0"


def test_ni_corps_ni_participants_ni_acl(client: TestClient, as_user: Headers) -> None:
    body = client.get(f"/items/mail:{USER_A}-2", headers=as_user(USER_A)).json()
    assert not {"content_text", "participants", "acl", "owner_user_id"} & set(body)
    assert "@" not in str(body)


def test_snippet_renvoye_tel_quel_en_texte_brut(client: TestClient, as_user: Headers) -> None:
    # L'API sert du JSON : l'échappement HTML revient à l'interface, au rendu.
    body = client.get(f"/items/mail:{USER_A}-1", headers=as_user(USER_A)).json()
    assert body["snippet"] == XSS


def test_item_d_un_autre_utilisateur_403(
    client: TestClient, as_user: Headers, audit: LogAuditSink
) -> None:
    response = client.get(f"/items/mail:{USER_B}-1", headers=as_user(USER_A))
    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"
    assert USER_B not in response.text
    event = audit.events[-1]
    assert (event.user_id, event.status, event.error_code) == (USER_A, "error", "forbidden")


def test_item_inconnu_404(client: TestClient, as_user: Headers) -> None:
    response = client.get("/items/mail:inconnu", headers=as_user(USER_A))
    assert response.status_code == 404
    assert response.json()["code"] == "item_not_found"


def test_chaque_consultation_ecrit_un_audit_view(
    client: TestClient, as_user: Headers, audit: LogAuditSink
) -> None:
    client.get(f"/items/mail:{USER_A}-2", headers=as_user(USER_A))
    event = audit.events[-1]
    assert (event.actor, event.action, event.status) == ("api", "view", "ok")
    assert event.item_ids == [f"mail:{USER_A}-2"] and event.count == 1


def test_aucune_route_n_expose_la_verite_terrain(client: TestClient, as_user: Headers) -> None:
    assert not any("truth" in path for path in client.get("/openapi.json").json()["paths"])
    response = client.get("/items/file:../truth.json", headers=as_user(USER_A))
    assert response.status_code == 404
    assert "vérité" not in response.text


def test_identifiant_invalide_422_sans_recopier_l_entree(
    client: TestClient, as_user: Headers
) -> None:
    response = client.get("/items/pas-un-id<b>", headers=as_user(USER_A))
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert "<b>" not in response.text


def test_donnees_de_demo_reelles(as_user: Headers) -> None:
    item = load_items(available_days()[0])[0]
    client = TestClient(create_app(auth_mode="demo"))
    response = client.get(f"/items/{item.item_id}", headers=as_user(DEFAULT_USER))
    assert response.status_code == 200
    assert response.json()["url"] == str(item.url)
