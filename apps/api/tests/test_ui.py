"""Interface (L3.3) : connexion de démo, XSS, lien source pour chaque texte, alertes."""

import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from alliance_api.auth import DEMO_COOKIE
from alliance_api.main import create_app
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A, USER_B, XSS, make_item, write_day
from contracts.models import Brief, Item
from packages.analyse import generate_brief


def _logged_in(client: TestClient, user: str = USER_A) -> TestClient:
    response = client.post("/login", data={"user_id": user}, follow_redirects=False)
    assert response.status_code == 303
    assert "samesite=strict" in response.headers["set-cookie"].lower()
    assert "httponly" in response.headers["set-cookie"].lower()
    return client


def _brief_page(client: TestClient) -> str:
    response = client.post("/generate", data={"date": DAY})
    assert response.status_code == 200, response.text
    return response.text


def test_sans_connexion_redirige_vers_la_connexion(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert (response.status_code, response.headers["location"]) == (303, "/login")


def test_identifiant_invalide_refuse(client: TestClient) -> None:
    response = client.post("/login", data={"user_id": "../etc"}, follow_redirects=False)
    assert response.status_code == 401
    assert DEMO_COOKIE not in response.headers.get("set-cookie", "")


def test_page_sans_brief_propose_de_le_generer(client: TestClient) -> None:
    page = _logged_in(client).get(f"/?date={DAY}").text
    assert "Générer le brief" in page


def test_brief_affiche_sections_score_raisons_et_sources(client: TestClient) -> None:
    page = _brief_page(_logged_in(client))
    for label in ("À traiter", "Réunions", "Retards"):
        assert label in page
    assert "Priorité 60/100" in page
    assert f'href="https://outlook.reseau-expertis.test/mail/id/{USER_A}-2"' in page


def test_script_d_un_mail_echappe_et_non_injecte(tmp_path: Path) -> None:
    write_day(tmp_path, USER_A, DAY, [make_item(USER_A, 1, title=XSS, snippet=XSS)])
    client = TestClient(create_app(item_store=DemoItemStore(tmp_path), auth_mode="demo"))
    page = _brief_page(_logged_in(client))
    assert "<script>" not in page
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in page


def test_chaque_entree_affichee_a_un_lien_source(client: TestClient) -> None:
    page = _brief_page(_logged_in(client))
    entries = re.findall(r'<li class="entry">(.*?)</ul>\s*</li>', page, flags=re.DOTALL)
    assert entries
    assert all('href="https://' in entry and "Voir la source" in entry for entry in entries)


def test_entree_sans_source_resoluble_masquee_et_signalee(data_root: Path) -> None:
    # Un item cité, puis disparu du store (ou d'un autre propriétaire) : on masque l'entrée.
    class Forgetful(DemoItemStore):
        def find(self, item_id: str) -> Item | None:
            return None if item_id.endswith("-1") else super().find(item_id)

    client = TestClient(create_app(item_store=Forgetful(data_root), auth_mode="demo"))
    page = _brief_page(_logged_in(client))
    assert "1 entrée(s) masquée(s)" in page
    assert f"{USER_A}-1" not in page


def test_warnings_du_brief_visibles(data_root: Path) -> None:
    def with_warning(items: Sequence[Item], user: str, day: date | str, **kw: object) -> Brief:
        data = generate_brief(items, user, day).model_dump(mode="json")
        data["warnings"] = ["source mail indisponible"]
        return Brief.model_validate(data)

    store = DemoItemStore(data_root)
    client = TestClient(create_app(item_store=store, generator=with_warning, auth_mode="demo"))
    assert "source mail indisponible" in _brief_page(_logged_in(client))


def test_jour_sans_collecte_message_lisible(client: TestClient) -> None:
    response = _logged_in(client).post("/generate", data={"date": "2026-01-01"})
    assert response.status_code == 404
    assert "aucune collecte pour ce jour" in response.text


def test_un_utilisateur_ne_voit_pas_le_brief_d_un_autre(client: TestClient) -> None:
    _brief_page(_logged_in(client, USER_A))
    page = _logged_in(client, USER_B).get(f"/?date={DAY}").text
    assert "Générer le brief" in page
    assert f"{USER_A}-" not in page


def test_deconnexion_efface_le_cookie(client: TestClient) -> None:
    client = _logged_in(client)
    client.post("/logout")
    assert client.get("/", follow_redirects=False).status_code == 303
