"""Page d'administration GET /admin (L3.7, étape B) : journal filtrable et chaîne d'un brief."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alliance_api.audit import COLLECT_FILE, JsonlAuditLog
from alliance_api.main import create_app
from alliance_api.sessions import MemorySessionStore
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A, XSS
from apps.api.tests.test_admin_chain import SECRET, _collect, _generator

ADMIN = "u_admin_001"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 25, 8, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def client(data_root: Path, clock: Clock) -> TestClient:
    collect = JsonlAuditLog(data_root, COLLECT_FILE)
    collect.write(_collect(USER_A, f"{DAY}T06:30:00+02:00", [f"mail:{USER_A}-1"]))
    collect.write(_collect(USER_A, f"{DAY}T06:31:00+02:00", [], error="GraphTimeout"))
    app = create_app(
        item_store=DemoItemStore(data_root),
        generator=_generator,
        collect_audit=collect,
        session_store=MemorySessionStore(clock=clock),
        auth_mode="demo",
    )
    client = TestClient(app)
    _login(client, USER_A)
    assert client.post("/generate", data={"date": DAY}, follow_redirects=False).status_code == 303
    client.post("/logout")
    return client


def _login(client: TestClient, user: str) -> None:
    assert client.post("/login", data={"user_id": user}, follow_redirects=False).status_code == 303


def _admin(client: TestClient, **params: str) -> str:
    _login(client, ADMIN)
    response = client.get("/admin", params=params)
    assert response.status_code == 200, response.text
    return response.text


def test_sans_session_vers_la_connexion(client: TestClient) -> None:
    response = client.get("/admin", follow_redirects=False)
    assert (response.status_code, response.headers["location"]) == (303, "/login")


def test_manager_refuse(client: TestClient) -> None:
    _login(client, USER_A)
    response = client.get("/admin")
    assert response.status_code == 403 and "Droits insuffisants" in response.text
    assert "<table" not in response.text


def test_journal_complet_puis_filtre(client: TestClient) -> None:
    page = _admin(client)
    assert "3 événements" in page and "connector:mail-graph" in page and "GraphTimeout" in page
    page = _admin(client, status="error")
    assert "1 événement<" in page and '<option value="error" selected>' in page


def test_chaine_du_brief_avec_utilisateur_et_jour(client: TestClient) -> None:
    page = _admin(client, user=USER_A, brief=DAY)
    assert f"Chaîne du brief b_{DAY}_{USER_A}" in page
    assert "modele-local 1, prompt p7" in page and "1800 en entrée, 62 en sortie" in page
    assert "Aucune consultation." in page  # générer depuis l'interface n'est pas consulter


def test_chaine_et_journal_filtre_independamment(client: TestClient) -> None:
    """Le brief du 30/09 a été généré « aujourd'hui » : la chaîne s'affiche même si le journal
    est filtré sur un autre jour."""
    page = _admin(client, user=USER_A, brief=DAY, date="2026-01-01")
    assert f"Chaîne du brief b_{DAY}_{USER_A}" in page and "0 événements" in page


def test_chaine_indisponible_sans_brief(client: TestClient) -> None:
    page = _admin(client, user=USER_A, brief="2026-09-29")
    assert "Chaîne indisponible : brief non généré pour ce jour." in page


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"date": "30/09/2026"}, "paramètres invalides : date"),
        ({"brief": "hier"}, "paramètres invalides : brief"),
        ({"action": "supprimer"}, "paramètres invalides : action ou statut"),
        ({"user": "../u_test_a"}, "paramètres invalides : user"),
    ],
)
def test_filtres_invalides(client: TestClient, params: dict[str, str], message: str) -> None:
    _login(client, ADMIN)
    response = client.get("/admin", params=params)
    assert response.status_code == 422 and message in response.text


def test_ni_contenu_ni_script(client: TestClient) -> None:
    page = _admin(client, user=USER_A, brief=DAY)
    assert SECRET not in page and XSS not in page and "<script>" not in page


def test_session_expiree_retour_sur_la_page(client: TestClient, clock: Clock) -> None:
    _login(client, ADMIN)
    clock.now += timedelta(hours=11)
    response = client.get("/admin", params={"status": "error"}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"].startswith("/login")
    response = client.post("/login", data={"user_id": ADMIN}, follow_redirects=False)
    assert response.headers["location"] == "/admin?status=error"
