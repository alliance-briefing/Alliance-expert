"""Parcours de connexion OpenID Connect (L3.4, #112, étape C2) : /auth/login → fournisseur →
/auth/callback, avec le faux fournisseur de test_oidc (aucun appel réseau)."""

from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from alliance_api.auth import DEMO_HEADER
from alliance_api.main import create_app
from alliance_api.oidc import STATE_COOKIE, load_accounts, pkce_challenge
from alliance_api.resume import PendingStore
from alliance_api.sessions import MemorySessionStore
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A
from apps.api.tests.test_oidc import FakeProvider, make_oidc
from apps.api.tests.test_resume import Clock

REFUSED = "/login?error=login_failed"


class FlowProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[dict[str, str]] = []  # demandes d'échange du code reçues

    def metadata(self) -> dict[str, Any]:
        return super().metadata() | {"authorization_endpoint": "https://login.example/authorize"}

    def exchange(self, form: dict[str, str]) -> dict[str, Any]:
        self.forms.append(form)
        return super().exchange(form)


@pytest.fixture
def provider() -> FlowProvider:
    return FlowProvider()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def client(data_root: Path, provider: FlowProvider, clock: Clock) -> TestClient:
    app = create_app(
        item_store=DemoItemStore(data_root),
        session_store=MemorySessionStore(clock=clock),
        pending_store=PendingStore(clock=clock),
        oidc_login=make_oidc(provider),
        auth_mode="oidc",
    )
    return TestClient(app)


def _start(client: TestClient, provider: FlowProvider, next_url: str = "/") -> dict[str, str]:
    """Clic sur « se connecter » : renvoie les paramètres envoyés au fournisseur."""
    start = client.get("/auth/login", params={"next": next_url}, follow_redirects=False)
    query = {k: v[0] for k, v in parse_qs(urlsplit(start.headers["location"]).query).items()}
    provider.nonce = query["nonce"]
    return query


def _back(client: TestClient, state: str) -> str:
    """Retour du fournisseur : renvoie la page d'arrivée."""
    params = {"code": "code-1", "state": state}
    return client.get("/auth/callback", params=params, follow_redirects=False).headers["location"]


def _connect(client: TestClient, provider: FlowProvider, next_url: str = "/") -> str:
    return _back(client, _start(client, provider, next_url)["state"])


def test_connexion_complete(client: TestClient, provider: FlowProvider) -> None:
    query = _start(client, provider, "/?date=2026-09-29")
    assert _back(client, query["state"]) == "/?date=2026-09-29"
    assert client.get("/me").json() == {"user_id": USER_A}
    assert STATE_COOKIE not in client.cookies
    # PKCE : le secret présenté à l'échange correspond à l'empreinte envoyée au départ.
    assert pkce_challenge(provider.forms[0]["code_verifier"]) == query["code_challenge"]


def test_reprise_apres_expiration_avec_microsoft(
    client: TestClient, provider: FlowProvider, clock: Clock
) -> None:
    _connect(client, provider)
    clock.advance(minutes=61)
    response = client.post("/generate", data={"date": DAY}, follow_redirects=False)
    assert response.headers["location"] == "/login?expired=1"
    assert _connect(client, provider) == f"/?date={DAY}"
    assert client.get(f"/briefs/{DAY}").status_code == 200  # l'action a été rejouée


def test_jeton_invalide_aucune_session(client: TestClient, provider: FlowProvider) -> None:
    provider.claims = {"aud": "autre-application"}
    assert _connect(client, provider) == REFUSED
    assert client.get("/me").status_code == 401


def test_compte_absent_du_fichier(client: TestClient, provider: FlowProvider) -> None:
    provider.claims = {"oid": "oid-inconnu"}
    assert _connect(client, provider) == "/login?error=unknown_account"
    assert client.get("/me").status_code == 401


def test_state_inconnu_ou_rejoue(client: TestClient, provider: FlowProvider) -> None:
    state = _start(client, provider)["state"]
    assert _back(client, "autre") == REFUSED
    client.cookies.set(STATE_COOKIE, state, path="/auth")  # effacé par le refus précédent
    assert _back(client, state) == "/"
    client.cookies.set(STATE_COOKIE, state, path="/auth")
    assert _back(client, state) == REFUSED  # usage unique


def test_retour_dans_un_autre_navigateur_refuse(client: TestClient, provider: FlowProvider) -> None:
    """Connexion imposée : un tiers lance la connexion puis fait ouvrir le lien de retour à la
    victime ; state et jeton sont valides, mais ce navigateur n'a pas le cookie du state."""
    state = _start(client, provider)["state"]
    assert _back(TestClient(client.app), state) == REFUSED


def test_refus_du_fournisseur(client: TestClient) -> None:
    response = client.get(
        "/auth/callback", params={"error": "access_denied"}, follow_redirects=False
    )
    assert response.headers["location"] == REFUSED


def test_redirection_externe_ignoree(client: TestClient, provider: FlowProvider) -> None:
    assert _connect(client, provider, "https://pirate.example") == "/"


def test_mode_demo_ferme_en_oidc(client: TestClient) -> None:
    assert client.get("/me", headers={DEMO_HEADER: USER_A}).status_code == 401
    assert client.post("/login", data={"user_id": USER_A}).status_code == 401


def test_routes_oidc_absentes_en_mode_demo(data_root: Path) -> None:
    client = TestClient(create_app(item_store=DemoItemStore(data_root), auth_mode="demo"))
    assert client.get("/auth/login", follow_redirects=False).status_code == 404


def test_configuration_incomplete_bloque_le_demarrage(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_REDIRECT_URI", "USERS_FILE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="variables manquantes"):
        create_app(item_store=DemoItemStore(data_root), auth_mode="oidc")


def test_exemple_du_depot_valide() -> None:
    assert load_accounts(Path(__file__).parents[3] / "infra" / "users.example.toml")
