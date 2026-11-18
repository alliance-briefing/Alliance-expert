"""Sessions côté serveur : durée, expiration, déconnexion (L3.4, #112, PR A2)."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alliance_api.auth import SESSION_COOKIE, requires
from alliance_api.main import create_app
from alliance_api.sessions import MemorySessionStore
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import USER_A


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 18, 8, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@pytest.fixture
def clock() -> Clock:
    return Clock()


def _app(root: Path, clock: Clock) -> TestClient:
    app = create_app(
        item_store=DemoItemStore(root),
        session_store=MemorySessionStore(clock=clock),
        auth_mode="demo",
    )
    return TestClient(app)


def _login(client: TestClient, user: str = USER_A) -> str:
    response = client.post("/login", data={"user_id": user}, follow_redirects=False)
    assert response.status_code == 303
    return client.cookies[SESSION_COOKIE]


# --- Stockage des sessions -------------------------------------------------------------


def test_session_valable_puis_expiree_apres_1h_d_inactivite(clock: Clock) -> None:
    store = MemorySessionStore(clock=clock)
    session = store.create(USER_A)
    clock.advance(minutes=59)
    assert store.lookup(session.session_id)[0] == "ok"  # l'activité repousse l'échéance
    clock.advance(minutes=59)
    assert store.lookup(session.session_id)[0] == "ok"
    clock.advance(minutes=60)
    status, expired = store.lookup(session.session_id)
    assert status == "expired" and expired is not None and expired.user == USER_A


def test_session_expiree_apres_10h_meme_avec_activite(clock: Clock) -> None:
    store = MemorySessionStore(clock=clock)
    session = store.create(USER_A)
    for _ in range(19):
        clock.advance(minutes=30)
        assert store.lookup(session.session_id)[0] == "ok"
    clock.advance(minutes=30)  # 10 h après la connexion
    assert store.lookup(session.session_id)[0] == "expired"


def test_deconnexion_efface_et_identifiants_imprevisibles(clock: Clock) -> None:
    store = MemorySessionStore(clock=clock)
    first, second = store.create(USER_A), store.create(USER_A)
    assert first.session_id != second.session_id and len(first.session_id) >= 40
    store.delete(first.session_id)
    assert store.lookup(first.session_id) == ("missing", None)


# --- Sessions dans l'application -------------------------------------------------------


def test_cookie_ne_contient_que_un_identifiant_aleatoire(data_root: Path, clock: Clock) -> None:
    client = _app(data_root, clock)
    session_id = _login(client)
    assert USER_A not in session_id
    assert client.get("/me").json() == {"user_id": USER_A}


def test_cookie_forge_refuse(data_root: Path, clock: Clock) -> None:
    client = _app(data_root, clock)
    client.cookies.set(SESSION_COOKIE, USER_A)
    response = client.get("/me")
    assert response.status_code == 401
    assert response.json()["code"] == "unauthenticated"


def test_session_expiree_401_explicite(data_root: Path, clock: Clock) -> None:
    client = _app(data_root, clock)
    _login(client)
    clock.advance(minutes=61)
    response = client.get("/me")
    assert response.status_code == 401
    assert response.json()["code"] == "session_expired"


def test_deconnexion_invalide_la_session_cote_serveur(data_root: Path, clock: Clock) -> None:
    client = _app(data_root, clock)
    session_id = _login(client)
    client.post("/logout", follow_redirects=False)
    client.cookies.set(SESSION_COOKIE, session_id)  # cookie volé et rejoué
    assert client.get("/me").status_code == 401


def test_role_relu_a_chaque_requete(data_root: Path, clock: Clock) -> None:
    client = _app(data_root, clock)
    client.app.get("/test-audit", dependencies=[requires("audit:read")])(lambda: {"ok": True})
    _login(client)
    assert client.get("/test-audit").status_code == 403
    policy = client.app.state.policy
    client.app.state.policy = replace(policy, users={**policy.users, USER_A: "admin"})
    assert client.get("/test-audit").status_code == 200  # sans se reconnecter
