"""Reprise après expiration de session (L3.4, #112, étape B) : rien n'est perdu."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alliance_api.auth import DEMO_HEADER
from alliance_api.main import create_app
from alliance_api.resume import REPLAYABLE, RESUME_COOKIE, PendingStore, safe_next
from alliance_api.sessions import MemorySessionStore
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A, USER_B


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


@pytest.fixture
def client(data_root: Path, clock: Clock) -> TestClient:
    app = create_app(
        item_store=DemoItemStore(data_root),
        session_store=MemorySessionStore(clock=clock),
        pending_store=PendingStore(clock=clock),
        auth_mode="demo",
    )
    return TestClient(app)


def _login(client: TestClient, user: str = USER_A) -> str:
    response = client.post("/login", data={"user_id": user}, follow_redirects=False)
    assert response.status_code == 303
    return response.headers["location"]


def _expire(client: TestClient, clock: Clock) -> None:
    _login(client)
    clock.advance(minutes=61)


def _brief_exists(client: TestClient, user: str) -> bool:
    response = client.get(f"/briefs/{DAY}", headers={DEMO_HEADER: user})
    return response.status_code == 200


def test_action_mise_de_cote_puis_rejouee_apres_reconnexion(
    client: TestClient, clock: Clock
) -> None:
    _expire(client, clock)
    response = client.post("/generate", data={"date": DAY}, follow_redirects=False)
    assert (response.status_code, response.headers["location"]) == (303, "/login?expired=1")
    assert RESUME_COOKIE in client.cookies
    assert not _brief_exists(client, USER_A)  # rien n'a été fait sans session

    page = client.get("/login?expired=1").text
    assert "Votre session a expiré" in page
    assert f'value="{USER_A}"' in page  # identifiant prérempli

    assert _login(client) == f"/?date={DAY}"  # retour sur la même page
    assert _brief_exists(client, USER_A)  # l'action a été rejouée
    assert RESUME_COOKIE not in client.cookies


def test_page_retrouvee_apres_reconnexion(client: TestClient, clock: Clock) -> None:
    _expire(client, clock)
    response = client.get("/?date=2026-09-29", follow_redirects=False)
    assert response.headers["location"] == "/login?expired=1"
    assert _login(client) == "/?date=2026-09-29"


def test_autre_personne_ne_rejoue_rien(client: TestClient, clock: Clock) -> None:
    _expire(client, clock)
    client.post("/generate", data={"date": DAY}, follow_redirects=False)
    assert _login(client, USER_B) == "/"
    assert not _brief_exists(client, USER_A) and not _brief_exists(client, USER_B)


def test_action_oubliee_apres_15_minutes(client: TestClient, clock: Clock) -> None:
    _expire(client, clock)
    client.post("/generate", data={"date": DAY}, follow_redirects=False)
    clock.advance(minutes=16)
    assert 'value="u_demo_001"' in client.get("/login").text  # plus de préremplissage
    assert _login(client) == "/"
    assert not _brief_exists(client, USER_A)


def test_jeton_a_usage_unique(client: TestClient, clock: Clock) -> None:
    _expire(client, clock)
    client.post("/generate", data={"date": DAY}, follow_redirects=False)
    token = client.cookies[RESUME_COOKIE]
    _login(client)
    client.cookies.set(RESUME_COOKIE, token)
    assert _login(client) == "/"


def test_sans_session_du_tout_pas_de_mise_de_cote(client: TestClient) -> None:
    response = client.post("/generate", data={"date": DAY}, follow_redirects=False)
    assert response.headers["location"] == "/login"
    assert RESUME_COOKIE not in client.cookies


def test_retour_de_feedback_rejouable(client: TestClient) -> None:
    client.post("/briefs", json={"date": DAY}, headers={DEMO_HEADER: USER_A})
    brief = client.get(f"/briefs/{DAY}", headers={DEMO_HEADER: USER_A}).json()
    section = next(s for s in brief["sections"] if s["entries"])
    data = {
        "brief_id": brief["brief_id"],
        "section": section["kind"],
        "rank": 1,
        "verdict": "useful",
    }
    REPLAYABLE["feedback"](client.app.state, USER_A, data)
    given = client.get(
        "/feedback", params={"brief_id": brief["brief_id"]}, headers={DEMO_HEADER: USER_A}
    )
    assert [f["verdict"] for f in given.json()] == ["useful"]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("/?date=2026-09-30", "/?date=2026-09-30"),
        ("//pirate.example", "/"),
        ("https://pirate.example", "/"),
        ("/\\pirate.example", "/"),
    ],
)
def test_jamais_de_redirection_vers_un_autre_site(url: str, expected: str) -> None:
    assert safe_next(url) == expected
