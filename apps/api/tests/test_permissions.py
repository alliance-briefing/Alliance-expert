"""Permissions configurables et protection CSRF (L3.4, #112, PR A1)."""

from pathlib import Path

import pytest
from alliance_api.auth import DEMO_HEADER, requires
from alliance_api.main import create_app
from alliance_api.permissions import DEFAULT_ROLES_FILE, Policy, load_policy
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A

ADMIN = "u_admin_001"


def _app(root: Path, policy: Policy | None = None) -> TestClient:
    app = create_app(item_store=DemoItemStore(root), policy=policy, auth_mode="demo")
    app.get("/test-audit", dependencies=[requires("audit:read")])(lambda: {"ok": True})
    return TestClient(app)


def _login(client: TestClient, user: str = USER_A) -> None:
    response = client.post("/login", data={"user_id": user}, follow_redirects=False)
    assert response.status_code == 303


# --- Politique de droits ---------------------------------------------------------------


def test_politique_par_defaut() -> None:
    policy = load_policy(DEFAULT_ROLES_FILE)
    assert policy.role_of(USER_A) == "manager" and policy.role_of(ADMIN) == "admin"
    assert policy.allows("admin", "audit:read") and not policy.allows("manager", "audit:read")
    assert not any(policy.allows(role, "brief:read_any") for role in policy.roles)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('default_role = "manager"\n[roles]\nmanager = ["brief:lire"]', "permissions inconnues"),
        ('default_role = "chef"\n[roles]\nmanager = []', "non défini"),
        ('default_role = "manager"\n[roles]\nmanager = []\n[users]\nu_x = "chef"', "non défini"),
    ],
)
def test_fichier_de_roles_errone_refuse(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "roles.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_policy(path)


def test_fichier_de_roles_choisi_par_variable(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "roles.toml"
    path.write_text('default_role = "admin"\n[roles]\nadmin = ["audit:read"]', encoding="utf-8")
    monkeypatch.setenv("ROLES_FILE", str(path))
    client = _app(data_root)
    assert client.get("/test-audit", headers={DEMO_HEADER: USER_A}).status_code == 200


def test_droits_modifiables_par_configuration(data_root: Path) -> None:
    lecture_seule = Policy(
        roles={"manager": frozenset({"brief:read_own"})}, users={}, default_role="manager"
    )
    response = _app(data_root, lecture_seule).post(
        "/briefs", json={"date": DAY}, headers={DEMO_HEADER: USER_A}
    )
    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


def test_permission_admin(data_root: Path) -> None:
    client = _app(data_root)
    assert client.get("/test-audit", headers={DEMO_HEADER: USER_A}).status_code == 403
    assert client.get("/test-audit", headers={DEMO_HEADER: ADMIN}).status_code == 200
    assert client.get("/test-audit").status_code == 401


# --- CSRF ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://pirate.example"},
        {"Origin": "null"},
        {"Referer": "https://pirate.example/page"},
    ],
)
def test_requete_d_une_autre_origine_refusee(data_root: Path, headers: dict[str, str]) -> None:
    client = _app(data_root)
    _login(client)
    response = client.post("/briefs", json={"date": DAY}, headers=headers)
    assert response.status_code == 403
    assert response.json()["code"] == "csrf_rejected"
    assert response.json()["correlation_id"]


@pytest.mark.parametrize(
    "headers", [{}, {"Origin": "http://testserver"}, {"Referer": "http://testserver/"}]
)
def test_meme_origine_ou_client_d_api_accepte(data_root: Path, headers: dict[str, str]) -> None:
    client = _app(data_root)
    _login(client)
    assert client.post("/briefs", json={"date": DAY}, headers=headers).status_code == 201


def test_origine_supplementaire_configurable(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://briefing.alliance.test")
    client = _app(data_root)
    _login(client)
    headers = {"Origin": "https://briefing.alliance.test"}
    assert client.post("/briefs", json={"date": DAY}, headers=headers).status_code == 201
