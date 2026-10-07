from alliance_api.auth import DEMO_HEADER
from alliance_api.main import create_app
from fastapi.testclient import TestClient

USER = "u_test_a"


def test_utilisateur_connecte(client: TestClient) -> None:
    response = client.get("/me", headers={DEMO_HEADER: USER})
    assert response.status_code == 200
    assert response.json() == {"user_id": USER}


def test_sans_utilisateur_401_au_format_standard(client: TestClient) -> None:
    response = client.get("/me")
    assert response.status_code == 401
    body = response.json()
    assert body["code"] == "unauthenticated"
    assert body["correlation_id"] == response.headers["X-Correlation-ID"]


def test_identifiant_utilisateur_hors_format_refuse(client: TestClient) -> None:
    assert client.get("/me", headers={DEMO_HEADER: "../u_test_a"}).status_code == 401


def test_api_fermee_si_le_mode_demo_n_est_pas_active() -> None:
    client = TestClient(create_app(auth_mode=""))
    assert client.get("/me", headers={DEMO_HEADER: USER}).status_code == 401


def test_health_reste_public() -> None:
    assert TestClient(create_app(auth_mode="")).get("/health").status_code == 200


def test_correlation_id_client_repris_s_il_est_sur(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Correlation-ID": "abc-123"})
    assert response.headers["X-Correlation-ID"] == "abc-123"
    response = client.get("/health", headers={"X-Correlation-ID": "<script>"})
    assert response.headers["X-Correlation-ID"] != "<script>"


def test_route_inconnue_404_au_format_standard(client: TestClient) -> None:
    body = client.get("/nulle-part").json()
    assert body["code"] == "not_found"
    assert set(body) == {"code", "message", "correlation_id"}
