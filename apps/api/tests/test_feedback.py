"""POST /feedback et GET /feedback (#93) : verdict par entrée, isolation, stockage JSONL."""

from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pytest
from alliance_api.feedback import Feedback, JsonlFeedbackStore
from alliance_api.main import create_app
from alliance_api.stores import DemoItemStore
from fastapi.testclient import TestClient

from apps.api.tests.factories import DAY, USER_A, USER_B
from contracts.models import Brief, Item
from packages.analyse import generate_brief

Headers = Callable[[str], dict[str, str]]
BRIEF_A = f"b_{DAY}_{USER_A}"


def _two_actions(items: Sequence[Item], user: str, day: date | str, **kw: object) -> Brief:
    """Brief connu : section « actions » à deux entrées, une par item."""
    data = generate_brief(items, user, day).model_dump(mode="json")
    data["sections"] = [
        {
            "kind": "actions",
            "entries": [
                {
                    "rank": n,
                    "priority_score": 90 - n,
                    "priority_reasons": [f"raison {n}"],
                    "text": f"Sujet {n}",
                    "source_item_ids": [item.item_id],
                }
                for n, item in enumerate(items, start=1)
            ],
        }
    ]
    return Brief.model_validate(data)


def _client(root: Path, **deps: object) -> TestClient:
    app = create_app(
        item_store=DemoItemStore(root), generator=_two_actions, auth_mode="demo", **deps
    )
    return TestClient(app)


@pytest.fixture
def api(data_root: Path, as_user: Headers) -> TestClient:
    client = _client(data_root)
    for user in (USER_A, USER_B):
        assert client.post("/briefs", json={"date": DAY}, headers=as_user(user)).status_code == 201
    return client


def _send(client: TestClient, user: str, **changes: object) -> object:
    body = {"brief_id": BRIEF_A, "section": "actions", "rank": 2, "verdict": "useful"} | changes
    return client.post("/feedback", json=body, headers={"X-Demo-User": user})


def test_retour_enregistre_avec_les_items_de_l_entree(api: TestClient, as_user: Headers) -> None:
    response = _send(api, USER_A)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["verdict"] == "useful"
    assert body["item_ids"] == [f"mail:{USER_A}-2"]
    assert "user_id" not in body
    listed = api.get("/feedback", params={"brief_id": BRIEF_A}, headers=as_user(USER_A)).json()
    assert [(f["section"], f["rank"], f["verdict"]) for f in listed] == [("actions", 2, "useful")]


def test_nouveau_clic_remplace_le_verdict(api: TestClient, as_user: Headers) -> None:
    _send(api, USER_A, rank=2, verdict="useful")
    _send(api, USER_A, rank=1, verdict="wrong")
    _send(api, USER_A, rank=2, verdict="not_useful")
    listed = api.get("/feedback", params={"brief_id": BRIEF_A}, headers=as_user(USER_A)).json()
    assert [(f["rank"], f["verdict"]) for f in listed] == [(1, "wrong"), (2, "not_useful")]


@pytest.mark.parametrize("brief_id", [BRIEF_A, f"b_{DAY}_u_inconnu", "b_2026-02-31_u_test_b", "x"])
def test_brief_d_un_autre_ou_inconnu_404(api: TestClient, as_user: Headers, brief_id: str) -> None:
    response = _send(api, USER_B, brief_id=brief_id)
    assert response.status_code == 404
    assert response.json()["code"] == "brief_not_found"
    response = api.get("/feedback", params={"brief_id": brief_id}, headers=as_user(USER_B))
    assert response.status_code == 404


def test_lecture_n_expose_pas_les_retours_d_un_autre(api: TestClient, as_user: Headers) -> None:
    _send(api, USER_A)
    brief_b = f"b_{DAY}_{USER_B}"
    listed = api.get("/feedback", params={"brief_id": brief_b}, headers=as_user(USER_B)).json()
    assert listed == []


def test_brief_non_genere_404(data_root: Path) -> None:
    response = _send(_client(data_root), USER_A)
    assert response.status_code == 404
    assert response.json()["code"] == "brief_not_found"


@pytest.mark.parametrize("changes", [{"rank": 3}, {"section": "meetings"}])
def test_entree_inconnue_422(api: TestClient, changes: dict[str, object]) -> None:
    response = _send(api, USER_A, **changes)
    assert response.status_code == 422
    assert response.json()["code"] == "unknown_entry"


@pytest.mark.parametrize(
    "changes", [{"verdict": "génial"}, {"rank": 0}, {"section": "autre"}, {"extra": 1}]
)
def test_requete_invalide_422(api: TestClient, changes: dict[str, object]) -> None:
    response = _send(api, USER_A, **changes)
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_sans_utilisateur_401(api: TestClient) -> None:
    body = {"brief_id": BRIEF_A, "section": "actions", "rank": 1, "verdict": "useful"}
    assert api.post("/feedback", json=body).status_code == 401
    assert api.get("/feedback", params={"brief_id": BRIEF_A}).status_code == 401


def test_jsonl_garde_l_historique_et_survit_au_redemarrage(
    data_root: Path, tmp_path: Path, as_user: Headers
) -> None:
    folder = tmp_path / "retours"
    client = _client(data_root, feedback_store=JsonlFeedbackStore(folder))
    client.post("/briefs", json={"date": DAY}, headers=as_user(USER_A))
    _send(client, USER_A, verdict="useful")
    _send(client, USER_A, verdict="wrong")

    lines = (folder / f"{USER_A}.jsonl").read_text(encoding="utf-8").splitlines()
    assert [Feedback.model_validate_json(line).verdict for line in lines] == ["useful", "wrong"]

    restarted = _client(data_root, feedback_store=JsonlFeedbackStore(folder))
    listed = restarted.get("/feedback", params={"brief_id": BRIEF_A}, headers=as_user(USER_A))
    assert [f["verdict"] for f in listed.json()] == ["wrong"]
