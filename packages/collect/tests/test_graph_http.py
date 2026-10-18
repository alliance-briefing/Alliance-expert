"""Tests des appels Graph sans réseau : Microsoft est simulé par une liste de réponses."""

from __future__ import annotations

import json

import pytest

from packages.collect.core.auth import AuthError, get_access_token
from packages.collect.core.graph_http import GraphError, get_all

TOKEN = "faux-jeton-secret"


class FakeGraph:
    """Renvoie les réponses prévues dans l'ordre et garde la trace des requêtes."""

    def __init__(self, *responses: tuple[int, dict, dict]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, url: str, headers: dict) -> tuple[int, dict, bytes]:
        self.requests.append((url, headers))
        status, response_headers, body = self.responses.pop(0)
        return status, response_headers, json.dumps(body).encode()


def _ok(values: list, next_link: str | None = None) -> tuple[int, dict, dict]:
    body = {"value": values} | ({"@odata.nextLink": next_link} if next_link else {})
    return 200, {}, body


def test_follows_every_page_without_loss_or_duplicate() -> None:
    fake = FakeGraph(_ok([1, 2], "https://graph.microsoft.com/v1.0/next?p=2"), _ok([3]))
    assert get_all("/me/messages", TOKEN, send=fake, sleep=lambda s: None) == [1, 2, 3]
    assert fake.requests[1][0].endswith("next?p=2")


def test_429_waits_the_retry_after_given_by_microsoft() -> None:
    waits: list[float] = []
    fake = FakeGraph((429, {"Retry-After": "3"}, {}), _ok(["ok"]))
    assert get_all("/x", TOKEN, send=fake, sleep=waits.append) == ["ok"]
    assert waits == [3]


def test_server_errors_wait_1_2_4_8_then_give_up() -> None:
    waits: list[float] = []
    fake = FakeGraph(*[(503, {}, {})] * 5)
    with pytest.raises(GraphError) as error:
        get_all("/x", TOKEN, send=fake, sleep=waits.append)
    assert waits == [1, 2, 4, 8] and len(fake.requests) == 5
    assert TOKEN not in str(error.value)  # le jeton ne fuit jamais dans l'erreur


def test_client_error_is_not_retried() -> None:
    waits: list[float] = []
    with pytest.raises(GraphError):
        get_all("/x", TOKEN, send=FakeGraph((401, {}, {})), sleep=waits.append)
    assert waits == []


def test_missing_token_gives_reauth_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPH_ACCESS_TOKEN", raising=False)
    with pytest.raises(AuthError, match="reauth_required"):
        get_access_token("u_demo_002")
    monkeypatch.setenv("GRAPH_ACCESS_TOKEN", " jeton ")
    assert get_access_token("u_demo_002") == "jeton"
