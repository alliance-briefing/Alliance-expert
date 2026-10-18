"""Appels en lecture à Microsoft Graph, partagés par les connecteurs (mails, calendrier…).

- pagination : suit @odata.nextLink jusqu'à la dernière page ;
- réessais sur 429 et 5xx : attente 1, 2, 4, 8 s (5 essais au total), Retry-After respecté ;
- le jeton n'apparaît jamais dans les erreurs : seul le code HTTP est donné.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

GRAPH = "https://graph.microsoft.com/v1.0"
DELAYS = (1, 2, 4, 8)

Response = tuple[int, dict[str, str], bytes]
Send = Callable[[str, dict[str, str]], Response]


class GraphError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"Microsoft Graph a répondu {status}")
        self.status = status


def urllib_send(url: str, headers: dict[str, str]) -> Response:
    """Envoie la requête GET ; une coupure réseau est traitée comme une panne serveur (599)."""
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()
    except urllib.error.URLError:
        return 599, {}, b""


def _retry_after(headers: dict[str, str], default: int) -> int:
    value = {k.lower(): v for k, v in headers.items()}.get("retry-after", "")
    return int(value) if value.isdigit() else default


def _get_page(
    url: str, headers: dict[str, str], send: Send, sleep: Callable[[float], None]
) -> dict:
    for attempt in range(len(DELAYS) + 1):
        status, response_headers, body = send(url, headers)
        if status == 200:
            return json.loads(body)
        retryable = status == 429 or status >= 500
        if not retryable or attempt == len(DELAYS):
            raise GraphError(status)
        sleep(_retry_after(response_headers, DELAYS[attempt]))
    raise AssertionError("inatteignable")


def get_all(
    path: str,
    token: str,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    send: Send = urllib_send,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """Tous les éléments `value` d'une requête Graph, toutes pages confondues."""
    url: str | None = GRAPH + path + ("?" + urllib.parse.urlencode(params) if params else "")
    all_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        **(headers or {}),
    }
    items: list[dict] = []
    while url:
        page = _get_page(url, all_headers, send, sleep)
        items.extend(page.get("value", []))
        url = page.get("@odata.nextLink")
    return items
