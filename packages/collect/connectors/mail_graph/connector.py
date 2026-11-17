"""Connecteur e-mails Microsoft Graph (guide L1.3).

Fenêtre : les mails reçus dans les 24 h de la collecte, plus ceux des 7 derniers jours
adressés directement à l'utilisateur (`to`) et toujours sans réponse de sa part au moment
de la collecte. Plafond : les 500 plus récents.

« Sans réponse » : aucun mail envoyé par l'utilisateur dans le même fil (`conversationId`)
après la réception et avant la collecte.

Deux sources au choix (variable MAIL_SOURCE) :
- « files » (par défaut) : un dossier au format alliance-graph-synth (`manifest.json` +
  `mail/AAAA-MM-JJ.json`), servi uniquement à l'utilisateur de démonstration (L1.7).
  MAIL_GRAPH_DIR=<alliance-graph-synth>/data pour le jeu complet de David ;
- « graph » : le vrai compte (boîte de réception et éléments envoyés), via `get_access_token`.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from contracts.models import Item
from packages.collect.connectors.mail_graph.mapping import address, message_to_item, received_at
from packages.collect.core.auth import get_access_token
from packages.collect.core.connector import Connector, Health, Window
from packages.collect.core.graph_http import Send, get_all, get_one, urllib_send
from packages.collect.core.normalize import normalize_all
from packages.collect.core.paris_time import parse_graph, to_paris

logger = logging.getLogger("collect.mail_graph")

LOOKBACK = timedelta(days=7)
CAP = 500
DEMO_USER = "u_demo_graph"  # persona fictive d'alliance-graph-synth (Hélène MARCHAND)
DEFAULT_DIR = Path(
    os.environ.get("MAIL_GRAPH_DIR", Path(__file__).parents[2] / "tests/fixtures/graph")
)
PREFER_UTC = {"Prefer": 'outlook.timezone="UTC"'}  # heures des invitations en UTC


@dataclass(frozen=True)
class Mailbox:
    """Ce qu'il faut pour choisir les mails : reçus, envoyés (pour les réponses), adresse."""

    owner_email: str
    received: list[dict] = field(default_factory=list)
    sent: list[dict] = field(default_factory=list)


Source = Callable[[str, datetime, datetime], Mailbox]  # (utilisateur, début, fin)


def select_messages(box: Mailbox, recent_from: datetime, now: datetime) -> list[dict]:
    """Les messages à collecter, du plus récent au plus ancien, 500 au plus, sans doublon."""
    me = box.owner_email.lower()
    replies: dict[str, list[datetime]] = {}
    for sent in box.sent:
        moment = parse_graph(sent["sentDateTime"])
        if moment <= now and sent.get("conversationId"):
            replies.setdefault(sent["conversationId"], []).append(moment)

    def unanswered(message: dict, moment: datetime) -> bool:
        return not any(r > moment for r in replies.get(message.get("conversationId") or "", []))

    def direct(message: dict) -> bool:
        return any(address(who) == me for who in message.get("toRecipients") or [])

    chosen: dict[str, tuple[datetime, dict]] = {}
    for message in box.received:
        if message.get("isDraft") or address(message.get("from") or {}) == me:
            continue  # brouillon ou mail envoyé par l'utilisateur lui-même
        moment = received_at(message)
        if not now - LOOKBACK < moment <= now:
            continue
        if moment > recent_from or (direct(message) and unanswered(message, moment)):
            chosen[message["id"]] = (moment, message)
    ordered = sorted(chosen.values(), key=lambda pair: (pair[0], pair[1]["id"]), reverse=True)
    if len(ordered) > CAP:
        logger.info("mail_graph : plafond atteint, %d messages écartés", len(ordered) - CAP)
    return [message for _, message in ordered[:CAP]]


def file_source(folder: Path = DEFAULT_DIR, owner: str = DEMO_USER) -> Source:
    """Lit `mail/AAAA-MM-JJ.json` (une journée de Paris par fichier) sur la période.

    Le dossier appartient à `owner` : tout autre utilisateur reçoit une boîte vide.
    """

    def fetch(user: str, start: datetime, end: datetime) -> Mailbox:
        manifest_path = Path(folder) / "manifest.json"
        if user != owner or not manifest_path.exists():
            return Mailbox(owner_email="")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        box = Mailbox(owner_email=manifest["persona"]["address"])
        day = to_paris(start).date()
        while day <= to_paris(end).date():
            path = Path(folder) / "mail" / f"{day.isoformat()}.json"
            if path.exists():
                for message in json.loads(path.read_text("utf-8"))["value"]:
                    is_sent = message.get("parentFolderId") == manifest["sentFolderId"]
                    (box.sent if is_sent else box.received).append(message)
            day += timedelta(days=1)
        return box

    return fetch


def _utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def graph_source(
    token_for: Callable[[str], str] = get_access_token, send: Send = urllib_send
) -> Source:
    """Le vrai compte : boîte de réception (messages entiers) et éléments envoyés (fil, date)."""

    def fetch(user: str, start: datetime, end: datetime) -> Mailbox:
        token = token_for(user)
        me = get_one("/me", token, {"$select": "mail,userPrincipalName"}, send=send)
        received = get_all(
            "/me/mailFolders/inbox/messages",
            token,
            {
                "$filter": f"receivedDateTime ge {_utc(start)} and receivedDateTime le {_utc(end)}",
                "$orderby": "receivedDateTime desc",
                "$top": "100",
            },
            PREFER_UTC,
            send=send,
        )
        sent = get_all(
            "/me/mailFolders/sentitems/messages",
            token,
            {
                "$filter": f"sentDateTime ge {_utc(start)}",
                "$select": "conversationId,sentDateTime",
                "$top": "100",
            },
            send=send,
        )
        return Mailbox(me.get("mail") or me.get("userPrincipalName") or "", received, sent)

    return fetch


class GraphMailConnector(Connector):
    name = "mail_graph"
    version = "0.1.0"

    def __init__(self, source: Source | None = None) -> None:
        if source is None:
            source = graph_source() if os.environ.get("MAIL_SOURCE") == "graph" else file_source()
        self.source = source

    def required_scopes(self) -> list[str]:
        return ["Mail.Read"]  # lecture seule (guide L1.2)

    def collect(self, user: str, window: Window) -> Iterator[Item]:
        now = window.end
        box = self.source(user, now - LOOKBACK, now)
        messages = select_messages(box, window.start, now)
        # un message invalide est rejeté seul et journalisé sans contenu (L1.6)
        result = normalize_all(
            messages, lambda m: message_to_item(dict(m), user, now), connector=self.name
        )
        yield from result.items

    def health(self) -> Health:
        try:
            self.source("health", datetime.now(UTC), datetime.now(UTC))
        except Exception as exc:  # noqa: BLE001
            return Health(False, type(exc).__name__)
        return Health(True, "source joignable")
