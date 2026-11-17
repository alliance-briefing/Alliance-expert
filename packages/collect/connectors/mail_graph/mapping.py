"""Traduction d'un message Microsoft Graph (`/me/messages`) en Item du contrat A 1.1.

- Extrait : corps HTML → texte, sans citations ni signature (core/text.py), sur une ligne,
  500 caractères au plus ; repli sur `bodyPreview` s'il ne reste rien.
- Corps complet non conservé (`content_text = None`) : rétention par défaut du guide (L1.7).
- Invitation à une réunion : même `correlation_id` que l'événement du calendrier (L1.6).
- Les destinataires en copie cachée ne sont jamais recopiés.
"""

from __future__ import annotations

import re
from datetime import datetime

from contracts.models import Item
from packages.collect.core.correlation import graph_invitation_correlation_id
from packages.collect.core.paris_time import parse_graph, to_paris
from packages.collect.core.text import (
    html_to_text,
    normalize_whitespace,
    strip_quotes_and_signature,
    truncate,
)

CONNECTOR = {"name": "mail_graph", "version": "0.1.0"}
_SPACES = re.compile(r"\s+")


def _one_line(text: str) -> str:
    """Une ligne sans caractère de contrôle (interdits par le contrat)."""
    return _SPACES.sub(" ", "".join(c if c.isprintable() else " " for c in text)).strip()


def snippet(message: dict) -> str:
    body = message.get("body") or {}
    content = body.get("content") or ""
    text = html_to_text(content) if body.get("contentType") == "html" else content
    text = strip_quotes_and_signature(normalize_whitespace(text))
    return truncate(_one_line(text or message.get("bodyPreview") or ""))


def address(who: dict) -> str:
    return (who.get("emailAddress") or {}).get("address", "").lower()


def _person(who: dict, role: str) -> dict:
    email = who["emailAddress"]
    return {"name": email.get("name") or email["address"], "email": email["address"], "role": role}


def _participants(message: dict) -> list[dict]:
    people = [_person(message["from"], "from")] if message.get("from") else []
    for role, key in (("to", "toRecipients"), ("cc", "ccRecipients")):
        people += [_person(who, role) for who in message.get(key) or []]
    return people


def received_at(message: dict) -> datetime:
    return parse_graph(message["receivedDateTime"])


def message_to_item(message: dict, owner: str, collected_at: datetime) -> Item:
    created = to_paris(received_at(message))
    modified = to_paris(
        parse_graph(message.get("lastModifiedDateTime") or message["receivedDateTime"])
    )
    return Item(
        item_id=f"mail:{message['id']}",
        source="mail",
        provider="microsoft",
        owner_user_id=owner,
        title=_one_line(message.get("subject") or ""),
        snippet=snippet(message),
        content_text=None,
        created_at=created,
        updated_at=max(created, modified),  # Exchange peut dater la modification avant la réception
        participants=_participants(message),
        thread_id=message.get("conversationId"),
        correlation_id=graph_invitation_correlation_id(message),
        native_importance=message.get("importance"),
        is_read=message.get("isRead"),
        url=message["webLink"],
        acl={"visible_to": [owner]},
        collected_at=to_paris(collected_at),
        connector=CONNECTOR,
    )
