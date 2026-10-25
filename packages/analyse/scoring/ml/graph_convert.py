"""Conversion des données Microsoft Graph synthétiques de David en Item (contrat A 1.1).

Sert uniquement à entraîner et mesurer le score appris : le connecteur mail officiel est
celui du lot 1. On réutilise son code en lecture seule (texte des corps, corrélation des
invitations, traduction des événements) pour produire les mêmes Item que lui.

Ce que le contrat A ne porte pas (destinataire direct ou en copie, mail envoyé…) va dans
`MailExtras`, à côté de l'Item : on ne modifie pas `contracts/`.
`flagged` et `categories` sont tirés au hasard par le générateur : gardés pour mémoire,
ils ne doivent pas servir de caractéristiques.

Usage : `uv run python -m packages.analyse.scoring.ml.graph_convert <dossier data/>`
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from contracts.models import Item
from packages.collect.connectors.calendar_graph.mapping import event_to_item
from packages.collect.core.correlation import graph_invitation_correlation_id
from packages.collect.core.paris_time import parse_graph, to_paris
from packages.collect.core.text import (
    html_to_text,
    normalize_whitespace,
    strip_quotes_and_signature,
    truncate,
)

OWNER = "u_graph_helene"
CONNECTOR = {"name": "graph_synth_convert", "version": "0.1.0"}
RecipientRole = Literal["from", "to", "cc", "bcc"]
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class MailExtras:
    """Champs Graph utiles au score mais absents du contrat A."""

    recipient_role: RecipientRole | None  # place du manager ; None : liste de diffusion
    is_sent: bool  # envoyé par le manager : contexte d'affaire, jamais noté
    inference_classification: str | None
    meeting_message_type: str | None
    has_attachments: bool
    flagged: bool  # aléatoire dans ces données : exclu des caractéristiques
    categories: tuple[str, ...]  # idem


@dataclass
class GraphDataset:
    items: list[Item] = field(default_factory=list)
    extras: dict[str, MailExtras] = field(default_factory=dict)  # par item_id (mails)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (identifiant, raison)


def _clean(text: str) -> str:
    """Une ligne, sans caractère de contrôle (interdits par le contrat dans le titre)."""
    return _SPACES.sub(" ", "".join(c if c.isprintable() else " " for c in text)).strip()


def _address(who: dict) -> str:
    return who["emailAddress"]["address"].lower()


def _body_text(mail: dict) -> str:
    """Corps sans citations ni signature ; repli sur l'aperçu Graph s'il ne reste rien."""
    body = mail.get("body") or {}
    content = body.get("content") or ""
    raw = html_to_text(content) if body.get("contentType") == "html" else content
    text = strip_quotes_and_signature(normalize_whitespace(raw))
    return text or normalize_whitespace(mail.get("bodyPreview") or "")


def _recipient_role(mail: dict, owner_email: str, is_sent: bool) -> RecipientRole | None:
    if is_sent:
        return "from"
    for role, key in (("to", "toRecipients"), ("cc", "ccRecipients"), ("bcc", "bccRecipients")):
        if any(_address(p) == owner_email for p in mail.get(key) or []):
            return role
    return None


def _participants(mail: dict, is_sent: bool) -> list[dict]:
    """Expéditeur, `to` et `cc` ; les `bcc` restent cachés (discrétion)."""
    sender = mail["from"]["emailAddress"]
    people = [{"name": sender["name"], "email": sender["address"], "role": "from"}]
    for role, key in (("to", "toRecipients"), ("cc", "ccRecipients")):
        for p in mail.get(key) or []:
            who = p["emailAddress"]
            people.append({"name": who["name"], "email": who["address"], "role": role})
    return people


def mail_to_item(
    mail: dict, owner: str, owner_email: str, sent_folder_id: str, collected_at: datetime
) -> tuple[Item, MailExtras]:
    """Un message `list-mail-messages` → (Item, MailExtras). Lève une erreur si invalide."""
    is_sent = mail["parentFolderId"] == sent_folder_id
    text = _body_text(mail)
    created = to_paris(parse_graph(mail["sentDateTime" if is_sent else "receivedDateTime"]))
    item = Item(
        item_id=f"mail:{mail['id']}",
        source="mail",
        provider="microsoft",
        owner_user_id=owner,
        title=_clean(mail.get("subject") or ""),
        snippet=truncate(_SPACES.sub(" ", text).strip()),
        content_text=text or None,
        created_at=created,
        # Exchange peut dater la modification avant la réception
        updated_at=max(created, to_paris(parse_graph(mail["lastModifiedDateTime"]))),
        participants=_participants(mail, is_sent),
        thread_id=mail.get("conversationId"),
        correlation_id=graph_invitation_correlation_id(mail),
        native_importance=mail.get("importance"),
        is_read=mail.get("isRead"),
        url=mail["webLink"],
        acl={"visible_to": [owner]},
        collected_at=to_paris(collected_at),
        connector=CONNECTOR,
    )
    extras = MailExtras(
        recipient_role=_recipient_role(mail, owner_email.lower(), is_sent),
        is_sent=is_sent,
        inference_classification=mail.get("inferenceClassification"),
        meeting_message_type=mail.get("meetingMessageType"),
        has_attachments=bool(mail.get("hasAttachments")),
        flagged=(mail.get("flag") or {}).get("flagStatus") == "flagged",
        categories=tuple(mail.get("categories") or ()),
    )
    return item, extras


def dataset_from_items(items: list[Item], owner_email: str) -> GraphDataset:
    """Un jeu déjà au format Item (démo du lot 1) : `MailExtras` déduits des participants.
    Les champs Graph absents (classement, signalement…) restent vides ; un mail dont le
    manager est l'expéditeur compte comme envoyé."""
    owner_email = owner_email.lower()
    dataset = GraphDataset()
    for item in sorted({i.item_id: i for i in items}.values(), key=lambda i: i.item_id):
        dataset.items.append(item)
        if item.source != "mail":
            continue
        roles = {p.role for p in item.participants if p.email.lower() == owner_email}
        role = next((r for r in ("from", "to", "cc") if r in roles), None)
        dataset.extras[item.item_id] = MailExtras(
            recipient_role=role,
            is_sent=role == "from",
            inference_classification=None,
            meeting_message_type=None,
            has_attachments=False,
            flagged=False,
            categories=(),
        )
    return dataset


def _read(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["value"]


def load_graph_dataset(data_dir: Path | str, owner: str = OWNER) -> GraphDataset:
    """Mails et événements de toute la période, dédoublonnés par identifiant.

    Les événements « toute la journée » figurent dans deux fichiers jour : on garde le
    premier. Un élément refusé par le contrat est compté dans `rejected`, jamais ignoré.
    """
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    owner_email = manifest["persona"]["address"]
    dataset, seen = GraphDataset(), set()
    for day, info in sorted(manifest["jours"].items()):
        collected_at = parse_graph(info["window_utc"][1])  # fin de la journée collectée
        for kind in ("mail", "calendar"):
            path = data_dir / kind / f"{day}.json"
            if not path.exists():
                continue
            for raw in _read(path):
                key = f"{kind}:{raw['id']}"
                if key in seen:
                    continue
                seen.add(key)
                try:
                    if kind == "mail":
                        item, extras = mail_to_item(
                            raw, owner, owner_email, manifest["sentFolderId"], collected_at
                        )
                        dataset.extras[item.item_id] = extras
                    else:
                        item = event_to_item(raw, owner, collected_at)
                except (ValidationError, ValueError, KeyError) as exc:
                    dataset.rejected.append((key, f"{type(exc).__name__}: {exc}"[:200]))
                    continue
                dataset.items.append(item)
    return dataset


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1])
        return 2
    dataset = load_graph_dataset(argv[0])
    by_source: dict[str, int] = {}
    for item in dataset.items:
        by_source[item.source] = by_source.get(item.source, 0) + 1
    sent = sum(e.is_sent for e in dataset.extras.values())
    print(f"Item : {len(dataset.items)} {by_source} (dont {sent} mails envoyés)")
    print(f"refusés : {len(dataset.rejected)}")
    for key, reason in dataset.rejected[:10]:
        print(f"  {key[:40]}… {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
