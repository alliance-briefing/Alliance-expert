"""Identifiant de corrélation d'une réunion (guide L1.6, contrat A 1.1).

Une même réunion arrive deux fois : en mail d'invitation et en événement du calendrier.
Les deux Item reçoivent le même `correlation_id`, calculé à partir du titre (sans
« Annulé : », « RE: »…) et de l'heure de début en UTC. Sur les données de David
(alliance-graph-synth), cette règle relie les 70 invitations à leur événement.

Les fonctions `*_graph_*` lisent directement le JSON Microsoft Graph : les connecteurs
mails (B) et calendrier (C) les appellent pendant leur traduction.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

# Préfixes ajoutés par Outlook devant le titre d'une réunion (FR et EN), à ignorer.
_PREFIXES = re.compile(
    r"^\s*((annul[ée]\s*:|canceled\s*:|cancelled\s*:|re\s*:|tr\s*:|fw\s*:|fwd\s*:"
    r"|mise à jour\s*:|updated\s*:|accepté\s*:|accepted\s*:)\s*)+",
    re.IGNORECASE,
)


def _slug(title: str) -> str:
    title = _PREFIXES.sub("", title)
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-") or "sans-titre"


def meeting_correlation_id(title: str, start: datetime) -> str:
    """`meet:<titre-normalisé>:<début UTC à la minute>Z`, identique pour l'invitation et l'événement."""
    if start.tzinfo is None:
        raise ValueError("l'heure de début doit avoir un fuseau horaire")
    return f"meet:{_slug(title)}:{start.astimezone(UTC):%Y-%m-%dT%H:%M}Z"


def parse_graph_datetime(value: Mapping[str, Any]) -> datetime:
    """Lit `{"dateTime": "2026-10-21T07:00:00.0000000", "timeZone": "UTC"}` (format Graph)."""
    if value.get("timeZone") != "UTC":
        raise ValueError("fuseau Graph non géré (demander les heures en UTC)")
    return datetime.fromisoformat(value["dateTime"]).replace(tzinfo=UTC)


def graph_event_correlation_id(event: Mapping[str, Any]) -> str:
    """Pour un événement `get-calendar-view`."""
    return meeting_correlation_id(event.get("subject") or "", parse_graph_datetime(event["start"]))


def graph_invitation_correlation_id(mail: Mapping[str, Any]) -> str | None:
    """Pour un mail `list-mail-messages` ; None si ce n'est pas une invitation à une réunion."""
    if not mail.get("meetingMessageType") or "startDateTime" not in mail:
        return None
    return meeting_correlation_id(
        mail.get("subject") or "", parse_graph_datetime(mail["startDateTime"])
    )
