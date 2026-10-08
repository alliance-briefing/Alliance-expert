"""Journal d'audit côté API (contrat C) : chaque appel authentifié écrit un AuditEvent.

On ne journalise que des identifiants et des compteurs (le modèle refuse les adresses).
Implémentation actuelle : une ligne JSON par événement sur le logger, plus les derniers
événements en mémoire (tests, future page d'administration L3.7). Remplaçable par une
table Postgres sans toucher aux endpoints.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from typing import Literal, Protocol

from contracts.models import AuditEvent

logger = logging.getLogger("alliance_api.audit")


class AuditSink(Protocol):
    def write(self, event: AuditEvent) -> None: ...


class LogAuditSink:
    def __init__(self, keep: int = 10_000) -> None:
        self.events: deque[AuditEvent] = deque(maxlen=keep)

    def write(self, event: AuditEvent) -> None:
        self.events.append(event)
        logger.info(event.model_dump_json())


def api_event(
    user: str,
    action: Literal["generate", "view"],
    item_ids: list[str],
    count: int,
    error_code: str | None = None,
) -> AuditEvent:
    """Événement écrit par l'API ; `error_code` renseigné = appel en erreur."""
    return AuditEvent(
        ts=datetime.now(UTC),
        actor="api",
        user_id=user,
        action=action,
        item_ids=item_ids,
        count=count,
        status="error" if error_code else "ok",
        error_code=error_code,
    )
