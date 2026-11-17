"""Journal d'audit de la collecte (guide L1.8, contrat C).

Chaque connecteur lancé par `runner.run` produit au moins un `AuditEvent` : qui (le
connecteur), pour quel utilisateur, quelle source, combien d'Items, lesquels, et le
statut. Jamais de contenu, de jeton ni d'adresse : seulement des identifiants et des
compteurs (le contrat C refuse tout « @ »). En cas d'échec, seul le type d'erreur est noté.

Stockage : `DATA_DIR/<user>/audit.jsonl`, un événement par ligne, **en ajout seulement**
(jamais réécrit). Un fichier par utilisateur : isolation et suppression simples (L1.7).

Traçabilité d'un Item du brief : l'événement qui cite son `item_id` dit quand et par quel
connecteur il a été collecté ; la version du connecteur est dans `item.connector.version`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from contracts.models import AuditEvent, Item
from packages.collect.core.store import user_folder

FILE_NAME = "audit.jsonl"

AuditSink = Callable[[AuditEvent], None]
Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def collect_events(
    connector: str, user: str, items: Iterable[Item], at: datetime
) -> list[AuditEvent]:
    """Une collecte réussie : un événement par source (un connecteur peut en avoir plusieurs).

    Aucun Item : un seul événement, `count = 0`, pour prouver que la source a été interrogée.
    """
    by_source: dict[str, list[str]] = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item.item_id)
    if not by_source:
        return [_event(connector, user, at, None, [], "ok")]
    return [
        _event(connector, user, at, source, sorted(ids), "ok")
        for source, ids in sorted(by_source.items())
    ]


def failure_event(connector: str, user: str, at: datetime, error: Exception) -> AuditEvent:
    """Une source en échec : le type d'erreur seulement, jamais son message."""
    return _event(connector, user, at, None, [], "error", type(error).__name__)


def _event(
    connector: str,
    user: str,
    at: datetime,
    source: str | None,
    item_ids: list[str],
    status: str,
    error_code: str | None = None,
) -> AuditEvent:
    return AuditEvent(
        ts=at,
        actor=f"connector:{connector}",
        user_id=user,
        action="collect",
        source=source,
        item_ids=item_ids,
        count=len(item_ids),
        status=status,
        error_code=error_code,
    )


class AuditLog:
    """Journal JSON Lines d'un dossier de données, un fichier par utilisateur."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path(self, user: str) -> Path:
        return user_folder(self.root, user) / FILE_NAME

    def write(self, event: AuditEvent) -> None:
        path = self.path(event.user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # « a » : ajout seulement ; une ligne complète par écriture, puis vidée sur disque
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(event.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read(self, user: str) -> list[AuditEvent]:
        path = self.path(user)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [AuditEvent.model_validate_json(line) for line in handle if line.strip()]
