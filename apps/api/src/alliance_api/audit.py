"""Journal d'audit côté API (contrat C) : chaque appel authentifié écrit un AuditEvent.

On ne journalise que des identifiants et des compteurs (le modèle refuse les adresses).

- `LogAuditSink` : une ligne JSON par événement sur le logger, plus les derniers événements
  en mémoire (tests, démonstration sans dossier ; perdus au redémarrage).
- `JsonlAuditLog` : un fichier JSON Lines par utilisateur, en ajout seul. Pour l'API,
  `AUDIT_DIR/<user>/api-audit.jsonl` ; avec `AUDIT_DIR` = `DATA_DIR`, le fichier est dans le
  dossier de l'utilisateur et `delete-user` (lot 1, L1.7) l'efface avec le reste.
  Le même format sert à **lire** le journal de collecte du lot 1
  (`DATA_DIR/<user>/audit.jsonl`), que l'API n'écrit jamais (L3.7).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import ValidationError

from alliance_api.auth import valid_user_id
from contracts.models import AuditEvent

logger = logging.getLogger("alliance_api.audit")

API_FILE = "api-audit.jsonl"  # journal de l'API (écrit ici)
COLLECT_FILE = "audit.jsonl"  # journal de collecte du lot 1 (lu seulement)


class AuditSink(Protocol):
    def write(self, event: AuditEvent) -> None: ...

    def read(self, user: str | None = None) -> list[AuditEvent]: ...


class AuditSource(Protocol):
    """Journal en lecture seule (celui du lot 1)."""

    def read(self, user: str | None = None) -> list[AuditEvent]: ...


class LogAuditSink:
    def __init__(self, keep: int = 10_000) -> None:
        self.events: deque[AuditEvent] = deque(maxlen=keep)

    def write(self, event: AuditEvent) -> None:
        self.events.append(event)
        logger.info(event.model_dump_json())

    def read(self, user: str | None = None) -> list[AuditEvent]:
        return [event for event in self.events if user is None or event.user_id == user]


class JsonlAuditLog:
    """`<root>/<user>/<file_name>`, une ligne par événement, jamais réécrit."""

    def __init__(self, root: Path, file_name: str = API_FILE) -> None:
        self.root = Path(root)
        self.file_name = file_name
        self._lock = threading.Lock()

    def _file(self, user: str) -> Path:
        if not valid_user_id(user):
            raise ValueError("identifiant d'utilisateur invalide")
        return self.root / user / self.file_name

    def write(self, event: AuditEvent) -> None:
        path = self._file(event.user_id)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as out:
                out.write(event.model_dump_json() + "\n")

    def read(self, user: str | None = None) -> list[AuditEvent]:
        """Événements de `user`, ou de tous les utilisateurs du dossier."""
        users = [user] if user is not None else self._users()
        events: list[AuditEvent] = []
        for name in users:
            events.extend(self._read_file(self._file(name)))
        return events

    def _users(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.name
            for p in self.root.iterdir()
            if valid_user_id(p.name) and (p / self.file_name).is_file()
        )

    @staticmethod
    def _read_file(path: Path) -> list[AuditEvent]:
        if not path.is_file():
            return []
        events = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                events.append(AuditEvent.model_validate_json(line))
            except ValidationError:
                # Une ligne illisible (écriture interrompue) ne doit pas masquer le reste.
                logger.warning("ligne %d illisible dans %s : ignorée", number, path.name)
        return events


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
