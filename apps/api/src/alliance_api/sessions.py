"""Sessions côté serveur (L3.4, #112).

Le cookie du navigateur ne contient qu'un identifiant aléatoire : l'utilisateur reste sur le
serveur, donc un cookie modifié ne permet pas de se faire passer pour un autre. Le rôle n'y
est pas figé : il est relu dans la politique de droits à chaque requête.

Durée limitée : au plus `max_age` après la connexion (10 h par défaut, une journée de travail)
et au plus `idle` sans activité (1 h). Une session expirée est gardée 24 h comme « expirée »
(et non « inconnue ») : l'interface peut dire « session expirée » et la reprise après
reconnexion (#112, PR B) sait qui était connecté. La déconnexion, elle, efface tout.

Aucune donnée de travail n'est rangée dans la session (briefs, retours : par utilisateur) :
une expiration ne fait rien perdre.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

Status = Literal["ok", "expired", "missing"]
Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Session:
    session_id: str
    user: str
    created_at: datetime
    last_seen: datetime


class SessionStore(Protocol):
    max_age: timedelta

    def create(self, user: str) -> Session: ...
    def lookup(self, session_id: str) -> tuple[Status, Session | None]:
        """« ok » (et l'activité est notée), « expired » (avec la session d'origine) ou
        « missing » (identifiant inconnu ou session fermée par déconnexion)."""
        ...

    def delete(self, session_id: str) -> None: ...


class MemorySessionStore:
    """Sessions dans le processus : perdues au redémarrage (il suffit de se reconnecter),
    propres à chaque worker. Postgres ensuite, derrière la même interface."""

    def __init__(
        self,
        max_age: timedelta = timedelta(hours=10),
        idle: timedelta = timedelta(hours=1),
        clock: Clock = utc_now,
        keep_expired: timedelta = timedelta(hours=24),
    ) -> None:
        self.max_age = max_age
        self.idle = idle
        self._clock = clock
        self._keep_expired = keep_expired
        self._live: dict[str, Session] = {}
        self._expired: dict[str, tuple[Session, datetime]] = {}
        self._lock = threading.Lock()

    def create(self, user: str) -> Session:
        now = self._clock()
        session = Session(secrets.token_urlsafe(32), user, now, now)
        with self._lock:
            self._purge(now)
            self._live[session.session_id] = session
        return session

    def lookup(self, session_id: str) -> tuple[Status, Session | None]:
        now = self._clock()
        with self._lock:
            session = self._live.get(session_id)
            if session is None:
                gone = self._expired.get(session_id)
                return ("expired", gone[0]) if gone else ("missing", None)
            if self._is_over(session, now):
                del self._live[session_id]
                self._expired[session_id] = (session, now)
                return "expired", session
            session = replace(session, last_seen=now)
            self._live[session_id] = session
            return "ok", session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._live.pop(session_id, None)
            self._expired.pop(session_id, None)

    def _is_over(self, session: Session, now: datetime) -> bool:
        return now - session.created_at >= self.max_age or now - session.last_seen >= self.idle

    def _purge(self, now: datetime) -> None:
        """Ménage à chaque connexion : la mémoire ne grossit pas indéfiniment."""
        for sid, session in list(self._live.items()):
            if self._is_over(session, now):
                del self._live[sid]
                self._expired[sid] = (session, now)
        for sid, (_, since) in list(self._expired.items()):
            if now - since >= self._keep_expired:
                del self._expired[sid]
