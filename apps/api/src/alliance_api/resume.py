"""Reprise après expiration de session (L3.4, #112, étape B).

Si la session expire pendant que la personne travaille, rien n'est perdu :
1. `defer` met de côté l'action en cours (ou seulement la page) pour 15 minutes, sous un jeton
   aléatoire à usage unique (cookie `resume`), puis renvoie vers la connexion ;
2. après reconnexion, `complete` rejoue l'action et renvoie sur la même page, **si et
   seulement si** c'est la même personne qui se reconnecte.

Seules les actions de `REPLAYABLE` sont rejouables ; elles passent par les mêmes fonctions
que l'API (`briefs.generate`, `feedback.record`), donc par les mêmes contrôles. Le contrôle
CSRF a eu lieu au moment du clic (middleware), avant la mise de côté.
Indépendant du mode de connexion : la connexion OpenID Connect appellera aussi `complete`.
"""

from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date as Date
from datetime import datetime, timedelta
from typing import Any

from fastapi import Request, Response
from fastapi.responses import RedirectResponse
from starlette.datastructures import State

from alliance_api import briefs, feedback
from alliance_api.errors import ApiError
from alliance_api.sessions import Clock, utc_now

logger = logging.getLogger("alliance_api")
RESUME_COOKIE = "resume"
PENDING_TTL = timedelta(minutes=15)

Replay = Callable[[State, str, dict[str, Any]], object]
REPLAYABLE: dict[str, Replay] = {
    "generate": lambda state, user, data: briefs.generate(
        state, user, Date.fromisoformat(data["date"])
    ),
    "feedback": lambda state, user, data: feedback.record(
        state, user, feedback.FeedbackRequest.model_validate(data)
    ),
}


@dataclass(frozen=True)
class Pending:
    user: str  # la personne dont la session a expiré
    next_url: str  # page où revenir
    action: str | None = None  # None : seulement revenir à la page
    data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


class PendingStore:
    """Actions mises de côté, en mémoire, 15 minutes au plus, reprises une seule fois."""

    def __init__(self, ttl: timedelta = PENDING_TTL, clock: Clock = utc_now) -> None:
        self.ttl = ttl
        self._clock = clock
        self._pending: dict[str, Pending] = {}
        self._lock = threading.Lock()

    def put(self, pending: Pending) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = self._clock()
            self._pending = {
                t: p for t, p in self._pending.items() if now - p.created_at < self.ttl
            }
            self._pending[token] = replace(pending, created_at=now)
        return token

    def peek(self, token: str) -> Pending | None:
        with self._lock:
            pending = self._pending.get(token)
        return pending if pending and self._fresh(pending) else None

    def take(self, token: str) -> Pending | None:
        with self._lock:
            pending = self._pending.pop(token, None)
        return pending if pending and self._fresh(pending) else None

    def _fresh(self, pending: Pending) -> bool:
        return self._clock() - pending.created_at < self.ttl


def safe_next(url: str) -> str:
    """Chemin interne uniquement : jamais de redirection vers un autre site."""
    if url.startswith("/") and not url.startswith("//") and "\\" not in url:
        return url
    return "/"


def defer(
    request: Request, user: str, next_url: str, action: str | None = None, data: dict | None = None
) -> Response:
    """Met de côté l'action de `user` (session expirée) et renvoie vers la connexion."""
    if action is not None and action not in REPLAYABLE:
        raise ValueError(f"action non rejouable : {action!r}")
    pending = Pending(user, safe_next(next_url), action, dict(data or {}))
    token = request.app.state.pending.put(pending)
    response = RedirectResponse("/login?expired=1", status_code=303)
    # Lax : le cookie doit revenir avec la redirection du fournisseur d'identité (étape C).
    response.set_cookie(
        RESUME_COOKIE,
        token,
        max_age=int(request.app.state.pending.ttl.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )
    return response


def pending_user(request: Request) -> str | None:
    """Qui avait une action en attente (pour préremplir la reconnexion), sans la consommer."""
    token = request.cookies.get(RESUME_COOKIE)
    pending = request.app.state.pending.peek(token) if token else None
    return pending.user if pending else None


def complete(request: Request, response: Response, user: str) -> str:
    """Après connexion de `user` : rejoue l'action en attente si elle est à lui, efface le
    jeton, et renvoie la page où revenir ("/" par défaut)."""
    token = request.cookies.get(RESUME_COOKIE)
    response.delete_cookie(RESUME_COOKIE, httponly=True, samesite="lax", path="/")
    pending = request.app.state.pending.take(token) if token else None
    if pending is None or pending.user != user:
        return "/"  # rien en attente, expiré, ou une autre personne : on ne rejoue rien
    if pending.action is not None:
        try:
            REPLAYABLE[pending.action](request.app.state, user, pending.data)
            logger.info("reprise user=%s action=%s ok", user, pending.action)
        except ApiError as exc:
            # La page d'arrivée montre l'état réel (ex. aucune collecte ce jour-là).
            logger.info("reprise user=%s action=%s refusée (%s)", user, pending.action, exc.code)
    return pending.next_url
