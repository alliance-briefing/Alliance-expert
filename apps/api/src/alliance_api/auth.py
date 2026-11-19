"""Utilisateur connecté, droits et protection CSRF (L3.4, #112).

Modes (`AUTH_MODE`) :
- `demo` : la page de connexion accepte un identifiant de démonstration, et les clients
  d'API peuvent envoyer l'en-tête X-Demo-User. Ce n'est PAS une sécurité (n'importe qui peut
  choisir n'importe quel identifiant) : données synthétiques, local, CI, démo hors ligne.
- `oidc` : connexion OpenID Connect (Microsoft pour la démo, voir `oidc.py`) ; l'en-tête
  X-Demo-User et le formulaire de démonstration sont ignorés.
- toute autre valeur : l'API est fermée (401).

Dans tous les modes, le navigateur reçoit une session côté serveur (`sessions.py`) : le
cookie ne porte qu'un identifiant aléatoire. Les routes ne connaissent que `CurrentUser`
et `requires(<permission>)` : changer de mode de connexion ne les touche pas.

CSRF : cookie SameSite=Strict, et toute requête qui modifie quelque chose (POST…) venant
d'une autre origine (en-tête Origin, à défaut Referer) est refusée (403).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Request, Response

from alliance_api.errors import ApiError, error_response

AUTH_MODES = frozenset({"demo", "oidc"})
DEMO_HEADER = "X-Demo-User"
SESSION_COOKIE = "session"
DEMO_COOKIE = SESSION_COOKIE  # ancien nom, encore utilisé par les tests de l'interface
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# L'identifiant sert aussi de nom de dossier dans les données de démo : format strict,
# sinon « ../ » permettrait de lire hors du dossier de l'utilisateur.
_USER_ID = re.compile(r"^u_[a-z0-9_]{1,60}$")


def valid_user_id(value: str) -> bool:
    return bool(_USER_ID.match(value))


@dataclass(frozen=True)
class Principal:
    user: str
    role: str


def _resolve(request: Request) -> tuple[Principal | None, bool]:
    """(utilisateur connecté ou None, session expirée ?), calculé une fois par requête."""
    cached = getattr(request.state, "auth", None)
    if cached is None:
        cached = request.state.auth = _lookup(request)
    return cached


def _lookup(request: Request) -> tuple[Principal | None, bool]:
    state = request.app.state
    if state.auth_mode not in AUTH_MODES:
        return None, False
    if state.auth_mode == "demo" and DEMO_HEADER in request.headers:
        user = request.headers[DEMO_HEADER]
        return (Principal(user, state.policy.role_of(user)) if valid_user_id(user) else None), False
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id:
        return None, False
    status, session = state.sessions.lookup(session_id)
    if status == "ok" and session is not None:
        return Principal(session.user, state.policy.role_of(session.user)), False
    return None, status == "expired"


def optional_user(request: Request) -> str | None:
    """L'utilisateur connecté, ou None (l'interface redirige alors vers la connexion)."""
    principal, _ = _resolve(request)
    return principal.user if principal else None


def session_expired(request: Request) -> bool:
    """Vrai si le navigateur présente une session qui a expiré (et non une absence de session)."""
    return _resolve(request)[1]


def expired_user(request: Request) -> str | None:
    """La personne dont la session vient d'expirer (pour reprendre son travail), sinon None."""
    if not session_expired(request):
        return None
    status, session = request.app.state.sessions.lookup(request.cookies[SESSION_COOKIE])
    return session.user if status == "expired" and session else None


def current_principal(request: Request) -> Principal:
    principal, expired = _resolve(request)
    if principal is None:
        if expired:
            raise ApiError(401, "session_expired", "session expirée, reconnectez-vous")
        raise ApiError(401, "unauthenticated", "authentification requise")
    return principal


def current_user(request: Request) -> str:
    return current_principal(request).user


CurrentUser = Annotated[str, Depends(current_user)]


def allowed(request: Request, permission: str) -> bool:
    """La permission est-elle accordée à l'utilisateur connecté ? (pour l'interface)"""
    principal, _ = _resolve(request)
    return principal is not None and request.app.state.policy.allows(principal.role, permission)


def requires(permission: str) -> Any:
    """Dépendance de route : 401 sans session, 403 si le rôle n'a pas la permission."""

    def check(request: Request) -> None:
        principal = current_principal(request)
        if not request.app.state.policy.allows(principal.role, permission):
            raise ApiError(403, "forbidden", "droits insuffisants")

    return Depends(check)


def _cookie_options(request: Request) -> dict[str, Any]:
    secure = request.url.scheme == "https"
    return {"httponly": True, "samesite": "strict", "secure": secure, "path": "/"}


def open_session(request: Request, response: Response, user: str) -> None:
    """Ouvre une session pour `user` et pose le cookie."""
    state = request.app.state
    session = state.sessions.create(user)
    max_age = int(state.sessions.max_age.total_seconds())
    response.set_cookie(
        SESSION_COOKIE, session.session_id, max_age=max_age, **_cookie_options(request)
    )


def close_session(request: Request, response: Response) -> None:
    """Déconnexion : la session est effacée côté serveur, le cookie devient inutilisable."""
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        request.app.state.sessions.delete(session_id)
    response.delete_cookie(SESSION_COOKIE, **_cookie_options(request))


def _origin_of(request: Request) -> str | None:
    origin = request.headers.get("origin")
    if origin is not None:
        return origin
    referer = request.headers.get("referer")
    if referer is None:
        return None  # client hors navigateur (curl, tests) : pas de risque CSRF
    parts = urlsplit(referer)
    return f"{parts.scheme}://{parts.netloc}"


def install_csrf(app: FastAPI) -> None:
    """Refuse les requêtes qui modifient quelque chose et viennent d'une autre origine.
    À installer AVANT `errors.install`, pour que l'identifiant de corrélation soit posé."""

    @app.middleware("http")
    async def _same_origin(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method in UNSAFE_METHODS:
            origin = _origin_of(request)
            own = f"{request.url.scheme}://{request.url.netloc}"
            if origin is not None and origin not in {own, *request.app.state.allowed_origins}:
                return error_response(
                    request, 403, "csrf_rejected", "requête venant d'une autre origine refusée"
                )
        return await call_next(request)
