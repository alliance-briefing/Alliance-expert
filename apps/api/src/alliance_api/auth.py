"""Utilisateur connecté — authentification PROVISOIRE, en attendant L3.4 (OAuth/OpenID).

Mode `demo` : l'utilisateur est lu dans l'en-tête X-Demo-User (clients d'API) ou dans le
cookie `demo_user` (navigateur, posé par la page de connexion de l'interface ; SameSite=Strict
en guise de protection CSRF en attendant L3.4). Ce n'est PAS une sécurité
(n'importe qui peut se faire passer pour n'importe qui) : ce mode sert aux données
synthétiques, en local et en CI. Tout autre mode refuse toutes les requêtes (401) :
sans configuration explicite, l'API est fermée.

L3.4 remplacera uniquement `current_user` par la lecture d'une session ; les endpoints,
qui ne connaissent que `Depends(current_user)`, ne changeront pas.

Droits (#112) : une route exige une permission (`requires("brief:read_own")`), jamais un
rôle ; le rôle de l'utilisateur et ses permissions viennent de `roles.toml` (permissions.py).

CSRF : cookie SameSite=Strict, et toute requête qui modifie quelque chose (POST…) venant
d'une autre origine (en-tête Origin, à défaut Referer) est refusée (403).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Request, Response

from alliance_api.errors import ApiError, error_response

DEMO_HEADER = "X-Demo-User"
DEMO_COOKIE = "demo_user"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# L'identifiant sert aussi de nom de dossier dans les données de démo : format strict,
# sinon « ../ » permettrait de lire hors du dossier de l'utilisateur.
_USER_ID = re.compile(r"^u_[a-z0-9_]{1,60}$")


def valid_user_id(value: str) -> bool:
    return bool(_USER_ID.match(value))


def optional_user(request: Request) -> str | None:
    """L'utilisateur connecté, ou None (l'interface redirige alors vers la connexion)."""
    if request.app.state.auth_mode != "demo":
        return None
    user = request.headers.get(DEMO_HEADER) or request.cookies.get(DEMO_COOKIE, "")
    return user if valid_user_id(user) else None


def current_user(request: Request) -> str:
    user = optional_user(request)
    if user is None:
        raise ApiError(401, "unauthenticated", "authentification requise")
    return user


CurrentUser = Annotated[str, Depends(current_user)]


def allowed(request: Request, permission: str) -> bool:
    """La permission est-elle accordée à l'utilisateur connecté ? (pour l'interface)"""
    user = optional_user(request)
    policy = request.app.state.policy
    return user is not None and policy.allows(policy.role_of(user), permission)


def requires(permission: str) -> Any:
    """Dépendance de route : 401 sans utilisateur, 403 si son rôle n'a pas la permission."""

    def check(request: Request) -> None:
        policy = request.app.state.policy
        if not policy.allows(policy.role_of(current_user(request)), permission):
            raise ApiError(403, "forbidden", "droits insuffisants")

    return Depends(check)


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
