"""Utilisateur connecté — authentification PROVISOIRE, en attendant L3.4 (OAuth/OpenID).

Mode `demo` : l'utilisateur est lu dans l'en-tête X-Demo-User (clients d'API) ou dans le
cookie `demo_user` (navigateur, posé par la page de connexion de l'interface ; SameSite=Strict
en guise de protection CSRF en attendant L3.4). Ce n'est PAS une sécurité
(n'importe qui peut se faire passer pour n'importe qui) : ce mode sert aux données
synthétiques, en local et en CI. Tout autre mode refuse toutes les requêtes (401) :
sans configuration explicite, l'API est fermée.

L3.4 remplacera uniquement `current_user` par la lecture d'une session ; les endpoints,
qui ne connaissent que `Depends(current_user)`, ne changeront pas.
"""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import Depends, Request

from alliance_api.errors import ApiError

DEMO_HEADER = "X-Demo-User"
DEMO_COOKIE = "demo_user"
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
