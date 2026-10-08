"""Utilisateur connecté — authentification PROVISOIRE, en attendant L3.4 (OAuth/OpenID).

Mode `demo` : l'utilisateur est lu dans l'en-tête X-Demo-User. Ce n'est PAS une sécurité
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
# L'identifiant sert aussi de nom de dossier dans les données de démo : format strict,
# sinon « ../ » permettrait de lire hors du dossier de l'utilisateur.
_USER_ID = re.compile(r"^u_[a-z0-9_]{1,60}$")


def current_user(request: Request) -> str:
    if request.app.state.auth_mode != "demo":
        raise ApiError(401, "unauthenticated", "authentification requise")
    user = request.headers.get(DEMO_HEADER, "")
    if not _USER_ID.match(user):
        raise ApiError(401, "unauthenticated", "authentification requise")
    return user


CurrentUser = Annotated[str, Depends(current_user)]
