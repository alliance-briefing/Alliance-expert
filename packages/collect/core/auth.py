"""Jeton Microsoft Graph pour les connecteurs (guide L1.2).

`get_access_token(user)` renvoie un jeton valide : celui rangé par `auth login` (chiffré),
renouvelé automatiquement s'il a expiré. Sans connexion utilisable (jamais connecté, accès
retiré, jeton de renouvellement expiré), elle lève `AuthError("reauth_required")` : le
runner en fait un warning, les autres sources continuent.

Dépannage seulement : un jeton copié depuis Graph Explorer dans GRAPH_ACCESS_TOKEN est
utilisé tel quel (il expire au bout d'environ 1 h). Ne jamais l'écrire dans un fichier.
"""

from __future__ import annotations

import os

from packages.collect.core.oauth import ReauthRequired, TokenStore, access_token, load_config


class AuthError(Exception):
    """Pas de jeton utilisable : l'utilisateur doit se reconnecter."""

    def __init__(self) -> None:
        super().__init__("reauth_required")


def get_access_token(user: str) -> str:
    manual = os.environ.get("GRAPH_ACCESS_TOKEN", "").strip()
    if manual:
        return manual
    try:
        return access_token(user, TokenStore(), load_config())
    except ReauthRequired as error:
        raise AuthError from error
