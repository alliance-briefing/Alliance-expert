"""Jeton Microsoft Graph — version provisoire en attendant L1.2 (rôle A, Rayan).

Rayan remplacera le corps de get_access_token par l'OAuth complet ; son nom et sa
signature ne changent pas, les connecteurs n'auront rien à modifier.

Essai réel sans OAuth : copier le jeton de Graph Explorer dans la variable d'environnement
GRAPH_ACCESS_TOKEN (il expire au bout d'environ 1 h). Ne jamais l'écrire dans un fichier.
"""

from __future__ import annotations

import os


class AuthError(Exception):
    """Pas de jeton utilisable : l'utilisateur doit se reconnecter."""

    def __init__(self) -> None:
        super().__init__("reauth_required")


def get_access_token(user: str) -> str:
    token = os.environ.get("GRAPH_ACCESS_TOKEN", "").strip()
    if not token:
        raise AuthError
    return token
