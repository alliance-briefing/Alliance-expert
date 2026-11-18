"""Rôles → permissions, lus dans un fichier TOML (L3.4, #112).

Le code ne teste jamais un rôle (« est-ce un admin ? ») mais une permission
(« peut-il lire le journal d'audit ? »). Changer les droits d'un rôle, ou en créer un,
se fait dans le fichier (`roles.toml` à côté de ce module, ou `ROLES_FILE`), sans toucher
aux routes. Le fichier est vérifié au démarrage : une permission mal orthographiée ou un
rôle inconnu arrêtent l'application au lieu d'ouvrir ou de fermer un accès en silence.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROLES_FILE = Path(__file__).parent / "roles.toml"

PERMISSIONS = frozenset(
    {
        "brief:read_own",  # lire ses briefs et leur historique
        "brief:generate_own",  # générer son brief
        "feedback:own",  # donner et relire ses retours utile / pas utile / faux
        "item:read_own",  # ouvrir ses items sources
        "audit:read",  # journal d'audit (identifiants seulement), page d'administration
        "brief:read_any",  # lire les briefs des autres : prévue, attribuée à aucun rôle
    }
)


@dataclass(frozen=True)
class Policy:
    roles: dict[str, frozenset[str]]
    users: dict[str, str]
    default_role: str

    def role_of(self, user: str) -> str:
        return self.users.get(user, self.default_role)

    def allows(self, role: str, permission: str) -> bool:
        return permission in self.roles.get(role, frozenset())


def load_policy(path: Path = DEFAULT_ROLES_FILE) -> Policy:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    roles = {name: frozenset(perms) for name, perms in data.get("roles", {}).items()}
    users = dict(data.get("users", {}))
    default_role = data.get("default_role", "")
    for name, perms in roles.items():
        unknown = perms - PERMISSIONS
        if unknown:
            raise ValueError(f"{path} : permissions inconnues pour {name!r} : {sorted(unknown)}")
    for role in {default_role, *users.values()}:
        if role not in roles:
            raise ValueError(f"{path} : rôle {role!r} utilisé mais non défini dans [roles]")
    return Policy(roles=roles, users=users, default_role=default_role)
