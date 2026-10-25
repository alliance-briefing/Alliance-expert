"""Qui est qui pour un manager : type d'expéditeur d'un mail (hiérarchie, client…).

Le contrat A ne donne ni l'organigramme ni les comptes clients : dans un vrai déploiement,
ils viendraient de l'annuaire et du CRM. Ici, une configuration par persona, versionnée.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SenderKind = Literal["hiérarchie", "direction", "collègue", "client", "fournisseur", "automate"]
SENDER_KINDS: tuple[SenderKind, ...] = (
    "hiérarchie",
    "direction",
    "collègue",
    "client",
    "fournisseur",
    "automate",
)
_AUTOMATED_LOCALS = ("noreply", "no-reply", "support", "alertes", "taches", "securite", "billets")


@dataclass(frozen=True)
class Persona:
    """Adresses en minuscules ; domaines sans le « @ »."""

    address: str
    internal_domain: str
    hierarchy: frozenset[str]  # N+1 et au-dessus
    direction: frozenset[str]  # pairs du comité de direction
    client_domains: frozenset[str]

    def sender_kind(self, email: str) -> SenderKind:
        email = email.lower()
        local, _, domain = email.partition("@")
        if email in self.hierarchy:
            return "hiérarchie"
        if email in self.direction:
            return "direction"
        # Une personne a une adresse « prénom.nom » ; un automate, « support@… », « alertes@… »
        person = "." in local and not local.startswith(_AUTOMATED_LOCALS)
        if not person:
            return "automate"
        if domain == self.internal_domain:
            return "collègue"
        if domain in self.client_domains:
            return "client"
        return "fournisseur"


# Hélène Marchand, directrice des opérations (données Graph de David), validé le 25/10/2026 :
# le DG est son N+1, la DAF et la DRH sont ses pairs de direction.
_DOM = "verdier-expertises.example"
GRAPH_HELENE = Persona(
    address=f"helene.marchand@{_DOM}",
    internal_domain=_DOM,
    hierarchy=frozenset({f"bruno.lefevre@{_DOM}"}),
    direction=frozenset({f"sandrine.moreau@{_DOM}", f"isabelle.garnier@{_DOM}"}),
    client_domains=frozenset(
        {
            "altair-groupe.example",
            "trois-rives.example",
            "saint-aubin.example",
            "brasserie-delacour.example",
            "noviel.example",
            "marchetti-transports.example",
            "agrinova.example",
        }
    ),
)
