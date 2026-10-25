"""Qui est qui pour un manager : type d'expéditeur d'un mail (hiérarchie, client…).

Le contrat A ne donne ni l'organigramme ni les comptes clients : dans un vrai déploiement,
ils viendraient de l'annuaire et du CRM. Ici, une configuration par persona, versionnée.
"""

from __future__ import annotations

import re
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
# Premier mot de l'adresse d'un automate (« support@… », « noreply-missions@… »). Une adresse
# sans « prénom.nom » n'en est pas forcément un : « contact@garage-tilleuls » est un garage.
_AUTOMATED_WORDS = frozenset(
    {
        "noreply",
        "no",
        "donotreply",
        "support",
        "alertes",
        "alerte",
        "alerts",
        "taches",
        "securite",
        "security",
        "billets",
        "notification",
        "notifications",
        "reporting",
        "newsletter",
        "redaction",
        "quotidien",
        "promo",
        "info",
        "system",
        "mailer",
        "admin",
    }
)


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
        if re.split(r"[.\-_+]", local)[0] in _AUTOMATED_WORDS:
            return "automate"
        if domain == self.internal_domain or domain.endswith("." + self.internal_domain):
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


# Camille Laurent, responsable de l'agence Lyon-Est d'un réseau d'expertise automobile (données
# de démo du lot 1), validé le 26/10/2026 : Bernard Mercier est son N+1 ; Stéphane Morel, Valérie
# Chevalier et Arnaud Petitjean ses pairs de direction régionale ; les assureurs et les assurés
# sont ses clients ; garages, carrosseries, loueurs et vendeurs de pièces, ses fournisseurs.
_DEMO = "reseau-expertis.test"
DEMO_CAMILLE = Persona(
    address=f"camille.laurent@{_DEMO}",
    internal_domain=_DEMO,
    hierarchy=frozenset({f"bernard.mercier@{_DEMO}"}),
    direction=frozenset(
        {f"stephane.morel@{_DEMO}", f"valerie.chevalier@{_DEMO}", f"arnaud.petitjean@{_DEMO}"}
    ),
    client_domains=frozenset(
        {
            "horizon-sud.test",
            "lumen-assurances.test",
            "assuralpes.test",
            "assurance-quebec.test",
            "particulier.test",
        }
    ),
)
