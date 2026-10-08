"""Annuaire fictif : un manager, son équipe, sa hiérarchie et ses interlocuteurs.

Tout est inventé. Les domaines en .test sont réservés (RFC 2606) : aucune adresse
ne peut exister réellement.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


def slug(text: str) -> str:
    text = text.replace("œ", "oe").replace("Œ", "Oe").replace("æ", "ae").replace("Æ", "Ae")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", ".", text.lower())
    return text.strip(".")


@dataclass(frozen=True)
class Person:
    key: str
    name: str
    email: str
    role: str  # manager, expert, assistant, directeur, fraude, rh, assureur, garage, client, partenaire, robot
    org: str
    title: str = ""
    automated: bool = False

    @property
    def first_name(self) -> str:
        return self.name.split(" ")[0]


@dataclass(frozen=True)
class ManagerProfile:
    user_id: str
    first_name: str
    last_name: str
    agency: str
    code: str  # préfixe des identifiants natifs, unique par manager (isolation)
    experts: tuple[tuple[str, str], ...]
    assistant: tuple[str, str]


PROFILES: dict[str, ManagerProfile] = {
    "u_demo_001": ManagerProfile(
        user_id="u_demo_001",
        first_name="Camille",
        last_name="Laurent",
        agency="Lyon-Est",
        code="dm1",
        experts=(
            ("Anaïs", "Lefèvre-Dubœuf"),
            ("Loïc", "Ngoma"),
            ("Maëlle", "Rousseau"),
            ("Jérôme", "Faure"),
            ("Karim", "Benali"),
            ("Sophie", "Nguyen"),
        ),
        assistant=("Noémie", "Girard"),
    ),
    "u_demo_002": ManagerProfile(
        user_id="u_demo_002",
        first_name="Thomas",
        last_name="Mbemba",
        agency="Grenoble",
        code="dm2",
        experts=(
            ("Clémence", "Bérard"),
            ("Yanis", "Haddad"),
            ("Léa", "Perrin"),
            ("François", "Ødegaard"),
        ),
        assistant=("Inès", "Carvalho"),
    ),
}

INTERNAL_DOMAIN = "reseau-expertis.test"

INSURERS = (
    ("Mutuelle Horizon Sud", "horizon-sud.test"),
    ("Assur'Alpes", "assuralpes.test"),
    ("Coopérative Lumen Assurances", "lumen-assurances.test"),
    ("Solvéo Assurances", "solveo.test"),
)
GARAGES = (
    ("Garage des Tilleuls", "Villeurbanne", "garage-tilleuls.test"),
    ("Carrosserie Martin & Fils", "Vénissieux", "carrosserie-martin.test"),
    ("Auto Sécurité Bron", "Bron", "autosecurite-bron.test"),
    ("Garage Saint-Exupéry", "Lyon 8e", "garage-st-exupery.test"),
    ("Carrosserie du Rhône", "Caluire-et-Cuire", "carrosserie-rhone.test"),
)
FLEETS = (
    ("Transports Rhodaniens", "transports-rhodaniens.test"),
    ("Location Pro Dauphiné", "locpro-dauphine.test"),
)
INSURER_CONTACTS = (
    ("Julie", "Marchand"),
    ("Olivier", "Da Silva"),
    ("Agnès", "Perret"),
    ("Mathieu", "Colin"),
    ("Samira", "Ouali"),
    ("Benoît", "Lemaître"),
    ("Chloé", "Vasseur"),
    ("Rémi", "Fontaine"),
)
CLIENTS = (
    ("Marc", "Delorme"),
    ("Fatou", "Sarr"),
    ("Élodie", "Chabert"),
    ("Pierre-Yves", "Morin"),
    ("Aurélie", "Joly"),
    ("Kevin", "Barbier"),
    ("Nadia", "Belkacem"),
    ("Hugo", "Lacroix"),
    ("Céline", "Roux"),
    ("Antoine", "Gauthier"),
)
COLLECTION_OWNERS = (("Jean-Claude", "Vermeil"), ("Odile", "Saint-Cyr"))


@dataclass
class Directory:
    manager: Person
    experts: list[Person]
    assistant: Person
    directeur: Person
    fraude: Person
    rh: Person
    insurers: list[Person]
    garages: list[Person]
    clients: list[Person]
    fleets: list[Person]
    collectors: list[Person]
    partner_montreal: Person
    peers: list[Person]
    robots: dict[str, Person] = field(default_factory=dict)

    def colleagues(self) -> list[Person]:
        return [*self.experts, self.assistant]


def _internal(first: str, last: str, role: str, title: str, org: str = "Réseau Expertis") -> Person:
    name = f"{first} {last}"
    return Person(
        key=slug(name),
        name=name,
        email=f"{slug(first)}.{slug(last)}@{INTERNAL_DOMAIN}",
        role=role,
        org=org,
        title=title,
    )


def _robot(key: str, name: str, email: str) -> Person:
    return Person(key=key, name=name, email=email, role="robot", org="Système", automated=True)


def build_directory(profile: ManagerProfile) -> Directory:
    manager = _internal(
        profile.first_name, profile.last_name, "manager", f"Responsable d'agence {profile.agency}"
    )
    experts = [_internal(f, n, "expert", "Expert automobile") for f, n in profile.experts]
    assistant = _internal(*profile.assistant, "assistant", "Assistante d'agence")

    insurers = []
    for index, (first, last) in enumerate(INSURER_CONTACTS):
        org, domain = INSURERS[index % len(INSURERS)]
        name = f"{first} {last}"
        insurers.append(
            Person(
                slug(name),
                name,
                f"{slug(first)}.{slug(last)}@{domain}",
                "assureur",
                org,
                "Gestionnaire sinistres",
            )
        )
    garages = [
        Person(slug(g), f"Accueil {g}", f"contact@{domain}", "garage", g, f"Garage – {city}")
        for g, city, domain in GARAGES
    ]
    clients = [
        Person(
            slug(f"{f} {n}"),
            f"{f} {n}",
            f"{slug(f)}.{slug(n)}@particulier.test",
            "client",
            "Particulier",
        )
        for f, n in CLIENTS
    ]
    fleets = [
        Person(slug(o), f"Service parc – {o}", f"parc@{d}", "client", o, "Gestion de flotte")
        for o, d in FLEETS
    ]
    collectors = [
        Person(
            slug(f"{f} {n}"),
            f"{f} {n}",
            f"{slug(f)}.{slug(n)}@collection.test",
            "client",
            "Particulier",
        )
        for f, n in COLLECTION_OWNERS
    ]
    montreal = Person(
        "genevieve.tremblay",
        "Geneviève Tremblay",
        "genevieve.tremblay@assurance-quebec.test",
        "partenaire",
        "Assurance Québec Partenaires",
        "Directrice des partenariats",
    )
    peers = [
        _internal("Stéphane", "Morel", "pair", "Responsable d'agence Villeurbanne"),
        _internal("Valérie", "Chevalier", "pair", "Responsable d'agence Saint-Étienne"),
        _internal("Arnaud", "Petitjean", "pair", "Responsable d'agence Valence"),
    ]
    robots = {
        "missions": _robot(
            "robot.missions", "Plateforme Missions", f"noreply-missions@{INTERNAL_DOMAIN}"
        ),
        "sharepoint": _robot(
            "robot.sharepoint", "SharePoint", f"no-reply@sharepoint.{INTERNAL_DOMAIN}"
        ),
        "teams": _robot(
            "robot.teams", "Microsoft Teams (démo)", f"noreply@teams.{INTERNAL_DOMAIN}"
        ),
        "reporting": _robot(
            "robot.reporting", "Reporting automatique", f"noreply-reporting@{INTERNAL_DOMAIN}"
        ),
        "securite": _robot(
            "robot.securite", "Service informatique", f"notifications-it@{INTERNAL_DOMAIN}"
        ),
        "newsletter_argus": _robot(
            "robot.argus", "La Lettre de l'Expertise", "newsletter@lettre-expertise.test"
        ),
        "newsletter_fournisseur": _robot(
            "robot.fournisseur", "PiècesAuto Pro", "newsletter@piecesauto-pro.test"
        ),
        "newsletter_interne": _robot(
            "robot.interne", "Communication interne", f"newsletter@{INTERNAL_DOMAIN}"
        ),
        "export": _robot("robot.export", "Robot d'export", f"noreply-export@{INTERNAL_DOMAIN}"),
    }
    return Directory(
        manager=manager,
        experts=experts,
        assistant=assistant,
        directeur=_internal(
            "Bernard", "Mercier", "directeur", "Directeur régional Auvergne-Rhône-Alpes"
        ),
        fraude=_internal("Hélène", "Dürr", "fraude", "Cellule anti-fraude"),
        rh=_internal("Gaël", "Kowalski", "rh", "Ressources humaines"),
        insurers=insurers,
        garages=garages,
        clients=clients,
        fleets=fleets,
        collectors=collectors,
        partner_montreal=montreal,
        peers=peers,
        robots=robots,
    )
