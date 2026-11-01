"""Le monde d'un manager : son organisation, ses correspondants, son calendrier de travail.

40 personas : 3 par métier d'entraînement (36) et 1 par métier réservé à T-gen (4).
Noms inventés, tirés de façon déterministe ; adresses `prenom.nom@domaine.test`.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from packages.collect.trainset.metiers import METIERS, Metier

PARIS = ZoneInfo("Europe/Paris")
START = date(2026, 11, 2)  # un lundi
N_WORKING_DAYS = 15
PERSONAS_PER_TRAIN_METIER = 3
PERSONAS_PER_RESERVED_METIER = 1

FEMALE = (
    "Claire",
    "Nora",
    "Inès",
    "Sophie",
    "Camille",
    "Léa",
    "Chloé",
    "Sarah",
    "Amina",
    "Julie",
    "Manon",
    "Élodie",
    "Laura",
    "Yasmine",
    "Hélène",
    "Mathilde",
    "Aurélie",
    "Fatou",
    "Lucie",
    "Émilie",
    "Céline",
    "Pauline",
    "Anaïs",
    "Marion",
    "Nadia",
    "Agathe",
    "Sonia",
    "Valérie",
)
MALE = (
    "Hugo",
    "Karim",
    "Julien",
    "Thomas",
    "Bruno",
    "Marc",
    "Yanis",
    "Nicolas",
    "Antoine",
    "Mehdi",
    "Romain",
    "Pierre",
    "Olivier",
    "Samuel",
    "David",
    "Lucas",
    "Bastien",
    "Rachid",
    "Vincent",
    "Jérôme",
    "Arnaud",
    "Cédric",
    "Florian",
    "Guillaume",
    "Paul",
    "Sylvain",
)
LAST = (
    "Martin",
    "Bernard",
    "Petit",
    "Moreau",
    "Lefort",
    "Haddad",
    "Garnier",
    "Faure",
    "Rousseau",
    "Blanc",
    "Guerin",
    "Muller",
    "Henry",
    "Roussel",
    "Nicolas",
    "Perrin",
    "Morin",
    "Mathieu",
    "Clement",
    "Gauthier",
    "Dumont",
    "Lopez",
    "Fontaine",
    "Chevalier",
    "Robin",
    "Masson",
    "Sanchez",
    "Diallo",
    "Benali",
    "Nguyen",
    "Lemoine",
    "Marchand",
    "Duval",
    "Denis",
    "Brunet",
    "Schmitt",
    "Colin",
    "Vidal",
    "Caron",
    "Picard",
    "Roger",
    "Lucas",
    "Barbier",
    "Leroy",
    "Arnaud",
    "Giraud",
    "Mercier",
    "Boyer",
    "Bonnet",
    "Dupont",
    "Lambert",
    "Fournier",
    "Girard",
)


def slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", ".", ascii_text.lower()).strip(".")


@dataclass(frozen=True)
class Person:
    name: str
    email: str
    kind: str  # manager, hierarchie, pair, equipe, externe_client, externe_fournisseur, automate
    role: str  # fonction lisible (« Responsable maintenance », « Client, Solvia »)
    female: bool = True

    @property
    def first(self) -> str:
        return self.name.split()[0]

    @property
    def last(self) -> str:
        return self.name.split()[-1]


@dataclass(frozen=True)
class Persona:
    key: str  # « logistique-2 »
    metier: Metier
    manager: Person
    boss: Person
    peers: tuple[Person, ...]
    team: tuple[Person, ...]
    clients: tuple[Person, ...]
    suppliers: tuple[Person, ...]
    automates: tuple[Person, ...]
    diligence: float  # probabilité de traiter une demande dans la journée

    @property
    def owner(self) -> str:
        return f"u_train_{self.key.replace('-', '_')}"

    @property
    def split(self) -> str:
        return "tgen" if self.metier.reserved else "train"

    @property
    def domain(self) -> str:
        return self.metier.domain


def _automates(domain: str) -> tuple[Person, ...]:
    return (
        Person("Microsoft Teams", f"noreply@teams.{domain}", "automate", "notification Teams"),
        Person("Portail RH", f"portail-rh@{domain}", "automate", "portail RH"),
        Person("Support informatique", f"support@{domain}", "automate", "outil de tickets"),
        Person(
            "Communication interne", f"communication@{domain}", "automate", "liste de diffusion"
        ),
        Person("Supervision", f"alertes@{domain}", "automate", "supervision"),
    )


def build_persona(metier: Metier, index: int, seed: int = 0) -> Persona:
    """Une persona déterministe : mêmes métier, index et graine, mêmes personnes."""
    rng = random.Random(f"trainset:persona:{metier.key}:{index}:{seed}")
    used: set[str] = set()

    def person(kind: str, role: str, domain: str) -> Person:
        while True:
            female = rng.random() < 0.5
            first = rng.choice(FEMALE if female else MALE)
            last = rng.choice(LAST)
            if f"{first} {last}" not in used:
                used.add(f"{first} {last}")
                return Person(
                    f"{first} {last}", f"{slug(first)}.{slug(last)}@{domain}", kind, role, female
                )

    d = metier.domain
    manager = person("manager", metier.title, d)
    boss = person("hierarchie", metier.boss_title, d)
    peers = tuple(person("pair", t, d) for t in metier.peer_titles)
    team = tuple(person("equipe", t, d) for t in metier.team_titles)
    clients = tuple(
        person("externe_client", f"{metier.client_label}, {org}", f"{slug(org)}.test")
        for org in metier.clients
    )
    suppliers = tuple(
        person("externe_fournisseur", f"Fournisseur, {org}", f"{slug(org)}.test")
        for org in metier.suppliers
    )
    return Persona(
        key=f"{metier.key}-{index}",
        metier=metier,
        manager=manager,
        boss=boss,
        peers=peers,
        team=team,
        clients=clients,
        suppliers=suppliers,
        automates=_automates(d),
        diligence=round(rng.uniform(0.45, 0.85), 2),
    )


def all_personas(seed: int = 0) -> list[Persona]:
    out = []
    for metier in METIERS:
        n = PERSONAS_PER_RESERVED_METIER if metier.reserved else PERSONAS_PER_TRAIN_METIER
        out.extend(build_persona(metier, i, seed) for i in range(1, n + 1))
    return out


def working_days(start: date = START, count: int = N_WORKING_DAYS) -> list[date]:
    out, day = [], start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def at(day: date, hour: float) -> datetime:
    """Heure de Paris, à la minute."""
    minutes = round(hour * 60)
    return datetime.combine(day, time(minutes // 60, minutes % 60), tzinfo=PARIS)


def brief_moment(day: date) -> datetime:
    """Le point de vue du guide : le manager lit son brief à 8 h 30."""
    return at(day, 8.5)


MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)
WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


def fr_day(day: date) -> str:
    return f"{WEEKDAYS[day.weekday()]} {day.day} {MONTHS[day.month - 1]}"
