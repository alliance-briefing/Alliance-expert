"""Objets du monde simulé, AVANT normalisation (équivalent des données brutes d'un fournisseur).

Chaque objet porte, en plus des données « métier », la vérité connue par construction :
l'affaire à laquelle il appartient (case_id), son importance intrinsèque (salience) et
les pièges qu'il contient (tags). Ces informations ne sortent JAMAIS dans les Item :
elles alimentent uniquement truth.json, utilisé pour l'évaluation (lot 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from collect.synth.people import Person


@dataclass
class SimMail:
    native_id: str
    thread_id: str
    subject: str
    html: str
    sender: Person
    to: list[Person]
    cc: list[Person]
    sent_at: datetime
    importance: str | None = "normal"
    case_id: str | None = None
    salience: int = 10
    requires_answer: bool = False
    answered_at: datetime | None = None
    read_at: datetime | None = None
    hint: str = ""  # formulation attendue dans le brief (vérité terrain)
    expires_at: datetime | None = None  # au-delà, le mail n'appelle plus d'action (réunion passée…)
    tags: set[str] = field(default_factory=set)
    filler: bool = False  # ajouté par la réparation des volumes


@dataclass
class SimEvent:
    native_id: str
    title: str
    start: datetime
    end: datetime
    organizer: Person
    attendees: list[Person]
    description: str = ""
    location: str = ""
    all_day: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    series_id: str | None = None
    case_id: str | None = None
    salience: int = 10
    hint: str = ""
    tags: set[str] = field(default_factory=set)
    filler: bool = False


@dataclass
class SimTask:
    native_id: str
    title: str
    notes: str
    created_at: datetime
    due_at: datetime | None
    assignee: Person
    creator: Person
    completed_at: datetime | None = None
    case_id: str | None = None
    salience: int = 20
    hint: str = ""
    tags: set[str] = field(default_factory=set)
    filler: bool = False


@dataclass
class SimFileVersion:
    modified_at: datetime
    modifier: Person


@dataclass
class SimFile:
    native_id: str
    name: str
    folder: str
    created_at: datetime
    versions: list[SimFileVersion]
    excerpt: str = ""  # uniquement pour texte, docx, pdf (guide L1.5)
    deleted_at: datetime | None = None
    case_id: str | None = None
    salience: int = 10
    hint: str = ""
    tags: set[str] = field(default_factory=set)
    filler: bool = False

    @property
    def extension(self) -> str:
        return self.name.rsplit(".", 1)[-1].lower() if "." in self.name else ""


@dataclass
class Case:
    """Une « affaire » : un sujet qui traverse plusieurs sources et plusieurs jours."""

    case_id: str
    kind: str
    label: str
    opened_on: datetime
    notes: dict[str, str] = field(default_factory=dict)


@dataclass
class TruthNote:
    """Fait particulier connu par construction (contradiction, conflit, corrélation…)."""

    kind: str
    day: str  # date d'export concernée (AAAA-MM-JJ) ou "*" pour toutes
    native_ids: list[str]
    detail: dict[str, object]
