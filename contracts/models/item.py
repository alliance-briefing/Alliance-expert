"""Contrat A — Item : sortie du lot 1, entrée du lot 2 (guide §2.2).

Source de vérité exécutable du contrat. Toute modification suit §2.5 du guide :
ajout d'un champ optionnel = version mineure, tout le reste = version majeure,
approbation d'un référent de chacun des 3 lots.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.1"
SUPPORTED_SCHEMA_VERSIONS = ("1.0", "1.1")
SNIPPET_MAX_CHARS = 500

Source = Literal["mail", "calendar", "task", "file"]
Provider = Literal["microsoft", "google", "demo"]
ParticipantRole = Literal["from", "to", "cc", "organizer", "attendee", "assignee", "modifier"]
Importance = Literal["low", "normal", "high"]
CalendarResponse = Literal["none", "organizer", "accepted", "tentative", "declined"]

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SEMVER = r"^\d+\.\d+\.\d+$"
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class _Strict(BaseModel):
    """Base commune : champ inconnu refusé, objet immuable après création."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Participant(_Strict):
    name: str
    email: str
    role: ParticipantRole

    @field_validator("email")
    @classmethod
    def _email_format(cls, value: str) -> str:
        if not _EMAIL.match(value):
            raise ValueError(f"adresse e-mail invalide : {value!r}")
        return value


class Acl(_Strict):
    visible_to: list[str] = Field(min_length=1)


class ConnectorInfo(_Strict):
    name: str = Field(min_length=1)
    version: str = Field(pattern=_SEMVER)


class CalendarAttributes(_Strict):
    """Détails d'un événement (contrat 1.1, ADR-0002). Toutes les clés sont fixées."""

    location: str | None = None
    online_url: HttpUrl | None = None
    is_all_day: bool = False
    is_cancelled: bool = False
    response: CalendarResponse | None = None
    series_id: str | None = None

    @field_validator("online_url")
    @classmethod
    def _https_only(cls, value: HttpUrl | None) -> HttpUrl | None:
        if value is not None and value.scheme != "https":
            raise ValueError("le lien de visio doit être en https")
        return value


class Attributes(_Strict):
    """Un bloc par source ; seul le bloc de la source de l'Item est autorisé (1.1 : calendar)."""

    calendar: CalendarAttributes | None = None


class Item(_Strict):
    """Un élément collecté (mail, événement, tâche, fichier) sous forme normalisée."""

    schema_version: Literal["1.0", "1.1"] = SCHEMA_VERSION
    item_id: str = Field(
        min_length=3, description="source + identifiant natif ; jamais un hash de contenu"
    )
    source: Source
    provider: Provider
    owner_user_id: str = Field(min_length=1)
    title: str = Field(description="Peut être vide (mail sans objet)")
    snippet: str = Field(
        max_length=SNIPPET_MAX_CHARS, description="Texte brut, sans balisage, sans signature"
    )
    content_text: str | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    start_at: AwareDatetime | None = None
    end_at: AwareDatetime | None = None
    due_at: AwareDatetime | None = None
    completed: bool | None = None
    participants: list[Participant] = Field(default_factory=list)
    thread_id: str | None = None
    correlation_id: str | None = Field(
        default=None,
        min_length=1,
        description="1.1 : identique sur les Item d'une même réunion (invitation + événement)",
    )
    native_importance: Importance | None = None
    is_read: bool | None = None
    url: HttpUrl
    acl: Acl
    collected_at: AwareDatetime
    connector: ConnectorInfo
    attributes: Attributes | None = Field(
        default=None, description="1.1 : détails propres à la source (ADR-0002)"
    )

    @field_validator("snippet", "title")
    @classmethod
    def _no_control_chars(cls, value: str) -> str:
        if _CONTROL_CHARS.search(value):
            raise ValueError("caractères de contrôle interdits")
        return value

    @field_validator("url")
    @classmethod
    def _https_only(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https":
            raise ValueError("le lien profond doit être en https")
        return value

    @model_validator(mode="after")
    def _coherence(self) -> Item:
        if not self.item_id.startswith(f"{self.source}:"):
            raise ValueError(f"item_id doit commencer par '{self.source}:'")
        if self.owner_user_id not in self.acl.visible_to:
            raise ValueError("le propriétaire doit figurer dans acl.visible_to")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at est antérieur à created_at")

        if self.source == "calendar":
            if self.start_at is None or self.end_at is None:
                raise ValueError("un événement exige start_at et end_at")
            if self.end_at < self.start_at:
                raise ValueError("end_at est antérieur à start_at")
        elif self.start_at is not None or self.end_at is not None:
            raise ValueError("start_at/end_at sont réservés au calendrier")

        if self.source == "task":
            if self.completed is None:
                raise ValueError("une tâche exige le champ completed")
        elif self.due_at is not None or self.completed is not None:
            raise ValueError("due_at/completed sont réservés aux tâches")

        if self.source != "mail" and self.is_read is not None:
            raise ValueError("is_read est réservé aux e-mails")

        if self.schema_version == "1.0" and (
            self.correlation_id is not None or self.attributes is not None
        ):
            raise ValueError("correlation_id et attributes exigent schema_version 1.1")
        if self.attributes is not None:
            blocks = {
                name for name, value in self.attributes.model_dump().items() if value is not None
            }
            if not blocks:
                raise ValueError("attributes vide : mettre null à la place")
            if blocks != {self.source}:
                raise ValueError(f"attributes ne peut contenir que le bloc « {self.source} »")
        return self

    def is_overdue(self, now: AwareDatetime) -> bool:
        """Règle du guide L1.4 : en retard = due_at < maintenant ET non terminée.

        Une tâche sans échéance n'est jamais en retard.
        """
        return (
            self.source == "task"
            and self.due_at is not None
            and self.due_at < now
            and self.completed is False
        )
