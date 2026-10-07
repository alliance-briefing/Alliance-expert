"""Contrat B — Brief : sortie du lot 2, entrée du lot 3 (guide §2.3).

Source de vérité exécutable du contrat. Toute modification suit §2.5 du guide :
ajout d'un champ optionnel = version mineure, tout le reste = version majeure,
approbation d'un référent de chacun des 3 lots.

Règle centrale (livrable 12, L3.3) : aucun texte du brief sans source. Chaque entrée
et chaque claim cite au moins un item, et les claims ne citent que des items de leur entrée.
"""

from __future__ import annotations

import re
from datetime import date as Date
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator

BRIEF_SCHEMA_VERSION = "1.0"
SECTION_ORDER = ("actions", "meetings", "overdue", "files")

SectionKind = Literal["actions", "meetings", "overdue", "files"]

_ITEM_ID = re.compile(r"^(mail|calendar|task|file):\S+$")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class _Strict(BaseModel):
    """Base commune : champ inconnu refusé, objet immuable après création."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_item_ids(values: list[str]) -> list[str]:
    for value in values:
        if not _ITEM_ID.match(value):
            raise ValueError(f"identifiant d'item invalide : {value!r}")
    if len(set(values)) != len(values):
        raise ValueError("identifiant d'item en double")
    return values


def _check_text(value: str) -> str:
    if not value.strip():
        raise ValueError("texte vide")
    if _CONTROL_CHARS.search(value):
        raise ValueError("caractères de contrôle interdits")
    return value


Text = Annotated[str, AfterValidator(_check_text)]
SourceItemIds = Annotated[list[str], Field(min_length=1), AfterValidator(_check_item_ids)]


class Generator(_Strict):
    """Ce qui a produit le brief. Un brief à base de règles (v0.1.0) met llm = "rules"."""

    llm: str = Field(min_length=1)
    llm_version: str | None = None
    prompt_version: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)


class Claim(_Strict):
    """Une affirmation factuelle de l'entrée, rattachée aux items qui la prouvent."""

    text: Text
    source_item_ids: SourceItemIds


class Entry(_Strict):
    rank: int = Field(ge=1, description="1 = le plus prioritaire de la section")
    priority_score: int = Field(ge=0, le=100)
    priority_reasons: list[Text] = Field(min_length=1)
    text: Text
    claims: list[Claim] = Field(default_factory=list)
    recommended_action: Text | None = None
    source_item_ids: SourceItemIds

    @model_validator(mode="after")
    def _claims_cite_entry_sources(self) -> Entry:
        allowed = set(self.source_item_ids)
        for claim in self.claims:
            unknown = set(claim.source_item_ids) - allowed
            if unknown:
                raise ValueError(f"claim citant un item absent de l'entrée : {sorted(unknown)}")
        return self


class Section(_Strict):
    kind: SectionKind
    entries: list[Entry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ranks(self) -> Section:
        ranks = [entry.rank for entry in self.entries]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"rangs attendus 1..n dans l'ordre, reçus {ranks}")
        scores = [entry.priority_score for entry in self.entries]
        if scores != sorted(scores, reverse=True):
            raise ValueError("priority_score doit décroître avec le rang")
        return self


class Stats(_Strict):
    items_in: int = Field(ge=0)
    items_out: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    cost_eur: float = Field(ge=0)


class Brief(_Strict):
    """Le brief du jour d'un utilisateur, prêt à afficher."""

    schema_version: Literal["1.0"] = BRIEF_SCHEMA_VERSION
    brief_id: str = Field(description="b_<date>_<owner_user_id>")
    owner_user_id: str = Field(min_length=1)
    date: Date
    generated_at: AwareDatetime
    generator: Generator
    sections: list[Section]
    stats: Stats
    warnings: list[Text] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coherence(self) -> Brief:
        expected_id = f"b_{self.date.isoformat()}_{self.owner_user_id}"
        if self.brief_id != expected_id:
            raise ValueError(f"brief_id attendu : {expected_id!r}")
        kinds = [section.kind for section in self.sections]
        if len(set(kinds)) != len(kinds):
            raise ValueError("une section apparaît deux fois")
        if kinds != sorted(kinds, key=SECTION_ORDER.index):
            raise ValueError(f"ordre des sections attendu : {', '.join(SECTION_ORDER)}")
        if self.stats.items_out > self.stats.items_in:
            raise ValueError("stats.items_out dépasse stats.items_in")
        return self

    def cited_item_ids(self) -> set[str]:
        """Tous les items cités par le brief (pour vérifier qu'ils existent et sont visibles)."""
        return {
            item_id
            for section in self.sections
            for entry in section.entries
            for item_id in entry.source_item_ids
        }
