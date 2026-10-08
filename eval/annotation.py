"""Annotations de référence du lot 3 (L3.5) : ce que le brief d'une journée doit contenir.

Une annotation = un fichier JSON par journée et par annotateur :
eval/annotations/<jeu>/<AAAA-MM-JJ>.<annotateur>.json. Elle est écrite à la main, à partir
des seuls Item de la journée, sans jamais lire truth.json (évaluation indépendante).
"""

from __future__ import annotations

from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.models import Item
from contracts.models.brief import SectionKind

ANNOTATION_SCHEMA = "annotation/1.0"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Expected(_Strict):
    """Un élément qui DOIT figurer dans le brief."""

    rank: int = Field(ge=1, description="1 = le plus important de la journée")
    section: SectionKind
    item_ids: list[str] = Field(min_length=1, description="items qui le prouvent")
    why: str = Field(min_length=3, description="pourquoi c'est important, en une phrase")


class Annotation(_Strict):
    schema_: Literal["annotation/1.0"] = Field(ANNOTATION_SCHEMA, alias="schema")
    user_id: str
    date: Date
    annotator: str = Field(pattern=r"^[a-z0-9_-]{2,40}$", description="pseudo, pas d'e-mail")
    annotated_at: Date
    must_include: list[Expected] = Field(min_length=3, max_length=7)
    must_exclude: list[str] = Field(default_factory=list, description="pièges à ne pas citer")
    notes: str = ""

    @model_validator(mode="after")
    def _coherence(self) -> Annotation:
        ranks = sorted(e.rank for e in self.must_include)
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"rangs attendus 1..{len(ranks)} sans trou ni doublon, reçus {ranks}")
        included = self.included_ids()
        if included & set(self.must_exclude):
            raise ValueError(
                f"items à la fois inclus et exclus : {sorted(included & set(self.must_exclude))}"
            )
        return self

    def included_ids(self) -> set[str]:
        return {item_id for e in self.must_include for item_id in e.item_ids}


def check_against_day(annotation: Annotation, items: list[Item]) -> list[str]:
    """Problèmes de l'annotation par rapport aux items réels de la journée (vide = OK)."""
    known = {item.item_id for item in items}
    problems = []
    for item_id in sorted((annotation.included_ids() | set(annotation.must_exclude)) - known):
        problems.append(f"{item_id} n'existe pas dans la journée du {annotation.date}")
    owners = {item.owner_user_id for item in items}
    if owners and owners != {annotation.user_id}:
        problems.append(f"user_id {annotation.user_id} différent du propriétaire des items")
    return problems


def cohen_kappa(a: Annotation, b: Annotation, items: list[Item]) -> float:
    """Accord entre deux annotateurs sur « cet item doit figurer dans le brief » (oui/non),
    calculé sur tous les items de la journée. 1 = accord parfait, 0 = hasard."""
    ids = [item.item_id for item in items]
    in_a, in_b = a.included_ids(), b.included_ids()
    n = len(ids)
    if n == 0:
        raise ValueError("journée sans items")
    agree = sum((i in in_a) == (i in in_b) for i in ids) / n
    pa, pb = sum(i in in_a for i in ids) / n, sum(i in in_b for i in ids) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if chance == 1 else (agree - chance) / (1 - chance)
