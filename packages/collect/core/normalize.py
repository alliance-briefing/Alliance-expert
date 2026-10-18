"""Couche de normalisation unique (guide L1.6).

Chaque objet d'un fournisseur (mail Graph, événement Graph…) passe par `normalize` :
la traduction propre à la source (écrite par B, C, D…) produit un Item ou un dict,
qui est validé par le contrat A **avant** de sortir. Un objet invalide n'est jamais
transmis au lot 2 : il devient un `Rejection`, journalisé sans aucun contenu.

Règles :
- rien n'est corrigé en silence : si la traduction donne un extrait de 501 caractères,
  l'objet est rejeté, pas tronqué ici ;
- un rejet ne garde que le type d'erreur et le nom des champs fautifs, jamais leur valeur
  (une valeur peut contenir une adresse ou un texte de mail) ;
- `deduplicate` garde un seul Item par `item_id` (ex. un événement « toute la journée »
  présent dans deux fichiers de jour), le plus récemment modifié.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from contracts.models import Item

logger = logging.getLogger("collect.normalize")

Translate = Callable[[Mapping[str, Any]], Item | Mapping[str, Any]]


@dataclass(frozen=True)
class Rejection:
    """Un objet fournisseur refusé, avec la raison (sans aucune donnée de l'objet)."""

    connector: str
    reason: str


@dataclass
class NormalizeResult:
    items: list[Item] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)


def _reason(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        fields = sorted({".".join(str(p) for p in err["loc"]) or "item" for err in exc.errors()})
        return f"contrat A refusé : {', '.join(fields)}"
    return f"traduction impossible ({type(exc).__name__})"


def normalize(raw: Mapping[str, Any], translate: Translate, *, connector: str) -> Item | Rejection:
    """Traduit puis valide un objet ; renvoie un Item conforme ou un Rejection explicite."""
    try:
        produced = translate(raw)
        if isinstance(produced, Item):
            # revalidation complète : un Item construit avec model_construct n'échappe pas au contrat
            return Item.model_validate(produced.model_dump(warnings=False))
        return Item.model_validate(produced)
    except (ValidationError, KeyError, TypeError, ValueError, AttributeError) as exc:
        rejection = Rejection(connector, _reason(exc))
        logger.warning("%s : objet rejeté — %s", connector, rejection.reason)
        return rejection


def normalize_all(
    raws: Iterable[Mapping[str, Any]], translate: Translate, *, connector: str
) -> NormalizeResult:
    result = NormalizeResult()
    for raw in raws:
        outcome = normalize(raw, translate, connector=connector)
        if isinstance(outcome, Item):
            result.items.append(outcome)
        else:
            result.rejections.append(outcome)
    return result


def deduplicate(items: Iterable[Item]) -> list[Item]:
    """Un seul Item par `item_id` : le plus récent (`updated_at`), ordre d'arrivée conservé."""
    kept: dict[str, Item] = {}
    for item in items:
        current = kept.get(item.item_id)
        if current is None or item.updated_at > current.updated_at:
            kept[item.item_id] = item
    return list(kept.values())
