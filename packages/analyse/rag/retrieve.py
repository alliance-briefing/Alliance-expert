"""Historique des affaires du jour (L2.6, étape R2) : les items passés vraiment liés.

Pour chaque affaire, chacun de ses items interroge l'index de l'utilisateur ; un item
passé garde sa meilleure similarité avec un item de l'affaire (un seul item très proche
suffit, un vecteur moyen brouillerait un mail, une tâche et une réunion).

Un item passé est rattaché :
- s'il cite la même référence que l'affaire (« même dossier 2026-3025 ») : lien certain,
  accepté quel que soit le score (niveau 2 de L2.1 étendu aux jours passés) ;
- sinon si sa similarité atteint `threshold`, sauf veto : l'affaire et l'item citent des
  références toutes différentes (« Dossier 2026-5394 : RAS » ≠ « Dossier 2026-4752 : RAS »,
  pourtant presque identiques pour le modèle).
Les liens certains passent avant les liens par similarité ; puis score, puis item_id.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from contracts.models import Item
from packages.analyse.aggregate.cases import Case, references
from packages.analyse.rag.index import HISTORY_DAYS, TOP_K, UserIndex

# Provisoire : prudent (peu de faux rattachements). À régler avec `explore --historique`.
DEFAULT_THRESHOLD = 0.85


@dataclass(frozen=True)
class PastLink:
    """Un item passé rattaché à une affaire du jour, et pourquoi."""

    item: Item
    score: float
    reason: str
    same_reference: bool


def history(
    cases: Sequence[Case],
    index: UserIndex,
    day: date,
    *,
    k: int = TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    days: int = HISTORY_DAYS,
    exclude: Iterable[str] = (),
) -> dict[str, list[PastLink]]:
    """Pour chaque affaire (par case_id), au plus `k` items passés liés.

    Les items des affaires et `exclude` ne sont jamais renvoyés comme historique.
    """
    ordered = sorted(cases, key=lambda case: case.case_id)
    queries = [item for case in ordered for item in case.items]
    skip = set(exclude) | {item.item_id for item in queries}
    candidates, scores = index.score_items(queries, day, days=days, exclude=skip)
    candidate_refs = [references(item.title) for item in candidates]
    result: dict[str, list[PastLink]] = {}
    row = 0  # les lignes de `scores` suivent l'ordre des items de `ordered`
    for case in ordered:
        rows = scores[row : row + len(case.items)]
        row += len(case.items)
        best = rows.max(axis=0) if len(rows) and len(candidates) else np.zeros(len(candidates))
        case_refs = set().union(*(references(item.title) for item in case.items))
        links = [
            link
            for item, score, refs in zip(candidates, best, candidate_refs, strict=True)
            if (link := _link(item, float(score), refs, case_refs, threshold)) is not None
        ]
        links.sort(key=lambda link: (not link.same_reference, -link.score, link.item.item_id))
        result[case.case_id] = links[:k]
    return result


SAME_REFERENCE, CLOSE_SUBJECT = "même référence", "sujet proche"
VETO, BELOW_THRESHOLD = "références différentes", "sous le seuil"


def verdict(score: float, refs: set[str], case_refs: set[str], threshold: float) -> str:
    """Pourquoi un item passé est retenu (SAME_REFERENCE, CLOSE_SUBJECT) ou écarté
    (VETO, BELOW_THRESHOLD) pour une affaire. Une seule règle, partagée avec l'exploration."""
    if refs & case_refs:
        return SAME_REFERENCE
    if refs and case_refs:
        return VETO
    return CLOSE_SUBJECT if score >= threshold else BELOW_THRESHOLD


def _link(
    item: Item, score: float, refs: set[str], case_refs: set[str], threshold: float
) -> PastLink | None:
    decision = verdict(score, refs, case_refs, threshold)
    if decision == SAME_REFERENCE:
        kind, _, number = min(refs & case_refs).partition(":")
        return PastLink(item, score, f"même {kind} {number}", True)
    if decision == CLOSE_SUBJECT:
        return PastLink(item, score, f"sujet proche ({score:.2f})", False)
    return None
