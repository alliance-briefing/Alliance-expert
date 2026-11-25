"""Métriques du brief contre les annotations de référence (L3.6).

Le brief se lit dans l'ordre où le manager le lit : sections dans l'ordre du contrat B,
puis rang dans la section (pas un tri par score, qui ignorerait l'ordre des sections).
Un élément de référence est retrouvé par une entrée qui cite au moins un de ses items ;
chaque élément n'est compté qu'une fois, par la première entrée qui le retrouve.

Le code est écrit par le lot 3 et n'importe rien du lot 2 : l'évaluation reste indépendante.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from contracts.models import Brief, Entry
from contracts.models.brief import SECTION_ORDER
from eval.annotation import Annotation, Expected

K_RECALL = 7
K_NDCG = 5
TOP_FAULTS = 3


def reading_order(brief: Brief) -> list[Entry]:
    """Les entrées du brief dans l'ordre de lecture : section (contrat B), puis rang."""
    sections = sorted(brief.sections, key=lambda s: SECTION_ORDER.index(s.kind))
    return [
        entry for section in sections for entry in sorted(section.entries, key=lambda e: e.rank)
    ]


def match(entries: Sequence[Entry], expected: Sequence[Expected]) -> list[Expected | None]:
    """Pour chaque entrée, l'élément de référence qu'elle retrouve (ou None).

    Une entrée qui touche plusieurs éléments est créditée du mieux classé encore libre ;
    un élément déjà retrouvé par une entrée précédente ne compte plus.
    """
    free = sorted(expected, key=lambda e: e.rank)
    found: list[Expected | None] = []
    for entry in entries:
        cited = set(entry.source_item_ids)
        hit = next((e for e in free if cited & set(e.item_ids)), None)
        if hit is not None:
            free.remove(hit)
        found.append(hit)
    return found


def _gain(element: Expected, n: int) -> int:
    """Gain gradué : le rang 1 vaut n, le dernier rang vaut 1."""
    return n - element.rank + 1


def ndcg(found: Sequence[Expected | None], n: int, k: int = K_NDCG) -> float:
    """NDCG@k de l'ordre du brief contre le classement de référence (n éléments)."""
    dcg = sum(
        _gain(hit, n) / math.log2(pos + 2) for pos, hit in enumerate(found[:k]) if hit is not None
    )
    ideal = sum((n - r) / math.log2(r + 2) for r in range(min(k, n)))
    return dcg / ideal


@dataclass(frozen=True)
class DayScore:
    """Scores d'une journée contre une annotation."""

    date: str
    annotator: str
    recall: float
    precision: float
    ndcg: float
    faults: tuple[str, ...]  # items « ne doit pas figurer » cités, dans l'ordre de lecture
    faults_top: int  # dont cités dans les TOP_FAULTS premières entrées
    missed: tuple[Expected, ...]  # éléments non retrouvés dans les K_RECALL premières entrées
    placed: tuple[tuple[int, int], ...]  # (rang de référence, position dans le brief)
    entries: int


def score_day(brief: Brief, annotation: Annotation) -> DayScore:
    entries = reading_order(brief)
    expected = annotation.must_include
    found = match(entries, expected)
    top = found[:K_RECALL]
    hits = [hit for hit in top if hit is not None]
    excluded = set(annotation.must_exclude)
    faults = [
        (pos, item_id)
        for pos, entry in enumerate(entries)
        for item_id in entry.source_item_ids
        if item_id in excluded
    ]
    return DayScore(
        date=annotation.date.isoformat(),
        annotator=annotation.annotator,
        recall=len(hits) / len(expected),
        precision=len(hits) / len(top) if top else 0.0,
        ndcg=ndcg(found, len(expected)),
        faults=tuple(item_id for _, item_id in faults),
        faults_top=sum(pos < TOP_FAULTS for pos, _ in faults),
        missed=tuple(e for e in expected if e not in hits),
        placed=tuple(sorted((hit.rank, pos + 1) for pos, hit in enumerate(found) if hit)),
        entries=len(entries),
    )


def injections_cited(brief: Brief, injection_ids: Sequence[str]) -> list[str]:
    """Items d'injection de prompt que le brief cite (une injection neutralisée n'est pas citée)."""
    cited = brief.cited_item_ids()
    return [item_id for item_id in injection_ids if item_id in cited]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(
    values: Sequence[float],
    seed: int,
    rounds: int = 2000,
    level: float = 0.95,
    statistic: Callable[[Sequence[float]], float] = mean,
) -> tuple[float, float]:
    """Intervalle de confiance par bootstrap sur les journées (percentiles, graine fixée)."""
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(rounds))
    low = stats[int((1 - level) / 2 * rounds)]
    high = stats[min(rounds - 1, int((1 + level) / 2 * rounds))]
    return low, high
