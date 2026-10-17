"""Mesure du regroupement par affaires (L2.1) contre les affaires annotées (`cases`).

Métrique principale, par paires : deux items de la même affaire forment une paire.
Précision = paires prédites qui sont aussi annotées ; rappel = paires annotées retrouvées.
Critère L2.1 : précision ≥ 0,9 et rappel ≥ 0,8, en cumulant les paires de toutes les
journées. Le B-cubed (par item) est donné à titre indicatif : une grosse affaire y pèse
moins lourd. Ne lit jamais truth.json : seuls les items et les annotations servent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

from contracts.models import Item
from eval.annotation import Annotation
from packages.analyse.aggregate.cases import group_into_cases

THRESHOLDS = {"precision": 0.9, "recall": 0.8}

Pair = tuple[str, str]


def pairs(groups: Iterable[Iterable[str]]) -> set[Pair]:
    """Toutes les paires d'items réunis dans un même groupe (ordre sans importance)."""
    return {pair for group in groups for pair in combinations(sorted(set(group)), 2)}


@dataclass(frozen=True)
class PairScore:
    predicted: int
    expected: int
    common: int

    @classmethod
    def of(cls, predicted: set[Pair], expected: set[Pair]) -> PairScore:
        return cls(len(predicted), len(expected), len(predicted & expected))

    def __add__(self, other: PairScore) -> PairScore:
        return PairScore(
            self.predicted + other.predicted,
            self.expected + other.expected,
            self.common + other.common,
        )

    @property
    def precision(self) -> float | None:
        """None si aucune paire prédite : la précision n'est pas définie."""
        return self.common / self.predicted if self.predicted else None

    @property
    def recall(self) -> float | None:
        return self.common / self.expected if self.expected else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0

    def passes(self) -> bool:
        p, r = self.precision, self.recall
        return (
            p is not None
            and r is not None
            and p >= THRESHOLDS["precision"]
            and r >= THRESHOLDS["recall"]
        )


def predicted_cases(items: Sequence[Item]) -> list[list[str]]:
    """Les affaires du lot 2 pour la journée (lecture seule de `group_into_cases`)."""
    return [case.item_ids for case in group_into_cases(items)]


def _annotated(annotation: Annotation) -> list[list[str]]:
    if annotation.cases is None:
        raise ValueError(f"{annotation.date} ({annotation.annotator}) : affaires non annotées")
    return annotation.cases


def score_day(annotation: Annotation, items: Sequence[Item]) -> PairScore:
    return PairScore.of(pairs(predicted_cases(items)), pairs(_annotated(annotation)))


def b_cubed(
    predicted: list[list[str]], expected: list[list[str]], ids: Iterable[str]
) -> tuple[float, float]:
    """(précision, rappel) B-cubed sur `ids`, chaque item absent d'un groupe étant seul."""
    ids = list(ids)
    if not ids:
        raise ValueError("journée sans items")

    def cluster_of(groups: list[list[str]]) -> dict[str, set[str]]:
        found = {item_id: set(group) for group in groups for item_id in group}
        return {item_id: found.get(item_id, {item_id}) for item_id in ids}

    pred, ref = cluster_of(predicted), cluster_of(expected)
    precision = sum(len(pred[i] & ref[i]) / len(pred[i]) for i in ids) / len(ids)
    recall = sum(len(pred[i] & ref[i]) / len(ref[i]) for i in ids) / len(ids)
    return precision, recall


def agreement(a: Annotation, b: Annotation) -> float | None:
    """Accord entre deux annotateurs sur les affaires : F1 par paires (symétrique)."""
    return PairScore.of(pairs(_annotated(a)), pairs(_annotated(b))).f1
