"""Regroupement des Item en affaires (L2.1, niveaux 1 et 2).

Deux items appartiennent à la même affaire s'ils partagent un lien :
- niveau 1, même fil : même `thread_id` ;
- niveau 1, même réunion : même `correlation_id` (contrat A 1.1), que le lot 1 donne à
  l'invitation reçue par mail et à l'événement du calendrier (L1.6) ;
- niveau 2, même référence dans le titre : par exemple « dossier 2026-3025 ».
Les liens sont transitifs (union-find) : un mail du fil T qui cite le dossier X réunit
le fil T et tous les items du dossier X.

Seul le titre est lu, jamais l'extrait : un extrait cite souvent d'autres dossiers
(« voir aussi… ») et fusionnerait des affaires sans rapport.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from contracts.models import Item

# Type de référence selon le mot qui précède le numéro. Dans les données de démo, les
# missions, rapports et devis portent le numéro de leur dossier ; une facture ou une
# commande a sa propre numérotation. Un mot absent de la table ne crée aucun lien.
REFERENCE_KINDS = {
    "dossier": "dossier",
    "mission": "dossier",
    "rapport": "dossier",
    "devis": "dossier",
    "facture": "facture",
    "commande": "commande",
}

_REFERENCE = re.compile(
    r"(?P<word>[^\W\d_]+)(?:\s+préliminaire)?\s+(?:n°\s*)?(?P<number>\d{4}-\d{3,4})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Case:
    """Une affaire : des items qui parlent du même sujet, et les liens qui les réunissent."""

    case_id: str
    items: tuple[Item, ...]
    links: tuple[str, ...]

    @property
    def item_ids(self) -> list[str]:
        return [item.item_id for item in self.items]


def references(title: str) -> set[str]:
    """Les références d'un titre, normalisées : « Rapport 2026-2367 » → {"dossier:2026-2367"}."""
    found = set()
    for match in _REFERENCE.finditer(title):
        kind = REFERENCE_KINDS.get(match["word"].lower())
        if kind:
            found.add(f"{kind}:{match['number']}")
    return found


def group_into_cases(items: Sequence[Item]) -> list[Case]:
    """Partitionne `items` en affaires, chaque item dans exactement une affaire.

    Deux utilisateurs ne partagent jamais une affaire, même avec un fil ou un dossier commun.
    Résultat trié par `case_id` ; dans une affaire, items par date de création.
    """
    ordered = sorted(items, key=lambda item: item.item_id)
    parent = list(range(len(ordered)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first_with_link: dict[tuple[str, str], int] = {}
    shared: Counter[tuple[str, str]] = Counter()
    for index, item in enumerate(ordered):
        for link in _links(item):
            key = (item.owner_user_id, link)
            shared[key] += 1
            if key in first_with_link:
                parent[root(index)] = root(first_with_link[key])
            else:
                first_with_link[key] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(ordered)):
        groups[root(index)].append(index)

    cases = [_case([ordered[i] for i in members], shared) for members in groups.values()]
    return sorted(cases, key=lambda case: case.case_id)


def _links(item: Item) -> set[str]:
    links = references(item.title)
    if item.thread_id:
        links.add(f"fil:{item.thread_id}")
    if item.correlation_id:
        links.add(f"réunion:{item.correlation_id}")
    return links


def _case(members: list[Item], shared: Counter[tuple[str, str]]) -> Case:
    # Un lien n'explique l'affaire que s'il relie au moins deux items.
    links = {
        link for item in members for link in _links(item) if shared[(item.owner_user_id, link)] > 1
    }
    return Case(
        case_id=f"case:{min(item.item_id for item in members)}",
        items=tuple(sorted(members, key=lambda item: (item.created_at, item.item_id))),
        links=tuple(sorted(links)),
    )
