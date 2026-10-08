"""Accès aux Item (contrat A) pour l'API, derrière une interface remplaçable.

`DemoItemStore` lit les journées factices du lot 1 via `contracts.demo_data`. Il ne lit
jamais `truth.json` : la vérité terrain est réservée à l'évaluation (L3.5).
Isolation : les items d'un jour ne sont lus que dans le dossier de l'utilisateur, puis
filtrés sur `owner_user_id` (double protection si un fichier était mal rangé).
"""

from __future__ import annotations

from datetime import date as Date
from functools import cached_property
from pathlib import Path
from typing import Protocol

from contracts.demo_data import DEMO_DIR, available_days, load_items
from contracts.models import Item


class ItemStore(Protocol):
    def items_for_day(self, user: str, day: Date) -> list[Item] | None:
        """Items de `user` collectés le matin de `day` ; None si aucune collecte ce jour-là."""
        ...

    def find(self, item_id: str) -> Item | None:
        """L'item, quel que soit son propriétaire (l'appelant vérifie les droits)."""
        ...


class DemoItemStore:
    def __init__(self, root: Path = DEMO_DIR) -> None:
        self.root = Path(root)

    def items_for_day(self, user: str, day: Date) -> list[Item] | None:
        if day.isoformat() not in available_days(user, self.root):
            return None
        items = load_items(day.isoformat(), user, self.root)
        return [item for item in items if item.owner_user_id == user]

    def find(self, item_id: str) -> Item | None:
        return self._index.get(item_id)

    @cached_property
    def _index(self) -> dict[str, Item]:
        """Tous les items de démo par identifiant ; un item vu plusieurs jours garde sa
        collecte la plus récente (jours parcourus dans l'ordre)."""
        index: dict[str, Item] = {}
        users = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        for user in users:
            for day in available_days(user, self.root):
                for item in load_items(day, user, self.root):
                    index[item.item_id] = item
        return index
