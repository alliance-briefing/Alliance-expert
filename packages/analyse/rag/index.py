"""Index RAG par utilisateur (L2.6, étape R1) : retrouver les items passés proches.

Trois garanties, testées :
- isolation : un index n'appartient qu'à un utilisateur et refuse les items d'un autre ;
  `IndexStore` en garde un par utilisateur (guide L2.6 : « une requête de A n'est jamais
  servie avec un extrait de B ») ;
- pas de fuite du futur : une recherche pour le brief du jour J ne renvoie que des items
  créés avant J (en heure de Paris), dans les `HISTORY_DAYS` jours qui précèdent ;
- suppression complète : `IndexStore.delete_user` efface l'index et les vecteurs en cache
  de l'utilisateur (guide L1.7).

Stockage en mémoire (NumPy) : quelques milliers d'items par utilisateur suffisent. Le choix
d'un stockage persistant (sqlite-vec, Qdrant, pgvector) sera tranché dans l'ADR RAG.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from contracts.models import Item
from packages.analyse.clock import to_paris
from packages.analyse.rag.embeddings import CacheKey, Embedder, embed_items

HISTORY_DAYS = 30  # profondeur d'historique décidée par le lot 2 (21/10)
TOP_K = 3  # items passés retenus par requête


@dataclass(frozen=True)
class Hit:
    """Un item passé retrouvé, et sa similarité cosinus avec la requête."""

    item: Item
    score: float


class UserIndex:
    """Les vecteurs des items d'un seul utilisateur, en mémoire."""

    def __init__(
        self,
        owner: str,
        embedder: Embedder,
        cache: MutableMapping[CacheKey, np.ndarray] | None = None,
    ) -> None:
        self.owner = owner
        self.embedder = embedder
        self._cache: MutableMapping[CacheKey, np.ndarray] = {} if cache is None else cache
        self._items: dict[str, Item] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._items)

    def add(self, items: Iterable[Item]) -> None:
        """Ajoute des items ; un item déjà présent est remplacé par sa version la plus récente.

        Un item d'un autre utilisateur est une erreur de programmation : on refuse tout
        l'ajout plutôt que de mélanger deux utilisateurs.
        """
        items = list(items)
        if any(item.owner_user_id != self.owner for item in items):
            raise ValueError(f"index de {self.owner} : items d'autres utilisateurs refusés")
        latest: dict[str, Item] = {}
        for item in items:
            known = latest.get(item.item_id) or self._items.get(item.item_id)
            if known is None or item.updated_at >= known.updated_at:
                latest[item.item_id] = item
        vectors = embed_items(list(latest.values()), self.embedder, self._cache)
        for item_id, item in latest.items():
            self._items[item_id] = item
            self._vectors[item_id] = vectors[item_id]

    def search(
        self,
        queries: np.ndarray,
        before: date,
        *,
        k: int = TOP_K,
        days: int = HISTORY_DAYS,
        exclude: Iterable[str] = (),
    ) -> list[list[Hit]]:
        """Pour chaque vecteur de requête, les `k` items passés les plus proches.

        Seuls comptent les items créés entre `before - days` et la veille de `before`
        (dates lues en heure de Paris), hors `exclude` (ex. les items du jour).
        Tri par similarité décroissante, puis par item_id : résultat déterministe.
        """
        queries = np.atleast_2d(queries)
        skip = set(exclude)
        first_day = before - timedelta(days=days)
        candidates = sorted(
            item_id
            for item_id, item in self._items.items()
            if item_id not in skip and first_day <= to_paris(item.created_at).date() < before
        )
        if not candidates or k <= 0:
            return [[] for _ in queries]
        matrix = np.stack([self._vectors[item_id] for item_id in candidates])
        scores = queries @ matrix.T
        results = []
        for row in scores:
            order = np.lexsort((np.arange(len(candidates)), -row))[:k]
            results.append([Hit(self._items[candidates[n]], float(row[n])) for n in order])
        return results

    def search_items(
        self,
        items: Sequence[Item],
        before: date,
        *,
        k: int = TOP_K,
        days: int = HISTORY_DAYS,
        exclude: Iterable[str] = (),
    ) -> list[list[Hit]]:
        """Comme `search`, en calculant les requêtes avec le même modèle que l'index."""
        if not items:
            return []
        if any(item.owner_user_id != self.owner for item in items):
            raise ValueError(f"index de {self.owner} : requête d'un autre utilisateur refusée")
        vectors = embed_items(items, self.embedder, self._cache)
        queries = np.stack([vectors[item.item_id] for item in items])
        return self.search(queries, before, k=k, days=days, exclude=exclude)

    def clear(self) -> None:
        """Efface les items, les vecteurs et les entrées du cache de cet utilisateur."""
        self._items.clear()
        self._vectors.clear()
        for key in [key for key in self._cache if key[1] == self.owner]:
            del self._cache[key]


class IndexStore:
    """Un index par utilisateur, qui partagent le même modèle et le même cache."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._cache: dict[CacheKey, np.ndarray] = {}
        self._indexes: dict[str, UserIndex] = {}

    def for_user(self, owner: str) -> UserIndex:
        if owner not in self._indexes:
            self._indexes[owner] = UserIndex(owner, self.embedder, self._cache)
        return self._indexes[owner]

    def delete_user(self, owner: str) -> None:
        """Suppression complète des données d'un utilisateur (guide L1.7)."""
        index = self._indexes.pop(owner, None)
        if index is not None:
            index.clear()
        for key in [key for key in self._cache if key[1] == owner]:
            del self._cache[key]

    def users(self) -> list[str]:
        return sorted(self._indexes)
