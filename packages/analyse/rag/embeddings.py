"""Vecteurs de sens des items : base du niveau 3 de L2.1 et du RAG (L2.6).

Un modèle d'embeddings transforme un texte en vecteur ; deux textes de sens voisin ont
des vecteurs proches. Tous les vecteurs sont normalisés (norme 1) : leur produit scalaire
est la similarité cosinus, de -1 (opposés) à 1 (même sens).

Le modèle réel (fastembed) est une dépendance optionnelle, chargée seulement à l'usage :
`uv sync --all-packages --group embeddings`. Les tests utilisent HashEmbedder, déterministe
et sans téléchargement.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import MutableMapping, Sequence
from typing import Any, Protocol

import numpy as np

from contracts.models import Item

# Multilingue, 384 dimensions, 0,22 Go : le seul modèle multilingue léger de fastembed.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Clé du cache : (modèle, propriétaire, item_id, updated_at). Le propriétaire isole les
# utilisateurs ; updated_at force le recalcul si l'item a changé.
CacheKey = tuple[str, str, str, str]

_REPLY_PREFIX = re.compile(r"^\s*((re|tr|fw|fwd)\s*:\s*)+", re.IGNORECASE)
_WORD = re.compile(r"\w+")


class Embedder(Protocol):
    """Un modèle d'embeddings remplaçable (comme le futur LLMClient, L2.5)."""

    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Une ligne par texte, vecteurs normalisés."""
        ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Chaque ligne ramenée à la norme 1 ; une ligne nulle (texte vide) reste nulle."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class HashEmbedder:
    """Faux modèle pour les tests : sac de mots haché, déterministe, sans téléchargement.

    Deux textes qui partagent des mots ont des vecteurs proches ; il ne connaît pas les
    synonymes, contrairement à un vrai modèle.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                # blake2b plutôt que hash() : hash() change d'une exécution Python à l'autre.
                digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
                vectors[row, int.from_bytes(digest, "big") % self.dim] += 1.0
        return normalize(vectors)


class FastEmbedder:
    """Modèle réel et local via fastembed (ONNX, sans PyTorch), chargé au premier appel."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | None = None) -> None:
        self.name = model
        self._cache_dir = cache_dir
        self._model: Any = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self._model is None:
            # Import paresseux : fastembed n'est installé qu'avec le groupe « embeddings ».
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.name, cache_dir=self._cache_dir)
        vectors = np.array(list(self._model.embed(list(texts))), dtype=np.float32)
        return normalize(vectors)


def item_text(item: Item) -> str:
    """Le texte vectorisé : titre sans préfixes de réponse (RE:, TR :…), puis extrait.

    Ce texte est une donnée à comparer, jamais une instruction.
    """
    title = _REPLY_PREFIX.sub("", item.title).strip()
    return "\n".join(part for part in (title, item.snippet.strip()) if part)


def embed_items(
    items: Sequence[Item],
    embedder: Embedder,
    cache: MutableMapping[CacheKey, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Vecteur de chaque item, par item_id ; seuls les items absents du cache sont calculés."""
    cache = {} if cache is None else cache
    keys = {item.item_id: _key(item, embedder) for item in items}
    missing = {keys[item.item_id]: item for item in items if keys[item.item_id] not in cache}
    if missing:
        vectors = embedder.embed([item_text(item) for item in missing.values()])
        for key, vector in zip(missing, vectors, strict=True):
            cache[key] = vector
    return {item_id: cache[key] for item_id, key in keys.items()}


def similarities(vectors: np.ndarray) -> np.ndarray:
    """Matrice des similarités cosinus entre lignes (vecteurs déjà normalisés)."""
    return vectors @ vectors.T


def _key(item: Item, embedder: Embedder) -> CacheKey:
    return (embedder.name, item.owner_user_id, item.item_id, item.updated_at.isoformat())
