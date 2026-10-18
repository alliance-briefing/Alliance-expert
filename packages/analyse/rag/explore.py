"""Exploration des embeddings sur une journée (L2.6) : ce que trouverait un niveau 3 de L2.1.

    uv run python -m packages.analyse.rag.explore 2026-10-07
    uv run python -m packages.analyse.rag.explore 2026-10-07 --texte titre --top 30
    uv run python -m packages.analyse.rag.explore 2026-10-07 --modele hash
    uv run --with model2vec python -m packages.analyse.rag.explore 2026-10-07 --modele model2vec

Affiche le temps de calcul, la distribution des similarités (paires déjà reliées par les
niveaux 1-2 contre les autres) et les paires les plus proches que les niveaux 1-2 ne
relient pas, à juger à la main. Outil d'étude : rien n'est branché dans le brief.
Ne lit jamais truth.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contracts.demo_data import DEFAULT_USER, DEMO_DIR, load_items
from contracts.models import Item
from packages.analyse.aggregate.cases import Case, group_into_cases, references
from packages.analyse.clock import to_paris
from packages.analyse.rag.embeddings import (
    DEFAULT_MODEL,
    Embedder,
    FastEmbedder,
    HashEmbedder,
    item_text,
    normalize,
    similarities,
)

MODEL2VEC_MODEL = "minishlab/potion-multilingual-128M"
PERCENTILES = (10, 50, 90, 99)
TITLE_WIDTH = 60


def title_text(item: Item) -> str:
    """Le titre seul (sans préfixes de réponse), pour comparer avec titre + extrait."""
    return item_text(item.model_copy(update={"snippet": ""}))


TEXTS: dict[str, Callable[[Item], str]] = {"complet": item_text, "titre": title_text}


class Model2VecEmbedder:
    """Vecteurs statiques (model2vec) : très petit et très rapide, à comparer en qualité.

    Non installé dans le dépôt : `uv run --with model2vec …`.
    """

    def __init__(self, model: str = MODEL2VEC_MODEL) -> None:
        self.name = model
        self._model: Any = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(self.name)
        return normalize(np.asarray(self._model.encode(list(texts)), dtype=np.float32))


def make_embedder(name: str) -> Embedder:
    """« hash » (faux modèle), « model2vec », sinon un nom de modèle fastembed."""
    if name == "hash":
        return HashEmbedder()
    if name == "model2vec":
        return Model2VecEmbedder()
    return FastEmbedder(name)


@dataclass(frozen=True)
class Pair:
    """Deux items proches que les niveaux 1-2 ne relient pas."""

    score: float
    first: Item
    second: Item
    shared_participant: bool
    days_apart: int
    # Les deux titres citent des références, toutes différentes (« dossier 2026-5394 » et
    # « dossier 2026-4752 ») : deux affaires distinctes, même si les textes se ressemblent.
    different_references: bool


def split_scores(
    vectors: np.ndarray, cases: Sequence[Case], items: Sequence[Item]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Similarités des paires déjà reliées (même affaire) et des autres, plus leurs indices."""
    case_of = {item_id: n for n, case in enumerate(cases) for item_id in case.item_ids}
    labels = np.array([case_of[item.item_id] for item in items])
    first, second = np.triu_indices(len(items), k=1)
    scores = similarities(vectors)[first, second]
    linked = labels[first] == labels[second]
    return scores[linked], scores[~linked], first[~linked], second[~linked]


def closest_unlinked(items: Sequence[Item], vectors: np.ndarray, top: int) -> list[Pair]:
    """Les `top` paires les plus proches hors des affaires des niveaux 1-2 ; ex aequo par id."""
    ordered = sorted(items, key=lambda item: item.item_id)
    order = {item.item_id: n for n, item in enumerate(items)}
    vectors = vectors[[order[item.item_id] for item in ordered]]
    _, scores, first, second = split_scores(vectors, group_into_cases(ordered), ordered)
    owner = _owner_addresses(ordered)
    best = np.lexsort((second, first, -scores))[:top]
    return [_pair(float(scores[n]), ordered[first[n]], ordered[second[n]], owner) for n in best]


def summary(scores: np.ndarray) -> str:
    if scores.size == 0:
        return "aucune paire"
    parts = [f"p{p} {np.percentile(scores, p):.2f}" for p in PERCENTILES]
    return f"n={scores.size}, min {scores.min():.2f}, {', '.join(parts)}, max {scores.max():.2f}"


def report(items: Sequence[Item], embedder: Embedder, text: str = "complet", top: int = 20) -> str:
    """Le rapport d'une journée, en texte."""
    ordered = sorted(items, key=lambda item: item.item_id)
    started = time.perf_counter()
    vectors = embedder.embed([TEXTS[text](item) for item in ordered])
    elapsed = time.perf_counter() - started
    cases = group_into_cases(ordered)
    linked, unlinked, _, _ = split_scores(vectors, cases, ordered)
    lines = [
        (
            f"Modèle : {embedder.name} · texte : {text} · {len(ordered)} items, "
            f"{len(cases)} affaires (niveaux 1-2)"
        ),
        f"Calcul des vecteurs : {elapsed:.1f} s (chargement du modèle compris)",
        "",
        "Similarités cosinus",
        f"- paires déjà reliées (même affaire) : {summary(linked)}",
        f"- autres paires : {summary(unlinked)}",
        "",
        f"{top} paires les plus proches non reliées (à juger : même affaire ?)",
    ]
    for pair in closest_unlinked(ordered, vectors, top):
        flags = [
            "participant commun" if pair.shared_participant else "aucun participant commun",
            f"{pair.days_apart} j",
        ]
        if pair.different_references:
            flags.append("RÉFÉRENCES DIFFÉRENTES")
        lines.append(
            f"  {pair.score:.2f} | {' | '.join(flags)} | "
            f"{_label(pair.first)}  <->  {_label(pair.second)}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exploration des embeddings (L2.6).")
    parser.add_argument("jour", help="AAAA-MM-JJ")
    parser.add_argument("--modele", default=DEFAULT_MODEL, help="hash, model2vec ou fastembed")
    parser.add_argument("--texte", choices=sorted(TEXTS), default="complet")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--utilisateur", default=DEFAULT_USER)
    parser.add_argument("--dossier", type=Path, default=DEMO_DIR, help="<user>/<jour>/items.jsonl")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # accents lisibles dans PowerShell
    items = load_items(args.jour, args.utilisateur, args.dossier)
    print(f"# Embeddings — {args.jour} ({args.utilisateur})")
    print(report(items, make_embedder(args.modele), args.texte, args.top))
    return 0


def _owner_addresses(items: Sequence[Item]) -> set[str]:
    """L'adresse du manager : la plus fréquente de la journée (il est dans presque tous les
    mails). L'ignorer évite que « participant commun » soit toujours vrai."""
    counts = Counter(p.email.lower() for item in items for p in item.participants)
    return {counts.most_common(1)[0][0]} if counts else set()


def _pair(score: float, first: Item, second: Item, owner: set[str]) -> Pair:
    emails = [{p.email.lower() for p in item.participants} - owner for item in (first, second)]
    days = abs((to_paris(first.created_at).date() - to_paris(second.created_at).date()).days)
    refs = [references(item.title) for item in (first, second)]
    different = bool(refs[0] and refs[1] and not refs[0] & refs[1])
    return Pair(score, first, second, bool(emails[0] & emails[1]), days, different)


def _label(item: Item) -> str:
    title = " ".join(item.title.split()) or "(sans objet)"
    if len(title) > TITLE_WIDTH:
        title = title[: TITLE_WIDTH - 1] + "…"
    return f"{item.source}: {title}"


if __name__ == "__main__":
    raise SystemExit(main())
