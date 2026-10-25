"""Deuxième série d'étiquettes « action » et « réponse » (étape A3, 26/10/2026).

Diagnostic : 42 demandes d'action, toutes dans le style de David ; le modèle lit le sujet du
mail plutôt que l'acte. Cette série ajoute de la variété sans fausser les mesures :
- démo du lot 1 (autre métier) : un mail par modèle de phrase **absent des 40 paires de démo**,
  pour que le test de transfert reste un mail jamais vu ;
- David : les mails des cartes des paires, hors paires réservées, où le modèle hésite le plus
  (probabilité la plus proche de 0,5 pour « action » ou « réponse »).

`uv run --group embeddings python -m packages.analyse.scoring.ml.sample_round2 <data>` écrit
`scoring/annotations/intent_sample_r2.jsonl`.
"""

from __future__ import annotations

import random
import sys
from collections import Counter

from contracts.models import Item
from packages.analyse.scoring.ml.annotate import DEMO_PAIRS_PATH, INTENT_R2_PATH, PAIRS_PATH
from packages.analyse.scoring.ml.graph_convert import GraphDataset, load_graph_dataset
from packages.analyse.scoring.ml.noise import noise_reason
from packages.analyse.scoring.ml.pairs import _probs, demo_dataset
from packages.analyse.scoring.ml.persona import DEMO_CAMILLE, GRAPH_HELENE, Persona
from packages.analyse.scoring.ml.sample import (
    TEXT_MAX_CHARS,
    load_sample,
    template,
    write_sample,
)

R2_PATH = INTENT_R2_PATH
N_UNCERTAIN = 50


def _row(item: Item, persona: Persona, dataset_name: str) -> dict:
    sender = next(p for p in item.participants if p.role == "from")
    return {
        "item_id": item.item_id,
        "template": template(item.title),
        "sender_kind": persona.sender_kind(sender.email),
        "sender": sender.name or sender.email,
        "title": item.title,
        "received_at": item.created_at.isoformat(),
        "text": (item.content_text or item.snippet)[:TEXT_MAX_CHARS],
        "round": 2,
        "dataset": dataset_name,
    }


def _received(dataset: GraphDataset) -> list[Item]:
    return [
        i
        for i in dataset.items
        if i.source == "mail" and not dataset.extras[i.item_id].is_sent and not noise_reason(i)
    ]


def demo_rows(dataset: GraphDataset, pairs: list[dict], seed: int = 0) -> list[dict]:
    """Un mail par (type d'expéditeur, modèle) absent des paires de démo."""
    by_id = {i.item_id: i for i in dataset.items}

    def key(item: Item) -> tuple[str, str]:
        sender = next(p.email for p in item.participants if p.role == "from")
        return DEMO_CAMILLE.sender_kind(sender), template(item.title)

    taken = {key(by_id[s["item_id"]]) for p in pairs for s in (p["a"], p["b"])}
    pools: dict[tuple[str, str], list[Item]] = {}
    for item in sorted(_received(dataset), key=lambda i: i.item_id):
        if key(item) not in taken:
            pools.setdefault(key(item), []).append(item)
    rng = random.Random(seed)
    return [_row(rng.choice(pools[k]), DEMO_CAMILLE, "démo") for k in sorted(pools)]


def uncertain_rows(
    dataset: GraphDataset,
    pairs: list[dict],
    probs: dict[str, dict[str, float]],
    exclude: set[str],
    n: int = N_UNCERTAIN,
) -> list[dict]:
    """Les cartes des paires d'entraînement où « action » ou « réponse » hésite le plus."""
    by_id = {i.item_id: i for i in dataset.items}
    heldout = {s["item_id"] for p in pairs if p.get("heldout") for s in (p["a"], p["b"])}
    cards = sorted({s["item_id"] for p in pairs for s in (p["a"], p["b"])} - heldout - exclude)

    def doubt(item_id: str) -> float:
        return min(abs(probs[f][item_id] - 0.5) for f in ("action", "reponse"))

    rows, seen = [], set()
    for item_id in sorted(cards, key=lambda i: (doubt(i), i)):
        item = by_id[item_id]
        text = (item.title, item.content_text or item.snippet)
        if text not in seen:  # deux cartes au texte identique : une seule
            seen.add(text)
            rows.append(_row(item, GRAPH_HELENE, "david"))
        if len(rows) == n:
            break
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-2])
        return 2
    david = load_graph_dataset(argv[0])
    demo = demo_dataset()
    labelled = {row["item_id"] for row in load_sample()}
    rows = uncertain_rows(david, load_sample(PAIRS_PATH), _probs(david), labelled)
    rows += demo_rows(demo, load_sample(DEMO_PAIRS_PATH))
    write_sample(rows, R2_PATH)
    print(f"{len(rows)} mails → {R2_PATH} {dict(Counter(r['dataset'] for r in rows))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
