"""Paires de mails à comparer : « ce matin, lequel traiter en premier ? ».

Deux mails du même matin (candidats du brief, `features.candidates`). Moitié tirées au hasard,
moitié « serrées » (le score par règles actuel les juge à 5 points ou moins) : ce sont les
cas qui apprennent le plus. Chaque mail est tiré par type d'expéditeur d'abord (rééquilibré),
côtés A/B au hasard (pas de biais de position) ; 24 paires réservées à la mesure.

`chips` est ce que l'annotateur a vu ; `features`, l'entrée du modèle, se recalcule avec le
modèle d'intentions courant (`rafraichir`), sans toucher aux paires ni à leur affichage.

    uv run --group embeddings python -m packages.analyse.scoring.ml.pairs construire <data>
    uv run --group embeddings python -m packages.analyse.scoring.ml.pairs rafraichir <data>
    uv run --group embeddings python -m packages.analyse.scoring.ml.pairs demo

`<data>` : le dossier `data/` d'alliance-graph-synth. `demo` tire 40 paires des données de démo
du lot 1 (persona Camille Laurent), toutes réservées : un test de transfert, jamais appris.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date, timedelta

import numpy as np

from contracts.models import Item
from packages.analyse.aggregate.cases import Case
from packages.analyse.scoring.ml.features import (
    MailFeatures,
    brief_instant,
    candidates,
    mail_features,
)
from packages.analyse.scoring.ml.graph_convert import (
    GraphDataset,
    dataset_from_items,
    load_graph_dataset,
)
from packages.analyse.scoring.ml.intents import (
    LEARNED,
    IntentModel,
    VectorCache,
    cache_path,
    embedder_for,
    intent_text,
)
from packages.analyse.scoring.ml.noise import noise_reason
from packages.analyse.scoring.ml.persona import DEMO_CAMILLE, GRAPH_HELENE, Persona
from packages.analyse.scoring.ml.sample import ANNOTATIONS, load_sample, write_sample
from packages.analyse.scoring.score import ScoringConfig, score_case

PAIRS_PATH = ANNOTATIONS / "pair_sample.jsonl"
DEMO_PAIRS_PATH = ANNOTATIONS / "pair_sample_demo.jsonl"
N_PAIRS, N_HELDOUT, CLOSE_GAP = 120, 24, 5
N_DEMO_PAIRS = 40
SHOWN_CHARS = 600
_KIND_LABEL = {
    "hiérarchie": "N+1 (hiérarchie)",
    "direction": "direction",
    "collègue": "collègue",
    "client": "client",
    "fournisseur": "fournisseur",
    "automate": "message automatique",
}


def intent_probs(
    items: Sequence[Item], model: IntentModel, embed: Callable[[list[str]], np.ndarray]
) -> dict[str, dict[str, float]]:
    texts = [intent_text(i.title, i.content_text or i.snippet) for i in items]
    probs = model.predict(embed(texts), texts) if items else {f: [] for f in LEARNED}
    return {f: {i.item_id: float(p) for i, p in zip(items, probs[f], strict=True)} for f in LEARNED}


def chips(f: MailFeatures, age_hours: float) -> list[str]:
    """La fiche en clair, telle qu'affichée sur la carte."""
    out = [_KIND_LABEL[f.sender_kind]]
    out += [
        label
        for label, p in (
            ("demande une action", f.action),
            ("attend une réponse", f.reponse),
            ("incident ou blocage", f.incident),
        )
        if p >= 0.5
    ]
    if f.alerte:
        out.append("alerte sécurité / banque")
    if f.deadline != "aucune":
        out.append(f"échéance {f.deadline} ({f.deadline_phrase})")
    out.append("pas encore répondu" if f.ball_in_court else "déjà répondu")
    if f.thread_size >= 3:
        out.append(f"fil de {f.thread_size} messages")
    if f.hierarchy_in_thread and f.sender_kind != "hiérarchie":
        out.append("N+1 dans le fil")
    if f.meeting_soon:
        out.append("réunion dans les 48 h")
    out.append(
        f"reçu il y a {age_hours:.0f} h"
        if age_hours < 48
        else f"reçu il y a {age_hours / 24:.0f} jours"
    )
    return out


def build_pairs(
    dataset: GraphDataset,
    probs: dict[str, dict[str, float]],
    persona: Persona = GRAPH_HELENE,
    seed: int = 0,
    n_pairs: int = N_PAIRS,
    n_heldout: int = N_HELDOUT,
    prefix: str = "p",
) -> list[dict]:
    rng = random.Random(seed)
    mornings = _mornings(dataset, probs, persona)
    days, used, seen, pairs = sorted(mornings), Counter(), set(), []
    while len(pairs) < n_pairs:
        close = len(pairs) % 2 == 1
        day = rng.choice(days)
        pool = mornings[day]
        a = _balanced(rng, pool)
        others = [
            b
            for b in pool
            if b[0].item_id != a[0].item_id and (not close or abs(b[2] - a[2]) <= CLOSE_GAP)
        ]
        if not others:
            continue
        b = _balanced(rng, others)
        key = frozenset((a[0].item_id, b[0].item_id))
        same_text = intent_text(a[0].title, a[0].snippet) == intent_text(b[0].title, b[0].snippet)
        if key in seen or same_text or used[a[0].item_id] >= 2 or used[b[0].item_id] >= 2:
            continue
        seen.add(key)
        used.update((a[0].item_id, b[0].item_id))
        if rng.random() < 0.5:
            a, b = b, a
        pairs.append(
            {
                "day": day.isoformat(),
                "kind": "serrée" if close else "hasard",
                "a": _side(*a, brief_instant(day)),
                "b": _side(*b, brief_instant(day)),
            }
        )
    heldout = set(rng.sample(range(len(pairs)), n_heldout))
    return [
        {"item_id": f"{prefix}{n:03d}", "heldout": n in heldout} | p for n, p in enumerate(pairs)
    ]


Option = tuple[Item, MailFeatures, int]  # (mail, fiche, score par règles)


def _mornings(
    dataset: GraphDataset, probs: dict[str, dict[str, float]], persona: Persona
) -> dict[date, list[Option]]:
    """Les candidats de chaque matin (au moins 4), avec leur fiche et leur score par règles."""
    config = ScoringConfig(user_emails=frozenset({persona.address}), hierarchy=persona.hierarchy)
    first = min(i.created_at for i in dataset.items).date() + timedelta(days=3)
    last = max(i.created_at for i in dataset.items).date()
    mornings: dict[date, list[Option]] = {}
    for n in range((last - first).days + 1):
        day = first + timedelta(days=n)
        now = brief_instant(day)
        items = candidates(dataset, day)
        if len(items) >= 4:
            feats = mail_features(items, dataset, now, persona, probs)
            rules = [score_case(Case(i.item_id, (i,)), now, config)[0] for i in items]
            mornings[day] = list(zip(items, feats, rules, strict=True))
    return mornings


def build_targeted_pairs(
    dataset: GraphDataset,
    probs: dict[str, dict[str, float]],
    exclude: set[frozenset[str]],
    persona: Persona = GRAPH_HELENE,
    seed: int = 1,
    n_pairs: int = 12,
    first_id: int = N_PAIRS,
) -> list[dict]:
    """2e série : une demande d'action du N+1 face à une demande d'action d'un autre type
    d'expéditeur (2 sur 3), ou face à un mail du N+1 sans demande (1 sur 3). Elle sert à
    apprendre l'interaction « demande d'action du N+1 » ; aucune n'est réservée."""
    rng = random.Random(seed)
    mornings = _mornings(dataset, probs, persona)
    asks = [
        (day, o)
        for day, opts in sorted(mornings.items())
        for o in opts
        if o[1].sender_kind == "hiérarchie" and o[1].action >= 0.5
    ]
    pairs, seen = [], set(exclude)
    for _ in range(50 * n_pairs):
        if len(pairs) == n_pairs or not asks:
            break
        day, a = rng.choice(asks)
        versus_boss = len(pairs) % 3 == 2
        others = [
            o
            for o in mornings[day]
            if o[0].item_id != a[0].item_id
            and (
                (o[1].sender_kind == "hiérarchie" and o[1].action < 0.5)
                if versus_boss
                else (o[1].sender_kind not in ("hiérarchie", "automate") and o[1].action >= 0.5)
            )
        ]
        if not others:
            continue
        b = rng.choice(others)
        key = frozenset((a[0].item_id, b[0].item_id))
        if key in seen:
            continue
        seen.add(key)
        if rng.random() < 0.5:
            a, b = b, a
        now = brief_instant(day)
        pairs.append(
            {"day": day.isoformat(), "kind": "ciblée N+1", "a": _side(*a, now), "b": _side(*b, now)}
        )
    return [
        {"item_id": f"p{first_id + n:03d}", "heldout": False, "round": 2} | p
        for n, p in enumerate(pairs)
    ]


def _balanced(rng: random.Random, options: list[tuple[Item, MailFeatures, int]]):
    """Un type d'expéditeur au hasard, puis un mail de ce type : sans ce rééquilibrage, les
    automates font près de la moitié des cartes et la hiérarchie 2 % (poids inapprenable)."""
    kind = rng.choice(sorted({f.sender_kind for _, f, _ in options}))
    return rng.choice([option for option in options if option[1].sender_kind == kind])


def _side(item: Item, f: MailFeatures, rule: int, now) -> dict:
    sender = next(p for p in item.participants if p.role == "from")
    return {
        "item_id": item.item_id,
        "sender": sender.name or sender.email,
        "sender_kind": f.sender_kind,
        "title": item.title,
        "text": (item.content_text or item.snippet)[:SHOWN_CHARS],
        "received_at": item.created_at.isoformat(),
        "chips": chips(f, (now - item.created_at).total_seconds() / 3600),
        "features": f.as_dict(),
        "rule_score": rule,
    }


def refresh_features(
    pairs: list[dict],
    dataset: GraphDataset,
    probs: dict[str, dict[str, float]],
    persona: Persona = GRAPH_HELENE,
) -> list[dict]:
    """Les mêmes paires, fiches recalculées (intentions du modèle courant) ; `chips` intact."""
    by_id = {item.item_id: item for item in dataset.items}
    out = []
    for pair in pairs:
        now = brief_instant(date.fromisoformat(pair["day"]))
        new = dict(pair)
        for side in ("a", "b"):
            item = by_id[pair[side]["item_id"]]
            f = mail_features([item], dataset, now, persona, probs)[0]
            new[side] = pair[side] | {"features": f.as_dict()}
        out.append(new)
    return out


def demo_dataset() -> GraphDataset:
    """Toutes les journées de démo du lot 1, dédoublonnées (u_demo_001, Camille Laurent)."""
    from contracts import demo_data

    items = [item for day in demo_data.available_days() for item in demo_data.load_items(day)]
    return dataset_from_items(items, DEMO_CAMILLE.address)


def _probs(dataset: GraphDataset) -> dict[str, dict[str, float]]:
    received = [
        i
        for i in dataset.items
        if i.source == "mail" and not dataset.extras[i.item_id].is_sent and not noise_reason(i)
    ]
    model = IntentModel.load()
    embedder = embedder_for(model.embedder_name)
    cache = VectorCache(cache_path(embedder))
    return intent_probs(received, model, lambda t: cache.embed(embedder, t))


def main(argv: list[str]) -> int:
    command = argv[0] if argv else ""
    if command == "demo" and len(argv) == 1:
        dataset = demo_dataset()
        pairs = build_pairs(
            dataset,
            _probs(dataset),
            DEMO_CAMILLE,
            seed=2,
            n_pairs=N_DEMO_PAIRS,
            n_heldout=N_DEMO_PAIRS,
            prefix="d",
        )
        pairs = [p | {"round": 3, "dataset": "démo", "manager": "Camille"} for p in pairs]
        write_sample(pairs, DEMO_PAIRS_PATH)
        print(f"{len(pairs)} paires de démo → {DEMO_PAIRS_PATH}, toutes réservées")
        return 0
    if command not in ("construire", "rafraichir") or len(argv) != 2:
        print(__doc__.split("\n\n")[3])
        return 2
    dataset = load_graph_dataset(argv[1])
    probs = _probs(dataset)
    if command == "rafraichir":
        pairs = refresh_features(load_sample(PAIRS_PATH), dataset, probs)
    else:
        pairs = build_pairs(dataset, probs)
        pairs += build_targeted_pairs(
            dataset, probs, {frozenset((p["a"]["item_id"], p["b"]["item_id"])) for p in pairs}
        )
    write_sample(pairs, PAIRS_PATH)
    kinds = Counter(p["kind"] for p in pairs)
    print(
        f"{len(pairs)} paires → {PAIRS_PATH} {dict(kinds)}, "
        f"{sum(p['heldout'] for p in pairs)} réservées à la mesure"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
