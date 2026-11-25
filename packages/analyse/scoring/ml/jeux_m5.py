"""M5 : les jeux de mesure gelés, la vérité fixée avant tout réglage (plan, P2 et P3).

- **T-gen** : les 4 personas réservées (métiers jamais vus à l'entraînement), corpus final
  (mails du LLM, autres sources par gabarits), au brief de 8 h 30 d'un jour ouvré :
  150 éléments (action, urgence, alerte), 150 paires d'éléments et 50 paires d'affaires
  (« ce matin, lequel d'abord ? »), en trois séries ;
- **T-démo** : les 40 paires de la démo du lot 1 (autre métier, autre persona), rejugées à
  l'aveugle, sans pastille de modèle.

Les cartes sont celles d'`annoter.py` : ce que verrait le modèle, jamais une étiquette du
simulateur (`check_blind`). Les éléments déjà montrés dans les échantillons de M3 sont
exclus : l'annotateur ne les juge pas de mémoire.

Le gel : `m5-gel.json` garde l'empreinte de chaque série (au tirage, avant toute réponse),
puis celle des réponses (`--geler`, une fois tout annoté). Un test vérifie que rien de gelé
ne change, et que rien de T-gen ni de T-démo ne peut entrer à l'entraînement.

Décision du 21/11 (David) : faute de temps, un seul annotateur, sans kappa.

    uv run python -m packages.analyse.scoring.ml.jeux_m5 --tirer
    uv run python -m packages.analyse.scoring.ml.annoter --serie m5-tgen-elements --annotateur theo
    uv run python -m packages.analyse.scoring.ml.jeux_m5 --geler
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path

from contracts.models import Item
from packages.analyse.scoring.ml.annoter import (
    SERIES,
    AnswerStore,
    Viewpoint,
    answers_path,
    card,
    case_card,
    load_series,
    order,
    viewpoint,
    write_series,
)
from packages.analyse.scoring.ml.persona import DEMO_CAMILLE
from packages.analyse.scoring.ml.sample import ANNOTATIONS
from packages.collect.trainset import load
from packages.collect.trainset.llm_corpus import OUT as LLM
from packages.collect.trainset.world import brief_moment, fr_day

CORPUS = LLM / "mail-v8"  # le corpus final de M3
DEMO_PAIRS = ANNOTATIONS / "pair_sample_demo.jsonl"
TIRAGE = SERIES / "m5-tirage.json"
GEL = SERIES / "m5-gel.json"
NAMES = {
    "elements": "m5-tgen-elements",
    "paires": "m5-tgen-paires",
    "affaires": "m5-tgen-affaires",
    "demo": "m5-tdemo-paires",
}
# Une moitié de mails (la source la plus fréquente), le reste en parts égales : assez de
# chaque source pour mesurer le modèle source par source.
ELEMENT_QUOTA = {"mail": 75, "calendar": 25, "task": 25, "file": 25}
N_PAIRS, N_CASES = 150, 50
ANNOTATOR = "theo"


# --- ce que montre un brief --------------------------------------------------------------------


def on_brief(item: Item, row: dict, moment: datetime, previous: datetime) -> bool:
    """L'élément figure-t-il au brief de `moment` ? Les fenêtres des connecteurs : mails et
    fichiers arrivés depuis le brief précédent, réunions des 48 h à venir, tâches en retard
    ou dues dans les 24 h ; et jamais rien qui ne soit pas encore visible."""
    if datetime.fromisoformat(row["visible_from"]) > moment:
        return False
    if item.source == "calendar":
        return item.start_at is not None and moment <= item.start_at < moment + timedelta(days=2)
    if item.source == "task":
        return item.due_at is not None and item.due_at < moment + timedelta(days=1)
    return previous < datetime.fromisoformat(row["visible_from"])


def _working_days(rows: Iterable[dict]) -> list[date]:
    days = {datetime.fromisoformat(r["visible_from"]).date() for r in rows}
    return sorted(d for d in days if d.weekday() < 5)


def briefs(
    corpus: Path = CORPUS, split: str = "tgen"
) -> list[tuple[str, date, Viewpoint, list[Item], list[Item]]]:
    """Les briefs d'un split (T-gen par défaut ; « train » pour les paires d'entraînement de
    M8) : (persona, jour, point de vue, éléments du brief, éléments connus). Les connus (déjà
    visibles) servent à la ligne « Cité dans » des fichiers."""
    out = []
    for key in load.personas(split, data=corpus):
        items = load.load_items(key, corpus)
        rows = {r["item_id"]: r for r in load.load_scenario(key, corpus)}
        days = _working_days(rows.values())
        for n, day in enumerate(days):
            moment = brief_moment(day)
            previous = brief_moment(days[n - 1]) if n else moment - timedelta(days=3)
            known = [
                i
                for i in items
                if datetime.fromisoformat(rows[i.item_id]["visible_from"]) <= moment
            ]
            shown = [i for i in known if on_brief(i, rows[i.item_id], moment, previous)]
            if shown:
                out.append((key, day, viewpoint(key, day), shown, known))
    return out


def seen_in_m3(series: Path = SERIES) -> set[str]:
    """Les éléments déjà montrés à l'annotateur dans les échantillons de M3."""
    return {
        item_id
        for path in series.glob("m3-*.tirage.json")
        for item_id in json.loads(path.read_text(encoding="utf-8")).get("elements", {}).values()
    }


# --- T-gen ---------------------------------------------------------------------------------------


def draw_tgen(corpus: Path = CORPUS, series: Path = SERIES, seed: int = 0) -> dict:
    """Les trois séries de T-gen et ce qu'il y a derrière chaque carte (sans étiquette)."""
    rng = random.Random(f"m5:tgen:{seed}")
    seen = seen_in_m3(series)
    all_briefs = briefs(corpus)
    rows = {
        key: {r["item_id"]: r for r in load.load_scenario(key, corpus)}
        for key in {b[0] for b in all_briefs}
    }
    # Éléments : un tirage par source, chaque élément une seule fois (au premier brief tiré).
    candidates = [(b, item) for b in all_briefs for item in b[3] if item.item_id not in seen]
    rng.shuffle(candidates)
    by_source: dict[str, list] = {s: [] for s in ELEMENT_QUOTA}
    taken: set[str] = set()
    for b, item in candidates:
        if item.item_id in taken or len(by_source[item.source]) >= ELEMENT_QUOTA[item.source]:
            continue
        by_source[item.source].append((b, item))
        taken.add(item.item_id)
    short = {s: len(v) for s, v in by_source.items() if len(v) < ELEMENT_QUOTA[s]}
    if short:
        raise ValueError(f"pas assez d'éléments jamais vus pour T-gen : {short}")
    chosen = [x for s in ELEMENT_QUOTA for x in by_source[s]]
    rng.shuffle(chosen)
    elements, behind_e = [], {}
    for n, ((_key, _day, vp, _shown, known), item) in enumerate(chosen, 1):
        task_id = f"t-e{n:03d}"
        elements.append(
            {"task_id": task_id, "kind": "element", "context": vp.context}
            | {"card": card(item, vp, known)}
        )
        behind_e[task_id] = item.item_id
    # Paires d'éléments : deux éléments du même brief, toutes sources mêlées.
    pairable = [b for b in all_briefs if sum(i.item_id not in seen for i in b[3]) >= 2]
    pairs, behind_p, used = [], {}, set()
    for _ in range(100 * N_PAIRS):  # borne : un corpus trop petit échoue au lieu de boucler
        if len(pairs) == N_PAIRS:
            break
        _key, _day, vp, shown, known = rng.choice(pairable)
        a, b = rng.sample([i for i in shown if i.item_id not in seen], 2)
        if frozenset((a.item_id, b.item_id)) in used:
            continue
        used.add(frozenset((a.item_id, b.item_id)))
        task_id = f"t-p{len(pairs) + 1:03d}"
        pairs.append(
            {"task_id": task_id, "kind": "paire", "context": vp.context}
            | {"a": card(a, vp, known), "b": card(b, vp, known)}
        )
        behind_p[task_id] = [a.item_id, b.item_id]
    if len(pairs) < N_PAIRS:
        raise ValueError(f"seulement {len(pairs)} paires d'éléments distinctes pour T-gen")

    # Paires d'affaires : deux affaires vraies du simulateur (décision du 17/11) présentes au
    # même brief ; chaque affaire montre ses éléments déjà visibles.
    def cases_of(b: tuple) -> dict[str, list[Item]]:
        key, _day, _vp, shown, known = b
        live = {rows[key][i.item_id]["case_id"] for i in shown}
        out: dict[str, list[Item]] = {}
        for item in known:
            row = rows[key][item.item_id]
            if row["case_type"] != "fond" and row["case_id"] in live:
                out.setdefault(row["case_id"], []).append(item)
        return out

    with_cases = [(b, c) for b in all_briefs if len(c := cases_of(b)) >= 2]
    cases, behind_a, used_cases = [], {}, set()
    for _ in range(100 * N_CASES):
        if len(cases) == N_CASES:
            break
        (_key, _day, vp, _shown, _known), found = rng.choice(with_cases)
        a, b = rng.sample(sorted(found), 2)
        if frozenset((a, b)) in used_cases:
            continue
        used_cases.add(frozenset((a, b)))
        task_id = f"t-a{len(cases) + 1:03d}"
        cases.append(
            {"task_id": task_id, "kind": "affaires", "context": vp.context}
            | {"a": case_card(found[a], vp), "b": case_card(found[b], vp)}
        )
        behind_a[task_id] = [[i.item_id for i in found[a]], [i.item_id for i in found[b]]]
    if len(cases) < N_CASES:
        raise ValueError(f"seulement {len(cases)} paires d'affaires distinctes pour T-gen")
    for name, tasks in (("elements", elements), ("paires", pairs), ("affaires", cases)):
        write_series(tasks, series / f"{NAMES[name]}.jsonl")  # refuse toute étiquette
    return {
        "personas": sorted({b[0] for b in all_briefs}),
        "briefs": len(all_briefs),
        "exclus_m3": len(seen),
        "elements": behind_e,
        "paires": behind_p,
        "affaires": behind_a,
    }


# --- T-démo --------------------------------------------------------------------------------------

_DEMO_RELATION = {
    "hiérarchie": "votre N+1",
    "direction": "direction",
    "collègue": "collègue de l'agence",
    "client": "client",
    "fournisseur": "partenaire",
    "automate": "envoi automatique",
}


def demo_items() -> dict[str, Item]:
    """Toutes les journées de démo du lot 1 (u_demo_001, Camille Laurent), dédoublonnées."""
    from contracts import demo_data

    out: dict[str, Item] = {}
    for day in demo_data.available_days():
        for item in demo_data.load_items(day):
            out.setdefault(item.item_id, item)
    return out


def demo_people(items: Iterable[Item]) -> dict[str, str]:
    """La relation de chaque correspondant à Camille (configuration versionnée de la persona :
    elle viendrait de l'annuaire dans un vrai déploiement)."""
    people = {}
    for item in items:
        for p in item.participants:
            if p.email.lower() != DEMO_CAMILLE.address:
                people[p.email] = _DEMO_RELATION[DEMO_CAMILLE.sender_kind(p.email)]
    return people


def demo_viewpoint(day: date, people: dict[str, str]) -> Viewpoint:
    """Camille Laurent, responsable de l'agence Lyon-Est, le jour du brief à 8 h 30."""
    context = (
        "Vous êtes Camille Laurent, responsable de l'agence Lyon-Est d'un réseau d'expertise "
        f"automobile (N+1 : Bernard Mercier). Brief du {fr_day(day)} à 8 h 30."
    )
    return Viewpoint(DEMO_CAMILLE.address, brief_moment(day), people, context)


def draw_tdemo(series: Path = SERIES, pairs_path: Path = DEMO_PAIRS, seed: int = 0) -> dict:
    """Les paires de démo, rejugées à l'aveugle : cartes neuves (aucune pastille de l'ancien
    outil), ordre et côtés tirés à nouveau."""
    rng = random.Random(f"m5:tdemo:{seed}")
    items = demo_items()
    rows = [json.loads(x) for x in pairs_path.read_text(encoding="utf-8").splitlines() if x]
    rng.shuffle(rows)
    people = demo_people(items.values())
    tasks, behind = [], {}
    for n, row in enumerate(rows, 1):
        a, b = items[row["a"]["item_id"]], items[row["b"]["item_id"]]
        if rng.random() < 0.5:
            a, b = b, a
        vp = demo_viewpoint(date.fromisoformat(row["day"]), people)
        task_id = f"d-p{n:03d}"
        tasks.append(
            {"task_id": task_id, "kind": "paire", "context": vp.context}
            | {"a": card(a, vp), "b": card(b, vp)}
        )
        behind[task_id] = [a.item_id, b.item_id]
    write_series(tasks, series / f"{NAMES['demo']}.jsonl")
    return {"paires": behind}


# --- tirage et gel -------------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_all(corpus: Path = CORPUS, series: Path = SERIES, seed: int = 0) -> dict:
    """Tire T-gen et T-démo, écrit le tirage et le gel des séries. Un jeu gelé ne se retire
    pas : refusé si le tirage existe déjà."""
    if (series / TIRAGE.name).exists():
        raise FileExistsError(f"{TIRAGE.name} existe : un jeu de mesure ne se retire pas")
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    info = {
        "graine": seed,
        "corpus": {"consigne": manifest["consigne"], "textes_sha256": manifest["textes"]["sha256"]},
        "annotateur": ANNOTATOR,
        "decision": "21/11 (David) : un seul annotateur, sans kappa ; priorité au modèle",
        "tgen": draw_tgen(corpus, series, seed),
        "tdemo": draw_tdemo(series, seed=seed),
    }
    (series / TIRAGE.name).write_text(
        json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    files = [series / f"{name}.jsonl" for name in NAMES.values()] + [series / TIRAGE.name]
    freeze = {"series": {p.name: _sha(p) for p in files}, "reponses": {}}
    (series / GEL.name).write_text(
        json.dumps(freeze, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return info


def missing_answers(series: Path = SERIES, annotator: str = ANNOTATOR) -> dict[str, int]:
    """Présentations encore sans réponse, par série (répétitions cachées comprises)."""
    out = {}
    for name in NAMES.values():
        tasks = load_series(series / f"{name}.jsonl")
        store = AnswerStore(answers_path(name, annotator, series))
        out[name] = sum(k not in store.answers for k in order(tasks, annotator, name))
    return out


def freeze_answers(series: Path = SERIES, annotator: str = ANNOTATOR) -> dict:
    """Ajoute au gel l'empreinte des réponses, une fois tout annoté. Refusé avant."""
    left = {k: v for k, v in missing_answers(series, annotator).items() if v}
    if left:
        raise ValueError(f"réponses manquantes : {left}")
    freeze = json.loads((series / GEL.name).read_text(encoding="utf-8"))
    if freeze["reponses"]:
        raise ValueError("réponses déjà gelées")
    for name in NAMES.values():
        path = answers_path(name, annotator, series)
        freeze["reponses"][path.name] = _sha(path)
    (series / GEL.name).write_text(
        json.dumps(freeze, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return freeze


def changed_since_freeze(series: Path = SERIES) -> list[str]:
    """Les fichiers gelés qui ont changé (ou disparu) depuis leur gel."""
    path = series / GEL.name
    if not path.exists():
        return []
    freeze = json.loads(path.read_text(encoding="utf-8"))
    files = {**freeze["series"], **freeze["reponses"]}
    return sorted(
        name
        for name, digest in files.items()
        if not (series / name).exists() or _sha(series / name) != digest
    )


def frozen_items(series: Path = SERIES) -> set[str]:
    """Tous les éléments de T-gen et de T-démo : à exclure de tout entraînement (M6, M7)."""
    path = series / TIRAGE.name
    if not path.exists():
        return set()
    info = json.loads(path.read_text(encoding="utf-8"))
    out = set(info["tgen"]["elements"].values())
    for pair in (*info["tgen"]["paires"].values(), *info["tdemo"]["paires"].values()):
        out.update(pair)
    for a, b in info["tgen"]["affaires"].values():
        out.update(a, b)
    return out


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if argv == ["--tirer"]:
        info = draw_all()
        t = info["tgen"]
        print(
            f"T-gen ({', '.join(t['personas'])}, {t['briefs']} briefs, {t['exclus_m3']} "
            f"éléments de M3 exclus) : {len(t['elements'])} éléments, {len(t['paires'])} "
            f"paires, {len(t['affaires'])} paires d'affaires ; T-démo : "
            f"{len(info['tdemo']['paires'])} paires."
        )
        for name in NAMES.values():
            print(
                "  uv run python -m packages.analyse.scoring.ml.annoter "
                f"--serie {name} --annotateur {ANNOTATOR}"
            )
        return 0
    if argv == ["--reste"]:
        print(missing_answers())
        return 0
    if argv == ["--geler"]:
        freeze = freeze_answers()
        print(f"Réponses gelées : {', '.join(freeze['reponses'])}")
        return 0
    print("usage : jeux_m5 --tirer | --reste | --geler")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
