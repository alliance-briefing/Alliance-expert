"""M6 : comprendre chaque élément (action principale, urgence déclarée, alerte), 4 sources.

La référence du plan est MiniLM gelé + régression logistique, sur processeur. Elle lit la carte
d'`annoter.card`, exactement ce que voit l'annotateur, et apprend sur les personas « train » du
simulateur (gabarits + LLM). Les jeux gelés ne servent qu'à la mesure finale (G1).

Décisions du 22/11 (plan, §11) :
- la cible est l'action lisible sur la carte : une résolution par le manager avant le brief ne
  se voit pas dans l'`Item`, elle relève de la fiche (« balle dans son camp », M8) ;
- T-démo de M5 n'a que des paires : on tire ici une série gelée de 60 éléments de démo
  (`m6-tdemo-elements`), jugée par Théo seul et à l'aveugle, avant toute mesure d'un modèle
  (P3), pour mesurer le transfert de l'action vers le vrai domaine.

    uv run python -m packages.analyse.scoring.ml.comprendre --tirer-demo
    uv run python -m packages.analyse.scoring.ml.annoter --serie m6-tdemo-elements --annotateur theo
    uv run python -m packages.analyse.scoring.ml.comprendre --geler-demo
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from datetime import date
from pathlib import Path

from contracts.models import Item
from packages.analyse.scoring.ml import jeux_m5
from packages.analyse.scoring.ml.annoter import (
    SERIES,
    AnswerStore,
    answers_path,
    card,
    load_series,
    order,
    write_series,
)
from packages.analyse.scoring.ml.sample import ANNOTATIONS

DEMO_SERIES = "m6-tdemo-elements"
DEMO_TIRAGE = f"{DEMO_SERIES}.tirage.json"
GEL = "m6-gel.json"
# La moitié en mails, comme T-gen ; 10 de chaque autre source suffisent pour lire le F1 macro
# de l'action sur l'ensemble (pas source par source : trop peu).
DEMO_QUOTA = {"mail": 30, "calendar": 10, "task": 10, "file": 10}
ANNOTATOR = jeux_m5.ANNOTATOR
# Un élément de démo cité dans un fichier d'annotation a déjà été montré à un annotateur
# (paires de M5, mails étiquetés en octobre) : il n'est pas rejugé de mémoire.
_DEMO_ID = re.compile(r"\b(?:mail|calendar|task|file):dm1-[A-Za-z0-9_-]+")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )


# --- T-démo, éléments ------------------------------------------------------------------------


def demo_briefs() -> list[tuple[date, list[Item]]]:
    """Les journées de démo du lot 1 : (jour, éléments collectés pour le brief de ce matin)."""
    from contracts import demo_data

    return [(date.fromisoformat(d), demo_data.load_items(d)) for d in demo_data.available_days()]


def shown_demo_items(annotations: Path = ANNOTATIONS) -> set[str]:
    """Éléments de démo déjà présents dans un fichier d'annotation (séries comprises)."""
    return {
        found
        for path in sorted(annotations.rglob("*.json*"))
        for found in _DEMO_ID.findall(path.read_text(encoding="utf-8"))
    }


def draw_tdemo_elements(
    briefs: list[tuple[date, list[Item]]] | None = None,
    series: Path = SERIES,
    annotations: Path = ANNOTATIONS,
    seed: int = 0,
) -> dict:
    """Tire la série gelée des éléments de démo et écrit tirage et gel. Chaque élément est
    montré au premier brief où il apparaît, avec les éléments déjà visibles ce matin-là (ligne
    « Cité dans » des fichiers). Refusé si le tirage existe : un jeu gelé ne se retire pas."""
    if (series / DEMO_TIRAGE).exists():
        raise FileExistsError(f"{DEMO_TIRAGE} existe : un jeu de mesure ne se retire pas")
    briefs = demo_briefs() if briefs is None else briefs
    rng = random.Random(f"m6:tdemo-elements:{seed}")
    people = jeux_m5.demo_people(i for _, items in briefs for i in items)
    first: dict[str, tuple[date, Item]] = {}
    for day, items in briefs:
        for item in items:
            first.setdefault(item.item_id, (day, item))
    excluded = shown_demo_items(annotations) | jeux_m5.frozen_items(series)
    candidates = sorted(i for i in first if i not in excluded)
    rng.shuffle(candidates)
    by_source: dict[str, list[str]] = {s: [] for s in DEMO_QUOTA}
    for item_id in candidates:
        source = first[item_id][1].source
        if len(by_source[source]) < DEMO_QUOTA[source]:
            by_source[source].append(item_id)
    short = {s: len(v) for s, v in by_source.items() if len(v) < DEMO_QUOTA[s]}
    if short:
        raise ValueError(f"pas assez d'éléments de démo jamais vus : {short}")
    chosen = [i for s in DEMO_QUOTA for i in by_source[s]]
    rng.shuffle(chosen)
    tasks, behind = [], {}
    for n, item_id in enumerate(chosen, 1):
        day, item = first[item_id]
        vp = jeux_m5.demo_viewpoint(day, people)
        known = [
            i
            for d, items in briefs
            if d <= day
            for i in items
            if i.created_at <= vp.moment or i.item_id == item_id
        ]
        known = list({i.item_id: i for i in known}.values())
        task_id = f"d-e{n:03d}"
        tasks.append(
            {"task_id": task_id, "kind": "element", "context": vp.context}
            | {"card": card(item, vp, known)}
        )
        behind[task_id] = item_id
    write_series(tasks, series / f"{DEMO_SERIES}.jsonl")  # refuse toute étiquette
    info = {
        "graine": seed,
        "annotateur": ANNOTATOR,
        "decision": "22/11 (Théo) : T-démo de M5 n'a que des paires ; série d'éléments pour M6",
        "exclus_deja_vus": len(excluded & set(first)),
        "elements": behind,
    }
    _write_json(series / DEMO_TIRAGE, info)
    files = [series / f"{DEMO_SERIES}.jsonl", series / DEMO_TIRAGE]
    _write_json(series / GEL, {"series": {p.name: _sha(p) for p in files}, "reponses": {}})
    return info


def missing_demo_answers(series: Path = SERIES, annotator: str = ANNOTATOR) -> int:
    """Présentations de la série démo encore sans réponse (répétitions cachées comprises)."""
    tasks = load_series(series / f"{DEMO_SERIES}.jsonl")
    store = AnswerStore(answers_path(DEMO_SERIES, annotator, series))
    return sum(k not in store.answers for k in order(tasks, annotator, DEMO_SERIES))


def freeze_demo_answers(series: Path = SERIES, annotator: str = ANNOTATOR) -> dict:
    """Ajoute au gel l'empreinte des réponses, une fois tout annoté. Refusé avant."""
    left = missing_demo_answers(series, annotator)
    if left:
        raise ValueError(f"réponses manquantes : {left}")
    freeze = json.loads((series / GEL).read_text(encoding="utf-8"))
    if freeze["reponses"]:
        raise ValueError("réponses déjà gelées")
    path = answers_path(DEMO_SERIES, annotator, series)
    freeze["reponses"][path.name] = _sha(path)
    _write_json(series / GEL, freeze)
    return freeze


def changed_since_freeze(series: Path = SERIES) -> list[str]:
    """Fichiers gelés (M5 et M6) qui ont changé ou disparu depuis leur gel."""
    changed = jeux_m5.changed_since_freeze(series)
    path = series / GEL
    if path.exists():
        freeze = json.loads(path.read_text(encoding="utf-8"))
        files = {**freeze["series"], **freeze["reponses"]}
        changed += [
            name
            for name, digest in files.items()
            if not (series / name).exists() or _sha(series / name) != digest
        ]
    return sorted(changed)


def frozen_items(series: Path = SERIES) -> set[str]:
    """Tous les éléments des jeux de mesure (M5 et série démo de M6) : jamais à l'entraînement."""
    out = jeux_m5.frozen_items(series)
    path = series / DEMO_TIRAGE
    if path.exists():
        out |= set(json.loads(path.read_text(encoding="utf-8"))["elements"].values())
    return out


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if argv == ["--tirer-demo"]:
        info = draw_tdemo_elements()
        print(
            f"{DEMO_SERIES} : {len(info['elements'])} éléments "
            f"({info['exclus_deja_vus']} éléments de démo déjà vus exclus)."
        )
        print(
            "  uv run python -m packages.analyse.scoring.ml.annoter "
            f"--serie {DEMO_SERIES} --annotateur {ANNOTATOR}"
        )
        return 0
    if argv == ["--reste-demo"]:
        print(missing_demo_answers())
        return 0
    if argv == ["--geler-demo"]:
        freeze = freeze_demo_answers()
        print(f"Réponses gelées : {', '.join(freeze['reponses'])}")
        return 0
    print("usage : comprendre --tirer-demo | --reste-demo | --geler-demo")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
