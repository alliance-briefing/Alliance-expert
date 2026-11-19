"""Sonde de l'écrivain LLM (M3) dans l'outil d'annotation unifié (M4).

Le trainset écrit les mails à annoter sans étiquette (`data/sonde/<dossier>/a_annoter.jsonl`) :
il n'a pas le droit d'importer le lot 2. Ce module en fait une série pour `annoter.py`, avec
ses cartes (`card`) : exactement ce que verra le modèle (objet, extrait de 500 caractères,
non lu, importance haute). Puis il compare les réponses aux étiquettes du simulateur
(`attendu.jsonl`).

    uv run python -m packages.analyse.scoring.ml.sonde_m3 --serie mail-v5-tirage2
    uv run python -m packages.analyse.scoring.ml.annoter --serie sonde-m3-mail-v5-tirage2 --annotateur theo
    uv run python -m packages.analyse.scoring.ml.sonde_m3 --comparer mail-v5-tirage2

Critère fixé d'avance (M3) : un texte est conforme si l'action, l'urgence et l'alerte annotées
sont celles du simulateur ; la sonde passe avec au moins 16 conformes sur 20, pour chaque
annotateur. Seules les premières présentations comptent (pas les répétitions cachées).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from contracts.models import Item
from packages.analyse.scoring.ml.annoter import (
    SERIES,
    card,
    first_answers,
    viewpoint,
    write_series,
)
from packages.collect.trainset.sonde import OUT as SONDES

THRESHOLD = 16  # conformes sur 20, fixé avant toute mesure
KEYS = ("action", "urgence", "alerte")


def series_name(folder: str) -> str:
    return f"sonde-m3-{folder}"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_series(folder: str, sondes: Path = SONDES, series: Path = SERIES) -> Path:
    """La série d'annotation d'une sonde : une tâche « élément » par mail, à l'aveugle."""
    tasks = []
    for row in _jsonl(sondes / folder / "a_annoter.jsonl"):
        vp = viewpoint(row["persona"], date.fromisoformat(row["jour"]))
        item = Item.model_validate(row["item"])
        tasks.append(
            {"task_id": row["case_id"], "kind": "element", "context": vp.context}
            | {"card": card(item, vp)}
        )
    path = series / f"{series_name(folder)}.jsonl"
    write_series(tasks, path)  # refuse toute étiquette (check_blind)
    return path


def compare(folder: str, sondes: Path = SONDES, series: Path = SERIES) -> str:
    """Textes conformes au simulateur, par annotateur, et le détail des écarts."""
    expected = {row["case_id"]: row for row in _jsonl(sondes / folder / "attendu.jsonl")}
    by_who = first_answers(series_name(folder), series)
    if not by_who:
        return f"Aucune réponse pour la série {series_name(folder)}."
    lines = []
    for who, answers in sorted(by_who.items()):
        done = [c for c in expected if c in answers]
        ok = [c for c in done if all(answers[c][k] == expected[c][k] for k in KEYS)]
        reached = len(ok) >= THRESHOLD and len(done) == len(expected)
        lines.append(
            f"{who} : {len(ok)} conformes sur {len(done)} annotés "
            f"(critère {THRESHOLD}/{len(expected)} : {'atteint' if reached else 'non atteint'})"
        )
        for key in KEYS:
            agree = sum(answers[c][key] == expected[c][key] for c in done)
            lines.append(f"  {key:<8} {agree}/{len(done)}")
        for c in done:
            if c not in ok:
                e = expected[c]
                diff = ", ".join(
                    f"{k} {answers[c][k]} au lieu de {e[k]}" for k in KEYS if answers[c][k] != e[k]
                )
                lines.append(f"  ≠ {c} [{e['emplacement']}] {diff}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if len(argv) == 2 and argv[0] == "--serie":
        path = build_series(argv[1])
        print(f"Série écrite : {path}")
        print(
            "Annoter : uv run python -m packages.analyse.scoring.ml.annoter "
            f"--serie {series_name(argv[1])} --annotateur theo"
        )
        return 0
    if len(argv) == 2 and argv[0] == "--comparer":
        print(compare(argv[1]))
        return 0
    print("usage : sonde_m3 --serie <dossier> | --comparer <dossier>")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
