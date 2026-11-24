"""M8 : priorité multi-sources, « ce matin, lequel traiter en premier ? ».

Étape 1, les paires d'entraînement. Personne dans l'équipe n'a le temps de les annoter :
Claude les juge (décision du 21/11 : Claude peut annoter les données d'entraînement), avec
les critères écrits avant dans le guide (section 4, 24/11). Les paires de T-gen et de T-démo,
jugées par Théo et gelées, restent la seule mesure : le modèle apprend le jugement de Claude
et il est jugé sur celui d'un humain (plan, §11).

Les paires sont tirées comme celles de T-gen (`jeux_m5.draw_tgen`) : deux éléments au hasard
du même brief, toutes sources mêlées, mais sur les personas « train » et sans aucun élément
d'un jeu gelé. Un tour est une série de l'outil d'annotation (`m8-paires-t<n>`), réservée à
l'annotateur `claude`, avec ses répétitions cachées (côtés inversés) pour mesurer sa
cohérence. Le tirage (`.tirage.json`) garde les éléments derrière chaque carte, jamais une
étiquette du simulateur.

    uv run python -m packages.analyse.scoring.ml.prioriser --tirer 1
    uv run python -m packages.analyse.scoring.ml.prioriser --montrer 1
    uv run python -m packages.analyse.scoring.ml.prioriser --reste
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from packages.analyse.scoring.ml import jeux_m5
from packages.analyse.scoring.ml.annoter import (
    SERIES,
    AnswerStore,
    answers_path,
    card,
    check_answer,
    load_series,
    order,
    swap,
    write_series,
)

CLAUDE = "claude"
PREFIX = "m8-paires-t"
ROUND_SIZE = 100


def round_name(n: int) -> str:
    return f"{PREFIX}{n}"


def drawn_rounds(series: Path = SERIES) -> list[int]:
    """Les tours déjà tirés, dans l'ordre."""
    return sorted(
        int(p.name[len(PREFIX) :].split(".")[0]) for p in series.glob(f"{PREFIX}*.tirage.json")
    )


def drawn_pairs(series: Path = SERIES) -> set[frozenset[str]]:
    """Les paires (éléments) des tours déjà tirés : un tour ne reprend jamais une paire."""
    out = set()
    for n in drawn_rounds(series):
        info = json.loads((series / f"{round_name(n)}.tirage.json").read_text(encoding="utf-8"))
        out.update(frozenset(p) for p in info["paires"].values())
    return out


def draw_round(
    n: int,
    corpus: Path = jeux_m5.CORPUS,
    series: Path = SERIES,
    size: int = ROUND_SIZE,
    seed: int = 0,
) -> dict:
    """Le tour `n` : `size` paires d'éléments du même brief, au hasard comme T-gen. Refusé
    si le tour existe : une série annotée ne se retire pas."""
    name = round_name(n)
    if (series / f"{name}.tirage.json").exists():
        raise FileExistsError(f"{name} existe déjà : un tour tiré ne se retire pas")
    rng = random.Random(f"m8:paires:{n}:{seed}")
    frozen = jeux_m5.frozen_items(series)
    used = drawn_pairs(series)
    all_briefs = jeux_m5.briefs(corpus, "train")
    pairable = [b for b in all_briefs if sum(i.item_id not in frozen for i in b[3]) >= 2]
    tasks, behind = [], {}
    for _ in range(100 * size):  # borne : un corpus trop petit échoue au lieu de boucler
        if len(tasks) == size:
            break
        _key, _day, vp, shown, known = rng.choice(pairable)
        a, b = rng.sample([i for i in shown if i.item_id not in frozen], 2)
        if frozenset((a.item_id, b.item_id)) in used:
            continue
        used.add(frozenset((a.item_id, b.item_id)))
        task_id = f"m8t{n}-p{len(tasks) + 1:03d}"
        tasks.append(
            {"task_id": task_id, "kind": "paire", "context": vp.context, "pour": [CLAUDE]}
            | {"a": card(a, vp, known), "b": card(b, vp, known)}
        )
        behind[task_id] = [a.item_id, b.item_id]
    if len(tasks) < size:
        raise ValueError(f"seulement {len(tasks)} paires distinctes pour le tour {n}")
    write_series(tasks, series / f"{name}.jsonl")  # refuse toute étiquette (check_blind)
    info = {
        "tour": n,
        "graine": seed,
        "selection": "hasard (comme T-gen)",
        "annotateur": CLAUDE,
        "personas": sorted({b[0] for b in all_briefs}),
        "exclus_geles": len(frozen),
        "paires": behind,
    }
    (series / f"{name}.tirage.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return info


def presentations(n: int, series: Path = SERIES) -> list[tuple[str, bool, dict]]:
    """Ce que voit l'annotateur, dans son ordre : (task_id, répétition, paire montrée). Une
    répétition revient côtés inversés, comme sur la page d'annotation."""
    name = round_name(n)
    tasks = {t["task_id"]: t for t in load_series(series / f"{name}.jsonl")}
    out = []
    for task_id, repeat in order(list(tasks.values()), CLAUDE, name):
        task = tasks[task_id]
        shown = task | ({"a": task["b"], "b": task["a"]} if repeat else {})
        out.append((task_id, repeat, shown))
    return out


def record(n: int, choices: dict[tuple[str, bool], str], series: Path = SERIES) -> int:
    """Enregistre des réponses données sur les paires **montrées** (A = carte de gauche) ;
    une répétition est ramenée au sens de la série (`swap`). Rend le nombre enregistré."""
    store = AnswerStore(answers_path(round_name(n), CLAUDE, series))
    valid = {(t, r) for t, r, _ in presentations(n, series)}
    for (task_id, repeat), choice in choices.items():
        if (task_id, repeat) not in valid:
            raise ValueError(f"{task_id} (répétition : {repeat}) n'est pas dans le tour {n}")
        answer = check_answer("paire", {"choix": choice})
        store.record(task_id, repeat, swap(answer) if repeat else answer)
    return len(choices)


def missing(series: Path = SERIES) -> dict[str, int]:
    """Présentations encore sans réponse, par tour."""
    out = {}
    for n in drawn_rounds(series):
        store = AnswerStore(answers_path(round_name(n), CLAUDE, series))
        out[round_name(n)] = sum(
            (t, r) not in store.answers for t, r, _ in presentations(n, series)
        )
    return out


def consistency(n: int, series: Path = SERIES) -> dict:
    """Cohérence de l'annotateur sur les répétitions cachées du tour."""
    answers = AnswerStore(answers_path(round_name(n), CLAUDE, series)).answers
    repeated = [t for (t, rep) in answers if rep and (t, False) in answers]
    same = sum(answers[t, True] == answers[t, False] for t in repeated)
    return {"repetitions": len(repeated), "identiques": same}


def show(task: dict) -> str:
    """Une paire en texte, pour l'annotation en ligne de commande."""

    def side(label: str, c: dict) -> list[str]:
        lines = [f"  [{label}] {c['source']} · {c['title']} · {c['when']}"]
        lines += [f"      {k} : {v}" for k, v in c["fields"]]
        if c["text"]:
            lines.append("      " + " ".join(c["text"].split())[:700])
        return lines

    return "\n".join([task["context"], *side("A", task["a"]), *side("B", task["b"])])


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if len(argv) == 2 and argv[0] == "--tirer":
        info = draw_round(int(argv[1]))
        print(
            f"{round_name(info['tour'])} : {len(info['paires'])} paires, "
            f"{len(info['personas'])} personas « train », {info['exclus_geles']} éléments gelés exclus"
        )
        return 0
    if len(argv) in (2, 4) and argv[0] == "--montrer":
        shown = presentations(int(argv[1]))
        start, end = (int(argv[2]), int(argv[3])) if len(argv) == 4 else (0, len(shown))
        for k, (task_id, repeat, task) in enumerate(shown[start:end], start):
            print(f"#{k} {task_id}{' (répétition)' if repeat else ''}\n{show(task)}\n")
        return 0
    if argv == ["--reste"]:
        print(missing())
        return 0
    print("usage : prioriser --tirer <tour> | --montrer <tour> [début fin] | --reste")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
