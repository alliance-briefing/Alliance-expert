"""M9 : réglage de la sélection du brief (seuil T, nombre maximal K) sur des journées simulées.

Règle du 25/11 (plan, §11), écrite avant tout calcul : des journées des personas « train »
(jamais T-gen, la démo ni le jeu du lot 3), le brief appris à 8 h 30, une vérité lue sur les
étiquettes du simulateur, le NDCG@7 dans l'ordre de lecture du brief, le choix G2 puis G3 avec
T = 50, K = 7 (les valeurs d'avant M9) comme référence.

Vérité simulée d'une journée, par affaire du simulateur, sur les éléments au brief (mêmes
fenêtres que le score appris, `learned.on_brief`) :
- 3 : une alerte, hors injection ;
- 2 : un acte (Décider, Faire, Répondre, Préparer, Relire) avec échéance dépassée ou du jour,
  réunion du jour ou ton urgent ;
- 1 : un autre acte ;
- un élément déjà résolu au moment du brief, ou envoyé par le manager, ne compte pas ;
- à exclure : injections et envois en masse.

Biais noté : N a vu ces personas à l'entraînement ; les valeurs absolues sont optimistes, seule
la comparaison entre réglages compte.

    uv run python -m packages.analyse.scoring.ml.selection --valider
    uv run python -m packages.analyse.scoring.ml.selection --mesurer-lot3 <annotations>
    uv run python -m packages.analyse.scoring.ml.selection --latence   # M10, ≤ 3 s par brief
    uv run python -m packages.analyse.scoring.ml.selection --demo 2026-10-07
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from contracts.models import Brief, Item
from packages.analyse.brief import SELECT_LIMIT, SELECT_THRESHOLD, generate_brief
from packages.analyse.scoring.learned import CaseScore, LearnedScorer, on_brief
from packages.analyse.scoring.ml import jeux_m5
from packages.collect.trainset import load

REPORT_PATH = Path(__file__).with_name("selection_validation.json")
THRESHOLDS = (40, 50, 60, 70)
LIMITS = (5, 7)
REFERENCE = (SELECT_THRESHOLD, SELECT_LIMIT)  # T = 50, K = 7
N_DAYS = 60
TOP = 7
ACTS = ("decider", "faire", "repondre", "preparer", "relire")


def truth_of_day(
    shown: Sequence[Item], rows: dict[str, dict], moment: datetime
) -> tuple[dict, set]:
    """({case_id du simulateur: pertinence}, éléments à exclure) pour un brief simulé."""
    grades: dict[str, int] = {}
    exclude: set[str] = set()
    end_of_day = moment.replace(hour=23, minute=59)
    for item in shown:
        row = rows[item.item_id]
        if row.get("trap") == "injection" or any(
            t.startswith("newsletter") for t in row.get("templates", ())
        ):
            exclude.add(item.item_id)
            continue
        resolved = row.get("resolved_at")
        if row.get("sent") or (resolved and datetime.fromisoformat(resolved) <= moment):
            continue
        due = item.due_at or (datetime.fromisoformat(row["deadline"]) if row["deadline"] else None)
        timely = (
            (due is not None and due <= end_of_day)
            or (
                item.source == "calendar"
                and item.start_at is not None
                and item.start_at <= end_of_day
            )
            or row["urgence"] == "urgent"
        )
        grade = (
            3
            if row["alerte"]
            else 2
            if row["action"] in ACTS and timely
            else (1 if row["action"] in ACTS else 0)
        )
        if grade:
            grades[row["case_id"]] = max(grades.get(row["case_id"], 0), grade)
    return grades, exclude


def reading_order(brief: Brief) -> list[list[str]]:
    """Les entrées dans l'ordre de lecture : sections dans l'ordre du contrat, puis rang."""
    return [
        e.source_item_ids for s in brief.sections for e in sorted(s.entries, key=lambda e: e.rank)
    ]


def ndcg(entries: list[list[str]], case_of: dict[str, str], grades: dict[str, int]) -> float:
    """NDCG@7 : chaque affaire du simulateur ne rapporte qu'une fois, à sa première entrée."""
    credited, gains = set(), []
    for ids in entries[:TOP]:
        cases = {case_of[i] for i in ids if i in case_of} - credited
        gain = max((grades.get(c, 0) for c in cases), default=0)
        credited |= cases
        gains.append(gain)
    dcg = sum((2**g - 1) / math.log2(k + 2) for k, g in enumerate(gains))
    ideal = sorted(grades.values(), reverse=True)[:TOP]
    idcg = sum((2**g - 1) / math.log2(k + 2) for k, g in enumerate(ideal))
    return dcg / idcg if idcg else 1.0


class CachedScorer:
    """Le score appris d'une journée, calculé une fois pour tous les réglages."""

    def __init__(self, scorer: LearnedScorer) -> None:
        self.scorer, self.memo = scorer, {}

    def score_cases(self, cases, items, user, now) -> dict[str, CaseScore]:
        key = (user, now)
        if key not in self.memo:
            self.memo[key] = self.scorer.score_cases(cases, items, user, now)
        return self.memo[key]


def sample_days(corpus: Path, n: int, seed: int = 0) -> list[tuple]:
    days = jeux_m5.briefs(corpus, "train")
    random.Random(f"m9:jours:{seed}").shuffle(days)
    return days[:n]


def valider(
    corpus: Path = jeux_m5.CORPUS,
    scorer: LearnedScorer | None = None,
    n_days: int = N_DAYS,
    out: Path | None = REPORT_PATH,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """NDCG@7 de chaque réglage (T, K) sur les journées simulées, choix G2 puis G3."""
    cached = CachedScorer(scorer or LearnedScorer())
    grid = [(t, k) for t in THRESHOLDS for k in LIMITS]
    scores: dict[tuple[int, int], list[float]] = {g: [] for g in grid}
    sizes: dict[tuple[int, int], list[int]] = {g: [] for g in grid}
    excluded_top3: dict[tuple[int, int], int] = dict.fromkeys(grid, 0)
    rows_of: dict[str, dict[str, dict]] = {}
    days = sample_days(corpus, n_days)
    for n, (key, day, vp, shown, known) in enumerate(days, 1):
        log(f"journée {n}/{len(days)} : {key} {day}")
        if key not in rows_of:
            rows_of[key] = {r["item_id"]: r for r in load.load_scenario(key, corpus)}
        rows = rows_of[key]
        items = [i.model_copy(update={"collected_at": vp.moment}) for i in known]
        user = items[0].owner_user_id
        on_day = [i for i in known if on_brief(i, vp.moment)]  # les éléments au brief
        grades, exclude = truth_of_day(on_day, rows, vp.moment)
        case_of = {i.item_id: rows[i.item_id]["case_id"] for i in known}
        for g in grid:
            brief = generate_brief(items, user, day, now=vp.moment, scorer=cached, select=g)
            order = reading_order(brief)
            scores[g].append(ndcg(order, case_of, grades))
            sizes[g].append(len(order))
            excluded_top3[g] += any(i in exclude for ids in order[:3] for i in ids)
    means = {g: float(np.mean(v)) for g, v in scores.items()}
    best = max(grid, key=lambda g: (means[g], g == REFERENCE))
    low, high = _paired_ci(np.array(scores[best]), np.array(scores[REFERENCE]))
    chosen = best if low > 0 else REFERENCE
    report = {
        "journees": len(days),
        "personas": sorted({d[0] for d in days}),
        "reglages": [
            {
                "seuil": t,
                "max": k,
                "ndcg7": round(means[(t, k)], 4),
                "entrees_moyennes": round(float(np.mean(sizes[(t, k)])), 2),
                "journees_avec_exclu_dans_top3": excluded_top3[(t, k)],
            }
            for t, k in grid
        ],
        "meilleur": {"seuil": best[0], "max": best[1]},
        "ecart_meilleur_reference_ic95": [round(low, 4), round(high, 4)],
        "retenu": {"seuil": chosen[0], "max": chosen[1]},
    }
    if out:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


def _paired_ci(
    a: np.ndarray, b: np.ndarray, draws: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Intervalle à 95 % de la moyenne de a − b, journées rééchantillonnées (G2)."""
    rng = np.random.default_rng(seed)
    diffs = [float((a[idx] - b[idx]).mean()) for idx in rng.integers(0, len(a), (draws, len(a)))]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


# --- mesure finale sur le jeu de développement du lot 3 ------------------------------------------

TASK = "M9, sélection du brief"
LOT3_BRANCH = "origin/l3/feat/annotations-dev"  # annotations du lot 3 (guide L3.5 v0.3)


def brief_metrics(brief: Brief, annotation) -> dict:
    """NDCG@7 (gain n − rang + 1 pour l'élément attendu de rang `rang`, chacun compté une
    fois, à sa première entrée), rappel des « doit figurer » sur tout le brief, et « ne doit
    pas figurer » cités par les 3 premières entrées, dans l'ordre de lecture."""
    expected = sorted(annotation.must_include, key=lambda e: e.rank)
    n = len(expected)
    gain = {e.rank: n - e.rank + 1 for e in expected}
    order = reading_order(brief)
    credited: set[int] = set()
    gains = []
    for k, ids in enumerate(order):
        found = {e.rank for e in expected if e.rank not in credited and set(e.item_ids) & set(ids)}
        if k < TOP:
            gains.append(max((gain[r] for r in found), default=0))
        credited |= found
    dcg = sum((2**g - 1) / math.log2(k + 2) for k, g in enumerate(gains))
    ideal = sorted(gain.values(), reverse=True)[:TOP]
    idcg = sum((2**g - 1) / math.log2(k + 2) for k, g in enumerate(ideal))
    excluded = set(annotation.must_exclude)
    return {
        "ndcg7": dcg / idcg,
        "rappel": len(credited) / n,
        "exclus_top3": sorted({i for ids in order[:3] for i in ids} & excluded),
        "entrees": len(order),
    }


def mesurer_lot3(
    annotations: Sequence[Path],
    scorer: LearnedScorer | None = None,
    load_items: Callable[[str, str], list[Item]] | None = None,
    journal: Path = Path(__file__).parent.parent / "annotations" / "journal-geles.jsonl",
    commit: Callable[[], str] | None = None,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """La mesure unique de M9 (règle du 25/11) : brief appris (score de M8, sélection T = 50,
    K = 7) contre brief par règles, sur chaque annotation du lot 3 ; au journal des jeux gelés,
    une seule fois par (modèle, tâche). Décide le mode par défaut du brief."""
    from eval.annotation import Annotation
    from packages.analyse.scoring.ml import comprendre, prioriser

    model_sha = comprendre._sha(prioriser.MODEL_PATH)
    prioriser._refuse_reading(journal, model_sha, TASK)
    revision = (
        commit
        or (
            lambda: comprendre.git_commit(
                (prioriser.MODEL_PATH, Path(__file__), Path(__file__).parents[2] / "brief.py")
            )
        )
    )()
    if load_items is None:
        from contracts import demo_data

        def load_items(day: str, user: str) -> list[Item]:
            return demo_data.load_items(day, user)

    scorer = scorer or LearnedScorer()
    rows, sources = [], {}
    for path in sorted(annotations):
        sources[path.name] = comprendre._sha(path)
        annotation = Annotation.model_validate_json(path.read_text(encoding="utf-8"))
        day = annotation.date.isoformat()
        log(f"{path.name}…")
        items = load_items(day, annotation.user_id)
        learned = generate_brief(items, annotation.user_id, day, scorer=scorer)
        rules = generate_brief(items, annotation.user_id, day, scorer=None)
        rows.append(
            {"annotation": path.name}
            | {f"appris_{k}": v for k, v in brief_metrics(learned, annotation).items()}
            | {f"regles_{k}": v for k, v in brief_metrics(rules, annotation).items()}
            | {"contrat_valide": _valid(learned) and _valid(rules)}
        )

    def mean(key: str) -> float:
        return round(float(np.mean([r[key] for r in rows])), 4)

    gaps = {
        m: _paired_ci(
            np.array([r[f"appris_{m}"] for r in rows]), np.array([r[f"regles_{m}"] for r in rows])
        )
        for m in ("ndcg7", "rappel")
    }
    summary = {
        "annotations": len(rows),
        "appris": {"ndcg7": mean("appris_ndcg7"), "rappel": mean("appris_rappel")},
        "regles": {"ndcg7": mean("regles_ndcg7"), "rappel": mean("regles_rappel")},
        "ecarts_ic95": {m: [round(lo, 4), round(hi, 4)] for m, (lo, hi) in gaps.items()},
        "appris_exclus_top3": sum(bool(r["appris_exclus_top3"]) for r in rows),
        "regles_exclus_top3": sum(bool(r["regles_exclus_top3"]) for r in rows),
        "contrat_valide": all(r["contrat_valide"] for r in rows),
    }
    accepted = (
        summary["appris"]["ndcg7"] > summary["regles"]["ndcg7"]
        and summary["appris"]["rappel"] > summary["regles"]["rappel"]
        and summary["appris_exclus_top3"] == 0
        and summary["contrat_valide"]
    )
    entry = {
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tache": TASK,
        "commit": revision,
        "modele": prioriser.MODEL_PATH.name,
        "modele_sha256": model_sha,
        "selection": {"seuil": SELECT_THRESHOLD, "max": SELECT_LIMIT},
        "annotations_sha256": sources,
        "resume": summary,
        "par_annotation": rows,
        "acceptation_atteinte": accepted,
        "mode_par_defaut": "appris" if accepted else "regles",
    }
    with journal.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _valid(brief: Brief) -> bool:
    try:
        Brief.model_validate(brief.model_dump())
    except ValueError:
        return False
    return True


# --- latence (M10) -------------------------------------------------------------------------------

LATENCY_PATH = Path(__file__).with_name("brief_latence.json")
LATENCY_LIMIT_S = 3.0  # acceptation de M10 : ≤ 3 s par brief sur processeur


def latence(
    days: Sequence[str] | None = None,
    out: Path | None = LATENCY_PATH,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """Temps de génération d'un brief sur les journées de démo, dans les deux modes : premier
    brief (chargement des modèles compris), puis médiane et 95e centile des suivants."""
    import platform
    import time

    from contracts import demo_data

    days = list(days or demo_data.available_days())
    loaded = [(day, demo_data.load_items(day)) for day in days]
    report: dict = {
        "journees": len(days),
        "processeur": platform.processor() or platform.machine(),
        "seuil_s": LATENCY_LIMIT_S,
    }
    for mode in ("regles", "appris"):
        times = []
        scorer = None
        for day, items in loaded:
            started = time.perf_counter()
            if mode == "appris" and scorer is None:
                scorer = LearnedScorer()  # chargement compté dans le premier brief
            generate_brief(items, demo_data.DEFAULT_USER, day, scorer=scorer)
            times.append(time.perf_counter() - started)
            log(f"{mode} {day} : {times[-1]:.2f} s")
        rest = np.array(times[1:])
        report[mode] = {
            "premier_s": round(times[0], 2),
            "mediane_s": round(float(np.median(rest)), 2),
            "p95_s": round(float(np.percentile(rest, 95)), 2),
            "max_s": round(float(rest.max()), 2),
            "dans_le_seuil": bool(np.percentile(rest, 95) <= LATENCY_LIMIT_S),
        }
    if out:
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    return report


def demo(day: str) -> str:
    """Le brief d'une journée de démo dans les deux modes, entrée par entrée (M10)."""
    from contracts import demo_data

    items = demo_data.load_items(day)
    lines = []
    for mode, scorer in (("règles", None), ("appris", LearnedScorer())):
        brief = generate_brief(items, demo_data.DEFAULT_USER, day, scorer=scorer)
        lines.append(f"=== brief {mode} du {day} ({brief.generator.llm})")
        for section in brief.sections:
            for e in section.entries:
                lines.append(f"[{section.kind} {e.rank}] {e.priority_score:>3}  {e.text}")
                lines.append(f"      raisons : {' ; '.join(e.priority_reasons)}")
                if e.recommended_action:
                    lines.append(f"      action : {e.recommended_action}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if argv == ["--valider"]:
        report = valider(log=lambda line: print(line, flush=True))
        print(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"→ {REPORT_PATH.name}")
        return 0
    if len(argv) == 2 and argv[0] == "--demo":
        print(demo(argv[1]))
        return 0
    if argv == ["--latence"]:
        print(json.dumps(latence(log=lambda line: print(line, flush=True)), indent=1))
        print(f"→ {LATENCY_PATH.name}")
        return 0
    if len(argv) == 2 and argv[0] == "--mesurer-lot3":
        paths = sorted(Path(argv[1]).glob("*.json"))
        entry = mesurer_lot3(paths, log=lambda line: print(line, flush=True))
        print(json.dumps({k: v for k, v in entry.items() if k != "par_annotation"}, indent=1))
        return 0
    print(
        "usage : selection --valider | --latence | --demo <jour> | --mesurer-lot3 <dossier des annotations>"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
