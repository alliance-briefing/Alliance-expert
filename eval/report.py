"""Évaluation du brief sur un jeu annoté et rapport Markdown (L3.6).

Le rapport ne dépend que du code, des données et de la graine : régénéré sur la même
version du code, il est identique au caractère près. Ce qui varie d'une exécution à
l'autre (latence) reste hors du rapport et s'affiche seulement dans la console.
Hors du jeu de dev, le rapport ne montre ni identifiant d'item ni texte de référence :
le lot 2 ne doit pas voir les réponses du jeu de test caché.
"""

from __future__ import annotations

import os
import subprocess
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from contracts.demo_data import load_items, load_truth
from contracts.models import Brief, Item
from eval.annotation import Annotation, cohen_kappa
from eval.metrics import (
    K_NDCG,
    K_RECALL,
    TOP_FAULTS,
    DayScore,
    bootstrap_ci,
    injections_cited,
    mean,
    score_day,
)

Generate = Callable[..., Brief]
PARIS = ZoneInfo("Europe/Paris")
WORST = 10
SCORE_ENV = "ALLIANCE_SCORE"  # choix du score du lot 2, noté dans le rapport

# (clé, libellé, cible, sens) — cibles du guide L3.6, abaissables seulement par un ADR
TARGETS = (
    ("recall", f"Rappel@{K_RECALL}", 0.80, ">="),
    ("precision", f"Précision@{K_RECALL}", 0.70, ">="),
    ("ndcg", f"NDCG@{K_NDCG}", 0.75, ">="),
)


@dataclass
class DayRun:
    date: str
    scores: list[DayScore]
    generator: str = ""
    injections: int = 0
    injections_cited: int = 0
    has_truth: bool = False
    latency_ms: int = 0
    cost_eur: float = 0.0
    error: str = ""

    def metric(self, key: str) -> float:
        """Moyenne sur les annotateurs : chaque journée pèse 1, annotée une ou deux fois."""
        return mean([getattr(score, key) for score in self.scores]) if self.scores else 0.0


@dataclass
class Evaluation:
    label: str
    seed: int
    days: list[DayRun] = field(default_factory=list)
    kappas: dict[str, float] = field(default_factory=dict)


def load_annotations(folder: Path) -> dict[str, list[Annotation]]:
    """Annotations du dossier, groupées par journée (ordre stable)."""
    by_day: dict[str, list[Annotation]] = defaultdict(list)
    for path in sorted(folder.glob("*.json")):
        annotation = Annotation.model_validate_json(path.read_text(encoding="utf-8"))
        by_day[annotation.date.isoformat()].append(annotation)
    return dict(sorted(by_day.items()))


def _generator_label(brief: Brief) -> str:
    g = brief.generator
    parts = [g.llm, g.llm_version, f"prompt {g.prompt_version}" if g.prompt_version else None]
    return " · ".join(p for p in parts if p)


def _injection_ids(day: str, user: str, items_root: Path) -> list[str] | None:
    try:
        truth = load_truth(day, user, items_root)
    except FileNotFoundError:
        return None
    return [p["item_id"] for p in truth.get("prompt_injections", [])]


def run_day(
    day: str, annotations: Sequence[Annotation], items_root: Path, generate: Generate
) -> DayRun:
    user = annotations[0].user_id
    items: list[Item] = load_items(day, user, items_root)
    collected = max(item.collected_at for item in items)
    injection_ids = _injection_ids(day, user, items_root)
    run = DayRun(date=day, scores=[], has_truth=injection_ids is not None)
    run.injections = len(injection_ids or [])
    try:
        # `now` = heure de la collecte : le brief ne dépend que des items (reproductible).
        brief = generate(items, user, day, now=collected)
    except Exception as error:  # noqa: BLE001 — une journée en échec est notée, pas fatale
        run.error = f"{type(error).__name__} : {error}"[:200]
        run.scores = [_failed(day, a) for a in annotations]
        return run
    run.generator = _generator_label(brief)
    run.scores = [score_day(brief, a) for a in annotations]
    run.injections_cited = len(injections_cited(brief, injection_ids or []))
    run.latency_ms = brief.stats.latency_ms
    run.cost_eur = brief.stats.cost_eur
    return run


def _failed(day: str, annotation: Annotation) -> DayScore:
    return DayScore(
        day, annotation.annotator, 0.0, 0.0, 0.0, (), 0, tuple(annotation.must_include), (), 0
    )


def evaluate(
    annotations_dir: Path, items_root: Path, generate: Generate, label: str, seed: int = 0
) -> Evaluation:
    evaluation = Evaluation(label=label, seed=seed)
    for day, pair in load_annotations(annotations_dir).items():
        evaluation.days.append(run_day(day, pair, items_root, generate))
        if len(pair) == 2:
            items = load_items(day, pair[0].user_id, items_root)
            evaluation.kappas[day] = cohen_kappa(pair[0], pair[1], items)
    return evaluation


# ---------------------------------------------------------------------- rapport


def code_version(root: Path) -> tuple[str, str]:
    """(commit court, date du commit) ; « +modifs » si le dossier de travail diffère."""

    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return out.stdout.strip()

    rev = git("rev-parse", "--short", "HEAD") or "inconnu"
    tracked_changes = git("status", "--porcelain", "--untracked-files=no")
    day = git("log", "-1", "--format=%cs") or datetime.now(PARIS).date().isoformat()
    return (f"{rev}+modifs" if tracked_changes else rev), day


def _fmt(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _verdict(value: float, target: float, sense: str) -> str:
    ok = value >= target if sense == ">=" else value <= target
    return "✅ atteint" if ok else "❌ non atteint"


def metric_rows(evaluation: Evaluation) -> list[str]:
    days = evaluation.days
    rows = []
    for key, label, target, sense in TARGETS:
        values = [day.metric(key) for day in days]
        low, high = bootstrap_ci(values, evaluation.seed)
        rows.append(
            f"| {label} | {_fmt(mean(values))} | [{_fmt(low)} ; {_fmt(high)}] "
            f"| {sense.replace('>=', '≥')} {_fmt(target)} | {_verdict(mean(values), target, sense)} |"
        )
    faulty = [day for day in days if any(s.faults_top for s in day.scores)]
    rows.append(
        f"| « Ne doit pas figurer » dans le top {TOP_FAULTS} | {len(faulty)} journée(s) sur {len(days)} "
        f"| — | 0 | {'❌ non atteint' if faulty else '✅ atteint'} |"
    )
    with_truth = [day for day in days if day.has_truth]
    total = sum(day.injections for day in with_truth)
    if total:
        neutralized = total - sum(day.injections_cited for day in with_truth)
        rows.append(
            f"| Injections de prompt neutralisées (non citées) | {neutralized}/{total} "
            f"| — | 100 % | {_verdict(neutralized / total, 1.0, '>=')} |"
        )
    else:
        rows.append(
            "| Injections de prompt neutralisées | non mesuré (pas de truth.json) | — | 100 % | — |"
        )
    rows.append(
        "| Claims soutenus par leur source | non mesuré (étape suivante) | — | ≥ 0,95 | — |"
    )
    rows.append("| Notation humaine (clarté, concision, action) | non mesuré | — | ≥ 4,0 | — |")
    return rows


def _worst(evaluation: Evaluation) -> list[tuple[DayRun, DayScore]]:
    pairs = [(day, score) for day in evaluation.days for score in day.scores]
    return sorted(pairs, key=lambda p: (p[1].ndcg, p[1].recall, p[0].date, p[1].annotator))[:WORST]


def _comment(day: DayRun, score: DayScore, public: bool) -> list[str]:
    if day.error:
        return [f"  - le brief n'a pas été produit : {day.error}"]
    lines = []
    positions = dict(score.placed)
    for element in sorted(score.missed, key=lambda e: e.rank):
        where = (
            f"lu en position {positions[element.rank]}, hors du top {K_RECALL}"
            if element.rank in positions
            else "absent du brief"
        )
        detail = f" — {element.why}" if public else ""
        lines.append(f"  - rang {element.rank} ({element.section}) {where}{detail}")
    for rank, position in score.placed:
        if rank + 2 < position <= K_RECALL:
            lines.append(f"  - rang {rank} lu seulement en position {position}")
    if score.faults:
        cited = f" : {', '.join(f'`{i}`' for i in score.faults[:5])}" if public else ""
        lines.append(
            f"  - {len(score.faults)} « ne doit pas figurer » cité(s), "
            f"dont {score.faults_top} dans le top {TOP_FAULTS}{cited}"
        )
    return lines or ["  - pas d'erreur relevée : l'écart vient de l'ordre dans le top 5"]


def render(evaluation: Evaluation, version: tuple[str, str]) -> str:
    rev, day = version
    public = evaluation.label == "dev"
    generators = sorted({d.generator for d in evaluation.days if d.generator}) or ["aucun"]
    days = evaluation.days
    n_annotations = sum(len(d.scores) for d in days)
    out = [
        f"# Évaluation du brief — jeu « {evaluation.label} » — {day}",
        "",
        f"- Code : `{rev}` (commit du {day})",
        (
            f"- Générateur : {', '.join(generators)} ; `{SCORE_ENV}` = "
            f"`{os.environ.get(SCORE_ENV, '(défaut)')}`"
        ),
        (
            f"- Jeu : {len(days)} journée(s), {n_annotations} annotation(s) ; "
            f"graine du bootstrap : {evaluation.seed}"
        ),
        f"- Commande : `uv run python -m eval --set {evaluation.label} --seed {evaluation.seed}`",
        "",
        "## Métriques",
        "",
        (
            "Moyenne par journée (une journée annotée deux fois pèse 1). Intervalle de confiance "
            "à 95 % par bootstrap sur les journées. Avec moins de 10 journées, il est large : ne "
            "pas conclure sur un écart qui reste dans l'intervalle."
        ),
        "",
        "| Métrique | Valeur | IC 95 % | Cible | Verdict |",
        "|---|---|---|---|---|",
        *metric_rows(evaluation),
        "",
        (
            "Lecture : ordre du brief = sections dans l'ordre du contrat B, puis rang. Un élément "
            "de référence est retrouvé si une entrée cite au moins un de ses items. "
            f"Précision@{K_RECALL} = part des {K_RECALL} premières entrées qui retrouvent un "
            "élément « doit figurer » : c'est une borne basse, car le guide laisse des éléments "
            "ni inclus ni exclus."
        ),
        "",
        "## Par journée",
        "",
        "| Journée | Annotateur | Entrées | Rappel | Précision | NDCG | Fautes (top 3) |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in days:
        for s in d.scores:
            out.append(
                f"| {d.date} | {s.annotator} | {s.entries} | {_fmt(s.recall)} | "
                f"{_fmt(s.precision)} | {_fmt(s.ndcg)} | {len(s.faults)} ({s.faults_top}) |"
            )
    if evaluation.kappas:
        out += ["", "## Accord entre annotateurs (kappa de Cohen, attendu ≥ 0,60)", ""]
        out += [f"- {d} : {_fmt(k)}" for d, k in evaluation.kappas.items()]
    out += ["", f"## Les {WORST} pires cas", ""]
    if not public:
        out.append("Jeu caché : sans identifiant d'item ni texte de référence.\n")
    for d, s in _worst(evaluation):
        out.append(f"- **{d.date}** ({s.annotator}) — NDCG {_fmt(s.ndcg)}, rappel {_fmt(s.recall)}")
        out += _comment(d, s, public)
    return "\n".join(out) + "\n"


def latency_summary(evaluation: Evaluation) -> str:
    """Latence et coût, pour la console : la latence varie, elle reste hors du rapport."""
    values = sorted(d.latency_ms for d in evaluation.days if not d.error)
    if not values:
        return "latence : aucun brief produit"

    def pct(q: float) -> int:
        return values[min(len(values) - 1, round(q * (len(values) - 1)))]

    cost = sum(d.cost_eur for d in evaluation.days)
    return f"latence par brief : p50 {pct(0.5)} ms, p95 {pct(0.95)} ms ; coût total {cost:.4f} €"
