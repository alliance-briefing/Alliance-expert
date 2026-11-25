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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from contracts.models import Item
from packages.analyse.aggregate.cases import Case
from packages.analyse.rag.embeddings import DEFAULT_MODEL, Embedder, FastEmbedder
from packages.analyse.scoring.ml import comprendre, jeux_m5
from packages.analyse.scoring.ml.annoter import (
    SERIES,
    AnswerStore,
    Viewpoint,
    answers_path,
    card,
    check_answer,
    final_labels,
    load_series,
    order,
    swap,
    write_series,
)
from packages.analyse.scoring.ml.intents import VectorCache, cache_path
from packages.analyse.scoring.ml.learned_score import fit_bradley_terry
from packages.analyse.scoring.score import ScoringConfig, score_case

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


Chooser = Callable[[list[dict]], list[int]]


def draw_round(
    n: int,
    corpus: Path = jeux_m5.CORPUS,
    series: Path = SERIES,
    size: int = ROUND_SIZE,
    seed: int = 0,
    pool: int | None = None,
    chooser: Chooser | None = None,
) -> dict:
    """Le tour `n` : `size` paires d'éléments du même brief. Sans `chooser`, au hasard comme
    T-gen ; avec, `pool` paires au hasard, dont `chooser` garde `size` (les plus incertaines
    pour le modèle du moment, `uncertain`). Refusé si le tour existe : une série annotée ne se
    retire pas."""
    name = round_name(n)
    if (series / f"{name}.tirage.json").exists():
        raise FileExistsError(f"{name} existe déjà : un tour tiré ne se retire pas")
    rng = random.Random(f"m8:paires:{n}:{seed}")
    frozen = jeux_m5.frozen_items(series)
    used = drawn_pairs(series)
    all_briefs = jeux_m5.briefs(corpus, "train")
    pairable = [b for b in all_briefs if sum(i.item_id not in frozen for i in b[3]) >= 2]
    wanted = pool if chooser else size
    found: list[dict] = []
    for _ in range(100 * wanted):  # borne : un corpus trop petit échoue au lieu de boucler
        if len(found) == wanted:
            break
        key, _day, vp, shown, known = rng.choice(pairable)
        a, b = rng.sample([i for i in shown if i.item_id not in frozen], 2)
        if frozenset((a.item_id, b.item_id)) in used:
            continue
        used.add(frozenset((a.item_id, b.item_id)))
        found.append(
            {"context": vp.context, "persona": key}
            | {"a": card(a, vp, known), "b": card(b, vp, known)}
            | {"ids": [a.item_id, b.item_id]}
        )
    if len(found) < wanted:
        raise ValueError(f"seulement {len(found)} paires distinctes pour le tour {n}")
    if chooser:
        found = [found[k] for k in chooser(found)][:size]
    tasks, behind = [], {}
    for k, pair in enumerate(found, 1):
        task_id = f"m8t{n}-p{k:03d}"
        tasks.append(
            {"task_id": task_id, "kind": "paire", "context": pair["context"], "pour": [CLAUDE]}
            | {"a": pair["a"], "b": pair["b"]}
        )
        behind[task_id] = pair["ids"]
    write_series(tasks, series / f"{name}.jsonl")  # refuse toute étiquette (check_blind)
    info = {
        "tour": n,
        "graine": seed,
        "selection": (
            f"les {size} plus incertaines parmi {pool} au hasard"
            if chooser
            else "hasard (comme T-gen)"
        ),
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


# --- étape 2 : fiche de priorité, Bradley-Terry, comparaison aux règles ------------------------

REPORT_PATH = Path(__file__).with_name("prioriser_validation.json")
L2_GRID = (10.0, 1.0, 0.1)  # du plus régularisé au moins : ordre de simplicité (G3)
PRIORITY_FOLDS = 4  # plis par métier, comme la validation dure de M6
POOL = 1000  # réservoir des tours 2 et 3 (règle du 24/11)
FROZEN = "*N figé*"  # « métier » des jeux gelés : la fiche prend le modèle N figé
_N_ACTIONS = tuple(a for a in comprendre.TARGETS["action"] if a != "rien")
PRIORITY_NAMES = (
    *comprendre.FEATURE_NAMES,
    "tache:echeance_aujourdhui",
    *(f"N:{a}" for a in _N_ACTIONS),
    "N:urgent",
    "N:sans_urgence",
    "N:alerte",
)


@dataclass
class Pair:
    """Une paire jugée : ses deux cartes, ses éléments, le brief et la réponse (A, B ou =)."""

    task_id: str
    persona: str
    vp: Viewpoint
    cards: tuple[dict, dict]
    items: tuple[Item, Item]
    label: str

    @property
    def metier(self) -> str:
        return comprendre.metier(self.persona)


def card_signals(c: dict) -> np.ndarray:
    """La partie lisible de la fiche : la fiche de M6, plus l'échéance du jour d'une tâche."""
    fields = {label: value for label, value in c["fields"]}
    today = c["source"] == "task" and fields.get("Échéance", "").startswith("aujourd'hui")
    return np.r_[comprendre.card_features(c), float(today)]


def n_columns(models: dict[str, comprendre.Softmax], X: np.ndarray) -> np.ndarray:
    """Les probabilités de N gardées dans la fiche : six actions (Rien est la référence),
    urgent, sans urgence, alerte."""
    action = models["action"].proba(X)
    urgence = models["urgence"].proba(X)
    keep = [comprendre.TARGETS["action"].index(a) for a in _N_ACTIONS]
    u = comprendre.TARGETS["urgence"]
    return np.hstack(
        [
            action[:, keep],
            urgence[:, [u.index("urgent"), u.index("sans_urgence")]],
            models["alerte"].proba(X)[:, [comprendre.TARGETS["alerte"].index("oui")]],
        ]
    )


class Understanding:
    """N, la compréhension retenue en M6, pour la fiche de priorité. Les éléments des paires
    d'entraînement sont aussi des exemples de N : leurs probabilités viennent de N appris sans
    leur métier (plis par métier), sinon Bradley-Terry apprendrait à croire des probabilités
    trop sûres d'elles, qu'il ne retrouverait pas sur T-gen (métiers jamais vus)."""

    def __init__(
        self,
        corpora: dict[str, Path] = comprendre.CORPORA,
        embedder: Embedder | None = None,
        cache: VectorCache | None = None,
        report_path: Path = comprendre.TEXT_REPORT_PATH,
        name: str = comprendre.CANDIDATE,
        n_folds: int = PRIORITY_FOLDS,
        log: Callable[[str], None] = lambda _: None,
        candidate_path: Path = comprendre.CANDIDATE_PATH,
    ) -> None:
        self.candidate_path = candidate_path
        variant = json.loads(report_path.read_text(encoding="utf-8"))["variantes"][name]
        self.fiche, self.texte, self.n_folds = variant["fiche"], variant["texte"], n_folds
        self.embedder = embedder or FastEmbedder(DEFAULT_MODEL)
        self.cache = cache or VectorCache(cache_path(self.embedder))
        log("compréhension : exemples des personas « train »…")
        examples, _ = comprendre.training_examples(corpora)
        X = self.design([e.card for e in examples])
        configs = {
            t: comprendre.Config(c["l2"], c["classes_pondérées"])
            for t, c in ((t, variant["metiers"][t]["configuration"]) for t in comprendre.TARGETS)
        }
        self.by_metier: dict[str, dict[str, comprendre.Softmax]] = {}
        for k, held in enumerate(comprendre.metier_folds({e.metier for e in examples}, n_folds)):
            log(f"compréhension sans les métiers {held} ({k + 1}/{n_folds})…")
            rows = [n for n, e in enumerate(examples) if e.metier not in held]
            models = {
                t: comprendre.fit_softmax(
                    X[rows], comprendre._codes([examples[n] for n in rows], t), len(c), configs[t]
                )
                for t, c in comprendre.TARGETS.items()
            }
            self.by_metier |= dict.fromkeys(held, models)

    def design(self, cards: Sequence[dict]) -> np.ndarray:
        return comprendre.design(cards, self.embedder, self.fiche, self.cache, self.texte)

    def features(self, cards: Sequence[dict], metiers: Sequence[str]) -> np.ndarray:
        """La fiche de priorité de chaque carte (ordre de PRIORITY_NAMES), hors pli ; le métier
        FROZEN prend le modèle N figé (jeux gelés : métiers jamais vus de toute façon)."""
        X = self.design(cards)
        out = np.zeros((len(cards), len(PRIORITY_NAMES)))
        for m in set(metiers):
            rows = [n for n, x in enumerate(metiers) if x == m]
            signals = np.array([card_signals(cards[n]) for n in rows])
            models = self.frozen() if m == FROZEN else self.by_metier[m]
            out[rows] = np.hstack([signals, n_columns(models, X[rows])])
        return out

    def frozen(self) -> dict[str, comprendre.Softmax]:
        return comprendre.Reference.load(self.candidate_path).models


def config_for(vp: Viewpoint) -> ScoringConfig:
    """Le barème des règles pour ce manager : son adresse et son N+1, ce qu'un déploiement
    connaîtrait (annuaire)."""
    hierarchy = frozenset(a for a, role in vp.people.items() if "N+1" in role)
    return ScoringConfig(user_emails=frozenset({vp.manager_email}), hierarchy=hierarchy)


def rule_score(item: Item, vp: Viewpoint) -> int:
    """Le score par règles de l'élément seul (L2.2), au moment du brief."""
    return score_case(Case(item.item_id, (item,)), vp.moment, config_for(vp))[0]


def round_pairs(n: int, corpus: Path = jeux_m5.CORPUS, series: Path = SERIES) -> list[Pair]:
    """Les paires jugées du tour `n` (premières présentations), avec leurs éléments et leur
    brief, retrouvés par le contexte affiché (une persona, un jour)."""
    name = round_name(n)
    behind = json.loads((series / f"{name}.tirage.json").read_text(encoding="utf-8"))["paires"]
    answers = AnswerStore(answers_path(name, CLAUDE, series)).answers
    briefs = {b[2].context: b for b in jeux_m5.briefs(corpus, "train")}
    out = []
    for task in load_series(series / f"{name}.jsonl"):
        if (task["task_id"], False) not in answers:
            continue
        key, _day, vp, shown, _known = briefs[task["context"]]
        by_id = {i.item_id: i for i in shown}
        a, b = (by_id[i] for i in behind[task["task_id"]])
        label = answers[task["task_id"], False]["choix"]
        out.append(Pair(task["task_id"], key, vp, (task["a"], task["b"]), (a, b), label))
    return out


def side_features(
    pairs: Sequence[Pair], understanding: Understanding, frozen: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Fiches de priorité des cartes A et des cartes B, une ligne par paire."""
    cards = [c for p in pairs for c in p.cards]
    metiers = [FROZEN if frozen else p.metier for p in pairs for _ in range(2)]
    F = understanding.features(cards, metiers)
    return F[0::2], F[1::2]


def pair_features(pairs: Sequence[Pair], understanding: Understanding) -> np.ndarray:
    """Différences de fiches A − B, une ligne par paire."""
    Fa, Fb = side_features(pairs, understanding)
    return Fa - Fb


def bt_rows(diffs: np.ndarray, labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lignes de Bradley-Terry : une égalité donne deux demi-comparaisons (règle du 24/11)."""
    rows, wins, weights = [], [], []
    for d, label in zip(diffs, labels, strict=True):
        if label == "=":
            rows += [d, d]
            wins += [1.0, 0.0]
            weights += [0.5, 0.5]
        else:
            rows.append(d)
            wins.append(float(label == "A"))
            weights.append(1.0)
    return np.array(rows), np.array(wins), np.array(weights)


def fit_priority(diffs: np.ndarray, labels: Sequence[str], l2: float) -> np.ndarray:
    return fit_bradley_terry(*bt_rows(diffs, labels), l2=l2)


def _accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float((pred == truth).mean())


def _ci(values: np.ndarray, draws: int = 1000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    return [round(float(x), 4) for x in np.percentile(means, [2.5, 97.5])]


def validate(
    pairs: Sequence[Pair], diffs: np.ndarray, n_folds: int = PRIORITY_FOLDS
) -> tuple[dict, float, np.ndarray]:
    """Validation croisée par métier : accord hors pli de chaque l2 sur les paires non
    équivalentes, choix G2 puis G3, et comparaison aux règles sur les mêmes paires. Rend le
    rapport, le l2 retenu et les poids appris sur toutes les paires."""
    labels = [p.label for p in pairs]
    decided = np.array([lab != "=" for lab in labels])
    truth = np.array([lab == "A" for lab in labels])[decided].astype(int)
    folds = comprendre.metier_folds({p.metier for p in pairs}, n_folds)
    preds = []
    for l2 in L2_GRID:
        margin = np.zeros(len(pairs))
        for held in folds:
            train = [n for n, p in enumerate(pairs) if p.metier not in held]
            test = [n for n, p in enumerate(pairs) if p.metier in held]
            beta = fit_priority(diffs[train], [labels[n] for n in train], l2)
            margin[test] = diffs[test] @ beta
        preds.append((margin[decided] > 0).astype(int))
    rank, tried = _select(preds, truth)
    rules = np.array([rule_score(p.items[0], p.vp) - rule_score(p.items[1], p.vp) for p in pairs])
    rule_hits = np.where(rules[decided] == 0, 0.5, ((rules[decided] > 0) == truth).astype(float))
    hits = (preds[rank] == truth).astype(float)
    beta = fit_priority(diffs, labels, L2_GRID[rank])
    report = {
        "paires": len(pairs),
        "paires_decidees": int(decided.sum()),
        "equivalentes": int((~decided).sum()),
        "plis_metiers": folds,
        "essais": [{"l2": l2} | t for l2, t in zip(L2_GRID, tried, strict=True)],
        "l2_retenu": L2_GRID[rank],
        "accord_appris": round(float(hits.mean()), 4),
        "accord_appris_ic95": _ci(hits),
        "accord_regles": round(float(rule_hits.mean()), 4),
        "egalites_des_regles": int((rules[decided] == 0).sum()),
        "ecart_appris_regles": round(float((hits - rule_hits).mean()), 4),
        "ecart_appris_regles_ic95": _ci(hits - rule_hits),
        "par_sources": _by_sources(pairs, decided, hits, rule_hits),
        "poids": dict(
            sorted(
                ((n, round(float(w), 3)) for n, w in zip(PRIORITY_NAMES, beta, strict=True)),
                key=lambda x: -abs(x[1]),
            )
        ),
    }
    return report, L2_GRID[rank], beta


def _select(preds: list[np.ndarray], truth: np.ndarray) -> tuple[int, list[dict]]:
    """G2 puis G3 : le plus régularisé dont l'écart apparié au meilleur contient zéro."""
    scores = [_accuracy(p, truth) for p in preds]
    best = int(np.argmax(scores))
    tried = [{"accord": round(s, 4)} for s in scores]
    for rank in range(best):
        low, high = comprendre.paired_diff_ci(_accuracy, preds[best], preds[rank], truth)
        tried[rank]["ecart_au_meilleur_ic95"] = [round(low, 4), round(high, 4)]
        if low <= 0:
            return rank, tried
    return best, tried


def _by_sources(
    pairs: Sequence[Pair], decided: np.ndarray, hits: np.ndarray, rule_hits: np.ndarray
) -> dict:
    kinds = [
        " / ".join(sorted(c["source"] for c in p.cards))
        for p, d in zip(pairs, decided, strict=True)
        if d
    ]
    out = {}
    for kind in sorted(set(kinds)):
        m = np.array([k == kind for k in kinds])
        out[kind] = {
            "paires": int(m.sum()),
            "appris": round(float(hits[m].mean()), 3),
            "regles": round(float(rule_hits[m].mean()), 3),
        }
    return out


def labeled_pairs(corpus: Path = jeux_m5.CORPUS, series: Path = SERIES) -> list[Pair]:
    return [p for n in drawn_rounds(series) for p in round_pairs(n, corpus, series)]


def valider(
    understanding: Understanding | None = None,
    corpus: Path = jeux_m5.CORPUS,
    series: Path = SERIES,
    out_report: Path | None = REPORT_PATH,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """Bradley-Terry sur toutes les paires annotées, validé par métier, contre les règles.
    Mesure intermédiaire : elle sert à tirer les tours suivants, pas à décider (§11)."""
    understanding = understanding or Understanding(log=log)
    pairs = labeled_pairs(corpus, series)
    log(f"{len(pairs)} paires annotées, fiches de priorité…")
    report, _, _ = validate(pairs, pair_features(pairs, understanding), understanding.n_folds)
    report = {"tours": drawn_rounds(series)} | report
    if out_report:
        out_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    return report


def uncertain(understanding: Understanding, beta: np.ndarray, size: int = ROUND_SIZE) -> Chooser:
    """Le choix des tours 2 et 3 : les `size` paires du réservoir dont l'écart de score est le
    plus faible pour le modèle du moment (à écart égal, l'ordre du tirage)."""

    def choose(candidates: list[dict]) -> list[int]:
        cards = [c for p in candidates for c in (p["a"], p["b"])]
        metiers = [comprendre.metier(p["persona"]) for p in candidates for _ in range(2)]
        F = understanding.features(cards, metiers)
        margin = np.abs((F[0::2] - F[1::2]) @ beta)
        return [int(k) for k in np.argsort(margin, kind="stable")[:size]]

    return choose


def draw_uncertain_round(
    n: int,
    understanding: Understanding | None = None,
    corpus: Path = jeux_m5.CORPUS,
    series: Path = SERIES,
    size: int = ROUND_SIZE,
    pool: int = POOL,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """Tours 2 et 3 : le modèle appris sur les tours déjà annotés (l2 choisi par `validate`)
    choisit les paires où il hésite le plus."""
    understanding = understanding or Understanding(log=log)
    pairs = labeled_pairs(corpus, series)
    _, l2, beta = validate(pairs, pair_features(pairs, understanding), understanding.n_folds)
    log(f"modèle du moment : {len(pairs)} paires, l2 = {l2}")
    return draw_round(
        n, corpus, series, size, pool=pool, chooser=uncertain(understanding, beta, size)
    )


# --- étape 3 : boosting comparé, modèle figé, mesure finale ------------------------------------

BOOST_ROUNDS = 200  # recette fixée le 24/11, sans grille
BOOST_STEP = 0.1
BOOST_MARGIN = 0.03  # gain d'accord exigé du boosting (plan, §6.1)
MODEL_PATH = Path(__file__).with_name("prioriser_modele.json")
FINAL_REPORT_PATH = Path(__file__).with_name("prioriser_comparaison.json")
FROZEN_PAIRS = {"T-gen": jeux_m5.NAMES["paires"], "T-démo": jeux_m5.NAMES["demo"]}
ACCEPT_AGREEMENT, ACCEPT_GAP = 0.80, 0.15  # acceptation de M8 sur T-gen
ELEMENT_TASK, CASE_TASK = "M8, priorité des éléments", "M8, score d'affaire"


@dataclass(frozen=True)
class Stump:
    """Une souche : `left` si la valeur `feature` est ≤ `threshold`, sinon `right`."""

    feature: int
    threshold: float
    left: float
    right: float

    def __call__(self, F: np.ndarray) -> np.ndarray:
        return np.where(F[:, self.feature] <= self.threshold, self.left, self.right)


def _thresholds(column: np.ndarray) -> np.ndarray:
    """Seuils candidats : les déciles distincts de la valeur, sauf le maximum."""
    cuts = np.unique(np.percentile(column, np.arange(10, 100, 10)))
    return cuts[cuts < column.max()]


def _best_stump(F: np.ndarray, target: np.ndarray) -> Stump:
    """La souche des moindres carrés sur la cible (le gradient négatif)."""
    best, found = -1.0, Stump(0, np.inf, 0.0, 0.0)
    total, n = target.sum(), len(target)
    for j in range(F.shape[1]):
        for t in _thresholds(F[:, j]):
            left = F[:, j] <= t
            k = int(left.sum())
            s = float(target[left].sum())
            gain = s * s / k + (total - s) ** 2 / (n - k)
            if gain > best:
                best, found = gain, Stump(j, float(t), s / k, float((total - s) / (n - k)))
    return found


def fit_boosting(
    Fa: np.ndarray, Fb: np.ndarray, labels: Sequence[str], rounds: int = BOOST_ROUNDS
) -> list[Stump]:
    """Boosting par paires : le score d'un élément est une somme de souches, appris sur la
    perte logistique de l'écart des deux scores (une égalité = deux demi-comparaisons)."""
    n = len(Fa)
    F = np.vstack([Fa, Fb])
    ia, y, w = [], [], []  # la carte B de la paire k est la ligne n + k
    for k, label in enumerate(labels):
        rows = [(1.0, 0.5), (-1.0, 0.5)] if label == "=" else [(1.0 if label == "A" else -1.0, 1.0)]
        for sign, weight in rows:
            ia.append(k)
            y.append(sign)
            w.append(weight)
    ia_, y_, w_ = np.array(ia), np.array(y), np.array(w)
    ib_ = ia_ + n
    score = np.zeros(2 * n)
    stumps = []
    for _ in range(rounds):
        margin = score[ia_] - score[ib_]
        g = -y_ * w_ / (1 + np.exp(y_ * margin))  # dérivée de la perte par rapport à l'écart
        grad = np.zeros(2 * n)
        np.add.at(grad, ia_, g)
        np.add.at(grad, ib_, -g)
        stump = _best_stump(F, -grad)
        stumps.append(stump)
        score += BOOST_STEP * stump(F)
    return stumps


def boost_score(stumps: Sequence[Stump], F: np.ndarray) -> np.ndarray:
    return BOOST_STEP * sum((s(F) for s in stumps), np.zeros(len(F)))


def compare(pairs: Sequence[Pair], Fa: np.ndarray, Fb: np.ndarray, l2: float, n_folds: int) -> dict:
    """Boosting contre Bradley-Terry (l2 retenu), hors pli par métier, sur les paires
    décidées : le boosting ne remplace Bradley-Terry que s'il gagne au moins BOOST_MARGIN avec
    un intervalle apparié qui exclut zéro (règle du 24/11)."""
    labels = [p.label for p in pairs]
    decided = np.array([lab != "=" for lab in labels])
    truth = np.array([lab == "A" for lab in labels])[decided].astype(int)
    bt, boost = np.zeros(len(pairs)), np.zeros(len(pairs))
    for held in comprendre.metier_folds({p.metier for p in pairs}, n_folds):
        train = [n for n, p in enumerate(pairs) if p.metier not in held]
        test = [n for n, p in enumerate(pairs) if p.metier in held]
        sub = [labels[n] for n in train]
        beta = fit_priority(Fa[train] - Fb[train], sub, l2)
        bt[test] = (Fa[test] - Fb[test]) @ beta
        stumps = fit_boosting(Fa[train], Fb[train], sub)
        boost[test] = boost_score(stumps, Fa[test]) - boost_score(stumps, Fb[test])
    p_bt, p_boost = ((m[decided] > 0).astype(int) for m in (bt, boost))
    low, high = comprendre.paired_diff_ci(_accuracy, p_boost, p_bt, truth)
    gain = _accuracy(p_boost, truth) - _accuracy(p_bt, truth)
    wins = gain >= BOOST_MARGIN and low > 0
    return {
        "accord_bradley_terry": round(_accuracy(p_bt, truth), 4),
        "accord_boosting": round(_accuracy(p_boost, truth), 4),
        "ecart_boosting_bt": round(gain, 4),
        "ecart_boosting_bt_ic95": [round(low, 4), round(high, 4)],
        "retenu": "boosting" if wins else "bradley-terry",
    }


def figer(
    understanding: Understanding | None = None,
    corpus: Path = jeux_m5.CORPUS,
    series: Path = SERIES,
    model_path: Path = MODEL_PATH,
    report_path: Path | None = FINAL_REPORT_PATH,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """La validation finale (Bradley-Terry, règles, boosting comparé), puis le modèle retenu
    appris sur toutes les paires et écrit avec ses poids : c'est ce fichier qu'on mesure."""
    understanding = understanding or Understanding(log=log)
    pairs = labeled_pairs(corpus, series)
    Fa, Fb = side_features(pairs, understanding)
    log(f"{len(pairs)} paires : Bradley-Terry, règles…")
    report, l2, beta = validate(pairs, Fa - Fb, understanding.n_folds)
    log("boosting comparé…")
    report["boosting"] = compare(pairs, Fa, Fb, l2, understanding.n_folds)
    labels = [p.label for p in pairs]
    model: dict = {
        "noms": list(PRIORITY_NAMES),
        "paires": len(pairs),
        "tours": drawn_rounds(series),
        "candidat_n_sha256": comprendre._sha(understanding.candidate_path),
    }
    if report["boosting"]["retenu"] == "boosting":
        stumps = fit_boosting(Fa, Fb, labels)
        model |= {"modele": "boosting", "souches": [s.__dict__ for s in stumps]}
    else:
        model |= {"modele": "bradley-terry", "l2": l2, "poids": [float(x) for x in beta]}
    model_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    report["modele"] = model_path.name
    if report_path:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    return report


def model_score(model: dict, F: np.ndarray) -> np.ndarray:
    """Le score de priorité d'éléments, d'après le modèle figé."""
    if model["modele"] == "boosting":
        return boost_score([Stump(**s) for s in model["souches"]], F)
    return F @ np.array(model["poids"])


def frozen_pairs(name: str, series: Path = SERIES, corpus: Path = jeux_m5.CORPUS) -> list[Pair]:
    """Les paires jugées par Théo d'un jeu gelé, avec leurs éléments et leur brief (retrouvés
    par le contexte affiché). Le tirage ne donne que les éléments, jamais une étiquette."""
    info = json.loads((series / jeux_m5.TIRAGE.name).read_text(encoding="utf-8"))
    labels = final_labels(name, series)
    tasks = [t for t in load_series(series / f"{name}.jsonl") if t["task_id"] in labels]
    if name == jeux_m5.NAMES["demo"]:
        behind = info["tdemo"]["paires"]
        items = jeux_m5.demo_items()
        people = jeux_m5.demo_people(items.values())
        days = {
            date.fromisoformat(json.loads(x)["day"])
            for x in jeux_m5.DEMO_PAIRS.read_text(encoding="utf-8").splitlines()
            if x
        }
        views = {(vp := jeux_m5.demo_viewpoint(d, people)).context: vp for d in days}
        found = {c: ("demo", vp, items) for c, vp in views.items()}
    else:
        behind = info["tgen"]["paires"]
        found = {
            b[2].context: (b[0], b[2], {i.item_id: i for i in b[3]})
            for b in jeux_m5.briefs(corpus, "tgen")
        }
    out = []
    for task in tasks:
        key, vp, by_id = found[task["context"]]
        a, b = (by_id[i] for i in behind[task["task_id"]])
        label = labels[task["task_id"]]["choix"]
        out.append(Pair(task["task_id"], key, vp, (task["a"], task["b"]), (a, b), label))
    return out


def measure_pairs(pairs: Sequence[Pair], model: dict, understanding: Understanding) -> dict:
    """Accord du modèle et des règles sur les paires départagées par l'annotateur."""
    Fa, Fb = side_features(pairs, understanding, frozen=True)
    margin = model_score(model, Fa) - model_score(model, Fb)
    decided = np.array([p.label != "=" for p in pairs])
    truth = np.array([p.label == "A" for p in pairs])[decided]
    hits = ((margin[decided] > 0) == truth).astype(float)
    rules = np.array([rule_score(p.items[0], p.vp) - rule_score(p.items[1], p.vp) for p in pairs])
    rule_hits = np.where(rules[decided] == 0, 0.5, ((rules[decided] > 0) == truth).astype(float))
    return {
        "paires": len(pairs),
        "paires_decidees": int(decided.sum()),
        "accord_appris": round(float(hits.mean()), 4),
        "accord_appris_ic95": _ci(hits),
        "accord_regles": round(float(rule_hits.mean()), 4),
        "egalites_des_regles": int((rules[decided] == 0).sum()),
        "ecart_appris_regles": round(float((hits - rule_hits).mean()), 4),
        "ecart_appris_regles_ic95": _ci(hits - rule_hits),
        "par_sources": _by_sources(pairs, decided, hits, rule_hits),
    }


def _refuse_reading(journal: Path, model_sha: str, task: str) -> None:
    """Un jeu gelé ne se lit qu'une fois par modèle et par tâche (G1) : le même modèle figé
    peut être mesuré sur les paires d'éléments, puis sur les paires d'affaires."""
    if not journal.exists():
        return
    for line in journal.read_text(encoding="utf-8").splitlines():
        if line:
            entry = json.loads(line)
            if entry["modele_sha256"] == model_sha and entry["tache"] == task:
                raise ValueError(f"ce modèle a déjà été mesuré ({task}) : voir le journal")


def mesurer(
    understanding: Understanding | None = None,
    series: Path = SERIES,
    corpus: Path = jeux_m5.CORPUS,
    model_path: Path = MODEL_PATH,
    journal: Path = comprendre.JOURNAL,
    commit: Callable[[], str] | None = None,
    sets: dict[str, str] = FROZEN_PAIRS,
    freezes: Sequence[str] = comprendre.FREEZES,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """La mesure finale de M8, une seule fois par modèle (G1), ajoutée au journal des jeux
    gelés ; refusée si un gel manque ou a changé, si le modèle est déjà au journal, ou si le
    code et le modèle ne sont pas commités."""
    comprendre.check_frozen(series, freezes)
    model_sha = comprendre._sha(model_path)
    _refuse_reading(journal, model_sha, ELEMENT_TASK)
    revision = (commit or (lambda: comprendre.git_commit((model_path, Path(__file__)))))()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    understanding = understanding or Understanding(log=log)
    results = {}
    for label, name in sets.items():
        log(f"{label} ({name})…")
        results[label] = {"serie": name} | measure_pairs(
            frozen_pairs(name, series, corpus), model, understanding
        )
    gen = results.get("T-gen")
    entry = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "tache": ELEMENT_TASK,
        "commit": revision,
        "modele": model_path.name,
        "modele_sha256": model_sha,
        "type": model["modele"],
        "jeux": results,
    }
    if gen:
        entry["acceptation"] = {
            "accord_min": ACCEPT_AGREEMENT,
            "ecart_regles_min": ACCEPT_GAP,
            "atteinte": gen["accord_appris"] >= ACCEPT_AGREEMENT
            and gen["ecart_appris_regles"] >= ACCEPT_GAP,
        }
    with journal.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


# --- échelle 0-100 du brief ----------------------------------------------------------------------

SCALE_PATH = Path(__file__).with_name("prioriser_echelle.json")


def echelle(
    corpus: Path = jeux_m5.CORPUS,
    series: Path = SERIES,
    embedder: Embedder | None = None,
    cache: VectorCache | None = None,
    model_path: Path = MODEL_PATH,
    candidate_path: Path = comprendre.CANDIDATE_PATH,
    out: Path = SCALE_PATH,
) -> dict:
    """L'échelle 0-100 du brief : les centiles du score brut sur les cartes des paires
    d'entraînement, fiches du N figé (comme au brief). Un score de 70 veut dire « passe avant
    70 % des éléments d'entraînement » ; 50 est l'élément médian. Le score brut, lui, sépare
    trop pour être lu tel quel (σ du score : 3 % au 10e centile, 100 % au 90e)."""
    reference = comprendre.Reference.load(candidate_path)
    embedder = embedder or FastEmbedder(reference.embedder_name)
    cache = cache or VectorCache(cache_path(embedder))
    cards = [c for p in labeled_pairs(corpus, series) for c in p.cards]
    X = comprendre.design(cards, embedder, reference.fiche, cache, reference.texte)
    signals = np.array([card_signals(c) for c in cards])
    F = np.hstack([signals, n_columns(reference.models, X)])
    model = json.loads(model_path.read_text(encoding="utf-8"))
    raw = model_score(model, F)
    scale = {
        "cartes": len(cards),
        "modele_sha256": comprendre._sha(model_path),
        "centiles": [round(float(v), 6) for v in np.percentile(raw, np.arange(101))],
    }
    out.write_text(json.dumps(scale, indent=1) + "\n", encoding="utf-8", newline="\n")
    return scale


def to_100(raw: np.ndarray, centiles: Sequence[float]) -> np.ndarray:
    """Le score brut placé sur l'échelle : son rang centile parmi les cartes d'entraînement."""
    values = np.asarray(centiles)
    lows = np.searchsorted(values, raw, side="left")
    highs = np.searchsorted(values, raw, side="right")
    return np.clip((lows + highs) / 2 - 0.5, 0, 100)  # à égalité, le milieu du palier


# --- score d'affaire ---------------------------------------------------------------------------


@dataclass
class CasePair:
    """Une paire d'affaires jugée : les éléments visibles de chaque affaire et leurs cartes."""

    task_id: str
    vp: Viewpoint
    cards: tuple[list[dict], list[dict]]
    items: tuple[list[Item], list[Item]]
    label: str


def frozen_case_pairs(
    name: str = jeux_m5.NAMES["affaires"], series: Path = SERIES, corpus: Path = jeux_m5.CORPUS
) -> list[CasePair]:
    """Les paires d'affaires jugées par Théo : chaque affaire a les éléments de sa carte (le
    tirage les donne, jamais une étiquette), mis en carte d'élément au moment du brief."""
    info = json.loads((series / jeux_m5.TIRAGE.name).read_text(encoding="utf-8"))
    behind = info["tgen"]["affaires"]
    labels = final_labels(name, series)
    briefs = {b[2].context: b for b in jeux_m5.briefs(corpus, "tgen")}
    out = []
    for task in load_series(series / f"{name}.jsonl"):
        if task["task_id"] not in labels:
            continue
        _key, _day, vp, _shown, known = briefs[task["context"]]
        by_id = {i.item_id: i for i in known}
        sides = tuple([by_id[i] for i in ids] for ids in behind[task["task_id"]])
        cards = tuple([card(i, vp, known) for i in side] for side in sides)
        label = labels[task["task_id"]]["choix"]
        out.append(CasePair(task["task_id"], vp, cards, sides, label))  # type: ignore[arg-type]
    return out


def case_scores(pairs: Sequence[CasePair], model: dict, understanding: Understanding) -> np.ndarray:
    """Score de chaque affaire (colonnes A, B) : le plus haut score de ses éléments."""
    cards = [c for p in pairs for side in p.cards for c in side]
    scores = iter(model_score(model, understanding.features(cards, [FROZEN] * len(cards))))
    return np.array([[max(next(scores) for _ in side) for side in p.cards] for p in pairs])


def measure_cases(pairs: Sequence[CasePair], model: dict, understanding: Understanding) -> dict:
    """Accord du score d'affaire et des règles (affaire entière) sur les paires départagées."""
    s = case_scores(pairs, model, understanding)
    margin = s[:, 0] - s[:, 1]
    decided = np.array([p.label != "=" for p in pairs])
    truth = np.array([p.label == "A" for p in pairs])[decided]
    hits = ((margin[decided] > 0) == truth).astype(float)

    def rules(items: list[Item], vp: Viewpoint, side: str, task_id: str) -> int:
        return score_case(Case(f"{task_id}-{side}", tuple(items)), vp.moment, config_for(vp))[0]

    diff = np.array(
        [
            rules(p.items[0], p.vp, "a", p.task_id) - rules(p.items[1], p.vp, "b", p.task_id)
            for p in pairs
        ]
    )
    rule_hits = np.where(diff[decided] == 0, 0.5, ((diff[decided] > 0) == truth).astype(float))
    return {
        "paires": len(pairs),
        "paires_decidees": int(decided.sum()),
        "elements_par_affaire": round(float(np.mean([len(i) for p in pairs for i in p.items])), 2),
        "accord_appris": round(float(hits.mean()), 4),
        "accord_appris_ic95": _ci(hits),
        "accord_regles": round(float(rule_hits.mean()), 4),
        "egalites_des_regles": int((diff[decided] == 0).sum()),
        "ecart_appris_regles": round(float((hits - rule_hits).mean()), 4),
        "ecart_appris_regles_ic95": _ci(hits - rule_hits),
    }


def mesurer_affaires(
    understanding: Understanding | None = None,
    series: Path = SERIES,
    corpus: Path = jeux_m5.CORPUS,
    model_path: Path = MODEL_PATH,
    journal: Path = comprendre.JOURNAL,
    commit: Callable[[], str] | None = None,
    freezes: Sequence[str] = comprendre.FREEZES,
    pairs: Sequence[CasePair] | None = None,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """La mesure unique du score d'affaire sur les paires d'affaires de T-gen (règle du 24/11) :
    le score appris reste le défaut du brief sauf écart aux règles significativement négatif."""
    comprendre.check_frozen(series, freezes)
    model_sha = comprendre._sha(model_path)
    _refuse_reading(journal, model_sha, CASE_TASK)
    revision = (commit or (lambda: comprendre.git_commit((model_path, Path(__file__)))))()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    understanding = understanding or Understanding(log=log)
    log("paires d'affaires de T-gen…")
    result = measure_cases(
        pairs if pairs is not None else frozen_case_pairs(series=series, corpus=corpus),
        model,
        understanding,
    )
    entry = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "tache": CASE_TASK,
        "commit": revision,
        "modele": model_path.name,
        "modele_sha256": model_sha,
        "type": model["modele"],
        "jeux": {"T-gen": {"serie": jeux_m5.NAMES["affaires"]} | result},
        "decision": (
            "règles par défaut"
            if result["ecart_appris_regles_ic95"][1] < 0
            else "score appris par défaut"
        ),
    }
    with journal.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


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
    if argv == ["--valider"]:
        report = valider(log=lambda line: print(line, flush=True))
        print(json.dumps({k: v for k, v in report.items() if k != "poids"}, ensure_ascii=False))
        print("poids les plus forts :", dict(list(report["poids"].items())[:12]))
        print(f"→ {REPORT_PATH.name}")
        return 0
    if argv == ["--figer"]:
        report = figer(log=lambda line: print(line, flush=True))
        print(json.dumps({k: v for k, v in report.items() if k != "poids"}, ensure_ascii=False))
        print(f"→ {MODEL_PATH.name}, {FINAL_REPORT_PATH.name}")
        return 0
    if argv == ["--mesurer"]:
        entry = mesurer(log=lambda line: print(line, flush=True))
        print(json.dumps(entry, ensure_ascii=False, indent=1))
        print(f"→ {comprendre.JOURNAL.name}")
        return 0
    if argv == ["--echelle"]:
        scale = echelle()
        print(f"{scale['cartes']} cartes ; médiane {scale['centiles'][50]} → {SCALE_PATH.name}")
        return 0
    if argv == ["--mesurer-affaires"]:
        entry = mesurer_affaires(log=lambda line: print(line, flush=True))
        print(json.dumps(entry, ensure_ascii=False, indent=1))
        print(f"→ {comprendre.JOURNAL.name}")
        return 0
    if len(argv) == 2 and argv[0] == "--tirer-incertain":
        info = draw_uncertain_round(int(argv[1]), log=lambda line: print(line, flush=True))
        print(f"{round_name(info['tour'])} : {len(info['paires'])} paires ({info['selection']})")
        return 0
    print(
        "usage : prioriser --tirer <tour> | --tirer-incertain <tour> | --montrer <tour> "
        "[début fin] | --reste | --valider | --figer | --mesurer | --mesurer-affaires | --echelle"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
