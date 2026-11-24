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
    ) -> None:
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
        """La fiche de priorité de chaque carte (ordre de PRIORITY_NAMES), hors pli."""
        X = self.design(cards)
        out = np.zeros((len(cards), len(PRIORITY_NAMES)))
        for m in set(metiers):
            rows = [n for n, x in enumerate(metiers) if x == m]
            signals = np.array([card_signals(cards[n]) for n in rows])
            out[rows] = np.hstack([signals, n_columns(self.by_metier[m], X[rows])])
        return out


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


def pair_features(pairs: Sequence[Pair], understanding: Understanding) -> np.ndarray:
    """Différences de fiches A − B, une ligne par paire."""
    cards = [c for p in pairs for c in p.cards]
    metiers = [p.metier for p in pairs for _ in range(2)]
    F = understanding.features(cards, metiers)
    return F[0::2] - F[1::2]


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
    if len(argv) == 2 and argv[0] == "--tirer-incertain":
        info = draw_uncertain_round(int(argv[1]), log=lambda line: print(line, flush=True))
        print(f"{round_name(info['tour'])} : {len(info['paires'])} paires ({info['selection']})")
        return 0
    print(
        "usage : prioriser --tirer <tour> | --tirer-incertain <tour> | --montrer <tour> "
        "[début fin] | --reste | --valider"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
