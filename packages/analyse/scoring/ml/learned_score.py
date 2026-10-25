"""Score de priorité appris (pipeline A) : un poids par caractéristique de la fiche.

Modèle de Bradley-Terry sur les comparaisons de Théo : P(A avant B) = σ(u(A) − u(B)), avec
u = poids · fiche (`features.vectorize`). C'est une régression logistique sur la différence
des fiches, sans biais (échanger A et B inverse la prédiction) ; une égalité compte pour une
demi-victoire de chaque côté. Les poids sont l'explication : les raisons d'un score sont
ses plus fortes contributions.

Entraînement : paires de David (Théo, avec pastilles) et des journées générées (Théo et Paul,
à l'aveugle ; accord, sinon arbitrage, sinon l'unique annotateur). Mesure principale : les 80
paires réservées des journées générées, jugées à l'aveugle : des paires jugées en voyant les
pastilles d'un modèle avantagent ce modèle. En complément : paires réservées de David,
transfert vers la démo du lot 1 (jamais apprise), score par règles (`score.py`).

`uv run python -m packages.analyse.scoring.ml.learned_score` : mesure, puis entraîne sur
toutes les paires et écrit `learned_score_weights.json`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from packages.analyse.scoring.ml.annotate import (
    DEMO_PAIRS_PATH,
    PAIR_LABELS_PATH,
    PAIRS_PATH,
    SYNTH_PAIRS_PATH,
    PairAnnotator,
)
from packages.analyse.scoring.ml.features import VECTOR_NAMES, MailFeatures, vectorize
from packages.analyse.scoring.ml.sample import load_sample

WEIGHTS_PATH = Path(__file__).with_name("learned_score_weights.json")
L2 = 1.0
MAX_REASONS, MIN_CONTRIBUTION = 3, 0.25
NO_SIGNAL = "aucun signal fort"


def load_pair_labels(path: Path = PAIR_LABELS_PATH) -> tuple[dict[str, str], dict[str, str]]:
    """({paire: réponse}, {paire: réponse à la répétition}) ; la dernière réponse compte."""
    first: dict[str, str] = {}
    repeats: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            if row["feature"] == PairAnnotator.FEATURE:
                (repeats if row["repeat"] else first)[row["item_id"]] = row["value"]
    return first, repeats


def merged_pair_labels(folder: Path) -> dict[str, str]:
    """Paires annotées à deux (`pair_labels_theo|paul.jsonl`) : la réponse commune, sinon
    l'arbitrage (`pair_labels_arbitrage.jsonl`), sinon celle de l'unique annotateur ; une
    paire en désaccord non arbitrée est écartée."""

    def first(name: str) -> dict[str, str]:
        path = folder / f"pair_labels_{name}.jsonl"
        return load_pair_labels(path)[0] if path.exists() else {}

    theo, paul, arbitration = first("theo"), first("paul"), first("arbitrage")
    out = {}
    for pair_id in sorted(set(theo) | set(paul)):
        given = {a[pair_id] for a in (theo, paul) if pair_id in a}
        value = arbitration.get(pair_id) if len(given) > 1 else next(iter(given))
        if value:
            out[pair_id] = value
    return out


def benchmarks(
    pairs: dict[str, dict], labels: dict[str, str], groups: dict[str, list[str]]
) -> dict[str, dict]:
    """Un seul modèle, appris sur toutes les paires jugées non réservées, mesuré sur chaque
    groupe de paires réservées ; comparé au score par règles, intervalle de confiance."""
    train_ids = sorted(i for i in labels if i in pairs and not pairs[i].get("heldout"))
    beta = fit_bradley_terry(*comparisons(pairs, labels, train_ids))
    out = {}
    for name, ids in groups.items():
        ids = sorted(i for i in ids if i in labels)
        learned, rules = _learned_hits(beta, pairs, labels, ids), _rule_hits(pairs, labels, ids)
        boot = np.random.default_rng(0).integers(0, len(learned), (10_000, len(learned)))
        out[name] = {
            "paires jugées (non équivalentes)": len(learned),
            "accord appris": float(learned.mean()),
            "IC 95 %": [float(x) for x in np.percentile(learned[boot].mean(1), [2.5, 97.5])],
            "accord règles": float(rules.mean()),
            "égalités du score par règles": int((rules == 0.5).sum()),
        }
    return out


def card_vector(side: dict) -> np.ndarray:
    return vectorize(MailFeatures(**side["features"]))


def comparisons(
    pairs: dict[str, dict], labels: dict[str, str], ids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(différences de fiches A − B, A d'abord ?, poids) ; une égalité donne deux demi-lignes."""
    rows, wins, weights = [], [], []
    for pair_id in ids:
        diff = card_vector(pairs[pair_id]["a"]) - card_vector(pairs[pair_id]["b"])
        if labels[pair_id] == "=":
            rows += [diff, diff]
            wins += [1.0, 0.0]
            weights += [0.5, 0.5]
        else:
            rows.append(diff)
            wins.append(float(labels[pair_id] == "A"))
            weights.append(1.0)
    return np.array(rows), np.array(wins), np.array(weights)


def fit_bradley_terry(
    diffs: np.ndarray, wins: np.ndarray, weights: np.ndarray, l2: float = L2, steps: int = 4000
) -> np.ndarray:
    """Poids par descente de gradient à pas fixe depuis zéro : déterministe."""
    beta = np.zeros(diffs.shape[1])
    total = weights.sum()
    for _ in range(steps):
        p = 1 / (1 + np.exp(-diffs @ beta))
        beta -= 0.5 * (diffs.T @ ((p - wins) * weights) / total + l2 * beta / total)
    return beta


def _learned_hits(beta: np.ndarray, pairs: dict, labels: dict, ids: list[str]) -> np.ndarray:
    """1 si l'ordre appris est celui de Théo, par paire non équivalente."""
    return np.array(
        [
            float((card_vector(pairs[i]["a"]) - card_vector(pairs[i]["b"])) @ beta > 0)
            == float(labels[i] == "A")
            for i in ids
            if labels[i] != "="
        ]
    )


def _rule_hits(pairs: dict, labels: dict, ids: list[str]) -> np.ndarray:
    """Idem pour le score par règles ; s'il ne départage pas (égalité), pile ou face : 0,5."""
    hits = []
    for i in ids:
        if labels[i] == "=":
            continue
        a, b = pairs[i]["a"]["rule_score"], pairs[i]["b"]["rule_score"]
        hits.append(0.5 if a == b else float((a > b) == (labels[i] == "A")))
    return np.array(hits)


def evaluate(pairs: dict[str, dict], labels: dict[str, str], repeats: dict[str, str]) -> dict:
    """La mesure publiée : entraînement sans les paires réservées, accord sur celles-ci.
    Seules comptent les paires présentes dans `pairs` (les données de David)."""
    labels = {i: v for i, v in labels.items() if i in pairs}
    repeats = {i: v for i, v in repeats.items() if i in pairs}
    train = sorted(i for i in labels if not pairs[i].get("heldout"))
    test = sorted(i for i in labels if pairs[i].get("heldout"))
    beta = fit_bradley_terry(*comparisons(pairs, labels, train))
    learned, rules = _learned_hits(beta, pairs, labels, test), _rule_hits(pairs, labels, test)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(learned), (10_000, len(learned)))
    loo = []
    for k in train:  # validation « une paire laissée de côté » sur l'entraînement
        if labels[k] != "=":
            rest = [i for i in train if i != k]
            loo.append(
                _learned_hits(
                    fit_bradley_terry(*comparisons(pairs, labels, rest)), pairs, labels, [k]
                )[0]
            )
    return {
        "paires réservées": len(learned),
        "accord appris": float(learned.mean()),
        "IC 95 %": [float(x) for x in np.percentile(learned[boot].mean(1), [2.5, 97.5])],
        "accord règles": float(rules.mean()),
        "égalités du score par règles": int((rules == 0.5).sum()),
        "écart appris − règles, IC 95 %": [
            float(x) for x in np.percentile((learned[boot] - rules[boot]).mean(1), [2.5, 97.5])
        ],
        "accord appris, validation croisée sur l'entraînement": float(np.mean(loo)),
        "cohérence de l'annotateur": float(np.mean([labels[i] == v for i, v in repeats.items()])),
    }


@dataclass(frozen=True)
class LearnedScore:
    """Score 0-100 d'un mail : 50 pour un mail médian, raisons = plus fortes contributions."""

    weights: dict[str, float]
    offset: float  # utilité médiane des mails des paires : elle vaut 50

    def utility(self, f: MailFeatures) -> float:
        return float(np.dot([self.weights[n] for n in VECTOR_NAMES], vectorize(f)))

    def score(self, f: MailFeatures) -> tuple[int, list[str]]:
        contributions = sorted(
            zip(
                VECTOR_NAMES,
                np.array([self.weights[n] for n in VECTOR_NAMES]) * vectorize(f),
                strict=True,
            ),
            key=lambda nc: (-nc[1], nc[0]),
        )
        reasons = [n for n, c in contributions if c >= MIN_CONTRIBUTION][:MAX_REASONS]
        value = 100 / (1 + np.exp(-(self.utility(f) - self.offset)))
        return round(float(value)), reasons or [NO_SIGNAL]

    def save(self, path: Path = WEIGHTS_PATH) -> None:
        data = {"poids": self.weights, "décalage": self.offset}
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )

    @classmethod
    def load(cls, path: Path = WEIGHTS_PATH) -> LearnedScore:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["poids"], data["décalage"])


def transfer(model: LearnedScore, pairs: dict[str, dict], labels: dict[str, str]) -> dict:
    """Test de transfert : le modèle appris sur David, jugé sur d'autres données (démo)."""
    ids = sorted(i for i in labels if i in pairs)
    beta = np.array([model.weights[n] for n in VECTOR_NAMES])
    learned, rules = _learned_hits(beta, pairs, labels, ids), _rule_hits(pairs, labels, ids)
    boot = np.random.default_rng(0).integers(0, len(learned), (10_000, len(learned)))
    return {
        "paires jugées (non équivalentes)": len(learned),
        "accord appris": float(learned.mean()),
        "IC 95 %": [float(x) for x in np.percentile(learned[boot].mean(1), [2.5, 97.5])],
        "accord règles": float(rules.mean()),
        "égalités du score par règles": int((rules == 0.5).sum()),
        "écart appris − règles, IC 95 %": [
            float(x) for x in np.percentile((learned[boot] - rules[boot]).mean(1), [2.5, 97.5])
        ],
    }


def train(pairs: dict[str, dict], labels: dict[str, str]) -> LearnedScore:
    """Modèle final : toutes les paires jugées (la mesure a été faite avant, sans les réservées)."""
    ids = sorted(i for i in labels if i in pairs)
    beta = fit_bradley_terry(*comparisons(pairs, labels, ids))
    cards = {s["item_id"]: card_vector(s) for p in pairs.values() for s in (p["a"], p["b"])}
    offset = float(np.median([v @ beta for v in cards.values()]))
    return LearnedScore(dict(zip(VECTOR_NAMES, map(float, beta), strict=True)), offset)


def main(argv: list[str]) -> int:
    david = {p["item_id"]: p for p in load_sample(PAIRS_PATH)}
    synth = (
        {p["item_id"]: p for p in load_sample(SYNTH_PAIRS_PATH)}
        if SYNTH_PAIRS_PATH.exists()
        else {}
    )
    labels, _ = load_pair_labels()
    labels |= merged_pair_labels(PAIR_LABELS_PATH.parent)
    pairs = david | synth
    groups = {
        "JUGE · paires réservées des journées générées (à l'aveugle)": [
            i for i, p in synth.items() if p.get("heldout")
        ],
        "paires réservées de David (avec pastilles)": [
            i for i, p in david.items() if p.get("heldout")
        ],
    }
    for name, report in benchmarks(pairs, labels, groups).items():
        print(f"\n{name}")
        _print(report)
    model = train(pairs, labels)
    model.save()
    print("\nPoids (modèle final, toutes les paires de David et des journées générées) :")
    for name, w in sorted(model.weights.items(), key=lambda nw: -nw[1]):
        print(f"  {w:+.2f}  {name}")
    if DEMO_PAIRS_PATH.exists():
        demo = {p["item_id"]: p for p in load_sample(DEMO_PAIRS_PATH)}
        if any(i in demo for i in labels):
            print("\nTransfert vers la démo du lot 1 (jamais apprise) :")
            _print(transfer(model, demo, labels))
    return 0


def _print(report: dict) -> None:
    for key, value in report.items():
        if isinstance(value, list):
            sign = "+" if "écart" in key else ""
            shown = f"[{value[0]:{sign}.2f} ; {value[1]:{sign}.2f}]"
        else:
            shown = f"{value:.2f}" if isinstance(value, float) else str(value)
        print(f"{key:<52} {shown}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
