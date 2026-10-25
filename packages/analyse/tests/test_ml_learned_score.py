"""Tests du score appris (Bradley-Terry sur les comparaisons de paires)."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from packages.analyse.scoring.ml.features import VECTOR_NAMES, MailFeatures
from packages.analyse.scoring.ml.learned_score import (
    NO_SIGNAL,
    LearnedScore,
    comparisons,
    evaluate,
    fit_bradley_terry,
    load_pair_labels,
    train,
    transfer,
)


def _features(n: int, action: float, kind: str = "collègue", deadline: str = "aucune") -> dict:
    return MailFeatures(
        item_id=f"mail:{n}",
        sender_kind=kind,
        action=action,
        reponse=0.1,
        incident=0.1,
        alerte=False,
        deadline=deadline,
        deadline_phrase=None,
        ball_in_court=True,
        thread_size=1,
        hierarchy_in_thread=False,
        fresh=True,
        meeting_soon=False,
    ).as_dict()


def _side(n, action, rule, **kw):
    return {"item_id": f"mail:{n}", "features": _features(n, action, **kw), "rule_score": rule}


def _pairs(n: int = 60):
    """Théo préfère toujours le mail qui demande une action ; les règles ne départagent pas."""
    rng = random.Random(0)
    pairs, labels = {}, {}
    for i in range(n):
        ask, info = _side(2 * i, 0.9, 50), _side(2 * i + 1, 0.1, 50)
        a, b, value = (ask, info, "A") if rng.random() < 0.5 else (info, ask, "B")
        pairs[f"p{i}"] = {"item_id": f"p{i}", "heldout": i % 5 == 0, "a": a, "b": b}
        labels[f"p{i}"] = value
    return pairs, labels


def test_egalite_compte_pour_deux_demi_victoires():
    pairs, labels = _pairs(3)
    labels["p1"] = "="
    diffs, wins, weights = comparisons(pairs, labels, ["p0", "p1", "p2"])
    assert len(diffs) == 4 and weights.tolist().count(0.5) == 2
    assert wins[1] + wins[2] == 1.0


def test_poids_symetriques_et_deterministes():
    pairs, labels = _pairs()
    data = comparisons(pairs, labels, sorted(labels))
    beta = fit_bradley_terry(*data)
    np.testing.assert_array_equal(beta, fit_bradley_terry(*data))
    flipped = fit_bradley_terry(-data[0], 1 - data[1], data[2])
    np.testing.assert_allclose(flipped, beta, atol=1e-9)  # échanger A et B ne change rien
    assert beta[VECTOR_NAMES.index("demande une action")] > 1


def test_mesure_sur_les_paires_reservees_contre_les_regles():
    pairs, labels = _pairs()
    report = evaluate(pairs, labels, repeats={"p0": labels["p0"], "p1": "="})
    assert report["paires réservées"] == 12
    assert report["accord appris"] == 1.0
    assert report["accord règles"] == 0.5 and report["égalités du score par règles"] == 12
    assert report["cohérence de l'annotateur"] == 0.5
    low, high = report["écart appris − règles, IC 95 %"]
    assert low == high == pytest.approx(0.5)


def test_score_0_100_raisons_et_enregistrement(tmp_path):
    pairs, labels = _pairs()
    model = train(pairs, labels)
    ask = MailFeatures(**_features(1, 0.95, kind="hiérarchie", deadline="aujourd'hui ou demain"))
    info = MailFeatures(**_features(2, 0.05))
    (s_ask, why_ask), (s_info, why_info) = model.score(ask), model.score(info)
    assert 0 <= s_info < 50 < s_ask <= 100
    assert why_ask[0] == "demande une action" and len(why_ask) <= 3
    assert why_info == [NO_SIGNAL]
    path = tmp_path / "w.json"
    model.save(path)
    assert LearnedScore.load(path) == model
    assert json.loads(path.read_text(encoding="utf-8"))["poids"]["demande une action"] > 1


def test_etiquettes_de_paires_derniere_reponse(tmp_path):
    path = tmp_path / "pl.jsonl"
    rows = [
        {"item_id": "p0", "feature": "paire", "value": "A", "repeat": False},
        {"item_id": "p0", "feature": "paire", "value": "B", "repeat": False},
        {"item_id": "p0", "feature": "paire", "value": "B", "repeat": True},
        {"item_id": "m0", "feature": "action", "value": "oui", "repeat": False},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert load_pair_labels(path) == ({"p0": "B"}, {"p0": "B"})


def test_transfert_juge_le_modele_sur_d_autres_paires_sans_les_apprendre():
    pairs, labels = _pairs()
    model = train(pairs, labels)
    other, other_labels = _pairs(10)
    other = {f"d{k}": v | {"item_id": f"d{k}"} for k, v in other.items()}
    other_labels = {f"d{k}": v for k, v in other_labels.items()}
    report = transfer(model, other, labels | other_labels)
    assert report["paires jugées (non équivalentes)"] == 10
    assert report["accord appris"] == 1.0 and report["accord règles"] == 0.5
    # les paires de démo ne changent ni la mesure ni le modèle de David
    repeats = {"p0": labels["p0"], "d0": "="}
    assert evaluate(pairs, labels | other_labels, repeats) == evaluate(pairs, labels, repeats)
    assert train(pairs, labels | other_labels) == model
