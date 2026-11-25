"""M8, étape 1 : paires d'entraînement tirées sur les personas « train », annotées par Claude."""

from __future__ import annotations

import json

import numpy as np
import pytest

from packages.analyse.rag.embeddings import HashEmbedder
from packages.analyse.scoring.ml import annoter, comprendre, jeux_m5, prioriser
from packages.analyse.scoring.ml.intents import VectorCache
from packages.collect.trainset import build
from packages.collect.trainset.world import all_personas


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Trois personas « train » de trois métiers, et une persona réservée, par gabarits."""
    root = tmp_path_factory.mktemp("corpus")
    by_metier = {}
    for p in all_personas():
        if p.split == "train":
            by_metier.setdefault(comprendre.metier(p.key), p)
    train = list(by_metier.values())[:3]
    tgen = [p for p in all_personas() if p.split == "tgen"][:1]
    personas = {}
    for p in (*train, *tgen):
        items, scenario = build.build_persona(p)
        for sub, rows in (
            ("items", [i.model_dump(mode="json") for i in items]),
            ("scenario", scenario),
        ):
            (root / sub).mkdir(exist_ok=True)
            (root / sub / f"{p.key}.jsonl").write_text(build._jsonl(rows), encoding="utf-8")
        personas[p.key] = {"split": p.split}
    (root / "manifest.json").write_text(json.dumps({"personas": personas}), encoding="utf-8")
    return root, {p.key for p in train}


@pytest.fixture
def series(corpus, tmp_path):
    """Un faux tirage de M5 dont un élément « train » est gelé : il ne doit jamais sortir."""
    root, train = corpus
    key = min(train)
    frozen = json.loads((root / "items" / f"{key}.jsonl").read_text("utf-8").splitlines()[0])
    tirage = {
        "tgen": {"elements": {"t-e001": frozen["item_id"]}, "paires": {}, "affaires": {}},
        "tdemo": {"paires": {}},
    }
    (tmp_path / jeux_m5.TIRAGE.name).write_text(json.dumps(tirage), encoding="utf-8")
    return tmp_path, frozen["item_id"]


def test_un_tour_tire_des_paires_d_entrainement_a_l_aveugle(corpus, series):
    root, train = corpus
    folder, frozen = series
    info = prioriser.draw_round(1, root, folder, size=20)
    assert set(info["personas"]) == train and len(info["paires"]) == 20
    tasks = annoter.load_series(folder / "m8-paires-t1.jsonl")  # check_blind
    assert all(t["pour"] == ["claude"] and t["kind"] == "paire" for t in tasks)
    ids = {i for pair in info["paires"].values() for i in pair}
    assert frozen not in ids
    owners = {t["a"]["item_id"].split(":")[0] for t in tasks}
    assert owners  # des cartes de vraies sources
    with pytest.raises(FileExistsError):
        prioriser.draw_round(1, root, folder, size=20)
    second = prioriser.draw_round(2, root, folder, size=20)
    first = {frozenset(p) for p in info["paires"].values()}
    assert not first & {frozenset(p) for p in second["paires"].values()}
    assert prioriser.drawn_rounds(folder) == [1, 2]


def test_les_reponses_suivent_les_cotes_montres(corpus, series):
    root, _ = corpus
    folder, _ = series
    prioriser.draw_round(1, root, folder, size=20)
    shown = prioriser.presentations(1, folder)
    repeats = [(t, task) for t, rep, task in shown if rep]
    assert len(shown) == 22 and len(repeats) == 2
    tasks = {t["task_id"]: t for t in annoter.load_series(folder / "m8-paires-t1.jsonl")}
    task_id, task = repeats[0]
    assert task["a"] == tasks[task_id]["b"]  # la répétition revient côtés inversés
    # même jugement sur les deux présentations : A à la première, B (la même carte) ensuite
    answers = {(t, r): "A" for t, r, _ in shown if not r}
    answers |= {(t, True): "B" for t, _ in repeats}
    assert prioriser.record(1, answers, folder) == 22
    assert prioriser.missing(folder) == {"m8-paires-t1": 0}
    assert prioriser.consistency(1, folder) == {"repetitions": 2, "identiques": 2}
    with pytest.raises(ValueError):
        prioriser.record(1, {("inconnue", False): "A"}, folder)
    with pytest.raises(ValueError):
        prioriser.record(1, {shown[0][:2]: "C"}, folder)
    assert "[A]" in prioriser.show(shown[0][2])


def _answer_all(n, folder):
    """Des réponses variées : A, B, puis équivalent, à tour de rôle."""
    shown = prioriser.presentations(n, folder)
    prioriser.record(n, {(t, r): "AB="[k % 3] for k, (t, r, _) in enumerate(shown)}, folder)


def test_la_fiche_de_priorite_a_ses_noms():
    task = {
        "source": "task",
        "title": "Valider le budget",
        "when": "créée hier",
        "fields": [["Échéance", "aujourd'hui à 23:59"], ["Statut", "à faire"]],
        "text": "",
    }
    signals = prioriser.card_signals(task)
    assert len(signals) == len(comprendre.FEATURE_NAMES) + 1 and signals[-1] == 1.0
    assert len(prioriser.PRIORITY_NAMES) == len(signals) + 9


def test_bradley_terry_valide_par_metier_puis_choisit_les_paires_incertaines(corpus, series):
    root, train = corpus
    folder, _ = series
    prioriser.draw_round(1, root, folder, size=24)
    _answer_all(1, folder)
    fit = comprendre.fit_softmax
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(comprendre, "fit_softmax", lambda X, y, k, c: fit(X, y, k, c, steps=30))
        understanding = prioriser.Understanding(
            {"gabarits": root, "llm": root}, HashEmbedder(), VectorCache(None), n_folds=3
        )
    assert set(understanding.by_metier) == {comprendre.metier(k) for k in train}
    pairs = prioriser.round_pairs(1, root, folder)
    assert len(pairs) == 24 and {p.label for p in pairs} == {"A", "B", "="}
    assert all(p.cards[0]["item_id"] == p.items[0].item_id for p in pairs)
    diffs = prioriser.pair_features(pairs, understanding)
    assert diffs.shape == (24, len(prioriser.PRIORITY_NAMES))
    out = folder / "v.json"
    report = prioriser.valider(understanding, root, folder, out)
    assert report["paires"] == 24 and report["paires_decidees"] == 16
    assert report["l2_retenu"] in prioriser.L2_GRID and len(report["essais"]) == 3
    assert 0 <= report["accord_regles"] <= 1 and set(report["poids"]) == set(
        prioriser.PRIORITY_NAMES
    )
    assert json.loads(out.read_text("utf-8")) == report
    # le tour 2 : les paires du réservoir où le modèle hésite le plus
    info = prioriser.draw_uncertain_round(2, understanding, root, folder, size=6, pool=20)
    assert len(info["paires"]) == 6 and "incertaines" in info["selection"]
    first = {
        frozenset(p)
        for p in json.loads((folder / "m8-paires-t1.tirage.json").read_text("utf-8"))[
            "paires"
        ].values()
    }
    assert not first & {frozenset(p) for p in info["paires"].values()}
    # le modèle figé, puis la mesure unique sur un faux jeu gelé (les paires du tour 1)
    _answer_all(2, folder)
    report = prioriser.figer(understanding, root, folder, folder / "m.json", folder / "c.json")
    assert report["boosting"]["retenu"] in ("bradley-terry", "boosting")
    model = json.loads((folder / "m.json").read_text("utf-8"))
    assert model["paires"] == 30 and model["noms"] == list(prioriser.PRIORITY_NAMES)
    journal = folder / "journal.jsonl"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(understanding, "frozen", lambda: next(iter(understanding.by_metier.values())))
        mp.setattr(
            prioriser, "frozen_pairs", lambda name, s, c: prioriser.round_pairs(1, root, folder)
        )

        def measure():
            return prioriser.mesurer(
                understanding,
                folder,
                root,
                folder / "m.json",
                journal,
                lambda: "abc1234",
                {"T-gen": "faux"},
                freezes=(),
            )

        entry = measure()
        assert entry["jeux"]["T-gen"]["paires_decidees"] == 16
        assert "atteinte" in entry["acceptation"] and entry["commit"] == "abc1234"
        assert json.loads(journal.read_text("utf-8")) == entry
        with pytest.raises(ValueError, match="déjà été mesuré"):
            measure()
        # le même modèle se mesure une fois sur les paires d'affaires (autre tâche du journal)
        cases = [
            prioriser.CasePair(
                p.task_id,
                p.vp,
                ([p.cards[0]], [p.cards[1], p.cards[0]]),
                ([p.items[0]], [p.items[1], p.items[0]]),
                p.label,
            )
            for p in prioriser.round_pairs(1, root, folder)
        ]

        def measure_cases():
            return prioriser.mesurer_affaires(
                understanding,
                folder,
                root,
                folder / "m.json",
                journal,
                lambda: "abc1234",
                freezes=(),
                pairs=cases,
            )

        case_entry = measure_cases()
        assert case_entry["tache"] == prioriser.CASE_TASK
        assert case_entry["jeux"]["T-gen"]["elements_par_affaire"] == 1.5
        assert case_entry["decision"] in ("règles par défaut", "score appris par défaut")
        with pytest.raises(ValueError, match="score d'affaire"):
            measure_cases()


def test_le_boosting_apprend_un_ordre_simple():
    rng = np.random.default_rng(0)
    Fa, Fb = rng.random((60, 3)), rng.random((60, 3))
    labels = ["A" if a > b else "B" for a, b in zip(Fa[:, 1], Fb[:, 1], strict=True)]
    labels[:5] = ["="] * 5
    stumps = prioriser.fit_boosting(Fa, Fb, labels, rounds=50)
    assert {s.feature for s in stumps} == {1}
    margin = prioriser.boost_score(stumps, Fa) - prioriser.boost_score(stumps, Fb)
    # deux valeurs du même décile restent à égalité : le seuil laisse cette marge
    assert np.mean((margin[5:] > 0) == (np.array(labels[5:]) == "A")) > 0.75


def test_le_choix_incertain_garde_les_plus_petits_ecarts():
    class Fixed:
        def features(self, cards, metiers):
            return np.array([[c["x"]] for c in cards], dtype=float)

    candidates = [
        {"a": {"x": a}, "b": {"x": b}, "persona": "hotel-1"}
        for a, b in ((5, 0), (1, 1), (2, 0), (0, 3))
    ]
    assert prioriser.uncertain(Fixed(), np.array([1.0]), 2)(candidates) == [1, 2]


def test_l_echelle_du_brief_est_le_rang_centile():
    centiles = list(np.linspace(-3.0, 7.0, 101))  # une échelle régulière, 0,1 par centile
    on_100 = prioriser.to_100(np.array([-10.0, -3.0, 2.0, 2.05, 7.0, 20.0]), centiles)
    assert on_100[0] == 0 and on_100[-1] == 100
    assert abs(on_100[2] - 50) <= 0.5 and on_100[3] >= on_100[2]
    assert np.all(np.diff(on_100) >= 0)
