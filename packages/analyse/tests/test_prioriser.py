"""M8, étape 1 : paires d'entraînement tirées sur les personas « train », annotées par Claude."""

from __future__ import annotations

import json

import pytest

from packages.analyse.scoring.ml import annoter, jeux_m5, prioriser
from packages.collect.trainset import build
from packages.collect.trainset.world import all_personas


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Deux personas « train » et une persona réservée, écrites par gabarits."""
    root = tmp_path_factory.mktemp("corpus")
    train = [p for p in all_personas() if p.split == "train"][:2]
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
