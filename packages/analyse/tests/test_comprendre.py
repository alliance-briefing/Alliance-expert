"""M6 : série gelée des éléments de démo (tirage à l'aveugle, éléments jamais vus, gel)."""

from __future__ import annotations

import json

import pytest

from packages.analyse.scoring.ml import annoter, comprendre, jeux_m5
from packages.analyse.scoring.ml.sample import ANNOTATIONS


@pytest.fixture(scope="module")
def briefs():
    return comprendre.demo_briefs()


@pytest.fixture(scope="module")
def drawn(briefs, tmp_path_factory):
    """Un tirage avec un faux fichier d'annotation qui a déjà montré presque tous les mails."""
    series = tmp_path_factory.mktemp("series")
    annotations = tmp_path_factory.mktemp("annotations")
    mails = sorted({i.item_id for _, items in briefs for i in items if i.source == "mail"})
    seen = set(mails[:-60])
    rows = [{"item_id": i} for i in sorted(seen)]
    (annotations / "deja.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    return series, annotations, comprendre.draw_tdemo_elements(briefs, series, annotations), seen


def test_la_serie_a_les_tailles_prevues_et_reste_a_l_aveugle(drawn):
    series, _, info, _ = drawn
    tasks = annoter.load_series(series / f"{comprendre.DEMO_SERIES}.jsonl")  # check_blind
    assert len(tasks) == sum(comprendre.DEMO_QUOTA.values())
    assert all(t["kind"] == "element" and "Camille Laurent" in t["context"] for t in tasks)
    sources = [t["card"]["source"] for t in tasks]
    assert {s: sources.count(s) for s in set(sources)} == comprendre.DEMO_QUOTA
    assert len(set(info["elements"].values())) == len(tasks)
    annoter.check_blind(json.loads((series / comprendre.DEMO_TIRAGE).read_text("utf-8")))


def test_les_elements_deja_montres_sont_exclus(drawn):
    _, _, info, seen = drawn
    assert not seen & set(info["elements"].values())
    assert info["exclus_deja_vus"] == len(seen)


def test_les_elements_de_demo_deja_annotes_sont_reperes():
    """Les paires de démo de M5 et les mails de démo étiquetés en octobre comptent comme vus."""
    shown = comprendre.shown_demo_items(ANNOTATIONS)
    pairs = [json.loads(x) for x in jeux_m5.DEMO_PAIRS.read_text("utf-8").splitlines() if x]
    assert {r[k]["item_id"] for r in pairs for k in ("a", "b")} <= shown


def test_le_tirage_est_deterministe_et_ne_se_refait_pas(briefs, drawn, tmp_path):
    series, annotations, _, _ = drawn
    comprendre.draw_tdemo_elements(briefs, tmp_path, annotations)
    for name in (f"{comprendre.DEMO_SERIES}.jsonl", comprendre.DEMO_TIRAGE, comprendre.GEL):
        assert (tmp_path / name).read_bytes() == (series / name).read_bytes()
    with pytest.raises(FileExistsError):
        comprendre.draw_tdemo_elements(briefs, tmp_path, annotations)


def test_le_gel_detecte_tout_changement_et_exclut_de_l_entrainement(drawn, tmp_path):
    series, _, info, _ = drawn
    for path in series.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    assert comprendre.frozen_items(tmp_path) == set(info["elements"].values())
    assert comprendre.changed_since_freeze(tmp_path) == []
    with pytest.raises(ValueError, match="réponses manquantes"):
        comprendre.freeze_demo_answers(tmp_path)
    name = comprendre.DEMO_SERIES
    tasks = annoter.load_series(tmp_path / f"{name}.jsonl")
    store = annoter.AnswerStore(annoter.answers_path(name, "theo", tmp_path))
    for task_id, repeat in annoter.order(tasks, "theo", name):
        store.record(task_id, repeat, {"action": "rien", "urgence": "normal", "alerte": False})
    assert comprendre.missing_demo_answers(tmp_path) == 0
    frozen = comprendre.freeze_demo_answers(tmp_path)
    assert list(frozen["reponses"]) == [f"{name}_theo.jsonl"]
    with pytest.raises(ValueError, match="déjà gelées"):
        comprendre.freeze_demo_answers(tmp_path)
    answers = annoter.answers_path(name, "theo", tmp_path)
    answers.write_text(answers.read_text("utf-8") + "\n", encoding="utf-8")
    assert comprendre.changed_since_freeze(tmp_path) == [answers.name]


def test_les_jeux_reels_n_ont_pas_change_depuis_leur_gel():
    assert comprendre.changed_since_freeze() == []
