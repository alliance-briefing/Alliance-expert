"""Sonde de l'écrivain LLM (M3) dans l'outil d'annotation unifié : série à l'aveugle, comparaison."""

from __future__ import annotations

import json

from contracts.models import Item
from packages.analyse.scoring.ml import annoter, sonde_m3
from packages.collect.trainset.build import DATA


def _sample_mail() -> Item:
    path = DATA / "sample" / "logistique-1.items.jsonl"
    items = [Item.model_validate_json(x) for x in path.read_text("utf-8").splitlines() if x]
    return next(i for i in items if i.source == "mail")


def _sonde(tmp_path, labels: list[dict]):
    """Une sonde de deux cas dans `tmp_path/sondes/essai`."""
    folder = tmp_path / "sondes" / "essai"
    folder.mkdir(parents=True)
    item = _sample_mail().model_dump(mode="json")
    rows = [
        {"case_id": f"sonde-{n:02d}", "persona": "logistique-1", "jour": "2026-11-03"}
        | {"item": item}
        for n in (1, 2)
    ]
    expected = [
        {"case_id": f"sonde-{n:02d}", "emplacement": "x"} | lab for n, lab in enumerate(labels, 1)
    ]
    for name, data in (("a_annoter.jsonl", rows), ("attendu.jsonl", expected)):
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
        (folder / name).write_text(text, encoding="utf-8")
    return tmp_path / "sondes", tmp_path / "series"


LABELS = [
    {"action": "faire", "urgence": "normal", "alerte": False},
    {"action": "relire", "urgence": "sans_urgence", "alerte": False},
]


def test_la_serie_est_a_l_aveugle_et_lisible_par_annoter(tmp_path):
    sondes, series = _sonde(tmp_path, LABELS)
    path = sonde_m3.build_series("essai", sondes, series)
    assert path.name == "sonde-m3-essai.jsonl"
    tasks = annoter.load_series(path)  # refuse toute étiquette (check_blind)
    assert [t["task_id"] for t in tasks] == ["sonde-01", "sonde-02"]
    assert all(t["kind"] == "element" and "8 h 30" in t["context"] for t in tasks)
    assert "faire" not in json.dumps(tasks) and "relire" not in json.dumps(tasks)


def test_la_comparaison_compte_les_textes_conformes(tmp_path):
    sondes, series = _sonde(tmp_path, LABELS)
    store = annoter.AnswerStore(annoter.answers_path("sonde-m3-essai", "theo", series))
    store.record("sonde-01", False, LABELS[0])
    store.record("sonde-02", False, LABELS[1] | {"urgence": "normal"})
    store.record("sonde-02", True, LABELS[1])  # une répétition cachée ne compte pas
    report = sonde_m3.compare("essai", sondes, series)
    assert "theo : 1 conformes sur 2 annotés (critère 16/2 : non atteint)" in report
    assert "≠ sonde-02 [x] urgence normal au lieu de sans_urgence" in report


def test_sans_reponse_la_comparaison_le_dit(tmp_path):
    sondes, series = _sonde(tmp_path, LABELS)
    assert sonde_m3.compare("essai", sondes, series).startswith("Aucune réponse")
