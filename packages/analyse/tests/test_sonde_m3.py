"""Sonde de l'écrivain LLM (M3) dans l'outil d'annotation unifié : série à l'aveugle, comparaison."""

from __future__ import annotations

import json

import pytest

from contracts.models import Item
from packages.analyse.scoring.ml import annoter, sonde_m3
from packages.collect.trainset import build
from packages.collect.trainset.build import DATA
from packages.collect.trainset.metiers import METIERS
from packages.collect.trainset.world import FEMALE, LAST, all_personas


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


# --- acceptation du corpus LLM : diversité, échantillon, conformité ----------------------------

CONSIGNE = "mail-test"


def _write(folder, key: str, items: list[Item], scenario: list[dict], manifest: dict) -> None:
    for sub, rows in (
        ("items", [i.model_dump(mode="json") for i in items]),
        ("scenario", scenario),
    ):
        (folder / sub).mkdir(parents=True, exist_ok=True)
        (folder / sub / f"{key}.jsonl").write_text(build._jsonl(rows), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    """Une persona écrite deux fois : gabarits, et un faux corpus LLM où un mail réécrit sur
    deux a un texte propre (retenu), les autres gardant les gabarits."""
    root = tmp_path_factory.mktemp("m3")
    persona = next(p for p in all_personas() if p.key == "logistique-1")
    items, scenario = build.build_persona(persona)
    mails = [
        r
        for r in scenario
        if r["source"] == "mail" and any(t.startswith("mail.") for t in r["templates"])
    ]
    chosen = {r["item_id"] for r in mails[::2]}
    llm_items = []
    for n, item in enumerate(items):
        if item.item_id in chosen:
            word = chr(97 + n % 26) + chr(97 + n // 26 % 26)  # des lettres : pas masquées
            text = f"Voici la version {word} du texte rédigé pour toi. Bonne journée, Julie"
            item = item.model_copy(update={"snippet": text})
        llm_items.append(item)
    llm_scenario = [
        r | {"templates": [f"llm:{CONSIGNE}"]} if r["item_id"] in chosen else r for r in scenario
    ]
    personas = {"personas": {persona.key: {"split": persona.split}}}
    gab, llm = root / "gabarits", root / "llm"
    _write(gab, persona.key, items, scenario, personas)
    counts = {"a_reecrire": len(mails), "taux_controles": 1.0, "ecrits": len(mails)}
    manifest = personas | {
        "consigne": CONSIGNE,
        "llm": counts | {"retenus": len(chosen)},
        "textes": {"sha256": "abc"},
    }
    _write(llm / CONSIGNE, persona.key, llm_items, llm_scenario, manifest)
    return gab, llm, chosen, {r["item_id"]: r for r in scenario}, len(mails)


def test_le_masque_garde_la_forme_de_la_phrase():
    metier = METIERS[0]
    obj, org = metier.objects[0].phrase, metier.suppliers[0]
    text = (
        f"Merci de valider {obj} d'ici vendredi 6 novembre. {FEMALE[0]} {LAST[0]} a appelé {org} "
        "ce matin. Merci ! Un petit point sur le planning avec toute l'équipe. Le texte est coup…"
    )
    assert sonde_m3.sentence_models(text) == [
        "merci de valider <objet> d'ici # # #.",
        "<nom> <nom> a appelé <org> ce matin.",
        "un petit point sur le planning avec toute l'équipe.",  # « petit » n'est pas un nom ici
    ]  # « Merci ! » (formule) et la phrase coupée par l'extrait sont écartées


def test_la_diversite_compare_les_memes_mails(corpora):
    gab, llm, chosen, _, _ = corpora
    d = sonde_m3.diversity(llm / CONSIGNE, gab)
    assert d["mails"] == len(chosen)
    assert d["llm"]["modeles_phrases"] == len(chosen)  # une phrase propre par mail
    assert d["gabarits"]["modeles_phrases"] < d["llm"]["modeles_phrases"]
    assert d["rapport_modeles"] > 1 and d["llm"]["part_distincte"] == 1.0
    report = sonde_m3.diversity_report(llm / CONSIGNE, gab)
    assert f"{len(chosen)} mails écrits des deux façons" in report


def test_l_echantillon_est_tire_au_hasard_parmi_les_textes_retenus(corpora, tmp_path):
    _, llm, chosen, _, to_rewrite = corpora
    path = sonde_m3.build_sample(CONSIGNE, series=tmp_path, llm=llm)
    assert path.name == f"m3-{CONSIGNE}-echantillon.jsonl"
    tasks = annoter.load_series(path)  # refuse toute étiquette (check_blind)
    size = min(sonde_m3.CAP, round(sonde_m3.SHARE * to_rewrite))
    assert len(tasks) == size and all(t["kind"] == "element" for t in tasks)
    info = json.loads((tmp_path / f"m3-{CONSIGNE}-echantillon.tirage.json").read_text("utf-8"))
    annoter.check_blind(info)  # le tirage ne dit pas non plus les réponses
    assert set(info["elements"].values()) <= chosen and len(set(info["elements"].values())) == size
    assert [t["card"]["item_id"] for t in tasks] == list(info["elements"].values())
    same = [r["item_id"] for _, r in sonde_m3.draw(llm / CONSIGNE)]
    assert same == list(info["elements"].values())  # graine fixe : même tirage
    with pytest.raises(FileExistsError):  # un échantillon de mesure ne se retire pas
        sonde_m3.build_sample(CONSIGNE, series=tmp_path, llm=llm)


def test_la_conformite_compare_aux_etiquettes_du_scenario(corpora, tmp_path):
    _, llm, _, scenario, _ = corpora
    sonde_m3.build_sample(CONSIGNE, series=tmp_path, llm=llm)
    name = f"m3-{CONSIGNE}-echantillon"
    assert sonde_m3.conformity(CONSIGNE, tmp_path, llm).startswith("Aucune réponse")
    behind = json.loads((tmp_path / f"{name}.tirage.json").read_text("utf-8"))["elements"]
    (first, a), (second, b) = list(behind.items())[:2]
    labels = {k: scenario[a][k] for k in sonde_m3.KEYS}
    wrong = {k: scenario[b][k] for k in sonde_m3.KEYS} | {"alerte": not scenario[b]["alerte"]}
    store = annoter.AnswerStore(annoter.answers_path(name, "theo", tmp_path))
    store.record(first, False, labels)
    store.record(second, False, wrong)
    report = sonde_m3.conformity(CONSIGNE, tmp_path, llm)
    assert report.startswith("theo : 1 conformes sur 2 annotés (50.0%")
    assert "critère 90% : non atteint" in report
    assert f"≠ {second} ({b}) alerte {wrong['alerte']} au lieu de {scenario[b]['alerte']}" in report


def test_l_intervalle_de_wilson():
    low, high = sonde_m3.wilson(90, 100)
    assert 0.82 < low < 0.9 < high < 0.95
    assert sonde_m3.wilson(0, 0) == (0.0, 1.0)
    assert sonde_m3.wilson(10, 10)[1] == pytest.approx(1.0)


def test_sans_corpus_assemble_l_outil_dit_quoi_faire(tmp_path):
    with pytest.raises(FileNotFoundError, match="--assembler"):
        sonde_m3.retained(tmp_path)
