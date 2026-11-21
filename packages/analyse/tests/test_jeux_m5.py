"""M5 : jeux de mesure gelés. Tirage à l'aveugle, étanchéité avec l'entraînement, gel."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from packages.analyse.scoring.ml import annoter, jeux_m5
from packages.collect.trainset import build
from packages.collect.trainset.build import MANIFEST
from packages.collect.trainset.world import all_personas, brief_moment


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Les 4 personas réservées (T-gen), écrites par gabarits, au format d'un corpus assemblé."""
    root = tmp_path_factory.mktemp("corpus")
    personas = {}
    for p in all_personas():
        if p.split != "tgen":
            continue
        items, scenario = build.build_persona(p)
        for sub, rows in (
            ("items", [i.model_dump(mode="json") for i in items]),
            ("scenario", scenario),
        ):
            (root / sub).mkdir(exist_ok=True)
            (root / sub / f"{p.key}.jsonl").write_text(build._jsonl(rows), encoding="utf-8")
        personas[p.key] = {"split": p.split}
    manifest = {"consigne": "mail-test", "textes": {"sha256": "abc"}, "personas": personas}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def drawn(corpus, tmp_path_factory):
    """Un tirage complet, avec un faux échantillon de M3 dont les éléments sont à exclure."""
    series = tmp_path_factory.mktemp("series")
    key = "hotel-1"
    seen = [
        r["item_id"]
        for r in json.loads(
            f"[{','.join((corpus / 'scenario' / f'{key}.jsonl').read_text('utf-8').splitlines())}]"
        )
        if r["source"] == "mail" and r["collecte"]
    ][:30]
    elements = {f"m3-{n:03d}": item_id for n, item_id in enumerate(seen)}
    (series / "m3-essai.tirage.json").write_text(json.dumps({"elements": elements}), "utf-8")
    return series, jeux_m5.draw_all(corpus, series), set(seen)


def _ids(info: dict) -> set[str]:
    t = info["tgen"]
    out = set(t["elements"].values())
    for pair in t["paires"].values():
        out.update(pair)
    for a, b in t["affaires"].values():
        out.update(a, b)
    return out


def test_t_gen_a_les_tailles_prevues_et_reste_a_l_aveugle(drawn):
    series, info, _ = drawn
    sizes = {
        "elements": sum(jeux_m5.ELEMENT_QUOTA.values()),
        "paires": jeux_m5.N_PAIRS,
        "affaires": jeux_m5.N_CASES,
        "demo": 40,
    }
    for name, size in sizes.items():
        tasks = annoter.load_series(series / f"{jeux_m5.NAMES[name]}.jsonl")  # check_blind
        assert len(tasks) == size
    annoter.check_blind(json.loads((series / "m5-tirage.json").read_text("utf-8")))
    sources = [i.split(":")[0] for i in info["tgen"]["elements"].values()]
    assert {s: sources.count(s) for s in set(sources)} == jeux_m5.ELEMENT_QUOTA
    assert len(set(info["tgen"]["elements"].values())) == sizes["elements"]


def test_t_gen_ne_contient_que_des_personas_reservees(drawn):
    """L'étanchéité : rien de T-gen n'appartient à une persona d'entraînement."""
    _, info, _ = drawn
    train = {
        k
        for k, v in json.loads(MANIFEST.read_text("utf-8"))["personas"].items()
        if v["split"] == "train"
    }
    owners = {i.split(":")[1].rsplit("-", 1)[0] for i in _ids(info)}
    assert owners == set(info["tgen"]["personas"]) and not owners & train


def test_les_elements_deja_vus_en_m3_sont_exclus(drawn):
    _, info, seen = drawn
    pairs = {i for p in info["tgen"]["paires"].values() for i in p}
    assert not seen & set(info["tgen"]["elements"].values()) and not seen & pairs


def test_t_demo_rejuge_les_paires_de_demo_a_l_aveugle(drawn):
    series, info, _ = drawn
    tasks = annoter.load_series(series / "m5-tdemo-paires.jsonl")
    original = [json.loads(x) for x in jeux_m5.DEMO_PAIRS.read_text("utf-8").splitlines() if x]
    assert {frozenset(p) for p in info["tdemo"]["paires"].values()} == {
        frozenset((r["a"]["item_id"], r["b"]["item_id"])) for r in original
    }
    assert all("Camille Laurent" in t["context"] for t in tasks)
    assert "chips" not in json.dumps(tasks)  # aucune pastille de l'ancien outil


def test_le_tirage_est_deterministe_et_ne_se_refait_pas(corpus, drawn, tmp_path):
    series, _, _ = drawn
    for path in series.glob("m3-*.tirage.json"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    jeux_m5.draw_all(corpus, tmp_path)
    for name in jeux_m5.NAMES.values():
        assert (tmp_path / f"{name}.jsonl").read_bytes() == (series / f"{name}.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        jeux_m5.draw_all(corpus, tmp_path)


def test_le_gel_detecte_tout_changement(corpus, drawn, tmp_path):
    series, _, _ = drawn
    for path in series.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    assert jeux_m5.changed_since_freeze(tmp_path) == []
    with pytest.raises(ValueError, match="réponses manquantes"):
        jeux_m5.freeze_answers(tmp_path)
    for name in jeux_m5.NAMES.values():
        tasks = annoter.load_series(tmp_path / f"{name}.jsonl")
        store = annoter.AnswerStore(annoter.answers_path(name, "theo", tmp_path))
        kinds = {t["task_id"]: t["kind"] for t in tasks}
        for task_id, repeat in annoter.order(tasks, "theo", name):
            answer = (
                {"action": "rien", "urgence": "normal", "alerte": False}
                if kinds[task_id] == "element"
                else {"choix": "A"}
            )
            store.record(task_id, repeat, answer)
    assert all(n == 0 for n in jeux_m5.missing_answers(tmp_path).values())
    frozen = jeux_m5.freeze_answers(tmp_path)
    assert len(frozen["reponses"]) == len(jeux_m5.NAMES)
    with pytest.raises(ValueError, match="déjà gelées"):
        jeux_m5.freeze_answers(tmp_path)
    answers = annoter.answers_path(jeux_m5.NAMES["elements"], "theo", tmp_path)
    answers.write_text(answers.read_text("utf-8") + "\n", encoding="utf-8")
    series_file = tmp_path / "m5-tgen-paires.jsonl"
    series_file.write_text(series_file.read_text("utf-8") + "\n", encoding="utf-8")
    assert jeux_m5.changed_since_freeze(tmp_path) == sorted([answers.name, series_file.name])


def test_les_fenetres_du_brief(corpus):
    key = "hotel-1"
    items = {
        i.item_id: i
        for i in build.build_persona(next(p for p in all_personas() if p.key == key))[0]
    }
    rows = {
        json.loads(x)["item_id"]: json.loads(x)
        for x in (corpus / "scenario" / f"{key}.jsonl").read_text("utf-8").splitlines()
    }
    mail = next(i for i in items.values() if i.source == "mail")
    seen = jeux_m5.datetime.fromisoformat(rows[mail.item_id]["visible_from"])
    moment = brief_moment(seen.date() + timedelta(days=1))
    assert jeux_m5.on_brief(mail, rows[mail.item_id], moment, seen - timedelta(minutes=1))
    assert not jeux_m5.on_brief(mail, rows[mail.item_id], moment, seen)  # déjà au brief d'avant
    assert not jeux_m5.on_brief(mail, rows[mail.item_id], seen - timedelta(hours=1), seen)


def test_le_gel_du_depot_est_intact():
    """Garde de la CI : une série ou des réponses gelées de M5 n'ont pas changé."""
    assert jeux_m5.changed_since_freeze() == []


def test_rien_de_gele_n_appartient_a_l_entrainement():
    """Garde de la CI : les éléments de T-gen et T-démo ne sont à aucune persona d'entraînement."""
    train = {
        k
        for k, v in json.loads(MANIFEST.read_text("utf-8"))["personas"].items()
        if v["split"] == "train"
    }
    owners = {i.split(":")[1].rsplit("-", 1)[0] for i in jeux_m5.frozen_items()}
    assert not owners & train
