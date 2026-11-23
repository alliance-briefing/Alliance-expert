"""M6 : série gelée des éléments de démo (tirage à l'aveugle, éléments jamais vus, gel)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime

import numpy as np
import pytest

from packages.analyse.rag.embeddings import HashEmbedder
from packages.analyse.scoring.ml import annoter, comprendre, jeux_m5
from packages.analyse.scoring.ml.intents import VectorCache
from packages.analyse.scoring.ml.sample import ANNOTATIONS
from packages.collect.trainset import build, load
from packages.collect.trainset.world import all_personas, brief_moment


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


# --- la référence --------------------------------------------------------------------------------

TRAIN_KEYS = ("agence-1", "agence-2", "chantier-1", "daf-1")


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    """Un corpus de gabarits (4 personas « train » et 1 réservée) et un corpus « LLM » où
    les mails d'une demande ont un autre texte, au format d'un corpus assemblé."""
    roots = {w: tmp_path_factory.mktemp(w) for w in ("gabarits", "llm")}
    personas = {}
    for p in all_personas():
        if p.key not in (*TRAIN_KEYS, "hotel-1"):
            continue
        items, scenario = build.build_persona(p)
        rewritten = [
            i.model_copy(update={"snippet": "Texte réécrit : " + i.snippet})
            if i.source == "mail" and n % 3 == 0
            else i
            for n, i in enumerate(items)
        ]
        for writer, its in (("gabarits", items), ("llm", rewritten)):
            for sub, rows in (
                ("items", [i.model_dump(mode="json") for i in its]),
                ("scenario", scenario),
            ):
                (roots[writer] / sub).mkdir(exist_ok=True)
                (roots[writer] / sub / f"{p.key}.jsonl").write_text(
                    build._jsonl(rows), encoding="utf-8"
                )
        personas[p.key] = {"split": p.split}
    for root in roots.values():
        (root / "manifest.json").write_text(json.dumps({"personas": personas}), "utf-8")
    return roots


@pytest.fixture(scope="module")
def examples(corpora):
    return comprendre.training_examples(corpora, frozen=set())[0]


def test_les_exemples_ne_viennent_que_des_personas_d_entrainement(examples):
    assert {e.persona for e in examples} == set(TRAIN_KEYS)
    assert all(e.text and e.writer in comprendre.CORPORA for e in examples)


def test_une_carte_identique_n_est_gardee_qu_une_fois(examples):
    by_item: dict[str, list] = {}
    for e in examples:
        by_item.setdefault(e.item_id, []).append(e)
    assert all(len({e.writer for e in group}) == len(group) for group in by_item.values())
    assert all(len({e.text for e in group}) == len(group) for group in by_item.values())
    llm = [e for e in examples if e.writer == "llm"]
    assert llm and all(e.source in ("mail", "file") for e in llm)  # fichier : « Cité dans »


def test_la_cible_est_l_action_du_fait_meme_si_traite_avant_le_brief(corpora, examples):
    rows = {
        r["item_id"]: r for key in TRAIN_KEYS for r in load.load_scenario(key, corpora["gabarits"])
    }
    assert all(e.labels["action"] == rows[e.item_id]["action"] for e in examples)
    first = {}
    for key in TRAIN_KEYS:
        items = load.load_items(key, corpora["gabarits"])
        first |= comprendre.first_briefs(items, {i.item_id: rows[i.item_id] for i in items})
    resolved_before = [
        e
        for e in examples
        if rows[e.item_id]["resolved_at"]
        and datetime.fromisoformat(rows[e.item_id]["resolved_at"]) <= brief_moment(first[e.item_id])
    ]
    # Des cas existent : « rien » au moment du brief, mais la carte n'en montre rien.
    assert resolved_before and any(e.labels["action"] != "rien" for e in resolved_before)


def test_chaque_carte_prend_l_etiquette_de_son_ecrivain(corpora, examples, tmp_path):
    """23/11 : une copie « pour suivre » se lit dans le texte du LLM, pas dans les gabarits."""
    target = next(e for e in examples if e.writer == "llm")
    llm = tmp_path / "llm"
    shutil.copytree(corpora["llm"], llm)
    path = llm / "scenario" / f"{target.persona}.jsonl"
    rows = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x]
    for row in rows:
        if row["item_id"] == target.item_id:
            row["action"] = "decider" if target.labels["action"] != "decider" else "faire"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    got = comprendre.persona_examples(
        target.persona, {"gabarits": corpora["gabarits"], "llm": llm}, set()
    )[0]
    by_writer = {e.writer: e.labels["action"] for e in got if e.item_id == target.item_id}
    assert by_writer["gabarits"] == target.labels["action"] != by_writer["llm"]


def test_les_elements_geles_sont_exclus(corpora, examples):
    frozen = {examples[0].item_id, examples[1].item_id}
    kept, excluded = comprendre.training_examples(corpora, frozen=frozen)
    assert excluded == 2 and not frozen & {e.item_id for e in kept}


def test_la_validation_coupe_par_persona_et_par_affaire(examples):
    held = comprendre.validation_personas(TRAIN_KEYS, n=2)
    assert len({k.rsplit("-", 1)[0] for k in held}) == 2
    assert held == comprendre.validation_personas(TRAIN_KEYS, n=2)
    train, val = comprendre.split(examples, held)
    assert not {examples[n].persona for n in train} & {examples[n].persona for n in val}
    assert not {examples[n].case_id for n in train} & {examples[n].case_id for n in val}


def test_la_regression_apprend_et_reste_deterministe():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 8)) * 3
    y = np.repeat(np.arange(3), 60)
    X = centers[y] + rng.normal(size=(180, 8))
    config = comprendre.Config(1e-3, balanced=True)
    model = comprendre.fit_softmax(X, y, 3, config, steps=300)
    assert (comprendre.predict(model, X) == y).mean() > 0.95
    again = comprendre.fit_softmax(X, y, 3, config, steps=300)
    assert np.array_equal(model.W, again.W) and np.array_equal(model.b, again.b)


def test_les_vecteurs_s_affichent_par_paquets_et_restent_en_cache():
    lines: list[str] = []
    cache, texts = VectorCache(None), [f"texte {n}" for n in range(25)]
    X = comprendre.embed_with_progress(cache, HashEmbedder(), texts, chunk=10, log=lines.append)
    assert [line.split(" (")[0] for line in lines] == [
        "vecteurs : 10/25",
        "vecteurs : 20/25",
        "vecteurs : 25/25",
    ]
    assert np.array_equal(X, HashEmbedder().embed(texts)) and len(cache.vectors) == 25


def test_le_seuil_d_alerte_garde_le_rappel():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    truth = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    t = comprendre.alert_threshold(p, truth, recall=0.8)
    assert t == 0.4 and ((p >= t) & (truth == 1)).sum() / truth.sum() >= 0.8
    assert comprendre.alert_threshold(p, np.zeros(10, dtype=int)) == 0.5


def test_les_mesures_par_classe():
    pred, truth = np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2])
    assert comprendre.per_class(pred, truth, 3)[0][:2] == (0.5, 1.0)
    assert comprendre.f1_macro(truth, truth, 3) == 1.0
    low, high = comprendre.bootstrap_ci(pred, truth, 3, draws=50)
    assert 0 <= low <= high <= 1


def test_l_entrainement_complet_ecrit_un_modele_relisible(corpora, tmp_path):
    model_path, report_path = tmp_path / "m.npz", tmp_path / "r.json"
    report = comprendre.entrainer(
        corpora, HashEmbedder(), VectorCache(None), model_path, report_path, n_validation=1
    )
    assert set(report["cibles"]) == set(comprendre.TARGETS)
    assert report["exemples"]["elements_geles_exclus"] == 0
    assert json.loads(report_path.read_text("utf-8")) == report
    assert "Texte" not in json.dumps(report)  # le rapport ne recopie aucun texte
    assert "F1 macro" in comprendre.summary(report)
    reference = comprendre.Reference.load(model_path)
    X = HashEmbedder().embed(["mail : Devis à valider\nMerci de valider le devis."])
    out = reference.predict(X)
    assert {t: v[0] in comprendre.TARGETS[t] for t, v in out.items()} == dict.fromkeys(
        comprendre.TARGETS, True
    )
    again = comprendre.entrainer(
        corpora, HashEmbedder(), VectorCache(None), tmp_path / "m2.npz", tmp_path / "r2.json", 1
    )
    assert again == report


# --- mesure finale sur les jeux gelés --------------------------------------------------------


@pytest.fixture(scope="module")
def trained(corpora, tmp_path_factory):
    folder = tmp_path_factory.mktemp("modele")
    model = folder / "reference.npz"
    comprendre.entrainer(
        corpora, HashEmbedder(), VectorCache(None), model, folder / "r.json", n_validation=1
    )
    return model


def _frozen_series(folder, answers):
    """Une série d'éléments jugée et gelée comme T-gen : séries, réponses, deux gels."""
    cards = [
        ("mail", "Devis à valider", "Merci de valider le devis avant jeudi."),
        ("mail", "Lettre interne", "Voici les nouvelles du mois, bonne lecture."),
        ("task", "Envoyer le rapport", ""),
        ("file", "Contrat.docx", "Contrat à relire."),
    ]
    tasks = [
        {
            "task_id": f"t-e{n:03d}",
            "kind": "element",
            "context": "Brief d'essai",
            "card": {
                "item_id": f"{source}:x-{n}",
                "source": source,
                "title": title,
                "when": "reçu hier à 10:00",
                "fields": [],
                "text": text,
            },
        }
        for n, (source, title, text) in enumerate(cards, 1)
    ]
    annoter.write_series(tasks, folder / "essai-m6.jsonl")
    store = annoter.AnswerStore(annoter.answers_path("essai-m6", "theo", folder))
    for task_id, answer in answers.items():
        store.record(task_id, False, answer)
    store.record("t-e001", True, answers["t-e001"])  # une répétition cachée, cohérente
    sha = comprendre._sha
    gel = {
        "series": {"essai-m6.jsonl": sha(folder / "essai-m6.jsonl")},
        "reponses": {store.path.name: sha(store.path)},
    }
    for name in comprendre.FREEZES:
        (folder / name).write_text(json.dumps(gel), encoding="utf-8")


ANSWERS = {
    "t-e001": {"action": "decider", "urgence": "normal", "alerte": False},
    "t-e002": {"action": "informer", "urgence": "sans_urgence", "alerte": False},
    "t-e003": {"action": "faire", "urgence": "normal", "alerte": False},
    "t-e004": {"action": "relire", "urgence": "urgent", "alerte": True},
}


def _measure(folder, model, journal):
    return comprendre.mesurer(
        folder, model, HashEmbedder(), journal, lambda: "abc1234", {"T-essai": "essai-m6"}
    )


def test_la_mesure_lit_la_verite_gelee_et_s_inscrit_au_journal(trained, tmp_path):
    _frozen_series(tmp_path, ANSWERS)
    journal = tmp_path / "journal.jsonl"
    entry = _measure(tmp_path, trained, journal)
    result = entry["jeux"]["T-essai"]
    assert result["elements"] == 4 and result["coherence_annotateur"]["repetitions"] == 1
    assert result["coherence_annotateur"]["action"] == 1.0
    assert result["alerte"]["classes"]["oui"]["n"] == 1
    assert sum(map(sum, result["action"]["confusion"])) == 4
    assert json.loads(journal.read_text("utf-8")) == entry
    assert entry["commit"] == "abc1234" and entry["modele_sha256"] == comprendre._sha(trained)
    assert "T-essai" in comprendre.measure_summary(entry)
    with pytest.raises(ValueError, match="déjà été mesuré"):
        _measure(tmp_path, trained, journal)


def test_la_mesure_refuse_une_verite_non_gelee_ou_modifiee(trained, tmp_path):
    _frozen_series(tmp_path, ANSWERS)
    gel = tmp_path / comprendre.GEL
    frozen = json.loads(gel.read_text("utf-8"))
    gel.write_text(json.dumps(frozen | {"reponses": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="pas encore gelées"):
        _measure(tmp_path, trained, tmp_path / "j.jsonl")
    gel.write_text(json.dumps(frozen), encoding="utf-8")
    answers = annoter.answers_path("essai-m6", "theo", tmp_path)
    answers.write_text(answers.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="modifiés"):
        _measure(tmp_path, trained, tmp_path / "j.jsonl")
    assert not (tmp_path / "j.jsonl").exists()  # rien n'a été lu ni journalisé


def test_les_vrais_jeux_gelés_sont_prets_a_mesurer():
    comprendre.check_frozen()
    for name in comprendre.FROZEN_SETS.values():
        texts, sources, truth = comprendre.series_truth(name)
        assert len(texts) == len(sources) == len(truth["action"]) > 0


# --- validation dure -----------------------------------------------------------------------------


def test_le_choix_garde_le_plus_simple_a_egalite_au_sens_de_g2():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 3, 300)
    simple = np.where(rng.random(300) < 0.3, (truth + 1) % 3, truth)
    almost = simple.copy()
    almost[np.flatnonzero(simple != truth)[0]] = truth[np.flatnonzero(simple != truth)[0]]
    rank, tried = comprendre.select("action", [simple, almost], truth)
    assert tried[1]["score"] > tried[0]["score"]  # meilleur strict, mais dans le bruit
    assert rank == 0 and tried[0]["ecart_au_meilleur_ic95"][0] <= 0
    bad = np.where(rng.random(300) < 0.6, (truth + 1) % 3, truth)
    assert comprendre.select("action", [bad, truth], truth)[0] == 1  # gain réel : retenu
    assert comprendre.select("action", [truth, truth], truth) == (0, [{"score": 1.0}] * 2)


def test_les_plis_de_metiers_partagent_les_metiers_et_sont_deterministes():
    metiers = [f"m{n}" for n in range(12)] * 3
    folds = comprendre.metier_folds(metiers, 4)
    assert sorted(m for f in folds for m in f) == sorted(set(metiers))
    assert [len(f) for f in folds] == [3] * 4 and folds == comprendre.metier_folds(metiers, 4)
    assert folds != comprendre.metier_folds(metiers, 4, seed=1)
    with pytest.raises(ValueError):
        comprendre.metier_folds(["a", "b"], 3)


def test_les_variantes_tenues_laissent_toujours_une_variante_par_famille():
    templates = [f"mail.decider.question.{n}" for n in range(9)]
    templates += ["task.faire.0", "task.faire.1", "teams.0", "llm:mail-v8", "lecture:avis"]
    assert comprendre.template_family("mail.decider.question.7") == "mail.decider.question"
    assert comprendre.template_family("llm:mail-v8") is None
    folds = comprendre.template_folds(templates, 4)
    held = set().union(*folds)
    assert sum(map(len, folds)) == len(held) == 11  # chaque variante dans un seul pli
    assert "teams.0" not in held  # famille d'une seule variante : jamais tenue
    for fold in folds:
        for family in ("mail.decider.question", "task.faire"):
            kept = [t for t in templates if t.startswith(family) and t not in fold]
            assert kept
    assert folds == comprendre.template_folds(reversed(templates), 4)


@pytest.fixture(scope="module")
def hard(corpora, tmp_path_factory):
    """La validation dure sur le petit corpus (3 métiers, 3 plis), apprentissages raccourcis ;
    les plis passés à `out_of_fold` sont gardés pour vérifier les découpages."""
    calls = []
    out_of_fold = comprendre.out_of_fold
    fit = comprendre.fit_softmax

    def spy(target, X, y, folds, log=lambda _: None):
        calls.append((target, folds))
        return out_of_fold(target, X, y, folds, log)

    path = tmp_path_factory.mktemp("dur") / "dur.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(comprendre, "out_of_fold", spy)
        mp.setattr(comprendre, "fit_softmax", lambda X, y, k, c: fit(X, y, k, c, steps=60))
        report = comprendre.valider_dur(corpora, HashEmbedder(), VectorCache(None), path, 3)
    return report, path, calls


def test_la_validation_dure_tient_des_metiers_entiers_a_l_ecart(examples, hard):
    report, _, calls = hard
    assert [c[0] for c in calls] == ["action", "urgence", "alerte", "action", "action", "action"]
    measured = []
    for fold in calls[0][1]:
        train, val = fold.train, fold.measured["v"]
        assert not {examples[n].metier for n in train} & {examples[n].metier for n in val}
        assert not {examples[n].case_id for n in train} & {examples[n].case_id for n in val}
        measured += val
    assert sorted(measured) == list(range(len(examples)))  # chaque carte mesurée une fois
    assert report["metiers"]["exemples"] == len(examples)
    assert set(report["metiers"]["cibles"]) == set(comprendre.TARGETS)
    assert len(report["metiers"]["cibles"]["action"]["par_pli"]) == 3


def test_les_variantes_tenues_ne_sont_jamais_vues_a_l_entrainement(examples, hard):
    report, _, calls = hard
    held = comprendre.template_folds((t for e in examples for t in e.templates), 3)
    base = calls[0][1]
    for fold, base_fold, h in zip(calls[3][1], base, held, strict=True):
        assert not any(h & set(examples[n].templates) for n in fold.train)
        assert all(h & set(examples[n].templates) for n in fold.measured["v"])
        assert set(fold.train) <= set(base_fold.train)
        assert set(fold.measured["v"]) <= set(base_fold.measured["v"])
    second = report["gabarits"]
    assert second["action"]["exemples"] == sum(len(f.measured["v"]) for f in calls[3][1]) > 0
    assert second["familles_jamais_tenues"] < second["familles"]


def test_l_ecrivain_croise_mesure_les_memes_mails_dans_les_deux_sens(examples, hard):
    report, _, calls = hard
    for g, w in zip(calls[4][1], calls[5][1], strict=True):
        assert {examples[n].writer for n in g.train} == {"gabarits"}
        assert {examples[n].writer for n in w.train} == {"llm"}
        assert g.measured == w.measured
        ids = {m: sorted(examples[n].item_id for n in g.measured[m]) for m in g.measured}
        assert ids["gabarits"] == ids["llm"]
        assert all(examples[n].source == "mail" for n in g.train + w.train)
    third = report["ecrivain"]
    assert set(third["cellules"]) == {
        f"{a}→{b}" for a in ("gabarits", "llm") for b in ("gabarits", "llm")
    }
    assert set(third["ecarts"]) == {"mesuré sur gabarits", "mesuré sur llm"}


def test_le_rapport_de_validation_dure_est_relisible_et_sans_texte(hard):
    report, path, _ = hard
    assert json.loads(path.read_text("utf-8")) == report
    assert "Texte" not in json.dumps(report)
    summary = comprendre.hard_summary(report)
    assert "1. métiers" in summary and "2. gabarits" in summary and "3. écrivain" in summary
