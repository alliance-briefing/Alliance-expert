"""M6, levier 3 : encodeur réajusté, sur un tout petit BERT créé sur place (sans téléchargement).

Sautés si torch ou transformers manquent (la CI ne les installe pas : groupe gpu ou nli)."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from packages.analyse.rag.embeddings import HashEmbedder
from packages.analyse.scoring.ml import annoter, comprendre, encodeur
from packages.analyse.scoring.ml.intents import VectorCache
from packages.analyse.tests import test_comprendre as base

# Les fixtures du petit corpus et du jeu réel d'essai, reprises du module de M6.
briefs, corpora, examples, dev = base.briefs, base.corpora, base.examples, base.dev
_answer = base._answer

TINY = encodeur.Recipe(epochs=1, batch=16, max_tokens=32)


def tiny_builder(texts):
    """Un BERT de 2 couches et un tokeniseur par mots, construits à partir des textes."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]", t.lower())})
    vocab = {w: n for n, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", *words])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()

    def build():
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]"
        )
        torch.manual_seed(0)
        config = transformers.BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
        return tokenizer, transformers.BertModel(config)

    return build


@pytest.fixture(scope="module")
def small(examples):
    """Un sous-ensemble court pour les tests d'apprentissage."""
    rows = list(range(0, len(examples), 7))[:120]
    texts = [comprendre.encoder_text(examples[n].card, "normalise") for n in rows]
    build = tiny_builder(texts)
    tokenizer, _ = build()
    tokens = tokenizer(texts, truncation=True, max_length=32)
    F = np.array([comprendre.card_features(examples[n].card) for n in rows])
    labels = {t: comprendre._codes([examples[n] for n in rows], t) for t in comprendre.TARGETS}
    return build, tokenizer, tokens, F, labels


def test_le_reseau_rend_trois_tetes_et_apprend_de_facon_reproductible(small):
    build, tokenizer, tokens, F, labels = small
    rows = list(range(len(F)))
    a = encodeur.train(build, tokens, tokenizer, F, labels, rows, 0, TINY)
    b = encodeur.train(build, tokens, tokenizer, F, labels, rows, 0, TINY)
    for (name, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(pa, pb), name
    proba = encodeur.predict_proba(a, tokens, tokenizer, F, rows[:10])
    assert {t: p.shape for t, p in proba.items()} == {
        t: (10, len(c)) for t, c in comprendre.TARGETS.items()
    }
    assert np.allclose(proba["action"].sum(axis=1), 1.0, atol=1e-5)


def test_l_apprenant_ne_reajuste_qu_une_fois_par_jeu_d_entrainement(small):
    build, tokenizer, tokens, F, labels = small
    learner = encodeur.FoldLearner(build, tokens, tokenizer, F, labels, 0, TINY, "cpu", print)
    folds = [
        comprendre.Fold(list(range(80)), {"v": list(range(80, 100))}),
        comprendre.Fold(list(range(20, 100)), {"v": list(range(20))}),
    ]
    out = {t: learner(t, None, None, folds)["v"] for t in comprendre.TARGETS}
    assert learner.trainings == 2  # trois cibles, un seul réseau par pli
    assert out["action"].index.tolist() == list(range(80, 100)) + list(range(20))
    assert out["alerte"].proba[0].shape == (40, 2)


def test_le_levier_3_compare_e_a_n_sur_les_memes_cartes(corpora, dev, examples, tmp_path):
    source, _, _ = dev
    for path in source.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    claude = annoter.AnswerStore(annoter.answers_path(comprendre.DEV_SERIES, "claude", tmp_path))
    for n, t in enumerate(annoter.load_series(tmp_path / f"{comprendre.DEV_SERIES}.jsonl")):
        claude.record(t["task_id"], False, _answer(("informer", "faire", "rien")[n % 3]))
    comprendre.freeze_dev(tmp_path)
    texts = [comprendre.encoder_text(e.card, "normalise") for e in examples]
    texts += [
        comprendre.encoder_text(t["card"], "normalise")
        for t in annoter.load_series(tmp_path / f"{comprendre.DEV_SERIES}.jsonl")
    ]
    fit = comprendre.fit_softmax
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(comprendre, "fit_softmax", lambda X, y, k, c: fit(X, y, k, c, steps=30))
        report = encodeur.levier_encodeur(
            corpora,
            HashEmbedder(),
            VectorCache(None),
            tmp_path,
            tmp_path / "e.json",
            n_folds=3,
            seeds=(0, 1),
            build=tiny_builder(texts),
            recipe=TINY,
            where="cpu",
            model_dir=tmp_path / "modeles",
        )
    assert "N" in report["variantes"] and "E0" in report["variantes"]
    assert report["variantes"]["E0"]["apprentissages"] == 3 + 3 + 6 + 1
    gaps = report["ecarts_apparies_action"]["E0-N"]
    assert set(gaps) == {"metiers", *comprendre.DECISIVE}
    decision = report["decision"]
    assert decision["graines_calculees"] == len(report["variantes"]) - 1
    assert decision["arret_anticipe"] == (decision["graines_calculees"] < 2)
    assert (tmp_path / "modeles" / "E0" / "reseau.pt").exists()
    assert json.loads((tmp_path / "e.json").read_text("utf-8")) == report


def test_la_sonde_donne_le_temps_sans_chiffre_de_mesure(corpora, examples):
    texts = [comprendre.encoder_text(e.card, "normalise") for e in examples]
    out = encodeur.sonde(corpora, tiny_builder(texts), TINY, "cpu", lambda _: None, 3)
    assert out["secondes_par_epoque"] > 0 and out["minutes_par_graine"] >= 0
    assert not any("f1" in k for k in out)


def test_export_onnx_et_latence(small, dev, tmp_path):
    pytest.importorskip("onnx")
    build, tokenizer, tokens, F, labels = small
    net = encodeur.train(build, tokens, tokenizer, F, labels, list(range(len(F))), 0, TINY)
    folder = tmp_path / "E0"
    folder.mkdir()
    torch.save(net.state_dict(), folder / "reseau.pt")
    (folder / "meta.json").write_text(
        json.dumps({"seuil_alerte": 0.5, "recette": {"max_tokens": 32}}), encoding="utf-8"
    )
    out = encodeur.exporter(folder, build=build)
    assert (out / "reseau-int8.onnx").exists() and (out / "tokenizer.json").exists()
    # le réseau exporté, servi par onnxruntime seul, rend les mêmes prédictions que torch
    cards = [examples_card for examples_card in _cards(dev)][:20]
    onnx_pred = encodeur.OnnxModel(out, "reseau.onnx", 32).predict(cards)
    texts = [comprendre.encoder_text(c, "normalise") for c in cards]
    toks = tokenizer(texts, truncation=True, max_length=32)
    Fc = np.array([comprendre.card_features(c) for c in cards])
    torch_pred = encodeur.predict_proba(net, toks, tokenizer, Fc, range(len(cards)))
    assert (onnx_pred["action"] == torch_pred["action"].argmax(axis=1)).mean() >= 0.95
    report = encodeur.latence(out, _series(dev, tmp_path), None)
    assert report["taille_mo"] > 0 and report["latence_ms_par_element"] > 0
    assert set(report) >= {"f1_action_int8", "f1_action_float32", "contraintes_tenues"}


def _cards(dev):
    series, _, _ = dev
    return [t["card"] for t in annoter.load_series(series / f"{comprendre.DEV_SERIES}.jsonl")]


def _series(dev, tmp_path):
    """Le jeu réel d'essai, annoté et gelé dans un dossier à part."""
    source, _, _ = dev
    folder = tmp_path / "series"
    folder.mkdir()
    for path in source.iterdir():
        (folder / path.name).write_bytes(path.read_bytes())
    claude = annoter.AnswerStore(annoter.answers_path(comprendre.DEV_SERIES, "claude", folder))
    for t in annoter.load_series(folder / f"{comprendre.DEV_SERIES}.jsonl"):
        claude.record(t["task_id"], False, _answer("informer"))
    comprendre.freeze_dev(folder)
    return folder
