"""Tests des intentions apprises et par règle (faux modèle d'embeddings, sans téléchargement)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from packages.analyse.rag.embeddings import HashEmbedder
from packages.analyse.scoring.ml.intents import (
    IntentModel,
    PrefixedEmbedder,
    VectorCache,
    cache_path,
    design,
    fit_logistic,
    folds,
    intent_text,
    is_alert,
    load_labels,
    nested_cv,
    proba,
    select,
    train,
)


def _write_labels(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def _row(item_id, feature, value, repeat=False):
    return {"item_id": item_id, "feature": feature, "value": value, "repeat": repeat}


def test_etiquettes_derniere_reponse_repetitions_et_doutes_ecartes(tmp_path):
    path = tmp_path / "labels.jsonl"
    _write_labels(
        path,
        [
            _row("a", "action", "non"),
            _row("a", "action", "oui"),  # correction : la dernière compte
            _row("a", "action", "non", repeat=True),  # répétition : sert à la cohérence
            _row("b", "action", "?"),
            _row("c", "action", "non"),
        ],
    )
    assert load_labels("action", path) == {"a": True, "c": False}


def test_question_scindee_herite_du_non_de_la_question_mere(tmp_path):
    path = tmp_path / "labels.jsonl"
    _write_labels(
        path,
        [
            _row("a", "probleme", "non"),
            _row("b", "probleme", "oui"),  # pas revu : ni oui ni non hérité
            _row("c", "probleme", "non"),
            _row("c", "incident", "oui"),  # revu : la revue l'emporte
            _row("d", "probleme", "non"),
            _row("d", "incident", "?"),  # doute à la revue : écarté
        ],
    )
    assert load_labels("incident", path) == {"a": False, "c": True}


@pytest.mark.parametrize(
    ("text", "kind", "expected"),
    [
        ("Connexion inhabituelle détectée\nUne connexion depuis Lyon.", "automate", True),
        ("Alerte : mouvement inhabituel\nUn mouvement de 2 850 €.", "automate", True),
        ("Votre mot de passe expire dans 7 jours", "automate", True),
        ("[INC-1] Mot de passe expiré (résolu)", "automate", False),
        ("Relevé de compte disponible", "automate", False),
        ("Connexion au VPN impossible", "collègue", False),  # incident, pas alerte
    ],
)
def test_alerte_par_regle(text, kind, expected):
    assert is_alert(text, kind) is expected


def test_texte_lu_tronque_comme_l_echantillon():
    assert intent_text("Objet", "x" * 2000) == "Objet\n" + "x" * 1200


def test_cache_de_vecteurs_calcule_une_fois_et_relu(tmp_path):
    calls = []

    class Counting(HashEmbedder):
        def embed(self, texts):
            calls.append(list(texts))
            return super().embed(texts)

    embedder = PrefixedEmbedder(Counting(), "query: ")
    assert embedder.name == "hash-256|query: "
    cache = VectorCache(tmp_path / "v.npz")
    first = cache.embed(embedder, ["peux-tu valider", "pour info", "peux-tu valider"])
    assert calls == [["query: peux-tu valider", "query: pour info"]] or calls == [
        ["query: pour info", "query: peux-tu valider"]
    ]
    np.testing.assert_allclose(first[0], first[2])
    relu = VectorCache(tmp_path / "v.npz")
    np.testing.assert_allclose(relu.embed(embedder, ["pour info"])[0], first[1])
    assert len(calls) == 1  # rien recalculé
    VectorCache(None).save()  # sans fichier : en mémoire seulement


def _corpus(n: int = 40):
    """Moitié demandes (« peux-tu … ? »), moitié informations : séparable par les mots."""
    texts, y = [], []
    for i in range(n):
        ask = i % 3 == 0
        texts.append(
            f"Objet {i}\n" + ("Peux-tu valider le devis ?" if ask else "Pour info, c'est fait.")
        )
        y.append(float(ask))
    return HashEmbedder(64).embed(texts), texts, np.array(y)


def test_plis_stratifies_et_regression_deterministe():
    E, texts, y = _corpus()
    fold = folds(y, 5, seed=0)
    assert all(abs(y[fold == k].sum() - y.sum() / 5) <= 1 for k in range(5))
    w1, w2 = fit_logistic(E, y, 0.1), fit_logistic(E, y, 0.1)
    np.testing.assert_array_equal(w1, w2)
    assert ((proba(w1, E) > 0.5) == (y == 1)).all()
    assert design(E, texts, "reponse", True).shape == (len(y), E.shape[1] + 1)


def test_selection_et_validation_imbriquee_sur_un_cas_separable():
    E, texts, y = _corpus()
    assert select(E, texts, y, "action").l2 in (0.01, 0.1)
    scores = nested_cv(E, texts, y, "action", repeats=2)
    assert scores.shape == (2, 3)
    assert scores[:, 2].mean() > 0.9


def test_entrainement_complet_rapport_et_modele_enregistre(tmp_path):
    _, texts, y = _corpus()
    sample = [
        {
            "item_id": f"m{i}",
            "title": t.split("\n")[0],
            "text": t.split("\n")[1],
            "sender_kind": "automate" if i == 1 else "collègue",
            "received_at": "2026-10-05T09:00:00+02:00",
        }
        for i, t in enumerate(texts)
    ]
    sample[1]["text"] = "Connexion inhabituelle détectée"
    sample[2]["text"] = "Merci de répondre avant le mardi 13 octobre."
    rows = []
    for i, ask in enumerate(y):
        value = "oui" if ask else "non"
        rows += [_row(f"m{i}", f, value) for f in ("action", "reponse", "probleme")]
        rows.append(_row(f"m{i}", "incident", value))
        rows.append(_row(f"m{i}", "echeance", "oui" if i == 2 else "non"))
        rows.append(_row(f"m{i}", "alerte", "oui" if i == 1 else "non"))
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, rows)
    embedder = HashEmbedder(64)
    model, report = train(sample, embedder.embed, labels, repeats=1)
    assert [line.split()[0] for line in report[1:]] == [
        "action",
        "reponse",
        "incident",
        "echeance",
        "alerte",
    ]
    assert report[-1].split()[-1] == "1.00" and report[-2].split()[-1] == "1.00"
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = IntentModel.load(path)
    assert loaded.hints == model.hints and loaded.embedder_name == model.embedder_name
    E = embedder.embed([intent_text(r["title"], r["text"]) for r in sample])
    probs = loaded.predict(E, [intent_text(r["title"], r["text"]) for r in sample])
    assert set(probs) == {"action", "reponse", "incident"}
    np.testing.assert_allclose(probs["action"], model.predict(E, texts)["action"], atol=1e-6)


def test_un_fichier_de_cache_par_modele_et_melange_refuse(tmp_path):
    small, big = HashEmbedder(8), HashEmbedder(16)
    assert cache_path(small) != cache_path(big)
    assert cache_path(PrefixedEmbedder(small, "query: ")).name == "vectors_hash_8_query.npz"
    cache = VectorCache(tmp_path / "v.npz")
    cache.embed(small, ["a"])
    with pytest.raises(ValueError):
        cache.embed(big, ["b"])
