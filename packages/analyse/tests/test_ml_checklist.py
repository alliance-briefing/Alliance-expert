"""Tests du banc CheckList des intentions (fausse prédiction, sans modèle)."""

from __future__ import annotations

import json

from packages.analyse.scoring.ml.checklist import SUITE_PATH, mail_text, report, run

SUITE = {
    "base": [
        {"cat": "demande", "title": "A", "body": "Peux-tu valider ?", "action": True},
        {"cat": "info", "title": "B", "body": "Pour info.", "action": False, "reponse": False},
    ],
    "invariance": [
        {
            "cat": "signature",
            "features": ["action"],
            "a": ["A", "Peux-tu valider ?"],
            "b": ["A", "Peux-tu valider ?\n\nCordialement,\nMarc"],
        },
    ],
    "direction": [
        {
            "cat": "tiers",
            "feature": "action",
            "dir": "down",
            "a": ["A", "Peux-tu valider ?"],
            "b": ["A", "Je demande à Julie de valider."],
        },
    ],
}


def fake_predict(texts):
    """« action » = présence d'un « ? » ; « reponse » toujours faible."""
    return {"action": [0.9 if "?" in t else 0.1 for t in texts], "reponse": [0.2 for _ in texts]}


def test_nettoyage_comme_un_vrai_mail():
    assert mail_text("A", "Peux-tu valider ?\n\nCordialement,\nMarc") == "A\nPeux-tu valider ?"
    assert mail_text("A", "Cordialement") == "A\nCordialement"  # rien ne reste : corps brut


def test_banc_base_invariance_direction_et_rapport():
    results = run(SUITE, fake_predict)
    assert [(r["type"], r["feature"], r["ok"]) for r in results] == [
        ("base", "action", True),
        ("base", "action", True),
        ("base", "reponse", True),
        ("invariance", "action", True),  # la signature est retirée avant la prédiction
        ("direction", "action", True),
    ]
    text = report(results)
    assert "base        action     2/2" in text and "Échecs :" in text


def test_echecs_listes_par_categorie():
    results = run(
        SUITE, lambda texts: {"action": [0.5] * len(texts), "reponse": [0.9] * len(texts)}
    )
    text = report(results)
    assert "2/2  base · info" in text  # action et reponse attendues non, prédites oui
    assert "[direction/action]" in text  # aucune baisse


def test_le_banc_versionne_est_bien_forme():
    suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    assert len(suite["base"]) >= 30 and suite["invariance"] and suite["direction"]
    for test in suite["base"]:
        assert {"cat", "title", "body"} <= set(test) and ({"action", "reponse"} & set(test))
    assert all(t["dir"] in ("up", "down") for t in suite["direction"])
