"""Tests du générateur de mails du lot 2 et de l'annotation à deux."""

from __future__ import annotations

import json
import threading
import urllib.request
from collections import Counter

import pytest

from packages.analyse.scoring.ml import annotate
from packages.analyse.scoring.ml.annotate import ANNOTATORS, assigned
from packages.analyse.scoring.ml.sample import write_sample
from packages.analyse.scoring.ml.synth import phrases as P
from packages.analyse.scoring.ml.synth.generator import _contract, _register, generate, labels
from packages.analyse.scoring.ml.synth.sectors import SECTOR_KEYS


def test_etiquettes_par_construction():
    assert labels(["demande implicite"]) == {"action": True, "reponse": False}
    assert labels(["demande explicite"]) == {"action": True, "reponse": None}  # à juger
    assert labels(["question", "information"]) == {"action": False, "reponse": True}
    assert labels(["demande explicite", "annulation"])["action"] is False
    assert labels(["question", "sans réponse"])["reponse"] is False
    assert labels(["engagement de l'expéditeur"]) == {"action": False, "reponse": False}


def test_registre_et_contractions():
    assert _register("[Peux-tu|Pouvez-vous] valider ?", tu=True) == "Peux-tu valider ?"
    assert _register("[Peux-tu|Pouvez-vous] valider ?", tu=False) == "Pouvez-vous valider ?"
    assert _contract("parle de le chantier à le comité") == "parle du chantier au comité"


def test_generation_deterministe_variee_et_equilibree():
    mails = generate(n=150, seed=1)
    assert mails == generate(n=150, seed=1)
    assert mails != generate(n=150, seed=2)
    assert Counter(m["sector"] for m in mails) == {k: 30 for k in SECTOR_KEYS}
    assert {m["acts"][0] for m in mails} == set(P.ACTS)  # tous les actes principaux
    share = sum(m["construction"]["action"] is True for m in mails) / len(mails)
    assert 0.25 < share < 0.55
    for m in mails:
        assert m["item_id"].startswith("synth:") and m["author"] == "claude"
        assert "{" not in m["text"] and "[" not in m["text"]  # gabarits tous remplis
        assert m["text"] and m["title"]
        assert " de le " not in m["text"] and " à le " not in m["text"]


def test_le_texte_montre_est_nettoye_comme_un_vrai_mail():
    mails = generate(n=300, seed=0)
    quoted = [m for m in mails if "a écrit :" in m["raw"]]
    assert quoted and all("a écrit :" not in m["text"] for m in quoted)


def _rows(n: int = 40) -> list[dict]:
    return [
        {
            "item_id": f"synth:{i:04d}",
            "sender": "X",
            "sender_kind": "client",
            "title": "T",
            "text": "Peux-tu valider ?",
            "construction": {"action": True, "reponse": None},
        }
        for i in range(n)
    ]


def test_repartition_communs_et_moities_disjointes():
    rows = _rows(200)
    theo, paul = ({r["item_id"] for r in assigned(rows, a)} for a in ANNOTATORS)
    shared = theo & paul
    assert theo | paul == {r["item_id"] for r in rows}
    assert 0.15 < len(shared) / len(rows) < 0.35
    assert abs(len(theo) - len(paul)) <= 1
    assert assigned(rows, "theo") == assigned(rows, "theo")
    with pytest.raises(ValueError):
        assigned(rows, "inconnu")


def test_serveur_par_annotateur_fichier_separe_sans_etiquette_de_construction(tmp_path):
    synth = tmp_path / "synth.jsonl"
    write_sample(_rows(), synth)
    httpd = annotate.serve(
        0,
        tmp_path / "s.jsonl",
        tmp_path / "intent_labels.jsonl",
        annotator="paul",
        synth_path=synth,
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/state") as res:
            state = json.loads(res.read())
        assert [f["key"] for f in state] == ["action", "reponse"]
        assert "Hélène" not in state[0]["question"]
        with urllib.request.urlopen(base + "/api/next?feature=action") as res:
            task = json.loads(res.read())["task"]
        assert "construction" not in task and "acts" not in task
        req = urllib.request.Request(
            base + "/api/answer", json.dumps(task | {"value": "oui"}).encode(), method="POST"
        )
        with urllib.request.urlopen(req) as res:
            assert res.status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert (tmp_path / "intent_labels_paul.jsonl").exists()
    assert not (tmp_path / "intent_labels.jsonl").exists()
