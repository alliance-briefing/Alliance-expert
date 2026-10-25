"""Tests de l'échantillon à étiqueter, de la persona et de la page d'annotation locale."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime

import pytest

from contracts.models import Item
from packages.analyse.scoring.ml import annotate
from packages.analyse.scoring.ml.annotate import Annotator, LabelStore, queue
from packages.analyse.scoring.ml.graph_convert import GraphDataset, MailExtras
from packages.analyse.scoring.ml.persona import GRAPH_HELENE
from packages.analyse.scoring.ml.sample import build_sample, load_sample, template, write_sample

AT = datetime(2026, 10, 7, 8, 0, tzinfo=UTC)
DOM = "verdier-expertises.example"


def _mail(item_id: str, title: str, sender: str, text: str = "Peux-tu valider ?") -> Item:
    return Item(
        item_id=f"mail:{item_id}",
        source="mail",
        provider="microsoft",
        owner_user_id="u1",
        title=title,
        snippet=text,
        content_text=text,
        created_at=AT,
        updated_at=AT,
        participants=[{"name": sender.split("@")[0], "email": sender, "role": "from"}],
        url="https://outlook.example/m",
        acl={"visible_to": ["u1"]},
        collected_at=AT,
        connector={"name": "test", "version": "0.1.0"},
    )


def _extras(sent: bool = False) -> MailExtras:
    return MailExtras("from" if sent else "to", sent, None, None, False, False, ())


@pytest.mark.parametrize(
    ("sender", "kind"),
    [
        (f"bruno.lefevre@{DOM}", "hiérarchie"),
        (f"SANDRINE.MOREAU@{DOM}", "direction"),
        (f"hugo.lambert@{DOM}", "collègue"),
        ("claire.vidal@agrinova.example", "client"),
        ("denis.morel@techplus-info.example", "fournisseur"),
        (f"noreply@portail-rh.{DOM}", "automate"),
        ("alertes@banque-lemanique.example", "automate"),
    ],
)
def test_type_d_expediteur(sender, kind):
    assert GRAPH_HELENE.sender_kind(sender) == kind


@pytest.mark.parametrize(
    ("title", "key"),
    [
        ("RE : Préparation du comité du samedi 7 novembre", "Préparation du comité du # # #"),
        ("TR : Planning de livraison — Refonte du CRM", "Planning de livraison"),
        ("Confirmation de votre voyage Nantes → Lille", "Confirmation de votre voyage Nantes"),
        ("Facture 2026-12", "Facture #-#"),
        ("", "(sans objet)"),
    ],
)
def test_modele_de_phrase(title, key):
    assert template(title) == key


def test_echantillon_un_mail_par_modele_hors_bruit_et_envoyes(tmp_path):
    items = [
        _mail("a1", "Facture 2026-1", "denis.morel@techplus-info.example"),
        _mail("a2", "RE : Facture 2026-2", "denis.morel@techplus-info.example"),
        _mail("b", "Facture 2026-3", f"bruno.lefevre@{DOM}"),  # même modèle, autre type
        _mail("n", "Lettre", "news@veille.example", "Se désabonner"),
        _mail("s", "Question", f"helene.marchand@{DOM}"),
    ]
    extras = {i.item_id: _extras(sent=i.item_id == "mail:s") for i in items}
    sample = build_sample(GraphDataset(items=items, extras=extras))
    assert Counter(row["sender_kind"] for row in sample) == {"hiérarchie": 1, "fournisseur": 1}
    assert sample == build_sample(GraphDataset(items=items, extras=extras))  # graine fixée
    write_sample(sample, tmp_path / "s.jsonl")
    assert load_sample(tmp_path / "s.jsonl") == sample


def _sample(n: int = 40) -> list[dict]:
    return [
        {
            "item_id": f"m{i}",
            "template": "t",
            "sender_kind": "client",
            "sender": "X",
            "title": f"Objet {i}",
            "text": "<b>texte</b>",
        }
        for i in range(n)
    ]


def test_file_d_attente_fixe_avec_10_pourcent_de_repetitions_tardives():
    tasks = queue(_sample(), "action")
    assert tasks == queue(_sample(), "action")
    assert tasks != queue(_sample(), "echeance")  # un ordre par question
    repeats = [i for i, (_, rep) in enumerate(tasks) if rep]
    assert len(repeats) == 4 and len(tasks) == 44
    for position in repeats:
        first = next(i for i, (mid, rep) in enumerate(tasks) if mid == tasks[position][0])
        assert first < 20 <= position  # vu dans la 1re moitié, répété dans la 2de


def test_reponses_ajoutees_la_derniere_compte_et_relues(tmp_path):
    path = tmp_path / "labels.jsonl"
    annotator = Annotator(_sample(5), LabelStore(path))
    first = annotator.next_task("action")
    assert first["text"] == "<b>texte</b>" and first["repeat"] is False
    annotator.answer({"feature": "action", "item_id": first["item_id"], "value": "non"})
    annotator.answer({"feature": "action", "item_id": first["item_id"], "value": "oui"})
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    relu = Annotator(_sample(5), LabelStore(path))
    assert relu.store.answers[(first["item_id"], "action", False)] == "oui"
    assert relu.next_task("action")["item_id"] != first["item_id"]
    assert [s["done"] for s in relu.state()] == [1, 0, 0, 0, 0]
    for bad in (
        {"feature": "x", "item_id": "m0", "value": "oui"},
        {"feature": "action", "item_id": "inconnu", "value": "oui"},
        {"feature": "action", "item_id": "m0", "value": "peut-être"},
    ):
        with pytest.raises(ValueError):
            relu.answer(bad)


def test_question_terminee_renvoie_none(tmp_path):
    annotator = Annotator(_sample(1), LabelStore(tmp_path / "l.jsonl"))
    for _ in range(2):
        task = annotator.next_task("echeance")
        if task:
            annotator.answer(task | {"value": "?"})
    assert annotator.next_task("echeance") is None


@pytest.fixture
def server(tmp_path):
    sample_path = tmp_path / "sample.jsonl"
    write_sample(_sample(3), sample_path)
    httpd = annotate.serve(0, sample_path, tmp_path / "labels.jsonl")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str):
    with urllib.request.urlopen(url) as res:
        return res.status, res.read()


def _post(url: str, data: object):
    req = urllib.request.Request(
        url, json.dumps(data).encode(), method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        return res.status, json.loads(res.read())


def test_serveur_page_etat_tache_reponse(server):
    status, page = _get(server + "/")
    assert (
        status == 200
        and b"textContent" in page
        and b"innerHTML" not in page.replace(b"jamais innerHTML", b"")
    )
    assert len(json.loads(_get(server + "/api/state")[1])) == 5
    task = json.loads(_get(server + "/api/next?feature=action")[1])["task"]
    assert _post(server + "/api/answer", task | {"value": "oui"}) == (200, {"ok": True})
    for url, code in ((server + "/api/next?feature=x", 400), (server + "/nope", 404)):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(url)
        assert err.value.code == code
    for path, body, code in (("/api/answer", {"value": "oui"}, 400), ("/autre", {}, 404)):
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(server + path, body)
        assert err.value.code == code


def test_question_de_revue_ne_pose_que_les_mails_listes(tmp_path):
    annotator = Annotator(_sample(10), LabelStore(tmp_path / "l.jsonl"), {"incident": {"m3"}})
    assert [mid for mid, _ in annotator.queues["incident"]] == ["m3"]
    assert len(annotator.queues["action"]) == 11  # 10 mails + 1 répétition
    assert annotate.FEATURES[2].review == "review_probleme.json"
