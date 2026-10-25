"""Tests des journées générées (niveau 2) et des paires à l'aveugle annotées à deux."""

from __future__ import annotations

import json
import threading
import urllib.request
from collections import Counter

from packages.analyse.scoring.ml import annotate
from packages.analyse.scoring.ml.features import candidates
from packages.analyse.scoring.ml.graph_convert import dataset_from_items
from packages.analyse.scoring.ml.noise import noise_reason
from packages.analyse.scoring.ml.sample import write_sample
from packages.analyse.scoring.ml.synth.days import (
    START,
    World,
    blind,
    directory,
    persona,
    working_days,
)
from packages.analyse.scoring.ml.synth.sectors import SECTORS

BTP = SECTORS[1]


def test_jours_ouvres_et_annuaire():
    days = working_days(START, 15)
    assert len(days) == 15 and all(d.weekday() < 5 for d in days)
    manager, people = directory(BTP)
    assert manager.email == "karim.mansouri@batirest.test"
    p = persona(BTP)
    assert p.sender_kind("gilles.arnaud@batirest.test") == "hiérarchie"
    assert p.sender_kind("bernard.collin@mairie.de.saint.ouen.les.pres.test") == "client"
    assert p.sender_kind("noreply@rh.batirest.test") == "automate"
    assert {c.kind for c in people} == {
        "hiérarchie",
        "direction",
        "collègue",
        "client",
        "fournisseur",
    }


def test_un_monde_deterministe_avec_fils_reponses_bruit_et_reunions():
    items = World(BTP).build()
    assert [i.item_id for i in items] == [i.item_id for i in World(BTP).build()]
    manager = persona(BTP).address
    mails = [i for i in items if i.source == "mail"]
    sent = [
        m for m in mails if any(p.role == "from" and p.email == manager for p in m.participants)
    ]
    assert sent  # la manager répond : « déjà répondu » existe
    threads = Counter(m.thread_id for m in mails)
    assert max(threads.values()) >= 3  # ouverture, réponse ou relance, clôture
    assert any(noise_reason(m) for m in mails)  # lettres d'information
    events = [i for i in items if i.source == "calendar"]
    invitations = {m.correlation_id for m in mails if m.correlation_id}
    assert events and {e.correlation_id for e in events} <= invitations
    assert all(i.owner_user_id == "u_synth_btp" for i in items)
    assert all(i.created_at.weekday() < 5 for i in mails)


def test_le_monde_alimente_les_candidats_du_brief():
    items = World(BTP).build()
    dataset = dataset_from_items(items, persona(BTP).address)
    assert any(dataset.extras[i.item_id].is_sent for i in items if i.source == "mail")
    assert len(candidates(dataset, working_days(START, 15)[5])) >= 4


def test_carte_a_l_aveugle_sans_intention_calculee():
    items = World(BTP).build()
    by_id = {i.item_id: i for i in items}
    mail = next(i for i in items if i.source == "mail" and "gilles" in i.participants[0].email)
    side = {
        "item_id": mail.item_id,
        "sender_kind": "hiérarchie",
        "chips": [
            "N+1 (hiérarchie)",
            "demande une action",
            "attend une réponse",
            "échéance aujourd'hui ou demain (avant le jeudi 5 novembre)",
            "déjà répondu",
            "fil de 3 messages",
            "reçu il y a 5 h",
        ],
    }
    roles = {mail.participants[0].email: "Gilles Arnaud · Directeur d'agence"}
    assert blind(side, roles, by_id)["chips"] == [
        "Gilles Arnaud · Directeur d'agence",
        "déjà répondu",
        "fil de 3 messages",
        "reçu il y a 5 h",
    ]
    auto = next(
        i for i in items if i.source == "mail" and i.participants[0].email.startswith("support")
    )
    chips = blind({"item_id": auto.item_id, "sender_kind": "automate", "chips": []}, roles, by_id)
    assert chips["chips"] == ["Support informatique · message automatique"]


def _pair(pid):
    side = {
        "item_id": "x",
        "sender": "S",
        "sender_kind": "client",
        "title": "T",
        "text": "t",
        "chips": ["Client"],
    }
    return {
        "item_id": pid,
        "day": "2026-11-05",
        "manager": "Karim",
        "blind": True,
        "a": side | {"title": "A"},
        "b": side | {"title": "B"},
    }


def test_serveur_paires_a_deux_fichier_separe(tmp_path):
    synth_pairs = tmp_path / "pairs.jsonl"
    write_sample([_pair(f"s{n:04d}") for n in range(20)], synth_pairs)
    httpd = annotate.serve(
        0,
        tmp_path / "s.jsonl",
        tmp_path / "intent_labels.jsonl",
        annotator="theo",
        pairs_mode=True,
        synth_pairs_path=synth_pairs,
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/pairs/next") as res:
            task = json.loads(res.read())["task"]
        assert task["manager"] == "Karim"
        req = urllib.request.Request(
            base + "/api/pairs/answer", json.dumps(task | {"value": "A"}).encode(), method="POST"
        )
        with urllib.request.urlopen(req) as res:
            assert res.status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert (tmp_path / "pair_labels_theo.jsonl").exists()
    assert not (tmp_path / "pair_labels.jsonl").exists()
