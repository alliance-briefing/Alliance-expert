"""Tests de la 2e série d'étiquettes action / réponse (étape A3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from contracts.models import Item
from packages.analyse.scoring.ml.annotate import Annotator, LabelStore
from packages.analyse.scoring.ml.graph_convert import GraphDataset, MailExtras
from packages.analyse.scoring.ml.sample_round2 import demo_rows, uncertain_rows

AT = datetime(2026, 10, 7, 9, 0, tzinfo=timezone(timedelta(hours=2)))


def _mail(item_id: str, title: str, sender: str, text: str = "Texte") -> Item:
    return Item(
        item_id=f"mail:{item_id}",
        source="mail",
        provider="demo",
        owner_user_id="u",
        title=title,
        snippet=text,
        created_at=AT,
        updated_at=AT,
        participants=[{"name": sender, "email": sender, "role": "from"}],
        url="https://x.example/m",
        acl={"visible_to": ["u"]},
        collected_at=AT,
        connector={"name": "t", "version": "0.1.0"},
    )


def _dataset(items):
    return GraphDataset(
        items=items,
        extras={i.item_id: MailExtras("to", False, None, None, False, False, ()) for i in items},
    )


def _card(item_id):
    return {"item_id": f"mail:{item_id}"}


def test_demo_un_mail_par_modele_absent_des_paires():
    expert = "sophie.nguyen@reseau-expertis.test"
    items = [
        _mail("a1", "Dossier 2026-1 : photos", expert),
        _mail("a2", "Dossier 2026-2 : photos", expert),  # même modèle que a1
        _mail("b", "Rapport à relire", expert),
        _mail("c", "Réunion de service", expert),
    ]
    pairs = [{"a": _card("c"), "b": _card("c")}]  # le modèle de c sert à la mesure
    rows = demo_rows(_dataset(items), pairs)
    assert sorted(r["template"] for r in rows) == ["Dossier #-# : photos", "Rapport à relire"]
    assert all(r["round"] == 2 and r["dataset"] == "démo" for r in rows)
    assert rows == demo_rows(_dataset(items), pairs)  # graine fixée


def test_david_les_plus_hesitants_hors_reserve_et_deja_etiquetes():
    sender = "hugo.lambert@verdier-expertises.example"
    items = [_mail(k, f"Objet {k}", sender, f"Texte {k}") for k in "abcde"]
    pairs = [
        {"heldout": False, "a": _card("a"), "b": _card("b")},
        {"heldout": False, "a": _card("c"), "b": _card("d")},
        {"heldout": True, "a": _card("e"), "b": _card("a")},  # e réservé : jamais étiqueté
    ]
    probs = {
        "action": {"mail:a": 0.9, "mail:b": 0.55, "mail:c": 0.1, "mail:d": 0.48, "mail:e": 0.5},
        "reponse": {"mail:a": 0.1, "mail:b": 0.9, "mail:c": 0.6, "mail:d": 0.1, "mail:e": 0.5},
    }
    rows = uncertain_rows(_dataset(items), pairs, probs, exclude={"mail:b"}, n=2)
    assert [r["item_id"] for r in rows] == ["mail:d", "mail:c"]


def test_la_2e_serie_vient_apres_et_ne_porte_que_sur_action_et_reponse(tmp_path):
    first = [
        {"item_id": f"m{n}", "sender": "x", "sender_kind": "client", "title": "t", "text": "t"}
        for n in range(20)
    ]
    second = [row | {"item_id": f"r{n}", "round": 2} for n, row in enumerate(first[:10])]
    before = Annotator(first, LabelStore(tmp_path / "a.jsonl")).queues
    after = Annotator(first + second, LabelStore(tmp_path / "b.jsonl")).queues
    for feature, tasks in after.items():
        assert tasks[: len(before[feature])] == before[feature]
        added = {item_id for item_id, _ in tasks[len(before[feature]) :]}
        assert added == (
            {f"r{n}" for n in range(10)} if feature in ("action", "reponse") else set()
        )
