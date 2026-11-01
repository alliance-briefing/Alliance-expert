"""Corpus d'entraînement simulé (M1-M2) : monde, étiqueteur, écrivain, déterminisme, chargeur."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta

import pytest

from packages.collect.trainset import build, load
from packages.collect.trainset import phrases as P
from packages.collect.trainset.facts import ACTIONS, Fact, label, label_on
from packages.collect.trainset.metiers import METIERS
from packages.collect.trainset.world import all_personas, at

DAY = date(2026, 11, 4)


@pytest.fixture(scope="module")
def personas():
    return {p.key: p for p in all_personas()}


@pytest.fixture(scope="module")
def built(personas):
    """Deux personas complètes : une d'entraînement, une de T-gen."""
    return {k: build.build_persona(personas[k]) for k in ("logistique-1", "hotel-1")}


def _fact(source: str, act: str = "rien", **kw) -> Fact:
    p = all_personas()[0]
    kw.setdefault("author", p.team[0])
    return Fact(
        item_id=f"{source}:t-1",
        source=source,
        at=at(DAY, 9.0),
        case_id="c",
        case_type="test",
        act=act,
        **kw,
    )


# --- le monde ---------------------------------------------------------------------------------


def test_quarante_personas_dont_quatre_reservees_a_tgen(personas):
    assert len(personas) == 40
    splits = Counter(p.split for p in personas.values())
    assert splits == {"train": 36, "tgen": 4}
    reserved = {m.key for m in METIERS if m.reserved}
    assert reserved == {"achats", "hotel", "laboratoire", "service_client"}
    assert all((p.split == "tgen") == (p.metier.key in reserved) for p in personas.values())
    assert len({p.owner for p in personas.values()}) == 40
    assert len({p.manager.email for p in personas.values()}) == 40


def test_aucun_domaine_reel_ni_expertise_automobile():
    for m in METIERS:
        assert m.domain.endswith(".test")
        assert "auto" not in m.key and "expert" not in m.company.lower()


# --- l'étiqueteur applique le guide -----------------------------------------------------------


def test_etiqueteur_reunions():
    meet = {"start": at(DAY, 14.0), "end": at(DAY, 15.0)}
    assert label(_fact("calendar", meta=meet | {"response": "accepted"}))["action"] == "rien"
    assert (
        label(_fact("calendar", meta=meet | {"organizer_me": True, "response": "organizer"}))[
            "action"
        ]
        == "preparer"
    )
    assert label(_fact("calendar", meta=meet | {"bring": True}))["action"] == "preparer"
    assert label(_fact("calendar", meta=meet | {"response": "none"}))["action"] == "repondre"
    conflict = meet | {"response": "tentative", "conflict": True}
    assert label(_fact("calendar", meta=conflict))["action"] == "repondre"
    cancelled = meet | {"cancelled": True, "organizer_me": True}
    assert label(_fact("calendar", meta=cancelled))["action"] == "rien"


def test_etiqueteur_taches_et_fichiers():
    assert label(_fact("task", "decider", meta={"completed": False}))["action"] == "decider"
    assert label(_fact("task", "faire", meta={"completed": True}))["action"] == "rien"
    assert label(_fact("file", meta={"by_me": True}))["action"] == "rien"
    assert label(_fact("file", meta={"linked_task_act": "decider"}))["action"] == "decider"
    assert label(_fact("file", meta={"linked_meeting": True}))["action"] == "preparer"
    assert label(_fact("file", meta={"to_sign": True}))["action"] == "decider"
    assert label(_fact("file", meta={"review": True}))["action"] == "relire"
    assert label(_fact("file", meta={"cr": True}))["action"] == "informer"


def test_etiqueteur_mails():
    assert label(_fact("mail", "repondre", secondary="decider"))["action"] == "decider"
    assert label(_fact("mail", "faire", to_manager=False))["action"] == "informer"
    injection = label(_fact("mail", "rien", alert=True, trap="injection"))
    assert injection == {"action": "rien", "urgence": "normal", "alerte": True}
    assert label(_fact("mail", "decider", sent=True))["action"] == "rien"
    assert label(_fact("mail", "decider", trap="doublon"))["action"] == "rien"
    assert label(_fact("mail", "relire", tone="sans_urgence"))["urgence"] == "sans_urgence"


def test_l_etiquette_depend_du_jour():
    row = {
        "visible_from": at(DAY, 9.0).isoformat(),
        "action": "decider",
        "urgence": "normal",
        "alerte": False,
        "resolved_at": at(DAY + timedelta(days=1), 11.0).isoformat(),
        "ends_at": None,
    }
    assert label_on(row, at(DAY, 8.5)) is None
    assert label_on(row, at(DAY + timedelta(days=1), 8.5))["action"] == "decider"
    assert label_on(row, at(DAY + timedelta(days=2), 8.5))["action"] == "rien"
    meeting = row | {
        "action": "preparer",
        "resolved_at": None,
        "ends_at": at(DAY, 15.0).isoformat(),
    }
    assert label_on(meeting, at(DAY + timedelta(days=1), 8.5))["action"] == "rien"


# --- le corpus --------------------------------------------------------------------------------


def test_items_valides_isoles_et_coherents(built, personas):
    for key, (items, rows) in built.items():
        owner = personas[key].owner
        assert len(items) == len(rows) > 300
        assert len({i.item_id for i in items}) == len(items)
        assert all(i.owner_user_id == owner and i.acl.visible_to == [owner] for i in items)
        assert [i.item_id for i in items] == [r["item_id"] for r in rows]
        ids = {i.item_id for i in items}
        assert all(set(r["links"]) <= ids for r in rows)
        assert all(r["action"] in ACTIONS for r in rows)


def test_toutes_les_actions_et_les_pieges_sont_presents(built):
    rows = [r for _, rs in built.values() for r in rs]
    assert {r["action"] for r in rows} == set(ACTIONS)
    by_source = {
        s: {r["action"] for r in rows if r["source"] == s}
        for s in ("mail", "calendar", "task", "file")
    }
    assert by_source["mail"] == set(ACTIONS)
    assert {"preparer", "repondre", "rien"} <= by_source["calendar"]
    assert {"decider", "faire", "rien"} <= by_source["task"]
    assert {"decider", "preparer", "relire", "informer", "rien"} <= by_source["file"]
    assert {"injection"} <= {r["trap"] for r in rows}
    assert any(r["alerte"] for r in rows)


def test_l_echeance_ecrite_n_est_pas_une_urgence(built):
    rows = [r for _, rs in built.values() for r in rs if r["source"] == "mail"]
    with_deadline = [r for r in rows if r["deadline"]]
    assert Counter(r["urgence"] for r in with_deadline)["normal"] > len(with_deadline) / 2


def test_le_texte_ne_trahit_pas_l_etiquette(built):
    """Le contrat n'expose ni l'affaire ni l'étiquette : aucun mot du scénario dans l'Item."""
    for items, _ in built.values():
        for item in items:
            blob = item.model_dump_json()
            assert "case_type" not in blob and "resolved_at" not in blob
            assert "{" not in item.title and "[" not in (item.content_text or "")


def test_gabarits_reserves_jamais_a_l_entrainement(built):
    def reserved(tid: str) -> bool:
        return int(tid.rsplit(".", 1)[1]) % P.RESERVED_EVERY == P.RESERVED_EVERY - 1

    train = [t for r in built["logistique-1"][1] for t in r["templates"]]
    tgen = [t for r in built["hotel-1"][1] for t in r["templates"]]
    assert train and not any(reserved(t) for t in train if not t.startswith("security"))
    assert any(reserved(t) for t in tgen)


def test_deterministe_et_conforme_au_manifeste(personas):
    a = build.render(personas["sante-2"])
    assert a == build.render(personas["sante-2"])
    assert build.verify(keys=["sante-2", "achats-1"]) == []


def test_chargeur(tmp_path):
    build.build_all(tmp_path)
    keys = load.personas("tgen", data=tmp_path)
    assert len(keys) == 4
    items = load.load_items(keys[0], data=tmp_path)
    labels = load.labels_at(keys[0], at(date(2026, 11, 10), 8.5), data=tmp_path)
    assert labels and set(labels) <= {i.item_id for i in items}
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["distribution"]["train"]["elements"] > 15000
    with pytest.raises(FileNotFoundError, match="python -m packages.collect.trainset"):
        load.load_items("inconnu", data=tmp_path)
