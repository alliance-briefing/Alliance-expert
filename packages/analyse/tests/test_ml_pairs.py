"""Tests de la fiche d'un mail à l'instant du brief, du tirage des paires et du mode paires."""

from __future__ import annotations

import json
import threading
import urllib.request
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from contracts.models import Item
from packages.analyse.clock import PARIS
from packages.analyse.rag.embeddings import HashEmbedder
from packages.analyse.scoring.ml import annotate
from packages.analyse.scoring.ml.annotate import LabelStore, PairAnnotator
from packages.analyse.scoring.ml.features import (
    VECTOR_NAMES,
    brief_instant,
    candidates,
    mail_features,
    vectorize,
)
from packages.analyse.scoring.ml.graph_convert import GraphDataset, MailExtras
from packages.analyse.scoring.ml.intents import IntentModel
from packages.analyse.scoring.ml.pairs import build_pairs, chips, intent_probs
from packages.analyse.scoring.ml.persona import GRAPH_HELENE
from packages.analyse.scoring.ml.sample import write_sample

DOM = "verdier-expertises.example"
ME = f"helene.marchand@{DOM}"
DAY = date(2026, 10, 14)
NOW = brief_instant(DAY)  # mercredi 14 octobre, 8 h 30


def _mail(item_id, sender, at, title="Point", text="Bonjour", thread=None, corr=None) -> Item:
    return Item(
        item_id=f"mail:{item_id}",
        source="mail",
        provider="microsoft",
        owner_user_id="u",
        title=title,
        snippet=text,
        content_text=text,
        created_at=at,
        updated_at=at,
        participants=[{"name": sender, "email": sender, "role": "from"}],
        thread_id=thread,
        correlation_id=corr,
        url="https://outlook.example/m",
        acl={"visible_to": ["u"]},
        collected_at=at,
        connector={"name": "t", "version": "0.1.0"},
    )


def _event(item_id, start, corr) -> Item:
    return Item(
        item_id=f"calendar:{item_id}",
        source="calendar",
        provider="microsoft",
        owner_user_id="u",
        title="Comité",
        snippet="",
        created_at=start - timedelta(days=5),
        updated_at=start - timedelta(days=5),
        start_at=start,
        end_at=start + timedelta(hours=1),
        correlation_id=corr,
        url="https://outlook.example/e",
        acl={"visible_to": ["u"]},
        collected_at=start,
        connector={"name": "t", "version": "0.1.0"},
    )


def _dataset() -> GraphDataset:
    h = timedelta(hours=1)
    items = [
        _mail(
            "ask",
            f"bruno.lefevre@{DOM}",
            NOW - 2 * h,
            "Document à relire",
            "Peux-tu le relire avant le jeudi 15 octobre ?",
            thread="t1",
        ),
        _mail(
            "old", f"hugo.lambert@{DOM}", NOW - 50 * h, "Question", "Une question ?", thread="t2"
        ),
        _mail("reply", ME, NOW - 40 * h, "RE : Question", "Voilà.", thread="t2"),
        _mail("t2b", f"lea.giraud@{DOM}", NOW - 30 * h, "RE : Question", "Merci", thread="t2"),
        _mail(
            "inv",
            f"sandrine.moreau@{DOM}",
            NOW - 5 * h,
            "Comité",
            "(aucune description)",
            corr="meet:comite",
        ),
        _mail(
            "sec",
            "securite@portail-it.example",
            NOW - 3 * h,
            "Connexion inhabituelle",
            "Connexion inhabituelle détectée.",
        ),
        _mail("news", "news@x.example", NOW - 1 * h, "Lettre", "Se désabonner"),
        _mail("future", f"hugo.lambert@{DOM}", NOW + h, "Après le brief", "…", thread="t2"),
        _mail("tooold", f"hugo.lambert@{DOM}", NOW - 80 * h, "Vieux", "…"),
        _event("e1", NOW + 26 * h, "meet:comite"),
    ]
    extras = {
        i.item_id: MailExtras(
            "from" if i.item_id == "mail:reply" else "to",
            i.item_id == "mail:reply",
            None,
            None,
            False,
            False,
            (),
        )
        for i in items
        if i.source == "mail"
    }
    return GraphDataset(items=items, extras=extras)


def _probs(dataset, value=0.2):
    ids = [i.item_id for i in dataset.items if i.source == "mail"]
    return {
        f: {i: (0.9 if i == "mail:ask" else value) for i in ids}
        for f in ("action", "reponse", "incident")
    }


def test_candidats_du_matin_hors_bruit_envois_futur_et_trop_vieux():
    ids = [i.item_id for i in candidates(_dataset(), DAY)]
    assert ids == ["mail:ask", "mail:inv", "mail:old", "mail:sec", "mail:t2b"]


def test_fiche_a_l_instant_du_brief():
    dataset = _dataset()
    feats = {
        f.item_id: f
        for f in mail_features(
            candidates(dataset, DAY), dataset, NOW, GRAPH_HELENE, _probs(dataset)
        )
    }
    ask = feats["mail:ask"]
    assert (ask.sender_kind, ask.action, ask.deadline) == (
        "hiérarchie",
        0.9,
        "aujourd'hui ou demain",
    )
    assert ask.deadline_phrase == "avant le jeudi 15 octobre"
    assert ask.fresh and ask.ball_in_court and ask.hierarchy_in_thread
    old = feats["mail:old"]  # Hélène a répondu après ; le message « futur » est ignoré
    assert not old.ball_in_court and old.thread_size == 3 and not old.fresh
    assert feats["mail:t2b"].ball_in_court  # message reçu après sa réponse
    assert feats["mail:inv"].meeting_soon and feats["mail:inv"].sender_kind == "direction"
    assert feats["mail:sec"].alerte and feats["mail:sec"].sender_kind == "automate"
    vector = vectorize(ask)
    assert len(vector) == len(VECTOR_NAMES)
    assert vector[VECTOR_NAMES.index("expéditeur : hiérarchie")] == 1.0
    assert vector[VECTOR_NAMES.index("échéance : aujourd'hui ou demain")] == 1.0
    assert "N+1 (hiérarchie)" in chips(ask, 2) and "reçu il y a 2 h" in chips(ask, 2)
    assert "reçu il y a 2 jours" in chips(old, 50) and "déjà répondu" in chips(old, 50)


def test_probabilites_d_intentions_par_item():
    dataset = _dataset()
    items = [i for i in dataset.items if i.source == "mail"]
    embedder = HashEmbedder(16)
    model = IntentModel(
        {f: np.zeros(17) for f in ("action", "reponse", "incident")},
        {f: False for f in ("action", "reponse", "incident")},
        "hash",
    )
    probs = intent_probs(items, model, embedder.embed)
    assert set(probs["action"]) == {i.item_id for i in items}
    assert probs["action"]["mail:ask"] == pytest.approx(0.5)
    assert intent_probs([], model, embedder.embed) == {"action": {}, "reponse": {}, "incident": {}}


def _many_mornings() -> GraphDataset:
    items = []
    for d in range(6):
        for n in range(6):
            at = datetime(2026, 10, 5 + d, 9, 0, tzinfo=PARIS) + timedelta(minutes=n)
            items.append(
                _mail(f"{d}-{n}", f"hugo.lambert@{DOM}", at, f"Objet {d}-{n}", f"Texte {d} {n}")
            )
    extras = {i.item_id: MailExtras("to", False, None, None, False, False, ()) for i in items}
    return GraphDataset(items=items, extras=extras)


def test_paires_du_meme_matin_deterministes_avec_reserve():
    dataset = _many_mornings()
    pairs = build_pairs(dataset, _probs(dataset), n_pairs=10, n_heldout=2)
    assert pairs == build_pairs(dataset, _probs(dataset), n_pairs=10, n_heldout=2)
    assert [p["item_id"] for p in pairs] == [f"p{n:03d}" for n in range(10)]
    assert sum(p["heldout"] for p in pairs) == 2
    assert {p["kind"] for p in pairs} == {"hasard", "serrée"}
    for p in pairs:
        assert p["a"]["item_id"] != p["b"]["item_id"]
        received = {datetime.fromisoformat(p[s]["received_at"]) for s in ("a", "b")}
        now = brief_instant(date.fromisoformat(p["day"]))
        assert all(now - timedelta(days=3) <= r < now for r in received)
    counts = {}
    for p in pairs:
        for s in ("a", "b"):
            counts[p[s]["item_id"]] = counts.get(p[s]["item_id"], 0) + 1
    assert max(counts.values()) <= 2


def _pair(pid):
    side = {
        "item_id": "x",
        "sender": "S",
        "sender_kind": "client",
        "title": "T",
        "text": "t",
        "chips": ["client"],
    }
    return {
        "item_id": pid,
        "day": "2026-10-14",
        "a": side | {"title": "A"},
        "b": side | {"title": "B"},
    }


def test_mode_paires_repetition_inversee_et_ramenee(tmp_path):
    pairs = [_pair(f"p{n}") for n in range(20)]
    annotator = PairAnnotator(pairs, LabelStore(tmp_path / "pl.jsonl"))
    assert annotator.state() == {"done": 0, "total": 22}
    while (task := annotator.next_task()) is not None:
        if task["repeat"]:
            assert task["a"]["title"] == "B"  # côtés inversés
            annotator.answer(task | {"value": "B"})  # « B d'abord » à l'écran = A d'origine
        else:
            annotator.answer(task | {"value": "A"})
    repeats = {k: v for k, v in annotator.store.answers.items() if k[2]}
    assert len(repeats) == 2 and set(repeats.values()) == {"A"}
    with pytest.raises(ValueError):
        annotator.answer({"pair_id": "p0", "value": "C"})


def test_serveur_page_et_api_des_paires(tmp_path):
    write_sample(
        [
            {
                "item_id": "m0",
                "template": "t",
                "sender_kind": "client",
                "sender": "X",
                "title": "O",
                "text": "t",
            }
        ],
        tmp_path / "s.jsonl",
    )
    write_sample([_pair("p0")], tmp_path / "pairs.jsonl")
    httpd = annotate.serve(
        0,
        tmp_path / "s.jsonl",
        tmp_path / "l.jsonl",
        tmp_path / "pairs.jsonl",
        tmp_path / "pl.jsonl",
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/paires") as res:
            assert b"textContent" in res.read()
        with urllib.request.urlopen(base + "/api/pairs/next") as res:
            task = json.loads(res.read())["task"]
        req = urllib.request.Request(
            base + "/api/pairs/answer", json.dumps(task | {"value": "="}).encode(), method="POST"
        )
        with urllib.request.urlopen(req) as res:
            assert json.loads(res.read()) == {"ok": True}
        with urllib.request.urlopen(base + "/api/pairs/state") as res:
            assert json.loads(res.read())["done"] == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_une_seconde_serie_vient_apres_sans_changer_la_premiere(tmp_path):
    first = [_pair(f"p{n}") for n in range(20)]
    second = [_pair(f"q{n}") | {"round": 2} for n in range(10)]
    before = PairAnnotator(first, LabelStore(tmp_path / "a.jsonl")).tasks
    after = PairAnnotator(first + second, LabelStore(tmp_path / "b.jsonl")).tasks
    assert after[: len(before)] == before
    assert {pid for pid, _ in after[len(before) :]} == {f"q{n}" for n in range(10)}
