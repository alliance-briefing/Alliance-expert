"""Tests de l'outil d'annotation unifié (M4) : cartes à l'aveugle, ordre et répétitions,
reprise après fermeture, accord, arbitrage, serveur local."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pytest

from packages.analyse.scoring.ml import annoter
from packages.analyse.scoring.ml.annoter import (
    AnswerStore,
    Session,
    agreement_report,
    answers_path,
    build_trial,
    card,
    case_card,
    check_answer,
    check_blind,
    cohen_kappa,
    disagreements,
    final_labels,
    first_answers,
    load_series,
    open_session,
    order,
    sample_corpus,
    viewpoint,
    when,
    write_series,
)

DAY = date(2026, 11, 3)


@pytest.fixture(scope="module")
def logistique():
    items, scenario = sample_corpus("logistique-1")
    return items, scenario, viewpoint("logistique-1", DAY)


@pytest.fixture(scope="module")
def trial(logistique):
    items, scenario, vp = logistique
    return build_trial(items, scenario, vp, prefix="t")


def _element(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "kind": "element",
        "context": "c",
        "card": {"item_id": task_id, "source": "mail", "title": "t", "when": "w"}
        | {"fields": [], "text": ""},
    }


def _pair(task_id: str, kind: str = "paire") -> dict:
    return {"task_id": task_id, "kind": kind, "context": "c", "a": {"n": "a"}, "b": {"n": "b"}}


def _series(folder: Path, tasks: list[dict], name: str = "s") -> str:
    write_series(tasks, folder / f"{name}.jsonl")
    return name


# --- Cartes --------------------------------------------------------------------------


def test_viewpoint_names_the_manager_and_relations(logistique):
    _, _, vp = logistique
    assert vp.context.startswith("Vous êtes ") and "mardi 3 novembre à 8 h 30" in vp.context
    assert vp.moment.hour == 8 and vp.moment.minute == 30
    roles = " ".join(vp.people.values())
    assert "(votre N+1)" in roles and "(votre équipe)" in roles and "(envoi automatique)" in roles


def test_when_reads_dates_from_the_brief_moment(logistique):
    _, _, vp = logistique
    tz = vp.moment.tzinfo
    assert when(datetime(2026, 11, 3, 7, 5, tzinfo=tz), vp) == "aujourd'hui à 07:05"
    assert when(datetime(2026, 11, 2, 18, 0, tzinfo=tz), vp) == "hier à 18:00"
    assert when(datetime(2026, 11, 4, 9, 0, tzinfo=tz), vp) == "demain à 09:00"
    assert when(datetime(2026, 11, 5, 10, 0, tzinfo=tz), vp) == "jeudi 5 novembre à 10:00"


def test_cards_cover_the_four_sources_without_labels(logistique):
    items, _, vp = logistique
    cards = {}
    for item in items:
        cards.setdefault(item.source, card(item, vp))
    assert set(cards) == {"mail", "calendar", "task", "file"}
    for c in cards.values():
        check_blind(c)
        assert set(c) == {"item_id", "source", "title", "when", "fields", "text"}
    labels = dict(cards["mail"]["fields"])
    assert "De" in labels and "À" in labels
    assert "Organisateur" in dict(cards["calendar"]["fields"])
    assert {"Statut", "Liste"} <= set(dict(cards["task"]["fields"]))
    assert {"Modifié par", "Emplacement"} <= set(dict(cards["file"]["fields"]))


def test_manager_appears_as_vous(logistique):
    items, _, vp = logistique
    sent = [
        card(i, vp)
        for i in items
        if any(p.role == "from" and p.email == vp.manager_email for p in i.participants)
    ]
    assert sent and all(dict(c["fields"])["De"] == "vous" for c in sent)


def test_late_task_is_flagged(logistique):
    items, _, vp = logistique
    task = next(i for i in items if i.source == "task" and i.due_at)
    later = annoter.Viewpoint(vp.manager_email, task.due_at.replace(year=2027), vp.people, "")
    assert dict(card(task, later)["fields"])["Échéance"].endswith("(en retard)")


def test_case_card_lists_items_in_order(logistique):
    items, scenario, vp = logistique
    case = next(r["case_id"] for r in scenario if r["case_type"] == "validation")
    members = [
        i
        for i in items
        if any(r["item_id"] == i.item_id and r["case_id"] == case for r in scenario)
    ]
    c = case_card(members, vp)
    check_blind(c)
    assert len(c["items"]) == len(members)
    assert not c["title"].startswith(("RE", "Re"))


def test_trial_is_blind_complete_and_deterministic(logistique, trial):
    items, scenario, vp = logistique
    assert Counter(t["kind"] for t in trial) == {"element": 12, "paire": 6, "affaires": 3}
    for task in trial:
        check_blind(task)
    assert build_trial(items, scenario, vp, prefix="t") == trial


# --- Séries --------------------------------------------------------------------------


def test_check_blind_refuses_labels_at_any_depth():
    with pytest.raises(ValueError, match="score"):
        check_blind({"a": {"items": [{"score": 3}]}})
    with pytest.raises(ValueError, match="action"):
        check_blind({"card": {"action": "faire"}})


def test_write_series_refuses_a_task_with_a_label(tmp_path):
    with pytest.raises(ValueError, match="case_type"):
        write_series([_element("e1") | {"case_type": "validation"}], tmp_path / "x.jsonl")


@pytest.mark.parametrize(
    "tasks, message",
    [
        ([_element("e1"), _element("e1")], "double"),
        ([_element("e1") | {"kind": "trio"}], "inconnue"),
        ([{"task_id": "p1", "kind": "paire", "a": {}}], "incomplète"),
        ([{"task_id": "e1", "kind": "element"}], "incomplète"),
    ],
)
def test_load_series_refuses_bad_files(tmp_path, tasks, message):
    path = tmp_path / "bad.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_series(path)


def test_order_is_stable_personal_and_repeats_late():
    tasks = [_element(f"e{n:03d}") for n in range(40)]
    plan = order(tasks, "theo", "s")
    assert plan == order(tasks, "theo", "s")
    assert plan != order(tasks, "paul", "s")
    repeats = [t for t, rep in plan if rep]
    assert len(repeats) == 4
    for task_id in repeats:
        assert plan.index((task_id, False)) < 20 <= plan.index((task_id, True))  # 1re / 2e moitié
    assert {t for t, _ in plan} == {t["task_id"] for t in tasks}


def test_order_keeps_only_tasks_for_the_annotator():
    tasks = [_element("e1") | {"pour": ["paul"]}, _element("e2")]
    assert {t for t, _ in order(tasks, "theo", "s")} == {"e2"}


def test_check_answer():
    assert check_answer("element", {"action": "faire", "urgence": "urgent", "alerte": True}) == {
        "action": "faire",
        "urgence": "urgent",
        "alerte": True,
    }
    assert check_answer("affaires", {"choix": "=", "x": 1}) == {"choix": "="}
    for kind, bad in [
        ("element", {"action": "dormir", "urgence": "normal", "alerte": False}),
        ("element", {"action": "faire", "urgence": "normal", "alerte": "non"}),
        ("paire", {"choix": "C"}),
        ("paire", "A"),
    ]:
        with pytest.raises(ValueError):
            check_answer(kind, bad)


# --- Session : réponses, répétitions, reprise ------------------------------------------


def test_repeated_pair_is_shown_swapped_and_stored_in_series_order(tmp_path):
    task = _pair("p1")
    store = AnswerStore(tmp_path / "a.jsonl")
    session = Session([task], store, [("p1", False), ("p1", True)])
    shown = session.state()["tasks"]
    assert shown[0]["a"] == {"n": "a"} and shown[1]["a"] == {"n": "b"}
    assert all("pour" not in t for t in shown)
    session.answer({"key": "p1|0", "answer": {"choix": "A"}})
    session.answer({"key": "p1|1", "answer": {"choix": "B"}})  # B à l'écran = A de la série
    assert store.answers == {("p1", False): {"choix": "A"}, ("p1", True): {"choix": "A"}}
    assert session.state()["answers"] == {"p1|0": {"choix": "A"}, "p1|1": {"choix": "B"}}


def test_answers_survive_a_restart_and_last_one_counts(tmp_path):
    path = tmp_path / "a.jsonl"
    session = Session([_element("e1")], AnswerStore(path), [("e1", False)])
    session.answer(
        {"key": "e1|0", "answer": {"action": "rien", "urgence": "normal", "alerte": False}}
    )
    session.answer(
        {"key": "e1|0", "answer": {"action": "faire", "urgence": "normal", "alerte": False}}
    )
    again = Session([_element("e1")], AnswerStore(path), [("e1", False)])
    assert again.state()["answers"]["e1|0"]["action"] == "faire"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_session_refuses_unknown_task(tmp_path):
    session = Session([_element("e1")], AnswerStore(tmp_path / "a.jsonl"), [("e1", False)])
    with pytest.raises(ValueError, match="inconnue"):
        session.answer({"key": "e1|1", "answer": {"action": "rien"}})
    with pytest.raises(ValueError, match="inconnue"):
        session.answer({"key": "zz|0"})


def test_open_session_checks_annotator(tmp_path):
    name = _series(tmp_path, [_element("e1")])
    with pytest.raises(ValueError, match="annotateur"):
        open_session(name, "bob", folder=tmp_path)
    session, title = open_session(name, "theo", folder=tmp_path)
    assert "theo" in title and len(session.state()["tasks"]) == 1


# --- Accord, arbitrage, étiquettes finales ---------------------------------------------


def _answer(folder: Path, name: str, who: str, task_id: str, answer: dict, repeat=False) -> None:
    AnswerStore(answers_path(name, who, folder)).record(task_id, repeat, answer)


def test_cohen_kappa():
    assert cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "a", "b", "b"]) == 0.0
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_agreement_arbitrage_and_final_labels(tmp_path):
    name = _series(tmp_path, [_element("e1"), _element("e2"), _pair("p1"), _pair("p2")])
    same = {"action": "faire", "urgence": "normal", "alerte": False}
    _answer(tmp_path, name, "theo", "e1", same)
    _answer(tmp_path, name, "paul", "e1", same)
    _answer(tmp_path, name, "theo", "e2", same)
    _answer(tmp_path, name, "paul", "e2", same | {"action": "repondre"})
    _answer(tmp_path, name, "theo", "p1", {"choix": "A"})
    _answer(tmp_path, name, "paul", "p1", {"choix": "A"})
    _answer(tmp_path, name, "theo", "p1", {"choix": "B"}, repeat=True)  # incohérent
    _answer(tmp_path, name, "guillaume", "p2", {"choix": "="})

    by_who = first_answers(name, tmp_path)
    assert set(by_who) == {"theo", "paul", "guillaume"}
    conflicts = disagreements(load_series(tmp_path / f"{name}.jsonl"), by_who)
    assert [c["task_id"] for c in conflicts] == ["e2"]
    assert {a["qui"] for a in conflicts[0]["avis"]} == {"theo", "paul"}

    report = agreement_report(name, tmp_path)
    assert "paul / theo, element : 2 en commun" in report
    assert "theo : cohérence sur les répétitions 0/1" in report
    assert "Désaccords à arbitrer : 1" in report

    assert final_labels(name, tmp_path) == {"e1": same, "p1": {"choix": "A"}, "p2": {"choix": "="}}

    session, _ = open_session(name, None, arbitrage=True, folder=tmp_path)
    shown = session.state()["tasks"]
    assert [t["task_id"] for t in shown] == ["e2"] and "avis" in shown[0]
    session.answer({"key": "e2|0", "answer": same})
    assert final_labels(name, tmp_path)["e2"] == same
    assert "Désaccords à arbitrer : 0" in agreement_report(name, tmp_path)


# --- Serveur local et ligne de commande ------------------------------------------------


def test_server_round_trip(tmp_path):
    name = _series(tmp_path, [_element("e1")])
    session, title = open_session(name, "paul", folder=tmp_path)
    server = annoter.serve(session, title, port=0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert b"<title>Annotation</title>" in urllib.request.urlopen(base + "/").read()
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert state["title"] == title and state["tasks"][0]["key"] == "e1|0"
        body = {"key": "e1|0", "answer": {"action": "rien", "urgence": "normal", "alerte": False}}
        request = urllib.request.Request(base + "/api/answer", json.dumps(body).encode())
        assert json.loads(urllib.request.urlopen(request).read()) == {"ok": True}
        for url, data in [("/api/answer", b'{"key": "x|0"}'), ("/api/answer", b"{")]:
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(urllib.request.Request(base + url, data))
            assert err.value.code == 400
        for url, data in [("/nulle-part", None), ("/autre", b"{}")]:
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(urllib.request.Request(base + url, data))
            assert err.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
    assert (
        AnswerStore(answers_path(name, "paul", tmp_path)).answers[("e1", False)]["action"] == "rien"
    )


def test_create_trial_writes_a_blind_series(tmp_path):
    path = annoter.create_trial(tmp_path)
    tasks = load_series(path)
    assert {t["kind"] for t in tasks} == {"element", "paire", "affaires"}
    assert len({t["context"] for t in tasks}) == 2  # deux personas


def test_main_commands(monkeypatch, capsys):
    monkeypatch.setattr(annoter, "create_trial", lambda: Path("essai.jsonl"))
    monkeypatch.setattr(annoter, "agreement_report", lambda series: f"rapport {series}")
    assert annoter.main(["--creer-essai"]) == 0
    assert annoter.main(["--accord", "essai"]) == 0
    assert annoter.main([]) == 2
    out = capsys.readouterr().out
    assert "essai.jsonl" in out and "rapport essai" in out


# --- « Cité dans » : les liens d'un fichier ----------------------------------------------


def test_file_words_keep_the_subject_only():
    assert annoter.file_words("Contrat_blanchisserie_v4.pdf") == {"contrat", "blanchisserie"}
    assert annoter.file_words("Procédure_arrivée_v2.docx") == {"procedure", "arrivee"}
    assert annoter.file_words("Tarifs_2027_v5.pptx") == {"tarifs", "2027"}
    assert annoter.file_words("Codir_logistique_04-11.pptx") == {"codir", "logistique"}
    assert annoter.file_words("Budget_v3.xlsx") == set()  # un seul mot : aucun lien


def test_file_card_cites_known_items_only():
    items, scenario = sample_corpus("hotel-1")
    vp = viewpoint("hotel-1", DAY)
    rows = {r["item_id"]: r for r in scenario}
    known = [
        i for i in items if datetime.fromisoformat(rows[i.item_id]["visible_from"]) <= vp.moment
    ]
    contract = next(i for i in known if i.title == "Contrat_blanchisserie_v4.pdf")
    line = dict(card(contract, vp, known)["fields"])["Cité dans"]
    assert "Blanchisserie Alpine" in line and line.count("«") == 3 and "autre" in line
    later = [i for i in items if i.created_at > vp.moment and i.source != "file"]
    assert later and all(f"« {i.title} » (" not in line or i in known for i in later)
    assert "Cité dans" not in dict(card(contract, vp)["fields"])  # sans boîte connue
    assert annoter.cited_in(contract, [], vp) == ""
