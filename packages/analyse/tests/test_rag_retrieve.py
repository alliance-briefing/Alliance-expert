"""Historique des affaires (L2.6, R2) : même référence, seuil, veto, exclusions, isolation."""

from __future__ import annotations

import random
from datetime import date

import pytest

from contracts.demo_data import available_days, load_items
from packages.analyse.aggregate.cases import Case, group_into_cases, references
from packages.analyse.rag.embeddings import HashEmbedder
from packages.analyse.rag.index import IndexStore, UserIndex
from packages.analyse.rag.retrieve import history, verdict
from packages.analyse.tests.factories import OTHER_USER, USER, make_mail, make_task

DAY = date(2026, 10, 7)
PAST = "2026-10-05T09:00:00+02:00"
TODAY = "2026-10-07T08:00:00+02:00"


def index_with(*items) -> UserIndex:
    index = IndexStore(HashEmbedder()).for_user(USER)
    index.add(items)
    return index


def case(*items) -> Case:
    return Case(case_id=f"case:{min(item.item_id for item in items)}", items=items)


def linked(result: dict, case_id: str) -> list[str]:
    return [link.item.item_id for link in result[case_id]]


# --- même référence, seuil, veto ---------------------------------------------------------------


def test_same_reference_is_linked_whatever_the_score() -> None:
    index = index_with(make_mail("photos", PAST, title="Dossier 2026-3025 : photos reçues"))
    today = case(make_mail("q", TODAY, title="Expertise Kia Niro – dossier 2026-3025"))

    (link,) = history([today], index, DAY, threshold=0.99)[today.case_id]

    assert link.item.item_id == "mail:photos"
    assert link.same_reference is True
    assert link.reason == "même dossier 2026-3025"


def test_different_references_are_vetoed_even_if_the_texts_match() -> None:
    index = index_with(make_mail("autre", PAST, title="Dossier 2026-4752 : RAS"))
    today = case(make_mail("q", TODAY, title="Dossier 2026-5394 : RAS"))

    assert history([today], index, DAY, threshold=0.1)[today.case_id] == []


def test_a_close_subject_without_reference_is_linked_above_the_threshold() -> None:
    index = index_with(
        make_mail("entretiens", PAST, title="Entretiens annuels : planning à compléter"),
        make_mail("conges", PAST, title="Congés de fin d'année"),
    )
    today = case(make_mail("q", TODAY, title="Compléter le planning des entretiens annuels"))

    (link,) = history([today], index, DAY, threshold=0.5)[today.case_id]

    assert link.item.item_id == "mail:entretiens"
    assert link.same_reference is False
    assert link.reason.startswith("sujet proche (")


def test_below_the_threshold_nothing_is_linked() -> None:
    index = index_with(make_mail("conges", PAST, title="Congés de fin d'année"))
    today = case(make_mail("q", TODAY, title="Compléter le planning des entretiens annuels"))

    assert history([today], index, DAY)[today.case_id] == []


def test_a_reference_on_one_side_only_is_not_a_veto() -> None:
    index = index_with(make_mail("p", PAST, title="photos du véhicule Kia Niro"))
    today = case(make_mail("q", TODAY, title="photos du véhicule Kia Niro, dossier 2026-3025"))

    assert linked(history([today], index, DAY, threshold=0.5), today.case_id) == ["mail:p"]


# --- tri, k, meilleure similarité ---------------------------------------------------------------


def test_certain_links_come_first_then_score_and_k() -> None:
    index = index_with(
        make_mail("ref-a", PAST, title="Dossier 2026-3025 : relance"),
        make_task("ref-b", None, title="Clôturer le dossier 2026-3025"),
        make_mail("proche-1", PAST, title="expertise kia niro garage saint exupéry"),
        make_mail("proche-2", PAST, title="expertise kia niro garage"),
    )
    today = case(
        make_mail("q", TODAY, title="Expertise kia niro garage saint exupéry dossier 2026-3025")
    )

    links = history([today], index, DAY, k=3, threshold=0.5)[today.case_id]

    # Les deux liens certains d'abord (entre eux, par score), puis le plus proche des autres.
    assert {link.item.item_id for link in links[:2]} == {"mail:ref-a", "task:ref-b"}
    assert all(link.same_reference for link in links[:2])
    assert links[0].score >= links[1].score
    assert links[2].item.item_id == "mail:proche-1"
    assert len(links) == 3


def test_the_best_item_of_the_case_decides() -> None:
    index = index_with(make_mail("p", PAST, title="planning des entretiens annuels"))
    today = case(
        make_mail("a", TODAY, title="congés de fin d'année"),
        make_mail("b", TODAY, title="planning des entretiens annuels"),
    )

    assert linked(history([today], index, DAY), today.case_id) == ["mail:p"]


# --- exclusions, futur, isolation, déterminisme -------------------------------------------------


def test_today_items_and_excluded_ids_are_never_history() -> None:
    today_mail = make_mail("q", TODAY, title="Dossier 2026-3025 : relance")
    yesterday = make_mail("hier", "2026-10-06T09:00:00+02:00", title="Dossier 2026-3025 : RAS")
    index = index_with(today_mail, yesterday, make_mail("x", PAST, title="Dossier 2026-3025"))

    result = history([case(today_mail)], index, DAY, exclude={"mail:x"})

    assert linked(result, "case:mail:q") == ["mail:hier"]


def test_items_of_the_brief_day_are_not_history() -> None:
    index = index_with(make_mail("tot", "2026-10-07T06:00:00+02:00", title="Dossier 2026-3025"))
    today = case(make_mail("q", TODAY, title="Dossier 2026-3025 : relance"))

    assert history([today], index, DAY)[today.case_id] == []


def test_a_case_of_another_user_is_refused() -> None:
    index = index_with(make_mail("p", PAST, title="Dossier 2026-3025"))
    foreign = case(make_mail("q", TODAY, title="Dossier 2026-3025", owner_user_id=OTHER_USER))

    with pytest.raises(ValueError, match="autre utilisateur"):
        history([foreign], index, DAY)


def test_same_cases_in_any_order_give_the_same_history() -> None:
    index = index_with(
        *[
            make_mail(f"p{n}", PAST, title=f"Dossier 2026-{1000 + n % 4} : point {n}")
            for n in range(12)
        ]
    )
    cases = [case(make_mail(f"q{n}", TODAY, title=f"Dossier 2026-{1000 + n}")) for n in range(4)]
    shuffled = cases[:]
    random.Random(2026).shuffle(shuffled)

    assert history(shuffled, index, DAY) == history(cases, index, DAY)


def test_no_case_or_empty_index() -> None:
    today = case(make_mail("q", TODAY, title="Dossier 2026-3025"))

    assert history([], index_with(), DAY) == {}
    assert history([today], index_with(), DAY) == {today.case_id: []}


# --- données de démo ------------------------------------------------------------------------------


def test_demo_history_respects_the_rules() -> None:
    days = available_days()
    index = IndexStore(HashEmbedder()).for_user(USER)
    for day in days[:5]:
        index.add(load_items(day))
    today = date.fromisoformat(days[5])
    cases = group_into_cases(load_items(days[5]))

    result = history(cases, index, today)

    assert set(result) == {c.case_id for c in cases}
    for c in cases:
        case_refs = set().union(*(references(item.title) for item in c.items))
        for link in result[c.case_id]:
            assert link.item.created_at.date() < today
            assert link.item.item_id not in c.item_ids
            if link.same_reference:
                assert references(link.item.title) & case_refs


@pytest.mark.parametrize(
    ("score", "refs", "case_refs", "expected"),
    [
        (0.1, {"dossier:1"}, {"dossier:1", "facture:2"}, "même référence"),
        (0.99, {"dossier:1"}, {"dossier:2"}, "références différentes"),
        (0.9, set(), {"dossier:2"}, "sujet proche"),
        (0.5, set(), set(), "sous le seuil"),
    ],
)
def test_verdict(score: float, refs: set, case_refs: set, expected: str) -> None:
    assert verdict(score, refs, case_refs, threshold=0.85) == expected
