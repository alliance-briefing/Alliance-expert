"""Regroupement en affaires (L2.1) : critères du guide, pièges et isolation."""

from __future__ import annotations

import random
import time

import pytest

from contracts.demo_data import available_days, load_items
from packages.analyse.aggregate.cases import Case, group_into_cases, references
from packages.analyse.tests.factories import OTHER_USER, make_event, make_mail, make_task

MORNING = "2026-10-07T08:00:00+02:00"


def groups(cases: list[Case]) -> list[list[str]]:
    return [case.item_ids for case in cases]


# --- références dans les titres ------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Dossier 2026-3025 : RAS", {"dossier:2026-3025"}),
        ("Mission d'expertise – dossier 2026-9372 – Citroën C4", {"dossier:2026-9372"}),
        ("Relire et signer le rapport 2026-2367", {"dossier:2026-2367"}),
        ("Transmettre le rapport préliminaire 2026-4837", {"dossier:2026-4837"}),
        ("Devis 2026-4900 au-dessus du seuil", {"dossier:2026-4900"}),
        ("RE : Question sur la facture 2026-0412", {"facture:2026-0412"}),
        ("Confirmation de commande n° 2026-0412", {"commande:2026-0412"}),
        ("Budget 2027 — arbitrages", set()),  # pas un numéro de référence
        ("Plan 2026-1234", set()),  # mot hors de la table : aucun lien
        ("", set()),
    ],
)
def test_references_are_typed_by_the_word_before_the_number(title: str, expected: set) -> None:
    assert references(title) == expected


# --- niveau 1 : même fil -------------------------------------------------------------------


def test_a_fifteen_message_thread_is_one_case() -> None:
    # Critère d'acceptation L2.1 : un fil de 15 messages produit une affaire et non 15.
    mails = [
        make_mail(f"m{n:02}", f"2026-10-07T08:{n:02}:00+02:00", thread_id="conv:long")
        for n in range(15)
    ]

    cases = group_into_cases(mails)

    assert len(cases) == 1
    assert cases[0].item_ids == [f"mail:m{n:02}" for n in range(15)]
    assert cases[0].links == ("fil:conv:long",)


def test_items_without_shared_link_stay_alone() -> None:
    items = [
        make_mail("a", MORNING, title="Question rapide", thread_id="conv:1"),
        make_mail("b", MORNING, title="Question rapide", thread_id="conv:2"),
        make_mail("c", MORNING, title="Congés"),
    ]

    cases = group_into_cases(items)

    assert groups(cases) == [["mail:a"], ["mail:b"], ["mail:c"]]
    assert all(case.links == () for case in cases)


# --- niveau 2 : même référence --------------------------------------------------------------


def test_mail_task_and_meeting_of_the_same_file_are_one_case() -> None:
    items = [
        make_mail("m1", MORNING, title="Dossier 2026-3025 : assuré injoignable"),
        make_task("t1", None, title="Affecter un expert au dossier 2026-3025"),
        make_event(
            "e1",
            "2026-10-08T09:00:00+02:00",
            "2026-10-08T10:00:00+02:00",
            title="Expertise Kia Niro – dossier 2026-3025",
        ),
    ]

    cases = group_into_cases(items)

    # Ordre par date de création : la tâche et l'événement (01/10) avant le mail (07/10).
    assert groups(cases) == [["calendar:e1", "task:t1", "mail:m1"]]
    assert cases[0].links == ("dossier:2026-3025",)


def test_report_and_file_with_the_same_number_are_one_case() -> None:
    items = [
        make_mail("m1", MORNING, title="Rapport 2026-2367 prêt pour relecture"),
        make_mail("m2", MORNING, title="Dossier 2026-2367 : RAS"),
    ]

    assert groups(group_into_cases(items)) == [["mail:m1", "mail:m2"]]


def test_invoice_and_order_with_the_same_number_stay_apart() -> None:
    items = [
        make_mail("facture", MORNING, title="Facture 2026-0412"),
        make_mail("question", MORNING, title="Question sur la facture 2026-0412"),
        make_mail("commande", MORNING, title="Confirmation de commande n° 2026-0412"),
    ]

    assert groups(group_into_cases(items)) == [
        ["mail:commande"],
        ["mail:facture", "mail:question"],
    ]


def test_a_number_only_in_the_snippet_creates_no_link() -> None:
    items = [
        make_mail("m1", MORNING, title="Dossier 2026-3025 : RAS"),
        make_mail("m2", MORNING, title="Point rapide", snippet="voir aussi le dossier 2026-3025"),
    ]

    assert groups(group_into_cases(items)) == [["mail:m1"], ["mail:m2"]]


# --- transitivité, isolation, déterminisme -------------------------------------------------


def test_links_are_transitive() -> None:
    # m1 et m2 : même fil ; m2 et t1 : même dossier ; donc une seule affaire.
    items = [
        make_mail("m1", "2026-10-06T08:00:00+02:00", title="Point", thread_id="conv:1"),
        make_mail("m2", MORNING, title="RE: Point dossier 2026-1111", thread_id="conv:1"),
        make_task("t1", None, title="Clôturer le dossier 2026-1111"),
    ]

    cases = group_into_cases(items)

    assert groups(cases) == [["task:t1", "mail:m1", "mail:m2"]]
    assert cases[0].links == ("dossier:2026-1111", "fil:conv:1")


def test_two_users_never_share_a_case() -> None:
    items = [
        make_mail("moi", MORNING, title="Dossier 2026-3025", thread_id="conv:1"),
        make_mail(
            "autre",
            MORNING,
            title="Dossier 2026-3025",
            thread_id="conv:1",
            owner_user_id=OTHER_USER,
        ),
    ]

    cases = group_into_cases(items)

    assert groups(cases) == [["mail:autre"], ["mail:moi"]]
    assert all(case.links == () for case in cases)


def test_same_items_in_any_order_give_the_same_cases() -> None:
    items = [
        make_mail(f"m{n}", MORNING, title=f"Dossier 2026-{1000 + n % 7}", thread_id=f"c{n % 5}")
        for n in range(40)
    ]
    shuffled = items[:]
    random.Random(2026).shuffle(shuffled)

    assert group_into_cases(shuffled) == group_into_cases(items)


def test_empty_input_gives_no_case() -> None:
    assert group_into_cases([]) == []


# --- performance et données de démo ----------------------------------------------------------


def test_five_hundred_items_are_grouped_in_less_than_two_seconds() -> None:
    # Critère d'acceptation L2.1 : ≤ 2 s pour 500 Item sans appel au LLM.
    items = [
        make_mail(
            f"m{n:03}",
            MORNING,
            title=f"Dossier 2026-{1000 + n % 120} : point",
            thread_id=f"conv:{n % 200}",
        )
        for n in range(500)
    ]

    started = time.perf_counter()
    cases = group_into_cases(items)

    assert time.perf_counter() - started < 2
    assert sum(len(case.items) for case in cases) == 500


@pytest.mark.parametrize("day", available_days())
def test_demo_day_items_are_each_in_exactly_one_case(day: str) -> None:
    items = load_items(day)

    cases = group_into_cases(items)

    grouped = [item_id for case in cases for item_id in case.item_ids]
    assert sorted(grouped) == sorted(item.item_id for item in items)
    assert len(cases) <= len(items)
