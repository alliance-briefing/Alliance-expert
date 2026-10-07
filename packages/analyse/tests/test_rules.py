"""Règles de sélection du stub de brief (L2.0) sur des Item synthétiques."""

from __future__ import annotations

from datetime import date

import pytest

from contracts.models import Item
from packages.analyse.aggregate.rules import meetings_of_day, overdue_tasks, unread_mails
from packages.analyse.tests.factories import COLLECTED_AT, at, make_event, make_mail, make_task

DAY = date(2026, 10, 7)
NOW = at(COLLECTED_AT)


def ids(items: list[Item]) -> list[str]:
    return [item.item_id for item in items]


# --- mails non lus -------------------------------------------------------------------------


def test_unread_mails_keeps_the_five_most_recent() -> None:
    mails = [make_mail(f"m{n}", f"2026-10-0{n}T09:00:00+02:00") for n in range(1, 8)]

    assert ids(unread_mails(mails)) == ["mail:m7", "mail:m6", "mail:m5", "mail:m4", "mail:m3"]


def test_unread_mails_ignores_read_mails_and_other_sources() -> None:
    items = [
        make_mail("lu", "2026-10-07T08:00:00+02:00", is_read=True),
        make_mail("non-lu", "2026-10-06T08:00:00+02:00"),
        make_task("t1", None),
    ]

    assert ids(unread_mails(items)) == ["mail:non-lu"]


def test_unread_mails_breaks_ties_by_item_id() -> None:
    same_time = "2026-10-07T08:00:00+02:00"
    mails = [make_mail("b", same_time), make_mail("a", same_time)]

    assert ids(unread_mails(mails)) == ["mail:a", "mail:b"]


def test_unread_mails_compares_instants_not_text() -> None:
    # 08:00+01:00 = 09:00+02:00 : plus récent que 08:30+02:00 malgré le texte.
    mails = [
        make_mail("paris-ete", "2026-10-07T08:30:00+02:00"),
        make_mail("autre-fuseau", "2026-10-07T08:00:00+01:00"),
    ]

    assert ids(unread_mails(mails)) == ["mail:autre-fuseau", "mail:paris-ete"]


# --- réunions du jour ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start_at", "end_at", "expected"),
    [
        ("2026-10-07T09:00:00+02:00", "2026-10-07T10:00:00+02:00", True),  # dans la journée
        ("2026-10-08T09:00:00+02:00", "2026-10-08T10:00:00+02:00", False),  # le lendemain
        ("2026-10-06T09:00:00+02:00", "2026-10-06T10:00:00+02:00", False),  # la veille
        ("2026-10-06T09:00:00+02:00", "2026-10-08T18:00:00+02:00", True),  # sur plusieurs jours
        ("2026-10-07T00:00:00+02:00", "2026-10-08T00:00:00+02:00", True),  # toute la journée
        ("2026-10-06T23:00:00+02:00", "2026-10-07T00:00:00+02:00", False),  # finit à minuit
        ("2026-10-07T09:00:00+02:00", "2026-10-07T09:00:00+02:00", True),  # durée nulle
    ],
)
def test_meetings_of_day_selects_events_occupying_the_day(
    start_at: str, end_at: str, expected: bool
) -> None:
    event = make_event("e1", start_at, end_at)

    assert (meetings_of_day([event], DAY) == [event]) is expected


def test_all_day_event_on_winter_time_change_stays_on_its_day() -> None:
    # Dimanche 25/10/2026 : la journée dure 25 h, de 00:00+02:00 à 00:00+01:00.
    event = make_event("hiver", "2026-10-25T00:00:00+02:00", "2026-10-26T00:00:00+01:00")

    assert meetings_of_day([event], date(2026, 10, 25)) == [event]
    assert meetings_of_day([event], date(2026, 10, 26)) == []


def test_meetings_of_day_are_sorted_by_start_then_id() -> None:
    items = [
        make_event("tard", "2026-10-07T14:00:00+02:00", "2026-10-07T15:00:00+02:00"),
        make_event("b-tot", "2026-10-07T09:00:00+02:00", "2026-10-07T10:00:00+02:00"),
        make_event("a-tot", "2026-10-07T09:00:00+02:00", "2026-10-07T09:30:00+02:00"),
        make_mail("m1", "2026-10-07T08:00:00+02:00"),
    ]

    assert ids(meetings_of_day(items, DAY)) == [
        "calendar:a-tot",
        "calendar:b-tot",
        "calendar:tard",
    ]


# --- tâches en retard ------------------------------------------------------------------------


def test_overdue_tasks_follow_the_contract_rule() -> None:
    items = [
        make_task("en-retard", "2026-10-05T12:00:00+02:00"),
        make_task("due-plus-tard-aujourd-hui", "2026-10-07T12:00:00+02:00"),
        make_task("sans-echeance", None),
        make_task("terminee", "2026-10-01T12:00:00+02:00", completed=True),
        make_mail("m1", "2026-10-01T08:00:00+02:00"),
    ]

    assert ids(overdue_tasks(items, NOW)) == ["task:en-retard"]


def test_overdue_tasks_are_sorted_oldest_due_first() -> None:
    items = [
        make_task("recente", "2026-10-06T12:00:00+02:00"),
        make_task("ancienne", "2026-09-28T12:00:00+02:00"),
        make_task("moyenne", "2026-10-02T12:00:00+02:00"),
    ]

    assert ids(overdue_tasks(items, NOW)) == ["task:ancienne", "task:moyenne", "task:recente"]
