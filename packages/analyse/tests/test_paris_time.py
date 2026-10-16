"""Dates lues en heure de Paris, quel que soit le décalage reçu (ex. UTC de Microsoft Graph)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from packages.analyse import generate_brief
from packages.analyse.aggregate.rules import meetings_of_day
from packages.analyse.clock import PARIS, is_all_day, to_paris
from packages.analyse.tests.factories import (
    COLLECTED_AT,
    USER,
    at,
    make_event,
    make_mail,
    make_task,
)

DAY = date(2026, 10, 7)
NOW = at(COLLECTED_AT)


def test_to_paris_keeps_the_instant_and_shows_the_paris_offset() -> None:
    utc = at("2026-10-07T07:00:00+00:00")

    paris = to_paris(utc)

    assert paris == utc
    assert paris.isoformat() == "2026-10-07T09:00:00+02:00"


# --- journées entières -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start_at", "end_at", "expected"),
    [
        ("2026-10-07T00:00:00+00:00", "2026-10-08T00:00:00+00:00", True),  # Graph, en UTC
        ("2026-10-07T00:00:00+02:00", "2026-10-08T00:00:00+02:00", True),  # demo, en Paris
        ("2026-10-25T00:00:00+02:00", "2026-10-26T00:00:00+01:00", True),  # journée de 25 h
        ("2026-10-07T09:00:00+00:00", "2026-10-07T10:00:00+00:00", False),
        ("2026-10-07T00:00:00+02:00", "2026-10-07T00:00:00+02:00", False),  # durée nulle
    ],
)
def test_is_all_day(start_at: str, end_at: str, expected: bool) -> None:
    assert is_all_day(make_event("e1", start_at, end_at)) is expected


def test_graph_all_day_event_stays_on_its_own_day() -> None:
    # Lue en heure de Paris, elle irait de 02:00 le 07/10 à 02:00 le 08/10.
    event = make_event("journee", "2026-10-07T00:00:00+00:00", "2026-10-08T00:00:00+00:00")

    assert meetings_of_day([event], DAY) == [event]
    assert meetings_of_day([event], date(2026, 10, 8)) == []
    brief = generate_brief([event], USER, DAY, now=NOW)
    assert brief.sections[1].entries[0].text.startswith("Réunion toute la journée")


# --- réunions, mails, tâches en UTC ------------------------------------------------------------


@pytest.mark.parametrize(
    ("start_at", "day", "expected"),
    [
        ("2026-10-07T07:00:00+00:00", date(2026, 10, 7), "Réunion à 09:00"),  # heure d'été
        ("2026-10-26T08:00:00+00:00", date(2026, 10, 26), "Réunion à 09:00"),  # heure d'hiver
        ("2026-10-06T22:30:00+00:00", date(2026, 10, 7), "Réunion à 00:30"),  # veille en UTC
    ],
)
def test_meeting_in_utc_is_shown_at_its_paris_time(start_at: str, day: date, expected: str) -> None:
    start = at(start_at)
    event = make_event("e1", start_at, (start + timedelta(hours=1)).isoformat())

    brief = generate_brief([event], USER, day, now=NOW)

    assert brief.sections[1].entries[0].text.startswith(expected)


def test_meeting_after_midnight_in_paris_is_not_in_the_previous_day() -> None:
    # 23:30 UTC le 07/10 = 01:30 le 08/10 à Paris.
    event = make_event("nuit", "2026-10-07T23:30:00+00:00", "2026-10-08T00:30:00+00:00")

    assert meetings_of_day([event], DAY) == []
    assert meetings_of_day([event], date(2026, 10, 8)) == [event]


def test_mail_and_task_dates_are_written_in_paris_time() -> None:
    items = [
        make_mail("m1", "2026-10-06T22:30:00+00:00"),
        make_task("t1", "2026-10-04T23:30:00+00:00"),
    ]

    actions, _, overdue = generate_brief(items, USER, DAY, now=NOW).sections

    assert actions.entries[0].priority_reasons == ["non lu, reçu le 07/10 à 00:30"]
    assert overdue.entries[0].priority_reasons == ["échéance dépassée depuis le 05/10"]


# --- generated_at --------------------------------------------------------------------------------


def test_generated_at_is_in_paris_time_even_if_now_is_given_in_utc() -> None:
    brief = generate_brief([], USER, DAY, now=at("2026-10-07T06:31:10+00:00"))

    assert brief.generated_at.isoformat() == "2026-10-07T08:31:10+02:00"


def test_generated_at_defaults_to_the_current_paris_time() -> None:
    brief = generate_brief([], USER, DAY)

    assert brief.generated_at.utcoffset() == brief.generated_at.astimezone(PARIS).utcoffset()
