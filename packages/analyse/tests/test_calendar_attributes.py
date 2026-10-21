"""Attributs de calendrier du contrat A 1.1 : « toute la journée » et réunions annulées."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from contracts.models import Item
from packages.analyse import generate_brief
from packages.analyse.aggregate.rules import meetings_of_day
from packages.analyse.clock import is_all_day
from packages.analyse.tests.factories import COLLECTED_AT, USER, at, make_event, make_item
from unittest.mock import MagicMock

DAY = date(2026, 10, 7)
NOW = at(COLLECTED_AT)


def event(native_id: str, start_at: str, end_at: str, **calendar: Any) -> Item:
    """Un événement 1.1 avec son bloc `attributes.calendar`."""
    return make_event(native_id, start_at, end_at, attributes={"calendar": calendar})


# --- toute la journée -------------------------------------------------------------------------


def test_the_all_day_attribute_is_trusted_over_the_deduction() -> None:
    # Minuit → minuit mais le lot 1 dit « pas toute la journée » : on le croit.
    midnight = event("e1", "2026-10-07T00:00:00+02:00", "2026-10-08T00:00:00+02:00")
    flagged = event("e2", "2026-10-07T00:00:00+02:00", "2026-10-08T00:00:00+02:00", is_all_day=True)

    assert is_all_day(midnight) is False
    assert is_all_day(flagged) is True


def test_without_attributes_the_deduction_still_applies() -> None:
    item_1_0 = make_event("e1", "2026-10-07T00:00:00+00:00", "2026-10-08T00:00:00+00:00")

    assert is_all_day(item_1_0) is True


def test_an_all_day_event_from_lot_1_stays_on_its_day_and_is_worded_so() -> None:
    # Le lot 1 écrit une journée entière de minuit à minuit, heure de Paris.
    formation = event(
        "formation",
        "2026-10-07T00:00:00+02:00",
        "2026-10-08T00:00:00+02:00",
        is_all_day=True,
    )

    assert meetings_of_day([formation], DAY) == [formation]
    assert meetings_of_day([formation], date(2026, 10, 8)) == []
    brief = generate_brief([formation], USER, DAY, now=NOW)
    assert brief.sections[1].entries[0].text.startswith("Réunion toute la journée")


# --- réunions annulées ------------------------------------------------------------------------


def test_a_cancelled_meeting_is_not_a_meeting_of_the_day() -> None:
    items = [
        event(
            "annulee", "2026-10-07T09:00:00+02:00", "2026-10-07T10:00:00+02:00", is_cancelled=True
        ),
        event("maintenue", "2026-10-07T11:00:00+02:00", "2026-10-07T12:00:00+02:00"),
        make_event("sans-attributs", "2026-10-07T14:00:00+02:00", "2026-10-07T15:00:00+02:00"),
    ]

    assert [item.item_id for item in meetings_of_day(items, DAY)] == [
        "calendar:maintenue",
        "calendar:sans-attributs",
    ]


def test_a_day_with_only_a_cancelled_meeting_has_nothing_to_report() -> None:
    cancelled = event(
        "annulee", "2026-10-07T09:00:00+02:00", "2026-10-07T10:00:00+02:00", is_cancelled=True
    )

    brief = generate_brief([cancelled], USER, DAY, now=NOW)

    assert brief.sections[1].entries == []
    assert brief.cited_item_ids() == set()


def test_is_all_day_avec_champ_none_est_deduit():
    """Si le champ is_all_day est None, on doit le déduire (minuit à minuit)."""
    item = make_item(
        native_id="test-all-day",
        source="calendar",
        start_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
        end_at=datetime(2026, 10, 8, tzinfo=timezone.utc),
    )

    # On crée un faux attribut avec is_all_day = None
    mock_attr = MagicMock()
    mock_attr.calendar.is_all_day = None

    # On utilise model_copy pour contourner le fait que l'Item est "frozen"
    item = item.model_copy(update={"attributes": mock_attr})

    assert is_all_day(item) is True


def test_meeting_annule_avec_champ_none_n_est_pas_filtre():
    """Si is_cancelled est None, l'événement ne doit pas être considéré comme annulé."""
    item = make_item(
        native_id="test-cancelled",
        source="calendar",
        start_at=datetime(2026, 10, 7, 10, 0, tzinfo=timezone.utc),
        end_at=datetime(2026, 10, 7, 11, 0, tzinfo=timezone.utc),
    )

    mock_attr = MagicMock()
    mock_attr.calendar.is_cancelled = None
    item = item.model_copy(update={"attributes": mock_attr})

    assert meetings_of_day([item], date(2026, 10, 7)) == [item]