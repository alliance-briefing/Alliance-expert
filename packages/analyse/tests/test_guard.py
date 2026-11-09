"""Garde L2.4 : claims acceptés quand ils sont soutenus, entrée dégradée ou retirée sinon."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from contracts.demo_data import available_days, load_items
from contracts.models import Brief, Claim
from packages.analyse import generate_brief
from packages.analyse.guard.check import check_claim, guard_brief
from packages.analyse.tests.factories import (
    COLLECTED_AT,
    OTHER_USER,
    USER,
    at,
    make_event,
    make_mail,
    make_task,
)

DAY = date(2026, 10, 7)
NOW = at(COLLECTED_AT)
MAIL = make_mail("m1", "2026-10-07T06:15:00Z", title="Devis 2026-3025")
TASK = make_task("t1", "2026-10-05T12:00:00+02:00", title="Relancer le garage")


def forged(items: list, **claim: Any) -> Brief:
    """Le brief de `items` dont le premier claim de « actions » est remplacé par `claim`."""
    data = generate_brief(items, USER, DAY, now=NOW).model_dump()
    data["sections"][0]["entries"][0]["claims"] = [claim]
    return Brief.model_validate(data)


@pytest.mark.parametrize("day", available_days())
def test_every_claim_of_the_demo_is_supported(day: str) -> None:
    items = load_items(day)
    brief = generate_brief(items, USER, day, now=NOW)

    assert guard_brief(brief, items) is brief
    assert not any("dégradée" in w or "retirée" in w for w in brief.warnings)


@pytest.mark.parametrize(
    ("text", "item"),
    [
        ("Mail non lu « Devis 2026-3025 », reçu le 07/10 à 08:15", MAIL),  # UTC lu à Paris
        ("Le devis 2026-3025 est arrivé le 07/10", MAIL),  # numéro et date libres
        ("Tâche « Relancer le garage », échéance dépassée le 05/10", TASK),
        (
            "Réunion « Salon » le 07/10, journée entière",
            make_event(
                "e1",
                "2026-10-07T00:00:00Z",
                "2026-10-08T00:00:00Z",
                title="Salon",
                attributes={"calendar": {"is_all_day": True}},
            ),
        ),
    ],
)
def test_supported_claims_are_accepted(text: str, item) -> None:
    claim = Claim(text=text, source_item_ids=[item.item_id])

    assert check_claim(claim, {item.item_id: item}, USER) is None


def test_count_claim_must_match_the_cited_items() -> None:
    mails = {m.item_id: m for m in [MAIL, make_mail("m2", "2026-10-06T09:00:00+02:00")]}

    good = Claim(text="Et 2 autres mails non lus", source_item_ids=list(mails))
    wrong = Claim(text="Et 3 autres mails non lus", source_item_ids=list(mails))

    assert check_claim(good, mails, USER) is None
    assert check_claim(wrong, mails, USER) == "compte faux : 3 pour 2 items cités"


def test_an_unsupported_claim_degrades_its_entry_and_warns() -> None:
    brief = forged(
        [MAIL],
        text="Mail non lu « Devis 2026-3025 », reçu le 07/10 à 09:15",  # heure fausse
        source_item_ids=[MAIL.item_id],
    )

    guarded = guard_brief(brief, [MAIL])

    entry = guarded.sections[0].entries[0]
    assert entry.text == "À consulter : « Devis 2026-3025 »"
    assert entry.claims == []
    assert entry.source_item_ids == [MAIL.item_id]
    assert guarded.warnings[-1] == "actions n°1 dégradée : date introuvable : 07/10 à 09:15"


def test_an_item_of_another_user_is_never_kept_as_a_source() -> None:
    foreign = make_mail("x1", "2026-10-07T07:00:00+02:00", owner_user_id=OTHER_USER)
    data = generate_brief([MAIL, foreign], OTHER_USER, DAY, now=NOW).model_dump()
    data["owner_user_id"], data["brief_id"] = USER, f"b_{DAY}_{USER}"
    brief = Brief.model_validate(data)  # brief de USER qui ne cite que l'item d'OTHER_USER

    guarded = guard_brief(brief, [MAIL, foreign])

    assert guarded.sections[0].entries == []
    assert guarded.warnings[-1] == "actions n°1 retirée : aucune source valide"


def test_ranks_stay_consecutive_after_a_removal() -> None:
    other = make_mail("m2", "2026-10-07T05:00:00+02:00", title="Facture")
    data = generate_brief([MAIL, other], USER, DAY, now=NOW).model_dump()
    data["sections"][0]["entries"][0]["source_item_ids"] = ["mail:inconnu"]
    data["sections"][0]["entries"][0]["claims"] = []
    brief = Brief.model_validate(data)

    guarded = guard_brief(brief, [MAIL, other])

    assert [e.rank for e in guarded.sections[0].entries] == [1]
    assert guarded.sections[0].entries[0].source_item_ids == ["mail:m2"]
