"""Tests métier de la partie B, avec les classes et données fournies."""

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from contracts.models import Item
from packages.collect.connectors.mail_local.connector import (
    DEFAULT_FIXTURES,
    LocalMailConnector,
    load_test_world,
    select_mails,
)
from packages.collect.core.connector import Window
from packages.collect.synth.clock import PARIS, collect_time
from packages.collect.synth.model import SimMail
from packages.collect.synth.people import PROFILES, build_directory
from packages.collect.synth.window import mail_item

DIRECTORY = build_directory(PROFILES["u_demo_001"])
NOW = datetime(2026, 10, 21, 8, 30, 2, tzinfo=UTC)


def sample_mail(sent_at, **kwargs):
    return SimMail(
        native_id="test-001",
        thread_id="thread-001",
        subject="Rapport",
        html="<p>Rapport disponible.</p>",
        sender=DIRECTORY.insurers[0],
        to=[DIRECTORY.manager],
        cc=[],
        sent_at=sent_at,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("age", "answered", "expected"),
    [
        (timedelta(hours=1), False, True),
        (timedelta(hours=1), True, True),
        (timedelta(hours=24), False, False),
        (timedelta(days=3), False, False),
        (timedelta(days=7), False, True),
        (timedelta(days=8), False, True),
        (timedelta(days=8), True, False),
        (timedelta(seconds=-1), False, False),
    ],
)
def test_selection_boundaries(age, answered, expected):
    mail = sample_mail(NOW - age, answered_at=NOW - timedelta(minutes=1) if answered else None)
    assert bool(select_mails([mail], DIRECTORY.manager, NOW)) is expected


def test_duplicates_and_future_answer():
    mail = sample_mail(NOW - timedelta(days=8), answered_at=NOW + timedelta(hours=1))
    assert select_mails([mail, mail], DIRECTORY.manager, NOW) == [mail]


def test_old_robot_cc_and_other_manager_excluded():
    old = sample_mail(NOW - timedelta(days=8))
    robot = replace(old, sender=DIRECTORY.robots["missions"])
    cc_only = replace(old, to=[], cc=[DIRECTORY.manager])
    unrelated = replace(old, to=[DIRECTORY.assistant])
    outgoing = replace(old, sender=DIRECTORY.manager)
    assert select_mails([robot, cc_only, unrelated, outgoing], DIRECTORY.manager, NOW) == []
    recent_cc = replace(cc_only, sent_at=NOW - timedelta(hours=1))
    assert select_mails([recent_cc], DIRECTORY.manager, NOW) == [recent_cc]


def test_actual_24_hours_across_daylight_saving():
    now = datetime(2026, 10, 26, 8, 30, tzinfo=PARIS)
    boundary = now.astimezone(UTC) - timedelta(hours=24)
    outside = sample_mail(boundary)
    inside = replace(outside, native_id="inside", sent_at=boundary + timedelta(seconds=1))
    assert select_mails([outside, inside], DIRECTORY.manager, now) == [inside]


def test_dates_need_timezone():
    with pytest.raises(ValueError, match="fuseau"):
        select_mails([], DIRECTORY.manager, NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="fuseau"):
        select_mails([sample_mail(NOW.replace(tzinfo=None))], DIRECTORY.manager, NOW)


@pytest.fixture(scope="module")
def world():
    return load_test_world(DEFAULT_FIXTURES, "u_demo_001")


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("", ""),
        ('<img src="cid:photo">', ""),
        ("<p>Cordialement,</p><p>" + "Signature " * 1000 + "</p>", ""),
        ("<script>danger()</script><p>Rapport disponible.</p>", "Rapport disponible."),
        ("<p>Oui.</p><p>&gt; Ancienne demande</p>", "Oui."),
        ("<p>un<p>deux", "un deux"),
    ],
)
def test_cleaning_and_empty_snippet(world, html, expected):
    mail = replace(sample_mail(NOW - timedelta(hours=1)), html=html, subject="")
    item = mail_item(world, mail, NOW)
    assert item.snippet == expected
    assert item.title == ""
    assert Item.model_validate_json(item.model_dump_json()) == item


def test_reconstructed_mails_match_delivered_data(world):
    day = "2026-10-21"
    by_id = {f"mail:{mail.native_id}": mail for mail in world.mails}
    path = DEFAULT_FIXTURES / "u_demo_001" / day / "items.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        delivered = Item.model_validate_json(line)
        if delivered.source == "mail":
            assert mail_item(world, by_id[delivered.item_id], delivered.collected_at) == delivered


def test_connector_with_existing_dataset():
    now = collect_time(date(2026, 10, 21))
    connector = LocalMailConnector()
    items = list(
        connector.collect("u_demo_001", Window(now.astimezone(UTC) - timedelta(hours=24), now))
    )
    assert items and len(items) <= 500
    assert len({item.item_id for item in items}) == len(items)
    assert any(
        item.created_at.astimezone(UTC) <= now.astimezone(UTC) - timedelta(days=7) for item in items
    )
    for item in items:
        assert item.source == "mail" and item.owner_user_id == "u_demo_001"
        assert item.acl.visible_to == ["u_demo_001"]
        assert len(item.snippet) <= 500 and item.content_text is None
        assert item.connector.name == "mail-local"
        Item.model_validate(json.loads(item.model_dump_json()))


def test_unknown_user_rejected():
    with pytest.raises(ValueError, match="manager absent"):
        load_test_world(DEFAULT_FIXTURES, "u_demo_002")
