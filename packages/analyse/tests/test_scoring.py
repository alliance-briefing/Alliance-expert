"""Tests des signaux du score de priorité (L2.2) : chaque défaut corrigé a son test.

Les Item sont fabriqués ici (`_item`) pour que le fichier se lise seul ; la fabrique
commune du dépôt (`factories.py`) pourra les remplacer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import permutations

import pytest

from contracts.models.item import Item
from packages.analyse.aggregate.cases import Case
from packages.analyse.scoring.score import ScoringConfig, score_case

PARIS = timezone(timedelta(hours=2))
NOW = datetime(2026, 10, 21, 9, 0, tzinfo=PARIS)
OWNER = "u1"
USER = "moi@alliance.test"
STANDARD = "élément professionnel standard"
URGENCY = "contient des mots signalant l'urgence"
NOISE = "message automatique ou newsletter"


def _item(
    item_id: str,
    source: str = "mail",
    *,
    title: str = "Point dossier",
    snippet: str = "Bonjour",
    people: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
    **extra: object,
) -> Item:
    """Un Item valide ; `people` = [(rôle, e-mail)] ; `extra` remplace n'importe quel champ."""
    data: dict[str, object] = {
        "item_id": f"{source}:{item_id}",
        "source": source,
        "provider": "demo",
        "owner_user_id": OWNER,
        "title": title,
        "snippet": snippet,
        "created_at": NOW - timedelta(hours=3),
        "updated_at": NOW - timedelta(hours=2),
        "participants": [
            {"name": email.split("@")[0], "email": email, "role": role} for role, email in people
        ],
        "url": "https://exemple.test/x",
        "acl": {"visible_to": [OWNER]},
        "collected_at": NOW,
        "connector": {"name": "demo", "version": "0.1.0"},
    }
    if source == "task":
        data["completed"] = False
    if source == "calendar":
        data["start_at"] = NOW + timedelta(hours=1)
        data["end_at"] = NOW + timedelta(hours=2)
    if source == "mail":
        data["is_read"] = True
    data.update(extra)
    return Item.model_validate(data)


def _score(
    *items: Item, config: ScoringConfig | None = None, now: datetime = NOW
) -> tuple[int, list[str]]:
    return score_case(Case(case_id="case:test", items=items), now, config)


def _meeting(item_id: str, start: datetime, end: datetime, **calendar: object) -> Item:
    extra: dict[str, object] = {"start_at": start, "end_at": end}
    if calendar:
        extra["attributes"] = {"calendar": calendar}
    return _item(item_id, "calendar", **extra)


# --- Cas de base -------------------------------------------------------------


def test_empty_case_gets_the_neutral_score_and_a_reason() -> None:
    assert _score() == (50, [STANDARD])


def test_plain_item_is_standard() -> None:
    assert _score(_item("1")) == (50, [STANDARD])


def test_same_input_same_output() -> None:
    item = _item("1", title="Urgent", native_importance="high")
    assert _score(item) == _score(item)


def test_naive_now_is_rejected() -> None:
    with pytest.raises(ValueError, match="fuseau"):
        score_case(Case(case_id="case:test", items=(_item("1"),)), datetime(2026, 10, 21, 9, 0))  # noqa: DTZ001 (naïf exprès)


# --- Signal 2 : importance native --------------------------------------------


def test_high_importance_wins_whatever_the_order() -> None:
    low = _item("1", native_importance="low")
    high = _item("2", native_importance="high")
    assert _score(low, high) == _score(high, low) == (70, ["marqué comme important"])


def test_low_importance_lowers_the_score_and_says_so() -> None:
    assert _score(_item("1", native_importance="low")) == (40, ["marqué comme peu important"])


# --- Signal 4 : échéances ----------------------------------------------------


def test_completed_task_due_soon_gets_no_deadline_bonus() -> None:
    done = _item("1", "task", completed=True, due_at=NOW + timedelta(hours=12))
    assert _score(done) == (50, [STANDARD])


def test_task_due_exactly_now_counts_as_due_within_24h() -> None:
    assert _score(_item("1", "task", due_at=NOW)) == (70, ["échéance dans moins de 24 h"])


def test_overdue_beats_an_upcoming_deadline_whatever_the_order() -> None:
    late = _item("1", "task", due_at=NOW - timedelta(days=1))
    soon = _item("2", "task", due_at=NOW + timedelta(hours=5))
    assert _score(late, soon) == _score(soon, late) == (75, ["tâche en retard"])


def test_nearest_deadline_decides_the_window() -> None:
    near = _item("1", "task", due_at=NOW + timedelta(hours=30))
    far = _item("2", "task", due_at=NOW + timedelta(hours=10))
    assert _score(near, far) == _score(far, near) == (70, ["échéance dans moins de 24 h"])


@pytest.mark.parametrize(
    ("hours", "expected"),
    [
        (1, (70, ["échéance dans moins de 24 h"])),
        (24, (70, ["échéance dans moins de 24 h"])),
        (24.5, (60, ["échéance dans moins de 48 h"])),
        (48, (60, ["échéance dans moins de 48 h"])),
        (49, (50, [STANDARD])),
    ],
)
def test_deadline_windows(hours: float, expected: tuple[int, list[str]]) -> None:
    task = _item("1", "task", due_at=NOW + timedelta(hours=hours))
    assert _score(task) == expected


def test_task_without_deadline_is_standard() -> None:
    assert _score(_item("1", "task")) == (50, [STANDARD])


# --- Signal 6 : réunions -----------------------------------------------------


def test_upcoming_meeting_within_48h() -> None:
    meeting = _meeting("1", NOW + timedelta(hours=5), NOW + timedelta(hours=6))
    assert _score(meeting) == (65, ["réunion à venir dans les 48 h"])


def test_meeting_in_progress_counts() -> None:
    meeting = _meeting("1", NOW - timedelta(minutes=30), NOW + timedelta(minutes=30))
    assert _score(meeting) == (65, ["réunion en cours"])


@pytest.mark.parametrize("hours", [49, -5])
def test_meeting_too_far_or_over_gets_no_bonus(hours: int) -> None:
    start = NOW + timedelta(hours=hours)
    assert _score(_meeting("1", start, start + timedelta(hours=1))) == (50, [STANDARD])


def test_cancelled_meeting_gets_no_bonus() -> None:
    meeting = _meeting("1", NOW + timedelta(hours=5), NOW + timedelta(hours=6), is_cancelled=True)
    assert _score(meeting) == (50, [STANDARD])


def test_declined_meeting_gets_no_bonus() -> None:
    meeting = _meeting("1", NOW + timedelta(hours=5), NOW + timedelta(hours=6), response="declined")
    assert _score(meeting) == (50, [STANDARD])


# --- Signal 7 : mail non lu adressé à l'utilisateur ---------------------------


def test_unread_mail_addressed_to_the_user_gets_a_bonus() -> None:
    mail = _item("1", is_read=False, people=[("from", "a@x.test"), ("to", USER)])
    config = ScoringConfig(user_emails=frozenset({USER}))
    assert _score(mail, config=config) == (60, ["mail non lu adressé directement"])


def test_user_email_is_case_insensitive() -> None:
    mail = _item("1", is_read=False, people=[("from", "a@x.test"), ("to", USER)])
    config = ScoringConfig(user_emails=frozenset({"Moi@Alliance.TEST"}))
    assert _score(mail, config=config)[0] == 60


def test_recipients_being_frequent_contacts_is_not_direct_addressing() -> None:
    mail = _item("1", is_read=False, people=[("from", "a@x.test"), ("to", "collegue@x.test")])
    config = ScoringConfig(
        frequent_contacts=frozenset({"collegue@x.test"}), user_emails=frozenset({USER})
    )
    assert _score(mail, config=config) == (50, [STANDARD])


def test_read_mail_gets_no_direct_bonus() -> None:
    mail = _item("1", is_read=True, people=[("to", USER)])
    assert _score(mail, config=ScoringConfig(user_emails=frozenset({USER}))) == (50, [STANDARD])


def test_without_user_emails_the_signal_is_off() -> None:
    mail = _item("1", is_read=False, people=[("to", USER)])
    assert _score(mail) == (50, [STANDARD])


# --- Signal 5 : expéditeur ----------------------------------------------------


def test_config_emails_are_case_insensitive() -> None:
    mail = _item("1", people=[("from", "chef@alliance.test")])
    config = ScoringConfig(hierarchy=frozenset({"Chef@Alliance.TEST"}))
    assert _score(mail, config=config) == (75, ["message de la hiérarchie"])


def test_config_accepts_plain_sets_and_is_not_modified() -> None:
    config = ScoringConfig(hierarchy={"A@X.test"})  # type: ignore[arg-type]
    mail = _item("1", people=[("from", "a@x.test")])
    assert _score(mail, config=config) == (75, ["message de la hiérarchie"])
    assert config.hierarchy == {"A@X.test"}


def test_blacklist_beats_hierarchy_in_the_same_case() -> None:
    blocked = _item("1", people=[("from", "spam@x.test")])
    boss = _item("2", people=[("from", "chef@x.test")])
    config = ScoringConfig(
        blacklist=frozenset({"spam@x.test"}), hierarchy=frozenset({"chef@x.test"})
    )
    assert _score(blocked, boss, config=config) == (30, ["expéditeur bloqué"])


def test_frequent_contact_bonus() -> None:
    mail = _item("1", people=[("from", "client@x.test")])
    config = ScoringConfig(frequent_contacts=frozenset({"client@x.test"}))
    assert _score(mail, config=config) == (60, ["contact fréquent"])


# --- Signaux 1 et 3 : mots-clés ----------------------------------------------


def test_urgency_needs_a_whole_word() -> None:
    assert _score(_item("1", title="Appel de l'urgentiste")) == (50, [STANDARD])


@pytest.mark.parametrize("title", ["ÉCHÉANCE vendredi", "echeance vendredi", "Échéance vendredi"])
def test_urgency_ignores_case_and_accents(title: str) -> None:
    assert _score(_item("1", title=title)) == (65, [URGENCY])


def test_urgency_keyword_with_several_words() -> None:
    assert _score(_item("1", title="À traiter dès que possible"))[0] == 65


@pytest.mark.parametrize("title", ["Newsletter d'octobre", "Les newsletters du mois"])
def test_newsletter_is_noise(title: str) -> None:
    assert _score(_item("1", title=title)) == (20, [NOISE])


def test_noreply_sender_is_noise() -> None:
    mail = _item("1", people=[("from", "noreply@portail.test")])
    assert _score(mail) == (20, [NOISE])


def test_noise_cancels_the_urgency_bonus() -> None:
    assert _score(_item("1", title="URGENT newsletter")) == (20, [NOISE])


def test_empty_keyword_lists_turn_the_text_signals_off() -> None:
    config = ScoringConfig(urgency_keywords=frozenset(), noise_keywords=frozenset())
    assert _score(_item("1", title="URGENT newsletter"), config=config) == (50, [STANDARD])


def test_keywords_given_as_plain_sets_work() -> None:
    config = ScoringConfig(urgency_keywords={"Bouclage"})  # type: ignore[arg-type]
    assert _score(_item("1", title="Bouclage demain"), config=config) == (65, [URGENCY])


def test_snippet_can_reveal_noise() -> None:
    assert _score(_item("1", snippet="Pour vous désabonner, cliquez ici")) == (20, [NOISE])


# --- Raisons, bornes, ordre --------------------------------------------------


def test_reasons_keep_the_heaviest_signals() -> None:
    config = ScoringConfig(
        hierarchy=frozenset({"chef@alliance.test"}), user_emails=frozenset({USER})
    )
    mail = _item(
        "1",
        title="Urgent : devis",
        native_importance="high",
        is_read=False,
        people=[("from", "chef@alliance.test"), ("to", USER)],
    )
    late = _item("2", "task", due_at=NOW - timedelta(days=1))
    meeting = _meeting("3", NOW + timedelta(hours=5), NOW + timedelta(hours=6))
    score, reasons = _score(mail, late, meeting, config=config)
    assert score == 100  # 50 + 20 + 15 + 25 + 25 + 15 + 10 = 160, borné à 100
    assert reasons == [
        "tâche en retard",
        "message de la hiérarchie",
        "marqué comme important",
        URGENCY,
    ]


def test_score_is_floored_at_zero() -> None:
    mail = _item(
        "1",
        title="Newsletter",
        native_importance="low",
        people=[("from", "spam@x.test")],
    )
    score, reasons = _score(mail, config=ScoringConfig(blacklist=frozenset({"spam@x.test"})))
    assert score == 0  # 50 - 30 - 20 - 10 = -10, borné à 0
    assert reasons == [NOISE, "expéditeur bloqué", "marqué comme peu important"]


def test_score_does_not_depend_on_item_order() -> None:
    config = ScoringConfig(hierarchy=frozenset({"chef@alliance.test"}))
    items = [
        _item("1", native_importance="low"),
        _item("2", native_importance="high", people=[("from", "chef@alliance.test")]),
        _item("3", "task", due_at=NOW + timedelta(hours=5)),
        _item("4", "task", due_at=NOW - timedelta(days=2)),
    ]
    results = {(s, tuple(r)) for s, r in (_score(*p, config=config) for p in permutations(items))}
    assert len(results) == 1
