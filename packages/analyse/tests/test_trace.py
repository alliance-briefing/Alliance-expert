"""Commande trace (L2.4) : claims, verdicts, items sources et preuves d'une entrée."""

from __future__ import annotations

from datetime import date

import pytest

from contracts.models import Brief
from packages.analyse import generate_brief
from packages.analyse.guard.trace import main, parse_brief_id, trace
from packages.analyse.tests.factories import COLLECTED_AT, USER, at, make_event, make_mail

DAY = date(2026, 10, 7)
MAIL = make_mail(
    "m1", "2026-10-07T06:15:00Z", title="Devis 2026-3025", snippet="Montant : 1 250 €" * 20
)
LU = make_mail("m0", "2026-10-06T09:00:00+02:00", title="Devis 2026-3025", is_read=True)
EVENT = make_event("e1", "2026-10-07T14:00:00+02:00", "2026-10-07T15:00:00+02:00", title="Point")
ITEMS = [MAIL, LU, EVENT]


def brief() -> Brief:
    return generate_brief(ITEMS, USER, DAY, now=at(COLLECTED_AT))


@pytest.mark.parametrize(
    ("brief_id", "expected"),
    [
        ("b_2026-10-07_u_demo_001", ("2026-10-07", "u_demo_001")),
        ("b_2026-10-07_x", ("2026-10-07", "x")),
    ],
)
def test_brief_id_gives_day_and_user(brief_id: str, expected: tuple[str, str]) -> None:
    assert parse_brief_id(brief_id) == expected


@pytest.mark.parametrize(
    "brief_id", ["2026-10-07_u", "b_2026-13-07_u", "b_2026-10-07", "x_2026-10-07_u"]
)
def test_invalid_brief_id_is_refused(brief_id: str) -> None:
    with pytest.raises(ValueError, match="identifiant de brief invalide"):
        parse_brief_id(brief_id)


def test_trace_shows_claims_verdict_sources_and_evidence() -> None:
    text = trace(brief(), 1, ITEMS, "actions")

    assert text.splitlines() == [
        "[actions n°1] Mail non lu : Devis 2026-3025 (score 50)",
        "  claim : Mail non lu « Devis 2026-3025 », reçu le 07/10 à 08:15",
        "    verdict : soutenu",
        "    mail:m1 (mail) « Devis 2026-3025 »",
        "      lien : https://demo.reseau-expertis.test/mail/m1",
        "      created_at : 07/10/2026 08:15 (citée)",
        "      updated_at : 07/10/2026 08:15 (citée)",
        f"      extrait : {MAIL.snippet[:200]}…",
        "  contexte : mail:m0 « Devis 2026-3025 »",
    ]


def test_without_section_every_entry_of_the_rank_is_traced() -> None:
    text = trace(brief(), 1, ITEMS)

    assert "[actions n°1]" in text
    assert "[meetings n°1] Réunion à 14:00 : Point" in text


def test_a_rejected_claim_and_a_degraded_entry_are_shown() -> None:
    data = brief().model_dump()
    entries = data["sections"][0]["entries"]
    entries[0]["claims"][0]["text"] = "Mail non lu « Devis 2026-3025 », reçu le 07/10 à 09:15"
    forged = Brief.model_validate(data)

    assert "verdict : REJETÉ : date introuvable : 07/10 à 09:15" in trace(forged, 1, ITEMS)

    entries[0]["claims"] = []
    degraded = Brief.model_validate(data)
    assert "aucun claim : entrée dégradée" in trace(degraded, 1, ITEMS, "actions")


def test_unknown_rank_is_an_error() -> None:
    with pytest.raises(LookupError, match="aucune entrée de rang 9 dans la section actions"):
        trace(brief(), 9, ITEMS, "actions")


def test_command_traces_a_demo_brief(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["b_2026-10-07_u_demo_001", "1", "--section", "meetings"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("[meetings n°1] Réunion toute la journée")
    assert "verdict : soutenu" in out


def test_command_reports_bad_input(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["b_2026-10-07_u_demo_001", "99"]) == 1
    assert "aucune entrée de rang 99" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        main(["b_2031-01-01_u_demo_001", "1"])  # journée absente
    assert "items.jsonl" in capsys.readouterr().err
