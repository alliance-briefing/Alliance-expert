"""Tests du score de priorité (L2.2)."""

from __future__ import annotations

from packages.analyse.aggregate.cases import group_into_cases
from packages.analyse.scoring.score import ScoringConfig, score_case
from packages.analyse.tests.factories import at, make_event, make_mail, make_task


def _score(items, now="2026-10-07T09:00:00+02:00", config=None):
    """Raccourci : groupe les items en affaires et retourne le score de la première."""
    cases = group_into_cases(items)
    return score_case(cases[0], at(now), config)


# --- Bruit ---


def test_newsletter_baisse_le_score():
    mail = make_mail("m1", "2026-10-07T08:00:00+02:00", title="Notre newsletter du mois")
    score, reasons = _score([mail])
    assert score < 50
    assert any("newsletter" in r for r in reasons)


def test_noreply_baisse_le_score():
    mail = make_mail(
        "m2",
        "2026-10-07T08:00:00+02:00",
        participants=[{"name": "Sys", "email": "noreply@service.com", "role": "from"}],
    )
    score, _ = _score([mail])
    assert score < 50


# --- Importance native ---


def test_importance_high_monte_le_score():
    mail = make_mail("m3", "2026-10-07T08:00:00+02:00", native_importance="high")
    score, reasons = _score([mail])
    assert score > 50
    assert any("important" in r for r in reasons)


def test_importance_low_baisse_le_score():
    mail = make_mail("m4", "2026-10-07T08:00:00+02:00", native_importance="low")
    score, _ = _score([mail])
    assert score < 50


# --- Mots d'urgence ---


def test_mot_urgent_monte_le_score():
    mail = make_mail("m5", "2026-10-07T08:00:00+02:00", title="URGENT : rapport à rendre")
    score, reasons = _score([mail])
    assert score > 50
    assert any("urgence" in r for r in reasons)


# --- Échéance ---


def test_tache_en_retard_monte_le_score():
    task = make_task("t1", due_at="2026-10-06T09:00:00+02:00", completed=False)
    score, reasons = _score([task])
    assert score > 50
    assert any("retard" in r for r in reasons)


def test_echeance_dans_24h_monte_le_score():
    task = make_task("t2", due_at="2026-10-07T18:00:00+02:00", completed=False)
    score, reasons = _score([task])
    assert score > 50
    assert any("24" in r for r in reasons)


def test_tache_completee_ne_monte_pas():
    task = make_task("t3", due_at="2026-10-06T09:00:00+02:00", completed=True)
    score, _ = _score([task])
    assert score == 50  # score neutre, aucun signal


# --- Expéditeur ---


def test_hierarchie_monte_le_score():
    config = ScoringConfig(hierarchy=frozenset({"boss@company.fr"}))
    mail = make_mail(
        "m6",
        "2026-10-07T08:00:00+02:00",
        participants=[{"name": "Boss", "email": "boss@company.fr", "role": "from"}],
    )
    score, reasons = _score([mail], config=config)
    assert score > 50
    assert any("hiérarchie" in r for r in reasons)


def test_blacklist_baisse_le_score():
    config = ScoringConfig(blacklist=frozenset({"spam@bad.com"}))
    mail = make_mail(
        "m7",
        "2026-10-07T08:00:00+02:00",
        participants=[{"name": "Spam", "email": "spam@bad.com", "role": "from"}],
    )
    score, reasons = _score([mail], config=config)
    assert score < 50
    assert any("bloqué" in r for r in reasons)


# --- Réunion ---


def test_reunion_dans_48h_monte_le_score():
    event = make_event(
        "e1",
        start_at="2026-10-07T14:00:00+02:00",
        end_at="2026-10-07T15:00:00+02:00",
    )
    score, reasons = _score([event])
    assert score > 50
    assert any("réunion" in r for r in reasons)


# --- Déterminisme ---


def test_memes_entrees_meme_sortie():
    mail = make_mail("m8", "2026-10-07T08:00:00+02:00", title="URGENT : client en attente")
    items = [mail]
    result1 = _score(items)
    result2 = _score(items)
    assert result1 == result2


# --- Bornes ---


def test_score_toujours_entre_0_et_100():
    config = ScoringConfig(hierarchy=frozenset({"boss@company.fr"}))
    mail = make_mail(
        "m9",
        "2026-10-07T08:00:00+02:00",
        title="URGENT URGENT URGENT deadline impératif",
        native_importance="high",
        participants=[{"name": "Boss", "email": "boss@company.fr", "role": "from"}],
    )
    score, _ = _score([mail], config=config)
    assert 0 <= score <= 100


# --- Raisons ---


def test_toujours_au_moins_une_raison():
    mail = make_mail("m10", "2026-10-07T08:00:00+02:00")
    _, reasons = _score([mail])
    assert len(reasons) >= 1


def test_jamais_plus_de_4_raisons():
    config = ScoringConfig(hierarchy=frozenset({"boss@company.fr"}))
    mail = make_mail(
        "m11",
        "2026-10-07T08:00:00+02:00",
        title="URGENT deadline impératif",
        native_importance="high",
        participants=[{"name": "Boss", "email": "boss@company.fr", "role": "from"}],
    )
    _, reasons = _score([mail], config=config)
    assert len(reasons) <= 4


def test_echeance_dans_48h_monte_le_score():
    task = make_task("t4", due_at="2026-10-08T18:00:00+02:00", completed=False)
    score, reasons = _score([task])
    assert score > 50
    assert any("48" in r for r in reasons)


def test_contact_frequent_monte_le_score():
    config = ScoringConfig(frequent_contacts=frozenset({"collegue@company.fr"}))
    mail = make_mail(
        "m12",
        "2026-10-07T08:00:00+02:00",
        participants=[{"name": "Collègue", "email": "collegue@company.fr", "role": "from"}],
    )
    score, reasons = _score([mail], config=config)
    assert score > 50
    assert any("fréquent" in r for r in reasons)


def test_mail_non_lu_contact_frequent_monte_le_score():
    config = ScoringConfig(frequent_contacts=frozenset({"collegue@company.fr"}))
    mail = make_mail(
        "m13",
        "2026-10-07T08:00:00+02:00",
        is_read=False,
        participants=[
            {"name": "Collègue", "email": "collegue@company.fr", "role": "from"},
            {"name": "Manager", "email": "collegue@company.fr", "role": "to"},
        ],
    )
    score, _ = _score([mail], config=config)
    assert score > 50
