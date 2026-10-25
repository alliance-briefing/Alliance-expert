"""Tests de la règle d'échéance (dates françaises écrites dans un mail)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from packages.analyse.scoring.ml.deadline import find_deadline

PARIS = timezone(timedelta(hours=2))
MON_5_OCT = datetime(2026, 10, 5, 9, 0, tzinfo=PARIS)  # un lundi


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Peux-tu le relire avant le mardi 13 octobre ?", date(2026, 10, 13)),
        ("La période de saisie est ouverte jusqu’au lundi 16 novembre.", date(2026, 11, 16)),
        ("Merci de m'envoyer vos éléments d'ici dimanche 8 novembre.", date(2026, 11, 8)),
        ("Retours attendus avant vendredi 16 octobre.", date(2026, 10, 16)),
        ("À rendre au plus tard le 1er décembre 2026.", date(2026, 12, 1)),
        ("Merci de valider avant le 12/10.", date(2026, 10, 12)),
        ("Peux-tu me répondre avant jeudi ?", date(2026, 10, 8)),
        ("Réponse attendue avant lundi.", date(2026, 10, 12)),  # lundi suivant, pas aujourd'hui
        ("Merci de le signer avant demain.", date(2026, 10, 6)),
        ("Nous sommes à 2 semaines de l'échéance.", date(2026, 10, 19)),
        ("Il reste dans trois jours de l'échéance.", date(2026, 10, 8)),
    ],
)
def test_echeances_reconnues(text, expected):
    deadline = find_deadline(text, MON_5_OCT)
    assert deadline is not None and deadline.when == expected


@pytest.mark.parametrize(
    "text",
    [
        "Le devis est valable jusqu'au jeudi 8 octobre.",
        "Nous revenons vers vous avant le jeudi 29 octobre.",
        "Je reviens vers toi avant vendredi.",
        "Nous restons disponibles pour la présenter avant le mercredi 21 octobre.",
        "On valide pour le samedi 10 octobre ?",
        "Le point du dimanche 25 octobre est déplacé en salle Horizon.",
        "Vous devriez le recevoir sous 48 heures.",
        "Avant le 31/02, rien.",  # date impossible
        "Merci de valider avant le 12/13.",  # mois impossible
    ],
)
def test_pas_d_echeance(text):
    assert find_deadline(text, MON_5_OCT) is None


def test_la_plus_proche_et_annee_suivante_en_fin_d_annee():
    text = "Avant le 20 octobre pour la v1. Avant le 9 octobre pour le plan."
    assert find_deadline(text, MON_5_OCT).when == date(2026, 10, 9)
    late_december = datetime(2026, 12, 28, 9, 0, tzinfo=PARIS)
    assert find_deadline("À rendre avant le 3 janvier.", late_december).when == date(2027, 1, 3)
    assert find_deadline("Avant le 29 février 2027.", MON_5_OCT) is None


def test_jours_restants_en_heure_de_paris():
    deadline = find_deadline("Avant le mardi 13 octobre.", MON_5_OCT)
    assert deadline.phrase == "Avant le mardi 13 octobre"
    assert deadline.days_left(MON_5_OCT) == 8
    assert deadline.days_left(datetime(2026, 10, 14, 23, 0, tzinfo=PARIS)) == -1
    assert find_deadline("Merci de répondre avant ce soir.", MON_5_OCT).days_left(MON_5_OCT) == 0
