"""Garde L2.4 : 20 cas adverses, tous rejetés (critère d'acceptation du guide).

Chaque cas est un claim plausible mais faux sur des items fictifs : date ou heure fausse,
montant inversé, numéro voisin, homonyme, information d'un autre utilisateur… La garde
doit le rejeter pour la bonne raison, et le brief qui le contient doit être dégradé.
"""

from __future__ import annotations

from datetime import date

import pytest

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

DEVIS = make_mail(
    "devis",
    "2026-10-07T06:15:00Z",  # 08:15 à Paris
    title="Devis 2026-3025",
    snippet="Montant : 1 250,00 € HT, validé par Camille Laurent.",
)
FACTURE = make_mail("facture", "2026-10-06T09:00:00+02:00", title="Facture 2026-118", is_read=True)
POINT = make_event(
    "point", "2026-10-07T14:00:00+02:00", "2026-10-07T15:00:00+02:00", title="Point dossier"
)
RELANCE = make_task("relance", "2026-10-09T12:00:00+02:00", title="Relancer le garage")
# Homonyme chez un autre utilisateur : même titre, autre contenu.
AUTRE = make_mail(
    "autre",
    "2026-10-07T07:00:00+02:00",
    title="Devis 2026-3025",
    snippet="Remise accordée de 15 %.",
    owner_user_id=OTHER_USER,
)
ITEMS = {item.item_id: item for item in (DEVIS, FACTURE, POINT, RELANCE, AUTRE)}

CASES = [
    # dates et heures
    ("date fausse", "Mail non lu « Devis 2026-3025 », reçu le 08/10 à 08:15", DEVIS, "date"),
    ("heure fausse", "Mail non lu « Devis 2026-3025 », reçu le 07/10 à 09:15", DEVIS, "date"),
    ("heure UTC non convertie", "Mail non lu « Devis 2026-3025 », reçu le 07/10 à 06:15", DEVIS, "date"),
    ("compte d'un autre type d'élément", "Et 1 autre tâche en retard", DEVIS, "type"),
    ("date et heure de deux champs", "Réunion « Point dossier » le 01/10 à 14:00", POINT, "date"),
    ("échéance fausse", "Tâche « Relancer le garage », échéance le 10/10", RELANCE, "date"),
    # montants et numéros
    ("montant inversé", "Le devis s'élève à 1 520,00 €", DEVIS, "nombre"),
    ("numéro de dossier voisin", "Le dossier 2026-3052 a un devis", DEVIS, "nombre"),
    ("information d'un autre utilisateur", "Remise de 15 % sur le devis", DEVIS, "nombre"),
    # titres et noms propres
    ("titre modifié", "Mail non lu « Devis 2026-3026 », reçu le 07/10 à 08:15", DEVIS, "titre"),
    ("titre tronqué", "Mail non lu « Devis », reçu le 07/10 à 08:15", DEVIS, "titre"),
    ("homonyme : prénom juste, nom faux", "Devis validé par Camille Durand", DEVIS, "nom propre"),
    ("nom propre inventé", "Mail non lu « Devis 2026-3025 », transféré par Julie", DEVIS, "nom propre"),
    # sources
    ("homonyme chez un autre utilisateur", "Mail non lu « Devis 2026-3025 »", AUTRE, "item cité d'un autre"),
    ("item inexistant", "Mail non lu « Devis 2026-3025 »", None, "item cité absent"),
    ("type d'élément faux", "Tâche « Devis 2026-3025 »", DEVIS, "type"),
    # états et comptes
    ("mail lu dit non lu", "Mail non lu « Facture 2026-118 », reçu le 06/10 à 09:00", FACTURE, "état"),
    ("tâche dite en retard avant l'échéance", "Tâche « Relancer le garage », échéance dépassée le 09/10", RELANCE, "état"),
    ("réunion dite annulée", "Réunion « Point dossier » le 07/10 à 14:00, annulée", POINT, "état"),
    ("compte gonflé", "Et 4 autres mails non lus", DEVIS, "compte"),
]  # fmt: skip


def test_there_are_twenty_distinct_cases() -> None:
    assert len(CASES) == 20
    assert len({name for name, *_ in CASES}) == 20


@pytest.mark.parametrize(("name", "text", "item", "reason"), CASES, ids=[c[0] for c in CASES])
def test_the_guard_rejects_the_adversarial_claim(name: str, text: str, item, reason: str) -> None:
    claim = Claim(text=text, source_item_ids=[item.item_id if item else "mail:fantome"])

    problem = check_claim(claim, ITEMS, USER)

    assert problem is not None and problem.startswith(reason), problem


ON_DEVIS = [(name, text) for name, text, item, _ in CASES if item is DEVIS]


@pytest.mark.parametrize(("name", "text"), ON_DEVIS, ids=[name for name, _ in ON_DEVIS])
def test_a_brief_carrying_the_claim_is_degraded(name: str, text: str) -> None:
    data = generate_brief([DEVIS], USER, date(2026, 10, 7), now=at(COLLECTED_AT)).model_dump()
    data["sections"][0]["entries"][0]["claims"] = [
        {"text": text, "source_item_ids": [DEVIS.item_id]}
    ]
    brief = Brief.model_validate(data)

    entry = guard_brief(brief, [DEVIS]).sections[0].entries[0]

    assert entry.text == "À consulter : « Devis 2026-3025 »"
    assert entry.claims == []
