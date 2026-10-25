"""Échéance fixée au manager, lue dans le texte d'un mail : « avant le mardi 13 octobre ».

Les données n'ont pas de tâches (`due_at`) : les échéances sont écrites dans les mails.
Une règle plutôt qu'un modèle : seulement 14 mails sur 181 en ont une, presque toujours sous
une forme explicite, et le score a besoin de la date elle-même (jours restants), pas d'un
simple oui/non. Mesurée sur les étiquettes « échéance » de Théo (25/10/2026).

Une échéance = un déclencheur (« avant », « jusqu'au », « d'ici », « au plus tard »,
« à N semaines de l'échéance ») suivi d'une date, dans une phrase qui n'est ni une validité
d'offre (« valable jusqu'au »), ni un engagement de l'expéditeur (« nous revenons vers vous
avant »). L'année manquante est déduite de la date de référence (date du mail).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from packages.analyse.clock import to_paris

MONTHS = {
    "janvier": 1,
    "février": 2,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "août": 8,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "décembre": 12,
    "decembre": 12,
}
WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
NUMBERS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "huit": 8}

_A = r"[’']"  # apostrophe droite ou typographique (« jusqu’au »)
_DATE = (
    rf"(?:(?P<wd>{'|'.join(WEEKDAYS)})\s+)?(?:(?P<day>\d{{1,2}})(?:er)?\s+(?P<month>"
    rf"{'|'.join(MONTHS)})(?:\s+(?P<year>\d{{4}}))?|(?P<num>\d{{1,2}})/(?P<nmonth>\d{{1,2}}))"
    rf"|(?P<alone>{'|'.join(WEEKDAYS)})|(?P<rel>demain|ce soir|aujourd{_A}hui)"
)
_TRIGGER = re.compile(
    rf"\b(?:avant(?:\s+le)?|jusqu{_A}(?:au|à)|d{_A}ici(?:\s+le)?|au plus tard(?:\s+le)?"
    rf"|à rendre(?:\s+le)?)\s+(?:{_DATE})\b",  # pas « pour le » : « on valide pour le 10 ? »
    re.IGNORECASE,
)
_COUNTDOWN = re.compile(
    rf"\b(?:à|dans)\s+(?P<n>\d+|{'|'.join(NUMBERS)})\s+(?P<unit>jours?|semaines?)\s+de\s+"
    rf"l{_A}échéance\b",
    re.IGNORECASE,
)
# Phrases où une date suit un déclencheur sans être une échéance pour le manager
_NOT_A_DEADLINE = re.compile(
    r"valable|valide jusqu|revenons vers|reviens vers|restons disponibles|devriez le recevoir"
    r"|sommes disponibles",
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Deadline:
    when: date
    phrase: str  # le passage reconnu, pour l'explication du score

    def days_left(self, now: datetime) -> int:
        """Jours calendaires entre aujourd'hui (heure de Paris) et l'échéance ; < 0 si dépassée."""
        return (self.when - to_paris(now).date()).days


def _with_year(day: int, month: int, reference: date) -> date | None:
    """La date (jour, mois) la plus proche de la référence : un mail du 28 décembre qui
    parle du « 3 janvier » vise l'année suivante."""
    candidates = []
    for year in (reference.year - 1, reference.year, reference.year + 1):
        try:
            candidates.append(date(year, month, day))
        except ValueError:
            continue
    return min(candidates, key=lambda d: abs((d - reference).days), default=None)


def _resolve(match: re.Match[str], reference: date) -> date | None:
    if match["day"]:
        if match["year"]:
            try:
                return date(int(match["year"]), MONTHS[match["month"].lower()], int(match["day"]))
            except ValueError:
                return None
        return _with_year(int(match["day"]), MONTHS[match["month"].lower()], reference)
    if match["num"]:
        if not 1 <= int(match["nmonth"]) <= 12:
            return None
        return _with_year(int(match["num"]), int(match["nmonth"]), reference)
    if match["alone"]:  # prochain jour de ce nom, strictement après la référence
        ahead = (WEEKDAYS.index(match["alone"].lower()) - reference.weekday() - 1) % 7 + 1
        return reference + timedelta(days=ahead)
    rel = match["rel"].lower()
    return reference + timedelta(days=1 if rel == "demain" else 0)


def find_deadline(text: str, written_at: datetime) -> Deadline | None:
    """L'échéance la plus proche annoncée dans `text`, ou None. `written_at` : date du mail."""
    reference = to_paris(written_at).date()
    found: list[Deadline] = []
    for sentence in _SENTENCES.split(text):
        if _NOT_A_DEADLINE.search(sentence):
            continue
        for match in _TRIGGER.finditer(sentence):
            when = _resolve(match, reference)
            if when is not None:
                found.append(Deadline(when, match.group(0)))
        for match in _COUNTDOWN.finditer(sentence):
            n = match["n"].lower()
            count = int(n) if n.isdigit() else NUMBERS[n]
            days = count * (7 if match["unit"].lower().startswith("semaine") else 1)
            found.append(Deadline(reference + timedelta(days=days), match.group(0)))
    return min(found, key=lambda d: d.when, default=None)
