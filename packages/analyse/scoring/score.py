"""Score de priorité expliqué pour une affaire (L2.2).

Classe une affaire de 0 à 100 avec 1 à 4 raisons lisibles, indépendamment du LLM.
Chaque signal se calcule sur toute l'affaire : le résultat ne dépend pas de l'ordre des
items, et les mêmes entrées donnent toujours la même sortie. Spécification détaillée :
tests/test_scoring.py.

Signaux (partant de 50) : bruit −30 · importance native +20 / −10 · mots d'urgence +15
(annulés par le bruit) · tâche en retard +25, sinon échéance la plus proche < 24 h +20 ou
< 48 h +10 · expéditeur bloqué −20, hiérarchie +25 ou contact fréquent +10 · réunion en
cours ou dans les 48 h +15 · mail non lu adressé au manager +10.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from contracts.models import Item
from packages.analyse.aggregate.cases import Case

NEUTRAL = 50
MAX_REASONS = 4
STANDARD = "élément professionnel standard"


@dataclass(frozen=True)
class ScoringConfig:
    """Barème modifiable sans toucher au code. Adresses et mots-clés : casse et accents
    ignorés ; un `set` simple est accepté et n'est jamais modifié."""

    # Adresses du manager : le contrat A ne les donne pas. Sans elles, le signal
    # « mail non lu adressé directement » est éteint.
    user_emails: frozenset[str] = frozenset()
    frequent_contacts: frozenset[str] = frozenset()
    hierarchy: frozenset[str] = frozenset()
    blacklist: frozenset[str] = frozenset()
    urgency_keywords: frozenset[str] = frozenset(
        {
            "urgent",
            "urgente",
            "urgence",
            "asap",
            "dès que possible",
            "deadline",
            "échéance",
            "impératif",
        }
    )
    noise_keywords: frozenset[str] = frozenset(
        {
            "newsletter",
            "noreply",
            "no-reply",
            "notification",
            "unsubscribe",
            "désabonner",
        }
    )


@dataclass(frozen=True)
class Signal:
    """Une contribution au score et la raison qui l'explique."""

    weight: int
    reason: str


def score_case(
    case: Case, now: datetime, config: ScoringConfig | None = None
) -> tuple[int, list[str]]:
    """(score 0-100, 1 à 4 raisons) pour une affaire, à l'instant `now` (avec fuseau).

    Les raisons gardent les signaux les plus lourds ; à poids égal, l'ordre des signaux
    ci-dessus. `now` est toujours passé en paramètre, jamais lu sur l'horloge.
    """
    if now.utcoffset() is None:
        raise ValueError("now doit avoir un fuseau horaire")
    config = config or ScoringConfig()
    items = case.items
    noise = _is_noise(items, config)
    candidates = [
        Signal(-30, "message automatique ou newsletter") if noise else None,
        _importance(items),
        None if noise else _urgency(items, config),
        _deadline(items, now),
        _sender(items, config),
        _meeting(items, now),
        _direct(items, config),
    ]
    signals = [signal for signal in candidates if signal is not None]
    score = max(0, min(100, NEUTRAL + sum(signal.weight for signal in signals)))
    ranked = sorted(signals, key=lambda signal: -abs(signal.weight))  # tri stable
    return score, [signal.reason for signal in ranked][:MAX_REASONS] or [STANDARD]


# --- signaux ------------------------------------------------------------------------------------


def _is_noise(items: Sequence[Item], config: ScoringConfig) -> bool:
    """Newsletter, notification automatique : mot-clé dans le titre, l'extrait ou l'adresse
    de l'expéditeur (« newsletters » compte comme « newsletter »)."""
    keywords = _normalized(config.noise_keywords)
    if not keywords:
        return False
    texts = [_normalize(f"{item.title} {item.snippet}") for item in items]
    texts += sorted(_addresses(items, "from"))
    return any(keyword in text for keyword in keywords for text in texts)


def _importance(items: Sequence[Item]) -> Signal | None:
    levels = {item.native_importance for item in items}
    if "high" in levels:
        return Signal(20, "marqué comme important")
    if "low" in levels:
        return Signal(-10, "marqué comme peu important")
    return None


def _urgency(items: Sequence[Item], config: ScoringConfig) -> Signal | None:
    """Un mot (ou une expression) d'urgence entier dans un titre : « urgentiste » ne compte pas."""
    titles = [_normalize(item.title) for item in items]
    for keyword in sorted(_normalized(config.urgency_keywords)):
        pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)")
        if any(pattern.search(title) for title in titles):
            return Signal(15, "contient des mots signalant l'urgence")
    return None


def _deadline(items: Sequence[Item], now: datetime) -> Signal | None:
    """Tâches non terminées : un retard l'emporte ; sinon l'échéance la plus proche."""
    open_tasks = [i for i in items if i.source == "task" and i.completed is False and i.due_at]
    if any(task.is_overdue(now) for task in open_tasks):
        return Signal(25, "tâche en retard")
    upcoming = [task.due_at - now for task in open_tasks if task.due_at and task.due_at >= now]
    if upcoming:
        nearest = min(upcoming)
        if nearest <= timedelta(hours=24):
            return Signal(20, "échéance dans moins de 24 h")
        if nearest <= timedelta(hours=48):
            return Signal(10, "échéance dans moins de 48 h")
    return None


def _sender(items: Sequence[Item], config: ScoringConfig) -> Signal | None:
    """Liste noire, puis hiérarchie, puis contacts fréquents : le premier trouvé décide."""
    senders = _addresses(items, "from")
    if senders & _normalized(config.blacklist):
        return Signal(-20, "expéditeur bloqué")
    if senders & _normalized(config.hierarchy):
        return Signal(25, "message de la hiérarchie")
    if senders & _normalized(config.frequent_contacts):
        return Signal(10, "contact fréquent")
    return None


def _meeting(items: Sequence[Item], now: datetime) -> Signal | None:
    """Réunion en cours, sinon qui commence dans les 48 h ; annulée ou déclinée : ignorée."""
    meetings = [item for item in items if item.source == "calendar" and _attended(item)]
    if any(m.start_at and m.end_at and m.start_at <= now < m.end_at for m in meetings):
        return Signal(15, "réunion en cours")
    horizon = now + timedelta(hours=48)
    if any(m.start_at and now < m.start_at <= horizon for m in meetings):
        return Signal(15, "réunion à venir dans les 48 h")
    return None


def _direct(items: Sequence[Item], config: ScoringConfig) -> Signal | None:
    """Un mail non lu dont le manager est destinataire principal (« to », pas « cc »)."""
    users = _normalized(config.user_emails)
    for item in items:
        if item.source == "mail" and item.is_read is False and _addresses([item], "to") & users:
            return Signal(10, "mail non lu adressé directement")
    return None


# --- outils -------------------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Minuscules sans accents : « Échéance » → « echeance »."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _normalized(values: Iterable[str]) -> frozenset[str]:
    return frozenset(_normalize(value).strip() for value in values if value.strip())


def _addresses(items: Iterable[Item], role: str) -> set[str]:
    return {_normalize(p.email) for item in items for p in item.participants if p.role == role}


def _attended(meeting: Item) -> bool:
    calendar = meeting.attributes.calendar if meeting.attributes else None
    if calendar is None:
        return True
    return calendar.is_cancelled is not True and calendar.response != "declined"
