"""Claims soutenus par leur source (L3.6), vérifiés par le lot 3 sans le code du lot 2.

Un texte (claim, ou texte affiché d'une entrée) est soutenu si chaque fait qu'il affirme
est vrai pour au moins un des items qu'il cite, à l'heure de la collecte (heure de Paris) :

1. source : chaque item cité existe dans la journée et appartient à l'utilisateur ;
2. titre : un texte entre « » est le titre exact d'un item cité (« (sans objet) » si vide) ;
3. type : « mail », « réunion », « tâche », « fichier » est la source d'un item cité ;
4. état : « non lu », « en retard » / « dépassée », « annulée » sont vrais à la collecte ;
5. date et heure, rattachées au bon champ : « reçu le » → `created_at`, « échéance » →
   `due_at`, sinon le début d'une réunion citée ; à défaut d'indice, n'importe quelle date
   de l'item, ou la date écrite dans son texte ;
6. nombre hors titre et hors date : il figure dans le titre ou le texte d'un item cité.

Limite : une paraphrase (« le client menace de saisir un avocat ») ne se vérifie pas de
façon déterministe. Pour un texte rédigé par un LLM, le taux est donc une borne haute.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from contracts.models import Brief, Item
from eval.metrics import reading_order

PARIS = ZoneInfo("Europe/Paris")
NO_TITLE = "(sans objet)"
_PLACEHOLDER = "\x00"

_QUOTE = re.compile(r"«([^«»]*)»")
_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b(?:\s+à\s+(\d{1,2})[:h](\d{2})\b)?")
_TIME = re.compile(r"\bà\s+(\d{1,2})[:h](\d{2})\b")
_NUMBER = re.compile(r"\d+(?:[\s.,]\d+)*")
_KINDS = {"mail": "mail", "réunion": "calendar", "tâche": "task", "fichier": "file"}
_KIND = re.compile(r"\b(mail|réunion|tâche|fichier)s?\b")
_SPACES = re.compile(r"\s+")


def title_of(item: Item) -> str:
    return item.title.strip() or NO_TITLE


def _strip_titles(text: str, items: Sequence[Item]) -> tuple[str, str | None]:
    """Retire du texte les titres des items cités (le plus long d'abord).
    Renvoie le reste et, s'il reste une citation « … » inconnue, le motif d'échec."""
    rest = text
    for title in sorted({title_of(i) for i in items}, key=len, reverse=True):
        rest = rest.replace(title, _PLACEHOLDER)
    for quoted in _QUOTE.findall(rest):
        if quoted.strip() != _PLACEHOLDER:
            return rest, f"titre cité absent des sources : « {quoted.strip()} »"
    return rest, None


def _paris(value: datetime | None) -> datetime | None:
    return value.astimezone(PARIS) if value is not None else None


def _moments(item: Item, cue: str) -> list[datetime]:
    """Dates de l'item qu'un indice (« reçu », « échéance », « réunion ») désigne."""
    if cue == "reçu":
        fields = [item.created_at]
    elif cue == "échéance":
        fields = [item.due_at]
    elif cue == "réunion":
        fields = [item.start_at] if item.source == "calendar" else []
    else:
        fields = [item.created_at, item.updated_at, item.start_at, item.end_at, item.due_at]
    return [m for m in (_paris(f) for f in fields) if m is not None]


def _cue(before: str, items: Sequence[Item]) -> str:
    before = before.lower()[-25:]
    if "reçu" in before or "envoyé" in before:
        return "reçu"
    if "échéance" in before:
        return "échéance"
    if items and all(i.source == "calendar" for i in items):
        return "réunion"
    return ""


def _haystack(item: Item) -> str:
    text = " ".join(filter(None, [item.title, item.snippet, item.content_text]))
    return _SPACES.sub(" ", text)


def _check_dates(rest: str, items: Sequence[Item]) -> tuple[str, str | None]:
    for found in list(_DATE.finditer(rest)):
        day, month, hour, minute = found.groups()
        cue = _cue(rest[: found.start()], items)
        ok = any(
            (m.day, m.month) == (int(day), int(month))
            and (hour is None or (m.hour, m.minute) == (int(hour), int(minute)))
            for item in items
            for m in _moments(item, cue)
        )
        written = found.group(0)
        # Sans indice, une date écrite dans le texte d'une source compte aussi.
        ok = ok or (not cue and any(written in _haystack(i) for i in items))
        if not ok:
            where = f" pour « {cue} »" if cue else ""
            return rest, f"date « {written} » introuvable dans les sources{where}"
    rest = _DATE.sub(" ", rest)
    for found in list(_TIME.finditer(rest)):
        hour, minute = (int(g) for g in found.groups())
        cue = _cue(rest[: found.start()], items)
        if not any(
            (m.hour, m.minute) == (hour, minute) for item in items for m in _moments(item, cue)
        ):
            return rest, f"heure « {found.group(0)} » introuvable dans les sources"
    return _TIME.sub(" ", rest), None


def _check_states(rest: str, items: Sequence[Item]) -> str | None:
    low = rest.lower()
    if "non lu" in low and not any(i.source == "mail" and i.is_read is False for i in items):
        return "« non lu » faux"
    if ("en retard" in low or "dépassée" in low) and not any(
        i.is_overdue(i.collected_at) for i in items
    ):
        return "« en retard » faux"
    cancelled = any(
        i.attributes and i.attributes.calendar and i.attributes.calendar.is_cancelled for i in items
    )
    if "annulée" in low and not cancelled:
        return "« annulée » faux"
    return None


def check_text(
    text: str, cited_ids: Iterable[str], day_items: Mapping[str, Item], user: str
) -> str | None:
    """None si le texte est soutenu par les items cités, sinon le premier fait en cause."""
    items = []
    for item_id in cited_ids:
        item = day_items.get(item_id)
        if item is None:
            return f"item cité absent de la journée : {item_id}"
        if item.owner_user_id != user:
            return f"item cité d'un autre utilisateur : {item_id}"
        items.append(item)
    rest, problem = _strip_titles(text, items)
    if problem:
        return problem
    for word in _KIND.findall(rest.lower()):
        if not any(i.source == _KINDS[word] for i in items):
            return f"type « {word} » faux"
    problem = _check_states(rest, items)
    if problem:
        return problem
    rest, problem = _check_dates(rest, items)
    if problem:
        return problem
    haystacks = [_haystack(i).replace(" ", " ").replace("\xa0", " ") for i in items]
    for number in _NUMBER.findall(rest.replace(_PLACEHOLDER, " ")):
        number = number.strip()
        if number and not any(number in h for h in haystacks):
            return f"nombre « {number} » introuvable dans les sources"
    return None


@dataclass(frozen=True)
class Unsupported:
    kind: str  # "claim" ou "texte"
    position: int  # position de l'entrée dans l'ordre de lecture (1 = première)
    text: str
    reason: str


@dataclass
class ClaimStats:
    claims: int = 0
    claims_ok: int = 0
    texts: int = 0
    texts_ok: int = 0
    unsupported: list[Unsupported] = field(default_factory=list)


def check_brief(brief: Brief, items: Sequence[Item]) -> ClaimStats:
    """Vérifie chaque claim et le texte affiché de chaque entrée du brief."""
    by_id = {item.item_id: item for item in items}
    user = brief.owner_user_id
    stats = ClaimStats()
    for position, entry in enumerate(reading_order(brief), start=1):
        for claim in entry.claims:
            stats.claims += 1
            reason = check_text(claim.text, claim.source_item_ids, by_id, user)
            if reason is None:
                stats.claims_ok += 1
            else:
                stats.unsupported.append(Unsupported("claim", position, claim.text, reason))
        stats.texts += 1
        reason = check_text(entry.text, entry.source_item_ids, by_id, user)
        if reason is None:
            stats.texts_ok += 1
        else:
            stats.unsupported.append(Unsupported("texte", position, entry.text, reason))
    return stats
