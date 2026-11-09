"""Garde anti-hallucination déterministe (L2.4) : aucun claim livré sans source vérifiable.

Un claim est accepté si chaque fait qu'il affirme se retrouve dans un item qu'il cite :
1. chaque item cité existe dans l'entrée du brief et appartient à son utilisateur ;
2. le titre entre « » est le titre exact d'un item cité ;
3. chaque date (jj/mm), avec son heure s'il y en a une, est une date d'un item cité
   (heure de Paris), ou figure dans son texte ;
4. chaque nombre, montant ou numéro hors titre figure dans le texte d'un item cité, sauf le
   compte « Et N autres… », qui doit égaler le nombre d'items cités ;
5. chaque nom propre hors titre figure dans le texte ou les participants d'un item cité ;
6. le type (mail, réunion…) et l'état (non lu, annulée, en retard) sont ceux d'un item cité.

Une entrée dont un claim échoue est dégradée (titre de sa source et lien, sans claim) et
un warning l'annonce ; une entrée dont aucune source n'est valide est retirée. Aucun LLM :
un LLM qui vérifie un LLM n'est pas une preuve.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime

from contracts.models import Brief, Claim, Entry, Item
from packages.analyse.clock import is_all_day, to_paris

NO_TITLE = "(sans objet)"  # titre affiché d'un élément sans titre (repris par brief.py)
DEGRADED = "À consulter : « {title} »"

_QUOTED = re.compile(r"«\s(.*)\s»")  # gourmand : un titre peut contenir des « »
_DATE = re.compile(r"\b(\d{2}/\d{2})(?: à (\d{2}:\d{2}))?")
_COUNT = re.compile(r"^Et (\d+) autres? (\w+)")
_NUMBER = re.compile(r"\d+(?:[\s.,\-]\d+)*")
_WORD = re.compile(r"[^\W\d_][\w'’-]*")
_SPACES = re.compile(r"\s+")
KINDS = {
    "mail": "mail",
    "réunion": "calendar",
    "tâche": "task",
    "fichier": "file",
}
STATES = {
    "non lu": lambda item: item.source == "mail" and item.is_read is False,
    "annulée": lambda item: _cancelled(item),
    "en retard": lambda item: item.is_overdue(item.collected_at),
    "dépassée": lambda item: item.is_overdue(item.collected_at),
}


def check_claim(claim: Claim, items: Mapping[str, Item], user: str) -> str | None:
    """Raison du rejet du claim, ou None s'il est soutenu par les items qu'il cite."""
    cited = [items.get(item_id) for item_id in claim.source_item_ids]
    if None in cited:
        return "item cité absent de l'entrée"
    sources = [item for item in cited if item is not None]
    if any(item.owner_user_id != user for item in sources):
        return "item cité d'un autre utilisateur"

    text = claim.text
    if quoted := _QUOTED.search(text):
        if all(_title(item) != quoted.group(1) for item in sources):
            return f"titre introuvable : « {quoted.group(1)} »"
        text = text[: quoted.start()] + text[quoted.end() :]

    if count := _COUNT.match(text):
        if int(count.group(1)) != len(sources):
            return f"compte faux : {count.group(1)} pour {len(sources)} items cités"
        kind = KINDS.get(count.group(2).lower().removesuffix("s"))
        if kind and any(item.source != kind for item in sources):
            return f"type faux : {count.group(2)}"
        text = text[count.end() :]
    elif kind := KINDS.get(text.split(" ", 1)[0].lower()):
        if all(item.source != kind for item in sources):
            return f"type faux : {text.split(' ', 1)[0]}"

    for match in _DATE.finditer(text):
        day, hour = match.groups()
        if not any(_has_moment(item, day, hour) for item in sources):
            return f"date introuvable : {match.group(0)}"
    text = _DATE.sub(" ", text)

    haystacks = [_haystack(item) for item in sources]
    for number in _NUMBER.findall(text):
        if not any(_squash(number) in _squash(h) for h in haystacks):
            return f"nombre introuvable : {number}"
    for word in _WORD.findall(text)[1:]:  # le premier mot porte la majuscule de phrase
        if word[0].isupper() and not any(
            re.search(rf"\b{re.escape(word)}\b", h) for h in haystacks
        ):
            return f"nom propre introuvable : {word}"
    for state, holds in STATES.items():
        if re.search(rf"\b{state}s?\b", text) and not any(holds(item) for item in sources):
            return f"état faux : {state}"
    return None


def guard_brief(brief: Brief, items: Sequence[Item]) -> Brief:
    """Le brief dont chaque claim est vérifié ; entrées dégradées ou retirées, annoncées."""
    user = brief.owner_user_id
    valid = {item.item_id: item for item in items if item.owner_user_id == user}
    warnings = list(brief.warnings)
    sections = []
    for section in brief.sections:
        entries: list[Entry] = []
        for entry in section.entries:
            label = f"{section.kind} n°{entry.rank}"
            sources = [item_id for item_id in entry.source_item_ids if item_id in valid]
            if not sources:
                warnings.append(f"{label} retirée : aucune source valide")
                continue
            in_entry = {item_id: valid[item_id] for item_id in sources}
            problem = next(
                (p for claim in entry.claims if (p := check_claim(claim, in_entry, user))), None
            )
            if problem or len(sources) < len(entry.source_item_ids):
                warnings.append(f"{label} dégradée : {problem or 'source invalide retirée'}")
                entry = _degraded(entry, [valid[item_id] for item_id in sources])
            entries.append(entry.model_copy(update={"rank": len(entries) + 1}))
        sections.append(section.model_copy(update={"entries": entries}))
    if warnings == brief.warnings:
        return brief
    data = brief.model_dump()
    data.update(sections=[s.model_dump() for s in sections], warnings=warnings)
    return Brief.model_validate(data)  # revalide le contrat B après correction


def _degraded(entry: Entry, sources: list[Item]) -> Entry:
    """Repli sûr : le titre de la première source et le lien (porté par la source), sans claim."""
    return entry.model_copy(
        update={
            "text": DEGRADED.format(title=_title(sources[0])),
            "claims": [],
            "recommended_action": None,
            "source_item_ids": [item.item_id for item in sources],
        }
    )


def _has_moment(item: Item, day: str, hour: str | None) -> bool:
    """Le jour (et l'heure) est celui d'une date de l'item, ou figure dans son texte."""
    if hour is None and day in _haystack(item):
        return True
    return any(
        f"{value:%d/%m}" == day and (hour is None or f"{value:%H:%M}" == hour)
        for value in moments(item).values()
    )


def moments(item: Item) -> dict[str, datetime]:
    """Les dates de l'item en heure de Paris ; une journée entière garde ses dates telles quelles."""
    all_day = item.source == "calendar" and is_all_day(item)
    fields = ("created_at", "updated_at", "start_at", "end_at", "due_at")
    return {
        name: value if all_day and name in ("start_at", "end_at") else to_paris(value)
        for name in fields
        if (value := getattr(item, name)) is not None
    }


def _haystack(item: Item) -> str:
    calendar = item.attributes.calendar if item.attributes else None
    location = calendar.location if calendar and calendar.location else ""
    names = " ".join(p.name for p in item.participants)
    return " ".join([item.title, item.snippet, item.content_text or "", location, names])


def _title(item: Item) -> str:
    return item.title.strip() or NO_TITLE


def _cancelled(item: Item) -> bool:
    calendar = item.attributes.calendar if item.attributes else None
    return calendar is not None and calendar.is_cancelled is True


def _squash(value: str) -> str:
    return _SPACES.sub("", value)
