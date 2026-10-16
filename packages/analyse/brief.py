"""Brief à base de règles, sans LLM (stub L2.0, guide §2.3), organisé par affaires (L2.1).

Construit un Brief valide (contrat B) à partir des Item d'un utilisateur : les mails non
lus les plus récents, les réunions du jour et les tâches en retard. Les éléments d'une
même affaire sont réunis dans une entrée (sauf les réunions, une par heure), et chaque
entrée cite aussi les autres éléments de son affaire. Les textes ne reprennent que les
titres et les dates, jamais le contenu (snippet) : un mail piégé ne peut donc rien
injecter dans le brief.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date as Date
from datetime import datetime
from datetime import time as Time

from contracts.models import Brief, Entry, Generator, Item, Section, Stats
from contracts.models.brief import SectionKind
from packages.analyse.aggregate.cases import Case, group_into_cases
from packages.analyse.aggregate.rules import meetings_of_day, overdue_tasks, unread_mails

# Score fixe par section : le vrai barème arrive avec L2.2 (score de priorité expliqué).
SCORES: dict[SectionKind, int] = {"overdue": 70, "actions": 60, "meetings": 50}
# Nombre maximal d'affaires (et non de mails) dans la section « à traiter ».
ACTIONS_LIMIT = 5
NO_TITLE = "(sans objet)"
NOTHING_TO_REPORT = "Aucun élément à signaler pour cette journée."
# Noms des éléments liés, au singulier et au pluriel, dans l'ordre d'affichage.
NOUNS = {
    "mail": ("mail", "mails"),
    "calendar": ("réunion", "réunions"),
    "task": ("tâche", "tâches"),
    "file": ("fichier", "fichiers"),
}

# Une règle de rédaction reçoit les éléments d'une entrée et renvoie (texte, raison).
Wording = Callable[[list[Item]], tuple[str, str]]


def generate_brief(
    items: Sequence[Item], user: str, date: Date | str, *, now: datetime | None = None
) -> Brief:
    """Brief du jour `date` pour `user`, construit par règles à partir de `items`.

    `now` fixe `generated_at` (utile aux tests) ; par défaut, l'heure courante.
    Le résultat ne dépend que des entrées, sauf `generated_at` et `stats.latency_ms`.
    """
    started = time.perf_counter()
    day = date if isinstance(date, Date) else Date.fromisoformat(date)
    mine = [item for item in items if item.owner_user_id == user]

    warnings = []
    if len(mine) < len(items):
        warnings.append(f"{len(items) - len(mine)} items ignorés : autre utilisateur")

    # « Maintenant » = l'heure de la collecte, pas l'horloge : le brief reste reproductible.
    collected_at = max((item.collected_at for item in mine), default=None)
    overdue = overdue_tasks(mine, collected_at) if collected_at else []

    case_of = {item.item_id: case for case in group_into_cases(mine) for item in case.items}
    actions = _by_case(unread_mails(mine, limit=None), case_of)[:ACTIONS_LIMIT]
    meetings = [[event] for event in meetings_of_day(mine, day)]
    sections = [
        _section("actions", actions, _mails_wording, case_of),
        _section("meetings", meetings, lambda group: _meeting_wording(group[0], day), case_of),
        _section("overdue", _by_case(overdue, case_of), _tasks_wording, case_of),
    ]
    if not any(section.entries for section in sections):
        warnings.append(NOTHING_TO_REPORT)

    cited = {item_id for s in sections for e in s.entries for item_id in e.source_item_ids}
    return Brief(
        brief_id=f"b_{day.isoformat()}_{user}",
        owner_user_id=user,
        date=day,
        generated_at=now or datetime.now().astimezone(),
        generator=Generator(llm="rules"),
        sections=sections,
        stats=Stats(
            items_in=len(items),
            items_out=len(cited),
            tokens_in=0,
            tokens_out=0,
            latency_ms=round((time.perf_counter() - started) * 1000),
            cost_eur=0.0,
        ),
        warnings=warnings,
    )


def _by_case(selected: list[Item], case_of: dict[str, Case]) -> list[list[Item]]:
    """Réunit les éléments sélectionnés par affaire, en gardant l'ordre de la sélection."""
    groups: dict[str, list[Item]] = {}
    for item in selected:
        groups.setdefault(case_of[item.item_id].case_id, []).append(item)
    return list(groups.values())


def _section(
    kind: SectionKind, groups: list[list[Item]], wording: Wording, case_of: dict[str, Case]
) -> Section:
    entries = []
    for rank, group in enumerate(groups, start=1):
        text, reason = wording(group)
        case = case_of[group[0].item_id]
        shown = {item.item_id for item in group}
        # Les autres éléments de l'affaire sont cités : la raison qui les annonce a sa source.
        others = [item for item in case.items if item.item_id not in shown]
        entries.append(
            Entry(
                rank=rank,
                priority_score=SCORES[kind],
                priority_reasons=[reason, _context(case, others)] if others else [reason],
                text=text,
                source_item_ids=list(dict.fromkeys(item.item_id for item in group + others)),
            )
        )
    return Section(kind=kind, entries=entries)


def _context(case: Case, others: list[Item]) -> str:
    """« dossier 2026-3025 : aussi 1 réunion, 1 tâche »."""
    counts = Counter(item.source for item in others)
    parts = [f"{counts[s]} {NOUNS[s][counts[s] > 1]}" for s in NOUNS if counts[s]]
    return f"{_case_name(case)} : aussi {', '.join(parts)}"


def _case_name(case: Case) -> str:
    for link in case.links:
        kind, _, reference = link.partition(":")
        if kind != "fil":
            return f"{kind} {reference}"
    return "même fil"


def _title(item: Item) -> str:
    return item.title.strip() or NO_TITLE


def _mails_wording(mails: list[Item]) -> tuple[str, str]:
    """`mails` du plus récent au plus ancien ; le plus récent donne le titre et la date."""
    received = mails[0].created_at.strftime("%d/%m à %H:%M")
    if len(mails) == 1:
        return f"Mail non lu : {_title(mails[0])}", f"non lu, reçu le {received}"
    return (
        f"{len(mails)} mails non lus : {_title(mails[0])}",
        f"non lus, le plus récent reçu le {received}",
    )


def _required(value: datetime | None, item: Item) -> datetime:
    """Horaire garanti par le contrat A pour cette source ; la vérification sert au typage."""
    if value is None:
        raise ValueError(f"horaire manquant : {item.item_id}")
    return value


def _meeting_wording(event: Item, day: Date) -> tuple[str, str]:
    start, end = _required(event.start_at, event), _required(event.end_at, event)
    if start.date() < day:
        when = f"en cours depuis le {start.strftime('%d/%m')}"
    elif start.time() == Time(0) and end.time() == Time(0):
        when = "toute la journée"
    else:
        when = f"à {start.strftime('%H:%M')}"
    return f"Réunion {when} : {_title(event)}", "réunion aujourd'hui"


def _tasks_wording(tasks: list[Item]) -> tuple[str, str]:
    """`tasks` de l'échéance la plus ancienne à la plus récente ; la plus ancienne d'abord."""
    due = _required(tasks[0].due_at, tasks[0]).strftime("%d/%m")
    if len(tasks) == 1:
        return f"Tâche en retard : {_title(tasks[0])}", f"échéance dépassée depuis le {due}"
    return (
        f"{len(tasks)} tâches en retard : {_title(tasks[0])}",
        f"échéances dépassées, la plus ancienne le {due}",
    )
