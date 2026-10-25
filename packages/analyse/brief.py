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

from contracts.models import Brief, Entry, Generator, Item, Section, Stats
from contracts.models.brief import SectionKind
from packages.analyse.aggregate.cases import Case, group_into_cases
from packages.analyse.aggregate.rules import meetings_of_day, overdue_tasks, unread_mails
from packages.analyse.clock import PARIS, is_all_day, to_paris
from packages.analyse.scoring.score import STANDARD, score_case

# Seuil de score à partir duquel une affaire sans mail non lu entre dans « À traiter ».
# 50 = score neutre : toute affaire avec au moins un signal net positif est incluse.
ACTIONS_THRESHOLD = 50
# Nombre maximal d'affaires dans la section « à traiter ».
ACTIONS_LIMIT = 7
# Score fixe pour les réunions : triées par heure, leur score ne porte pas l'ordre.
MEETINGS_SCORE = 50
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

    cases = group_into_cases(mine)
    case_of: dict[str, Case] = {item.item_id: case for case in cases for item in case.items}
    _now = collected_at  # alias pour le score (même valeur que pour les retards)

    # Score par affaire. Sans collected_at (liste vide), on ne peut pas scorer.
    case_scores: dict[str, tuple[int, list[str]]] = (
        {case.case_id: score_case(case, _now) for case in cases} if _now else {}
    )

    # Réunions du jour par item_id.
    today_events: set[str] = {e.item_id for e in meetings_of_day(mine, day)}
    # Affaires qui ont au moins une réunion du jour.
    meeting_case_ids: set[str] = {case_of[eid].case_id for eid in today_events if eid in case_of}

    # Tâches en retard par affaire.
    overdue_ids: set[str] = {t.item_id for t in overdue}
    overdue_case_ids: set[str] = {case_of[tid].case_id for tid in overdue_ids if tid in case_of}

    # Mails non lus, triés du plus récent au plus ancien (ordre conservé pour le rendu).
    unread_sorted = unread_mails(mine, limit=None)
    unread_ids: set[str] = {m.item_id for m in unread_sorted}
    unread_case_ids: set[str] = {case_of[mid].case_id for mid in unread_ids if mid in case_of}
    # Regroupement des mails non lus par affaire, dans l'ordre du plus récent au plus ancien.
    unread_by_case: dict[str, list[Item]] = {}
    for m in unread_sorted:
        unread_by_case.setdefault(case_of[m.item_id].case_id, []).append(m)

    # Tâches en retard regroupées par affaire, dans l'ordre due_at le plus ancien en premier.
    overdue_by_case: dict[str, list[Item]] = {}
    for t in overdue:  # overdue_tasks renvoie déjà (due_at, item_id) croissant
        overdue_by_case.setdefault(case_of[t.item_id].case_id, []).append(t)

    # --- Partition des affaires : meetings > overdue > actions ---
    # Réunions : une entrée par événement (une par heure), pas une par affaire.
    meeting_events = meetings_of_day(mine, day)  # déjà triés par heure
    meeting_groups: list[list[Item]] = [[event] for event in meeting_events]

    non_meeting_cases = [c for c in cases if c.case_id not in meeting_case_ids]

    overdue_cases = [c for c in non_meeting_cases if c.case_id in overdue_case_ids]
    overdue_cases.sort(key=lambda c: (-case_scores.get(c.case_id, (0,))[0], c.case_id))
    overdue_groups: list[list[Item]] = [overdue_by_case[c.case_id] for c in overdue_cases]

    actions_cases = [
        c
        for c in non_meeting_cases
        if c.case_id not in overdue_case_ids
        and (
            c.case_id in unread_case_ids or case_scores.get(c.case_id, (0,))[0] > ACTIONS_THRESHOLD
        )
    ]
    actions_cases.sort(key=lambda c: (-case_scores.get(c.case_id, (0,))[0], c.case_id))
    actions_cases = actions_cases[:ACTIONS_LIMIT]
    # Pour chaque affaire : les mails non lus (newest-first) ; si aucun, l'item le plus ancien.
    actions_groups: list[list[Item]] = [
        unread_by_case.get(c.case_id) or [c.items[0]] for c in actions_cases
    ]

    sections = [
        _section(
            "actions",
            actions_groups,
            _mails_wording,
            case_of,
            case_scores,
            overdue_ids,
            today_events,
            _now,
        ),
        _section(
            "meetings",
            meeting_groups,
            lambda group: _meeting_wording(group[0], day),
            case_of,
            case_scores,
            overdue_ids,
            today_events,
            _now,
            fixed_score=MEETINGS_SCORE,
        ),
        _section(
            "overdue",
            overdue_groups,
            _tasks_wording,
            case_of,
            case_scores,
            overdue_ids,
            today_events,
            _now,
        ),
    ]
    if not any(section.entries for section in sections):
        warnings.append(NOTHING_TO_REPORT)

    cited = {item_id for s in sections for e in s.entries for item_id in e.source_item_ids}
    return Brief(
        brief_id=f"b_{day.isoformat()}_{user}",
        owner_user_id=user,
        date=day,
        generated_at=to_paris(now or datetime.now(PARIS)),
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


def _section(
    kind: SectionKind,
    groups: list[list[Item]],
    wording: Wording,
    case_of: dict[str, Case],
    case_scores: dict[str, tuple[int, list[str]]],
    overdue_ids: set[str],
    today_events: set[str],
    now: datetime | None,
    *,
    fixed_score: int | None = None,
) -> Section:
    entries = []
    for rank, group in enumerate(groups, start=1):
        text, fallback_reason = wording(group)
        case = case_of[group[0].item_id]

        score, score_reasons = case_scores.get(case.case_id, (fixed_score or 50, [STANDARD]))
        displayed_score = fixed_score if fixed_score is not None else score

        # Raisons du score, sans STANDARD (décision 3).
        filtered_reasons = [r for r in score_reasons if r != STANDARD]

        shown = {item.item_id for item in group}
        others = [item for item in case.items if item.item_id not in shown]
        context = _context(case, others, now, today_events) if others else None

        # Raisons finales (max 4) : jusqu'à 3 raisons du score, puis la raison de contexte.
        # Repli sur la raison de rédaction si le score est neutre (STANDARD filtré).
        if filtered_reasons:
            reasons: list[str] = filtered_reasons[:3]
            if context:
                reasons = reasons[:3] + [context]
        else:
            reasons = [context] if context else [fallback_reason]

        # Tous les items de l'affaire sont cités, sans doublon (contrat B, décision 6).
        # Les réunions annulées sont incluses (décision 5).
        cancelled = _cancelled_in_case(case, today_events)
        all_items = list(dict.fromkeys(item.item_id for item in group + others + cancelled))

        entries.append(
            Entry(
                rank=rank,
                priority_score=displayed_score,
                priority_reasons=reasons,
                text=text,
                source_item_ids=all_items,
            )
        )
    return Section(kind=kind, entries=entries)


def _cancelled_in_case(case: Case, today_events: set[str]) -> list[Item]:
    """Réunions annulées de l'affaire, non comptées dans today_events."""
    return [
        item
        for item in case.items
        if item.source == "calendar" and _is_cancelled(item) and item.item_id not in today_events
    ]


def _is_cancelled(event: Item) -> bool:
    calendar = event.attributes.calendar if event.attributes else None
    return calendar is not None and calendar.is_cancelled is True


def _context(
    case: Case,
    others: list[Item],
    now: datetime | None,
    today_events: set[str],
) -> str | None:
    """« dossier 2026-3025 : aussi 1 mail, 1 tâche en retard, 1 réunion annulée »."""
    counts: Counter[str] = Counter()
    for item in others:
        if item.source == "calendar":
            if _is_cancelled(item) and item.item_id not in today_events:
                counts["calendar_cancelled"] += 1
            elif item.item_id in today_events:
                counts["calendar"] += 1
            # réunions annulées déjà dans le groupe (today_events) ignorées ici
        elif item.source == "task":
            key = "task_overdue" if (now and item.is_overdue(now)) else "task"
            counts[key] += 1
        else:
            counts[item.source] += 1

    if not counts:
        return None

    parts = []
    if counts["mail"]:
        n = counts["mail"]
        parts.append(f"{n} {NOUNS['mail'][n > 1]}")
    if counts["calendar"]:
        n = counts["calendar"]
        parts.append(f"{n} {NOUNS['calendar'][n > 1]}")
    if counts["calendar_cancelled"]:
        n = counts["calendar_cancelled"]
        parts.append(f"{n} réunion{'s' if n > 1 else ''} annulée{'s' if n > 1 else ''}")
    if counts["task_overdue"]:
        n = counts["task_overdue"]
        parts.append(f"{n} tâche{'s' if n > 1 else ''} en retard")
    if counts["task"]:
        n = counts["task"]
        parts.append(f"{n} {NOUNS['task'][n > 1]}")
    if counts["file"]:
        n = counts["file"]
        parts.append(f"{n} {NOUNS['file'][n > 1]}")

    return f"{_case_name(case)} : aussi {', '.join(parts)}"


def _case_name(case: Case) -> str:
    """Le nom le plus parlant : une référence (« dossier 2026-3025 »), sinon le type de lien.

    Les identifiants de fil et de réunion sont techniques : on ne les affiche pas.
    """
    kinds = {link.partition(":")[0] for link in case.links}
    for link in case.links:
        kind, _, reference = link.partition(":")
        if kind not in ("fil", "réunion"):
            return f"{kind} {reference}"
    return "même réunion" if "réunion" in kinds else "même fil"


def _title(item: Item) -> str:
    return item.title.strip() or NO_TITLE


def _mails_wording(mails: list[Item]) -> tuple[str, str]:
    """`mails` du plus récent au plus ancien ; le plus récent donne le titre et la date."""
    received = to_paris(mails[0].created_at).strftime("%d/%m à %H:%M")
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
    all_day = is_all_day(event)
    # Une journée entière garde ses dates ; une réunion se lit en heure de Paris.
    start = _required(event.start_at, event)
    start = start if all_day else to_paris(start)
    if start.date() < day:
        when = f"en cours depuis le {start.strftime('%d/%m')}"
    elif all_day:
        when = "toute la journée"
    else:
        when = f"à {start.strftime('%H:%M')}"
    return f"Réunion {when} : {_title(event)}", "réunion aujourd'hui"


def _tasks_wording(tasks: list[Item]) -> tuple[str, str]:
    """`tasks` de l'échéance la plus ancienne à la plus récente ; la plus ancienne d'abord."""
    due = to_paris(_required(tasks[0].due_at, tasks[0])).strftime("%d/%m")
    if len(tasks) == 1:
        return f"Tâche en retard : {_title(tasks[0])}", f"échéance dépassée depuis le {due}"
    return (
        f"{len(tasks)} tâches en retard : {_title(tasks[0])}",
        f"échéances dépassées, la plus ancienne le {due}",
    )
