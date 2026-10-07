"""Brief à base de règles, sans LLM (stub L2.0, guide §2.3).

Construit un Brief valide (contrat B) à partir des Item d'un utilisateur : les mails non
lus les plus récents, les réunions du jour et les tâches en retard. Les textes ne
reprennent que les titres et les dates, jamais le contenu (snippet) : un mail piégé
ne peut donc rien injecter dans le brief.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import date as Date
from datetime import datetime
from datetime import time as Time

from contracts.models import Brief, Entry, Generator, Item, Section, Stats
from contracts.models.brief import SectionKind
from packages.analyse.aggregate.rules import meetings_of_day, overdue_tasks, unread_mails

# Score fixe par section : le vrai barème arrive avec L2.2 (score de priorité expliqué).
SCORES: dict[SectionKind, int] = {"overdue": 70, "actions": 60, "meetings": 50}
NO_TITLE = "(sans objet)"
NOTHING_TO_REPORT = "Aucun élément à signaler pour cette journée."

# Une règle de rédaction renvoie (texte de l'entrée, raison de priorité).
Wording = Callable[[Item], tuple[str, str]]


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

    sections = [
        _section("actions", unread_mails(mine), _mail_wording),
        _section("meetings", meetings_of_day(mine, day), lambda e: _meeting_wording(e, day)),
        _section("overdue", overdue, _task_wording),
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


def _section(kind: SectionKind, selected: list[Item], wording: Wording) -> Section:
    entries = []
    for rank, item in enumerate(selected, start=1):
        text, reason = wording(item)
        entries.append(
            Entry(
                rank=rank,
                priority_score=SCORES[kind],
                priority_reasons=[reason],
                text=text,
                source_item_ids=[item.item_id],
            )
        )
    return Section(kind=kind, entries=entries)


def _title(item: Item) -> str:
    return item.title.strip() or NO_TITLE


def _mail_wording(mail: Item) -> tuple[str, str]:
    received = mail.created_at.strftime("%d/%m à %H:%M")
    return f"Mail non lu : {_title(mail)}", f"non lu, reçu le {received}"


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


def _task_wording(task: Item) -> tuple[str, str]:
    due = _required(task.due_at, task).strftime("%d/%m")
    return f"Tâche en retard : {_title(task)}", f"échéance dépassée depuis le {due}"
