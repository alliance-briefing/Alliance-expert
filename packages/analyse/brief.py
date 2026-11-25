"""Brief à base de règles, sans LLM (stub L2.0, guide §2.3), organisé par affaires (L2.1).

Construit un Brief valide (contrat B) à partir des Item d'un utilisateur : les mails non
lus les plus récents, les réunions du jour et les tâches en retard. Les éléments d'une
même affaire sont réunis dans une entrée (sauf les réunions, une par heure), et chaque
entrée cite aussi les autres éléments de son affaire. Les textes ne reprennent que les
titres et les dates, jamais le contenu (snippet) : un mail piégé ne peut donc rien
injecter dans le brief. Le texte de chaque entrée est découpé en claims (L2.4) : un fait
par élément montré, cité par ce seul élément, pour que la garde puisse le vérifier.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date as Date
from datetime import datetime
from typing import Literal

from contracts.models import Brief, Claim, Entry, Generator, Item, Section, Stats
from contracts.models.brief import SECTION_ORDER, SectionKind
from packages.analyse.aggregate.cases import Case, group_into_cases
from packages.analyse.aggregate.rules import meetings_of_day, overdue_tasks, unread_mails
from packages.analyse.clock import PARIS, is_all_day, to_paris
from packages.analyse.guard.check import NO_TITLE, guard_brief
from packages.analyse.scoring.learned import (
    GENERATOR,
    CaseScore,
    LearnedScorer,
    learned_case_scores,
)
from packages.analyse.scoring.score import STANDARD, score_case

# Seuil de score à partir duquel une affaire sans mail non lu entre dans « À traiter ».
# 50 = score neutre : toute affaire avec au moins un signal net positif est incluse.
ACTIONS_THRESHOLD = 50
# Nombre maximal d'affaires dans la section « à traiter ».
ACTIONS_LIMIT = 7
# Sélection du brief appris (M9, plan §11) : seuil T et nombre maximal K d'entrées, réglés
# sur des journées simulées ; au moins SELECT_MIN entrées quand il y a de quoi.
SELECT_THRESHOLD = 50
SELECT_LIMIT = 7
SELECT_MIN = 3
# Action recommandée (M9) : verbe selon l'action que N prévoit, suivi du seul titre.
ACTION_VERBS = {
    "decider": "Décider ou valider",
    "faire": "Traiter",
    "repondre": "Répondre",
    "preparer": "Préparer",
    "relire": "Relire",
}
ALERT_ACTION = "Vérifier en priorité"
ALERT_REASON = "alerte (sécurité, incident)"
# Score fixe pour les réunions : triées par heure, leur score ne porte pas l'ordre.
MEETINGS_SCORE = 50
# Claims détaillés par entrée ; au-delà, les éléments restants forment un seul claim.
CLAIMS_LIMIT = 3
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
    items: Sequence[Item],
    user: str,
    date: Date | str,
    *,
    now: datetime | None = None,
    scorer: LearnedScorer | Literal["auto"] | None = "auto",
    select: tuple[int, int] = (SELECT_THRESHOLD, SELECT_LIMIT),
) -> Brief:
    """Brief du jour `date` pour `user`, construit par règles à partir de `items`.

    `now` fixe `generated_at` (utile aux tests) ; par défaut, l'heure courante.
    `scorer` : score appris des affaires ("auto" : selon `ALLIANCE_SCORE`, None : règles).
    `select` : seuil et nombre maximal d'entrées du brief appris (réglage de M9).
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
    # Score appris (M8) pour les affaires qui ont un élément à noter ; les autres
    # gardent le score par règles. Repli annoncé si le modèle manque.
    learned: dict[str, CaseScore] = {}
    learned_mode = False  # score appris actif (même sans élément à noter ce jour-là)
    if _now:
        scored, unavailable = learned_case_scores(cases, mine, user, _now, scorer)
        learned_mode = scored is not None
        learned = scored or {}
        if unavailable:
            warnings.append(unavailable)
    case_scores.update({case_id: (s.score, s.reasons) for case_id, s in learned.items()})

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

    if learned_mode:
        sections = _learned_sections(
            learned, case_of, case_scores, overdue_ids, overdue_by_case, unread_by_case,
            today_events, _now, select,
        )  # fmt: skip
    else:
        sections = _rule_sections(
            cases, day, mine, case_of, case_scores, overdue_ids, overdue_by_case,
            unread_by_case, unread_case_ids, meeting_case_ids, overdue_case_ids, today_events, _now,
        )  # fmt: skip
    if not any(section.entries for section in sections):
        warnings.append(NOTHING_TO_REPORT)

    cited = {item_id for s in sections for e in s.entries for item_id in e.source_item_ids}
    brief = Brief(
        brief_id=f"b_{day.isoformat()}_{user}",
        owner_user_id=user,
        date=day,
        generated_at=to_paris(now or datetime.now(PARIS)),
        # « rules » quand le score appris n'est pas utilisé (repli, ALLIANCE_SCORE).
        generator=(
            Generator(
                llm=GENERATOR,
                llm_version=next(iter(learned.values())).model if learned else None,
            )
            if learned_mode
            else Generator(llm="rules")
        ),
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
    # Porte dure (L2.4) : aucun claim livré sans source vérifiée dans les items de l'utilisateur.
    return guard_brief(brief, mine)


def _learned_sections(
    learned: dict[str, CaseScore],
    case_of: dict[str, Case],
    case_scores: dict[str, tuple[int, list[str]]],
    overdue_ids: set[str],
    overdue_by_case: dict[str, list[Item]],
    unread_by_case: dict[str, list[Item]],
    today_events: set[str],
    now: datetime | None,
    select: tuple[int, int] = (SELECT_THRESHOLD, SELECT_LIMIT),
) -> list[Section]:
    """Le brief appris (M9) : les affaires notées par M8, alertes en tête puis par score ; une
    affaire entre si son score atteint SELECT_THRESHOLD, avec au moins SELECT_MIN et au plus
    SELECT_LIMIT entrées. La section suit l'élément qui porte le score."""
    ranked = sorted(learned.items(), key=lambda kv: (not kv[1].alert, -kv[1].score, kv[0]))
    chosen = [
        case_id
        for n, (case_id, s) in enumerate(ranked)
        if s.alert or s.score >= select[0] or n < SELECT_MIN
    ][: select[1]]
    by_id = {item.item_id: item for case in case_of.values() for item in case.items}
    # Une alerte tenue en tête prend la priorité maximale : le contrat B veut des scores qui
    # décroissent avec le rang, et c'est bien la priorité qu'on lui donne.
    case_scores = dict(case_scores)
    for case_id in chosen:
        if learned[case_id].alert:
            reasons = case_scores[case_id][1]
            case_scores[case_id] = (100, [ALERT_REASON, *[r for r in reasons if r != ALERT_REASON]])
    groups: dict[SectionKind, list[list[Item]]] = {k: [] for k in SECTION_ORDER}
    recommended: dict[str, str | None] = {}
    for case_id in chosen:
        best = by_id[learned[case_id].best_item_id]
        if best.source == "calendar" and best.item_id in today_events:
            kind, group = "meetings", [best]
        elif best.item_id in overdue_ids:
            kind, group = "overdue", overdue_by_case[case_id]
        elif best.source == "file":
            kind, group = "files", [best]
        else:
            kind, group = "actions", unread_by_case.get(case_id) or [best]
        groups[kind].append(group)
        recommended[case_id] = _recommended(learned[case_id], best)
    wordings: dict[SectionKind, Wording] = {
        "actions": _mails_wording,
        "meetings": lambda group: (
            _meeting_wording(group[0], to_paris(group[0].start_at).date())
            if group[0].start_at
            else _mails_wording(group)
        ),
        "overdue": _tasks_wording,
        "files": _mails_wording,
    }
    return [
        _section(
            kind,
            groups[kind],
            wordings[kind],
            case_of,
            case_scores,
            overdue_ids,
            today_events,
            now,
            recommended=recommended,
        )
        for kind in SECTION_ORDER
    ]


def _recommended(score: CaseScore, item: Item) -> str | None:
    """L'action recommandée, par gabarit : un verbe selon l'action prévue par N et le seul
    titre de l'élément (jamais son contenu : rien à injecter)."""
    if score.alert:
        return f"{ALERT_ACTION} : {_title(item)}"
    if item.source == "task" and score.action in ACTION_VERBS:
        return _title(item)  # l'intitulé d'une tâche est déjà une action (« Appeler… »)
    verb = ACTION_VERBS.get(score.action)
    return f"{verb} : {_title(item)}" if verb else None


def _rule_sections(
    cases: list[Case],
    day: Date,
    mine: list[Item],
    case_of: dict[str, Case],
    case_scores: dict[str, tuple[int, list[str]]],
    overdue_ids: set[str],
    overdue_by_case: dict[str, list[Item]],
    unread_by_case: dict[str, list[Item]],
    unread_case_ids: set[str],
    meeting_case_ids: set[str],
    overdue_case_ids: set[str],
    today_events: set[str],
    _now: datetime | None,
) -> list[Section]:
    """Le brief par règles d'avant M9 (L2.0, L2.1), gardé tel quel : c'est la référence de la
    comparaison de M9 et le repli quand le score appris manque."""
    learned: dict[str, CaseScore] = {}
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
    # Sans mail non lu : le mail qui porte le score appris, sinon le premier élément.
    actions_groups: list[list[Item]] = [
        unread_by_case.get(c.case_id) or [_best_item(c, learned)] for c in actions_cases
    ]

    return [
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
    recommended: dict[str, str | None] | None = None,
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
                claims=_claims(group, now),
                recommended_action=(recommended or {}).get(case.case_id),
                source_item_ids=all_items,
            )
        )
    return Section(kind=kind, entries=entries)


def _claims(group: list[Item], now: datetime | None) -> list[Claim]:
    """Le texte de l'entrée découpé en faits : un claim par élément montré, cité par lui seul.

    Les éléments de contexte (autres éléments de l'affaire) n'y sont pas : ils sont résumés
    dans les raisons. Au-delà de CLAIMS_LIMIT, un seul claim compte les éléments restants
    et les cite tous : l'affirmation reste vérifiable sans allonger l'entrée.
    """
    claims = [
        Claim(text=_fact(item, now), source_item_ids=[item.item_id])
        for item in group[:CLAIMS_LIMIT]
    ]
    rest = group[CLAIMS_LIMIT:]
    if rest:
        n, first = len(rest), rest[0]
        plural = "s" if n > 1 else ""
        state = _state(first, now)
        text = f"Et {n} autre{plural} {NOUNS[first.source][n > 1]}"
        if state:
            text += f" {state}{plural}"
        claims.append(Claim(text=text, source_item_ids=[item.item_id for item in rest]))
    return claims


def _state(item: Item, now: datetime | None) -> str:
    """État qui justifie la présence d'un élément dans le brief, au singulier."""
    if item.source == "mail" and item.is_read is False:
        return "non lu"
    if now is not None and item.is_overdue(now):
        return "en retard"
    return ""


def _fact(item: Item, now: datetime | None) -> str:
    """Un fait vérifiable sur un élément : type, titre entre « » et date en heure de Paris."""
    title = f"« {_title(item)} »"
    if item.source == "mail":
        unread = " non lu" if item.is_read is False else ""
        return f"Mail{unread} {title}, reçu le {_when(item.created_at)}"
    if item.source == "calendar":
        start = _required(item.start_at, item)
        when = f"le {start:%d/%m}, journée entière" if is_all_day(item) else f"le {_when(start)}"
        cancelled = ", annulée" if _is_cancelled(item) else ""
        return f"Réunion {title} {when}{cancelled}"
    if item.source == "task":
        if item.due_at is None:
            return f"Tâche {title}, sans échéance"
        late = "dépassée " if _state(item, now) == "en retard" else ""
        return f"Tâche {title}, échéance {late}le {to_paris(item.due_at):%d/%m}"
    return f"Fichier {title}, modifié le {to_paris(item.updated_at):%d/%m}"


def _when(value: datetime) -> str:
    return to_paris(value).strftime("%d/%m à %H:%M")


def _best_item(case: Case, learned: dict[str, CaseScore]) -> Item:
    if case.case_id in learned:
        best = learned[case.case_id].best_item_id
        return next(item for item in case.items if item.item_id == best)
    return case.items[0]


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
    """`mails` du plus récent au plus ancien ; le plus récent donne le titre et la date.

    Une affaire sans mail non lu montre son mail le mieux noté, ou son premier élément :
    le texte ne dit « non lu » que s'il l'est."""
    first = mails[0]
    received = _when(first.created_at)
    if first.source != "mail":
        noun = NOUNS[first.source][0]
        updated = to_paris(first.updated_at).strftime("%d/%m")
        return f"{noun.capitalize()} : {_title(first)}", f"dernière mise à jour le {updated}"
    if first.is_read is not False:
        return f"Mail : {_title(first)}", f"reçu le {received}"
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
