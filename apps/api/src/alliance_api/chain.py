"""Chaîne d'un brief pour l'administration (L3.7, étape B) : collecte → génération → consultation.

Sans changer le contrat C, les trois journaux sont rapprochés par les identifiants d'items :
- collecte : les événements du lot 1 jusqu'au jour du brief qui ont apporté au moins un item
  cité, plus ceux du jour du brief (heure de Paris), échecs et collectes vides compris ; un
  mail non lu est recollecté les jours suivants, ces collectes-là sont exclues ;
- génération : modèle, prompt et compteurs lus dans le brief lui-même (contrat B), et les
  générations réussies de l'API dont les items sont exactement ceux du brief ;
- consultation : les lectures de l'API qui couvrent les items du brief (« brief » si ce sont
  exactement les siens, « historique » si la liste des 30 jours les contenait).

Limites (sans identifiant de corrélation) : un brief qui ne cite aucun item n'a ni génération
ni consultation rattachable ; une génération en échec n'a pas de jour et n'apparaît pas.
Aucun texte du brief ni contenu de message : identifiants, compteurs et métadonnées.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date as Date
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from contracts.models import AuditEvent, Brief

PARIS = ZoneInfo("Europe/Paris")


class CollectStep(BaseModel):
    ts: datetime
    connector: str
    source: str | None
    count: int  # items collectés par cet événement
    cited: int  # dont cités par le brief
    status: Literal["ok", "error"]
    error_code: str | None = None


class Generation(BaseModel):
    brief_id: str
    generated_at: datetime
    llm: str
    llm_version: str | None
    prompt_version: str | None
    items_in: int
    items_out: int
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_eur: float
    warnings: int  # nombre d'avertissements (leur texte reste dans le brief)
    cited_items: int
    api_calls: list[datetime]  # générations journalisées par l'API


class Consultation(BaseModel):
    ts: datetime
    user_id: str
    kind: Literal["brief", "historique"]


class BriefChain(BaseModel):
    user_id: str
    date: Date
    collection: list[CollectStep]
    generation: Generation
    consultations: list[Consultation]


def brief_chain(
    brief: Brief, api_events: Iterable[AuditEvent], collect_events: Iterable[AuditEvent]
) -> BriefChain:
    cited = brief.cited_item_ids()
    api = list(api_events)
    return BriefChain(
        user_id=brief.owner_user_id,
        date=brief.date,
        collection=_collection(brief, cited, collect_events),
        generation=_generation(brief, cited, api),
        consultations=_consultations(cited, api),
    )


def _collection(brief: Brief, cited: set[str], events: Iterable[AuditEvent]) -> list[CollectStep]:
    steps = [
        CollectStep(
            ts=event.ts,
            connector=event.actor.removeprefix("connector:"),
            source=event.source,
            count=event.count,
            cited=len(cited.intersection(event.item_ids)),
            status=event.status,
            error_code=event.error_code,
        )
        for event in events
        if event.action == "collect"
        and event.user_id == brief.owner_user_id
        and _day(event) <= brief.date  # une collecte postérieure n'a pas pu servir au brief
        and (cited.intersection(event.item_ids) or _day(event) == brief.date)
    ]
    return sorted(steps, key=lambda step: step.ts)


def _day(event: AuditEvent) -> Date:
    return event.ts.astimezone(PARIS).date()


def _generation(brief: Brief, cited: set[str], events: list[AuditEvent]) -> Generation:
    calls = [
        event.ts
        for event in events
        if event.action == "generate"
        and event.status == "ok"
        and cited
        and set(event.item_ids) == cited
    ]
    return Generation(
        brief_id=brief.brief_id,
        generated_at=brief.generated_at,
        llm=brief.generator.llm,
        llm_version=brief.generator.llm_version,
        prompt_version=brief.generator.prompt_version,
        **brief.stats.model_dump(),
        warnings=len(brief.warnings),
        cited_items=len(cited),
        api_calls=sorted(calls),
    )


def _consultations(cited: set[str], events: list[AuditEvent]) -> list[Consultation]:
    found = []
    for event in events:
        ids = set(event.item_ids)
        if event.action != "view" or event.status != "ok" or not cited or not cited <= ids:
            continue
        kind: Literal["brief", "historique"] = "brief" if ids == cited else "historique"
        found.append(Consultation(ts=event.ts, user_id=event.user_id, kind=kind))
    return sorted(found, key=lambda c: c.ts)
