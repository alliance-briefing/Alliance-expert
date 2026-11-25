"""Administration (L3.7) : journal d'audit et chaîne d'un brief, réservés à `audit:read`.

`GET /admin/audit` réunit le journal de l'API (génération, consultation) et celui de la
collecte du lot 1 (lu seulement), filtrés par utilisateur, jour (Europe/Paris), action et
statut, du plus récent au plus ancien. Les événements du contrat C ne contiennent que des
identifiants et des compteurs : aucun contenu de message n'est exposé ici.

`GET /admin/chain?user=…&date=…` : collecte, génération et consultations d'un brief
(voir `chain.py`). L'administrateur ne lit pas le texte du brief (`brief:read_any`).
"""

from __future__ import annotations

import logging
from datetime import date as Date
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
from starlette.datastructures import State

from alliance_api.auth import CurrentUser, requires, valid_user_id
from alliance_api.chain import BriefChain, brief_chain
from alliance_api.errors import ApiError
from contracts.models import AuditEvent

PARIS = ZoneInfo("Europe/Paris")
LIMIT_MAX = 1000

logger = logging.getLogger("alliance_api.admin")
router = APIRouter(prefix="/admin", tags=["admin"])


class AuditPage(BaseModel):
    total: int  # événements correspondant aux filtres, avant la limite
    events: list[AuditEvent]


@router.get("/audit", response_model=AuditPage, dependencies=[requires("audit:read")])
def audit_log(
    admin: CurrentUser,
    request: Request,
    user: Annotated[str | None, Query()] = None,
    day: Annotated[Date | None, Query(alias="date")] = None,
    action: Annotated[Literal["collect", "generate", "view"] | None, Query()] = None,
    status: Annotated[Literal["ok", "error"] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=LIMIT_MAX)] = 200,
) -> AuditPage:
    events = filtered_events(request.app.state, user, day, action, status)
    # Le contrat C n'a pas d'action pour la lecture du journal : trace sur le logger.
    logger.info("journal d'audit lu par %s (%d événements)", admin, len(events))
    return AuditPage(total=len(events), events=events[:limit])


@router.get("/chain", response_model=BriefChain, dependencies=[requires("audit:read")])
def chain(
    admin: CurrentUser,
    request: Request,
    user: Annotated[str, Query()],
    day: Annotated[Date, Query(alias="date")],
) -> BriefChain:
    found = chain_for(request.app.state, user, day)
    logger.info("chaîne du brief %s %s lue par %s", user, day, admin)
    return found


def filtered_events(
    state: State, user: str | None, day: Date | None, action: str | None, status: str | None
) -> list[AuditEvent]:
    """Les deux journaux réunis et filtrés, du plus récent au plus ancien (API et page)."""
    _check_user(user)
    events = [
        event
        for event in [*state.audit.read(user), *state.collect_audit.read(user)]
        if (day is None or event.ts.astimezone(PARIS).date() == day)
        and (action is None or event.action == action)
        and (status is None or event.status == status)
    ]
    return sorted(events, key=lambda event: event.ts, reverse=True)


def chain_for(state: State, user: str, day: Date) -> BriefChain:
    """Chaîne du brief en cache de `user` pour `day` ; 404 s'il n'a pas été généré."""
    _check_user(user)
    brief = state.briefs.get(user, day)  # lecture directe : pas une consultation du manager
    if brief is None:
        raise ApiError(404, "brief_not_found", "brief non généré pour ce jour")
    return brief_chain(brief, state.audit.read(user), state.collect_audit.read(user))


def _check_user(user: str | None) -> None:
    if user is not None and not valid_user_id(user):
        raise ApiError(422, "invalid_request", "paramètres invalides : user")
