"""Administration (L3.7) : journal d'audit filtrable, réservé à la permission `audit:read`.

`GET /admin/audit` réunit le journal de l'API (génération, consultation) et celui de la
collecte du lot 1 (lu seulement), filtrés par utilisateur, jour (Europe/Paris), action et
statut, du plus récent au plus ancien. Les événements du contrat C ne contiennent que des
identifiants et des compteurs : aucun contenu de message n'est exposé ici.
"""

from __future__ import annotations

import logging
from datetime import date as Date
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from alliance_api.auth import CurrentUser, requires, valid_user_id
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
    if user is not None and not valid_user_id(user):
        raise ApiError(422, "invalid_request", "paramètres invalides : user")
    state = request.app.state
    events = [
        event
        for event in [*state.audit.read(user), *state.collect_audit.read(user)]
        if (day is None or event.ts.astimezone(PARIS).date() == day)
        and (action is None or event.action == action)
        and (status is None or event.status == status)
    ]
    events.sort(key=lambda event: event.ts, reverse=True)
    # Le contrat C n'a pas d'action pour la lecture du journal : trace sur le logger.
    logger.info("journal d'audit lu par %s (%d événements)", admin, len(events))
    return AuditPage(total=len(events), events=events[:limit])
