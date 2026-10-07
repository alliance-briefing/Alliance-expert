"""POST /briefs et GET /briefs/{date} : génère puis sert le brief du jour (contrat B).

L'API ne connaît du générateur que sa signature et son contrat de sortie : elle ne regarde
ni `generator.llm`, ni le nombre de sections. Le passage au LLM (v0.2.0) ne change rien ici.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date as Date
from datetime import datetime
from typing import Literal, Protocol

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
from starlette.datastructures import State

from alliance_api.audit import api_event
from alliance_api.auth import CurrentUser
from alliance_api.errors import ApiError
from contracts.models import Brief, Item

logger = logging.getLogger("alliance_api")
router = APIRouter()


class BriefGenerator(Protocol):
    def __call__(
        self, items: Sequence[Item], user: str, date: Date | str, *, now: datetime | None = None
    ) -> Brief: ...


class BriefStore(Protocol):
    def get(self, user: str, day: Date) -> Brief | None: ...
    def put(self, brief: Brief) -> None: ...


class MemoryBriefStore:
    """Cache du brief par (utilisateur, jour), dans le processus : perdu au redémarrage,
    propre à chaque worker. Suffisant pour le squelette ; Postgres ensuite (ADR)."""

    def __init__(self) -> None:
        self._briefs: dict[tuple[str, Date], Brief] = {}

    def get(self, user: str, day: Date) -> Brief | None:
        return self._briefs.get((user, day))

    def put(self, brief: Brief) -> None:
        self._briefs[(brief.owner_user_id, brief.date)] = brief


class BriefRequest(BaseModel):
    date: Date


class BriefStatus(BaseModel):
    """État de la génération. Aujourd'hui synchrone (« done ») ; quand la génération LLM
    passera en tâche de fond, le même format renverra « pending » puis « done »."""

    status: Literal["done"]
    brief_id: str
    date: Date


@router.post("/briefs", status_code=201, response_model=BriefStatus)
def create_brief(
    body: BriefRequest, user: CurrentUser, request: Request, response: Response
) -> BriefStatus:
    brief = generate(request.app.state, user, body.date)
    response.headers["Location"] = f"/briefs/{body.date.isoformat()}"
    return BriefStatus(status="done", brief_id=brief.brief_id, date=brief.date)


@router.get("/briefs/{day}", response_model=Brief)
def read_brief(day: Date, user: CurrentUser, request: Request) -> Brief:
    return view(request.app.state, user, day)


def generate(state: State, user: str, day: Date) -> Brief:
    """Génère, vérifie, met en cache et journalise le brief (partagé par l'API et l'interface)."""
    items = state.items.items_for_day(user, day)
    if items is None:
        # Pas de collecte ce jour-là : un brief « rien à signaler » serait faux.
        state.audit.write(api_event(user, "generate", [], 0, "no_items"))
        raise ApiError(404, "no_items", "aucune collecte pour ce jour")
    try:
        brief = state.generator(items, user, day)
        _check(brief, user, day, items)
    except Exception:
        logger.exception("génération en échec (user=%s, date=%s)", user, day)
        state.audit.write(api_event(user, "generate", [], 0, "generation_failed"))
        raise ApiError(500, "generation_failed", "la génération du brief a échoué") from None
    state.briefs.put(brief)
    cited = sorted(brief.cited_item_ids())
    state.audit.write(api_event(user, "generate", cited, len(cited)))
    return brief


def view(state: State, user: str, day: Date) -> Brief:
    """Brief en cache de `user` pour `day`, journalisé comme consultation."""
    brief = state.briefs.get(user, day)
    if brief is None:
        state.audit.write(api_event(user, "view", [], 0, "brief_not_found"))
        raise ApiError(404, "brief_not_found", "brief non généré pour ce jour")
    cited = sorted(brief.cited_item_ids())
    state.audit.write(api_event(user, "view", cited, len(cited)))
    return brief


def _check(brief: Brief, user: str, day: Date, items: Sequence[Item]) -> None:
    """Garde côté API, quel que soit le générateur : le brief est bien celui demandé et
    chaque item cité fait partie des items de l'utilisateur, donc a un lien source."""
    if (brief.owner_user_id, brief.date) != (user, day):
        raise ValueError("brief d'un autre utilisateur ou d'un autre jour")
    unknown = brief.cited_item_ids() - {item.item_id for item in items}
    if unknown:
        raise ValueError(f"{len(unknown)} items cités absents des items de l'utilisateur")
