"""Interface de consultation du brief (L3.3, ADR-0001) : pages HTML rendues par Jinja.

Deux règles tenues ici, quel que soit le générateur :
- aucun texte du brief n'est affiché sans son lien source : une entrée dont un item cité
  est introuvable (ou appartient à un autre utilisateur) est masquée et signalée ;
- tout texte est échappé par Jinja (autoescape) : un `<script>` dans un mail s'affiche
  comme du texte, il ne s'exécute pas.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.datastructures import State

from alliance_api.auth import DEMO_COOKIE, optional_user, valid_user_id
from alliance_api.briefs import generate, view
from alliance_api.errors import ApiError
from contracts.models import Brief, Entry

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

PARIS = ZoneInfo("Europe/Paris")
SESSION_SECONDS = 8 * 3600
LABELS = {"actions": "À traiter", "meetings": "Réunions", "overdue": "Retards", "files": "Fichiers"}


@dataclass(frozen=True)
class Source:
    url: str
    title: str
    snippet: str


@dataclass(frozen=True)
class EntryView:
    entry: Entry
    sources: list[Source]


@dataclass(frozen=True)
class SectionView:
    label: str
    entries: list[EntryView]


def _today() -> Date:
    return datetime.now(PARIS).date()


def _sections(state: State, user: str, brief: Brief) -> tuple[list[SectionView], int]:
    """Sections prêtes à afficher, et le nombre d'entrées masquées faute de source."""
    hidden, sections = 0, []
    for section in brief.sections:
        entries = []
        for entry in section.entries:
            items = [state.items.find(item_id) for item_id in entry.source_item_ids]
            if any(item is None or item.owner_user_id != user for item in items):
                hidden += 1
                continue
            sources = [Source(str(i.url), i.title, i.snippet) for i in items if i is not None]
            entries.append(EntryView(entry, sources))
        sections.append(SectionView(LABELS.get(section.kind, section.kind), entries))
    return sections, hidden


def _page(request: Request, user: str, day: Date, error: ApiError | None = None) -> Response:
    state = request.app.state
    brief, sections, hidden = None, [], 0
    try:
        brief = view(state, user, day)
        sections, hidden = _sections(state, user, brief)
    except ApiError as exc:
        if exc.code != "brief_not_found":
            raise
    context = {
        "user": user,
        "day": day,
        "brief": brief,
        "sections": sections,
        "hidden": hidden,
        "error": error,
    }
    status = error.status if error else 200
    return templates.TemplateResponse(request, "brief.html", context, status_code=status)


@router.get("/", response_class=HTMLResponse)
def home(request: Request, day: Annotated[Date | None, Query(alias="date")] = None) -> Response:
    user = optional_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return _page(request, user, day or _today())


@router.post("/generate", response_class=HTMLResponse)
def generate_page(request: Request, day: Annotated[Date, Form(alias="date")]) -> Response:
    user = optional_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    try:
        generate(request.app.state, user, day)
    except ApiError as exc:
        return _page(request, user, day, error=exc)
    return RedirectResponse(f"/?date={day.isoformat()}", status_code=303)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> Response:
    return templates.TemplateResponse(request, "login.html", {"error": False})


@router.post("/login", response_class=HTMLResponse)
def login(request: Request, user_id: Annotated[str, Form()]) -> Response:
    if request.app.state.auth_mode != "demo" or not valid_user_id(user_id):
        return templates.TemplateResponse(request, "login.html", {"error": True}, status_code=401)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        DEMO_COOKIE, user_id, max_age=SESSION_SECONDS, httponly=True, samesite="strict"
    )
    return response


@router.post("/logout")
def logout() -> Response:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(DEMO_COOKIE, httponly=True, samesite="strict")
    return response
