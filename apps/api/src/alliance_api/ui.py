"""Interface de consultation du brief (L3.3, ADR-0001) : pages HTML rendues par Jinja.

Deux règles tenues ici, quel que soit le générateur :
- aucun texte du brief n'est affiché sans son lien source : une entrée dont un item cité
  est introuvable (ou appartient à un autre utilisateur) est masquée et signalée ;
- tout texte est échappé par Jinja (autoescape) : un `<script>` dans un mail s'affiche
  comme du texte, il ne s'exécute pas.

Organisation, pour modifier l'interface sans toucher au reste :
- textes et icônes des sections : `SECTIONS` et `SOURCES` ci-dessous ;
- mise en page : `templates/` (brief.html = page, _entry.html = une entrée et son détail) ;
- apparence et animations : `static/app.css` et `static/app.js`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timedelta
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
from contracts.models import Brief, Entry, Item

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
STATIC_DIR = Path(__file__).parent / "static"

PARIS = ZoneInfo("Europe/Paris")
SESSION_SECONDS = 8 * 3600
# « L'essentiel du jour » : 3 à 7 éléments classés (guide d'annotation v0.3, § But).
# Au-delà, les entrées restent consultables dans un bloc replié, jamais supprimées.
ESSENTIEL_MAX = 7


@dataclass(frozen=True)
class SectionStyle:
    label: str  # titre de la carte
    subtitle: str  # précision affichée à côté du titre
    icon: str  # nom d'icône (templates/_icons.html)
    empty: str  # message quand la section est vide


SECTIONS = {
    "actions": SectionStyle("À traiter", "actions à mener", "circle-check", "Rien à traiter."),
    "meetings": SectionStyle("Réunions", "aujourd'hui", "calendar", "Aucune réunion aujourd'hui."),
    "overdue": SectionStyle("Retards", "tâches en souffrance", "alarm", "Aucune tâche en retard."),
    "files": SectionStyle("Fichiers", "créés ou modifiés", "file", "Aucun fichier récent."),
}
SOURCES = {
    "mail": ("Messagerie", "mail", "mails reçus"),
    "calendar": ("Agenda", "calendar", "événements"),
    "task": ("Tâches", "check", "tâches suivies"),
    "file": ("Documents", "file", "fichiers"),
}
LABELS = {kind: style.label for kind, style in SECTIONS.items()}

DAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
MONTHS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
          "septembre", "octobre", "novembre", "décembre")  # fmt: skip


@dataclass(frozen=True)
class Source:
    url: str
    title: str
    snippet: str
    kind: str = "mail"
    sender: str | None = None
    recipients: tuple[str, ...] = ()
    when: str | None = None
    hour: str | None = None
    start_day: Date | None = None
    due: str | None = None
    unread: bool = False
    important: bool = False


@dataclass(frozen=True)
class EntryView:
    entry: Entry
    sources: list[Source]
    kind: str = "actions"
    day: Date | None = None  # jour du brief, pour situer une réunion d'un autre jour

    @property
    def uid(self) -> str:
        return f"{self.kind}-{self.entry.rank}"

    @property
    def main(self) -> Source:
        return self.sources[0]

    @property
    def headline(self) -> str:
        return self.main.title or self.entry.text

    @property
    def meta(self) -> str:
        """Nature et repère de l'élément principal : « Réunion à 09:00 », « Mail de … »."""
        s = self.main
        if s.kind == "calendar":
            if s.start_day and self.day and s.start_day > self.day:
                when = (
                    "demain"
                    if s.start_day == self.day + timedelta(days=1)
                    else f"le {long_date(s.start_day)}"
                )
                return f"Réunion {when} à {s.hour}"
            return "Réunion sur la journée" if s.hour == "Journée" else f"Réunion à {s.hour}"
        if s.kind == "task":
            return f"Tâche, échéance {s.due}" if s.due else "Tâche"
        if s.kind == "mail":
            return f"Mail de {s.sender}" if s.sender else "Mail"
        return "Fichier"

    @property
    def level(self) -> str:
        score = self.entry.priority_score
        return "haute" if score >= 70 else "moyenne" if score >= 50 else "basse"


@dataclass(frozen=True)
class SectionView:
    label: str
    entries: list[EntryView]
    kind: str = "actions"

    @property
    def style(self) -> SectionStyle:
        return SECTIONS.get(self.kind, SectionStyle(self.label, "", "file", "Rien ici."))


@dataclass(frozen=True)
class Slot:
    """Un événement de l'agenda du jour ; `uid` renvoie à son entrée s'il est dans le brief."""

    hour: str
    title: str
    place: str
    url: str
    uid: str | None


@dataclass(frozen=True)
class Feed:
    """Une source connectée et ce qu'elle a apporté ce matin."""

    label: str
    icon: str
    unit: str
    count: int


def _today() -> Date:
    return datetime.now(PARIS).date()


def long_date(day: Date) -> str:
    return f"{DAYS[day.weekday()]} {day.day} {MONTHS[day.month - 1]}"


def _hour(moment: datetime) -> str:
    return moment.astimezone(PARIS).strftime("%H:%M")


def _when(item: Item) -> str | None:
    """Moment utile de l'item, en clair (heure d'une réunion, date d'un mail…)."""
    start = item.start_at.astimezone(PARIS) if item.start_at else None
    end = item.end_at.astimezone(PARIS) if item.end_at else None
    if start and end:
        last_day = (end - timedelta(seconds=1)).date()
        if last_day > start.date():
            return f"Du {long_date(start.date())} au {long_date(last_day)}"
        if end - start >= timedelta(hours=23):
            return f"Toute la journée, {long_date(start.date())}"
        return f"{long_date(start.date()).capitalize()}, {_hour(start)} – {_hour(end)}"
    moment = item.created_at.astimezone(PARIS)
    return f"{long_date(moment.date()).capitalize()} à {_hour(moment)}"


def _source(item: Item) -> Source:
    def names(*roles: str) -> list[str]:
        return [p.name or p.email for p in item.participants if p.role in roles]

    sender = (names("from") or names("organizer") or [None])[0]
    start, end = item.start_at, item.end_at
    all_day = bool(start and end and end - start >= timedelta(hours=23))
    return Source(
        url=str(item.url),
        title=item.title,
        snippet=item.snippet,
        kind=item.source,
        sender=sender,
        recipients=tuple(names("to", "cc", "attendee", "assignee")),
        when=_when(item),
        hour="Journée" if all_day else _hour(start) if start else None,
        start_day=start.astimezone(PARIS).date() if start else None,
        due=f"{long_date(item.due_at.astimezone(PARIS).date())}" if item.due_at else None,
        unread=item.is_read is False,
        important=item.native_importance == "high",
    )


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
            sources = [_source(i) for i in items if i is not None]
            entries.append(EntryView(entry, sources, section.kind, brief.date))
        sections.append(SectionView(LABELS.get(section.kind, section.kind), entries, section.kind))
    return sections, hidden


def _ranked(sections: list[SectionView]) -> list[EntryView]:
    """Toutes les entrées, de la plus importante à la moins importante, toutes sections
    confondues (à score égal : ordre des sections, puis rang dans la section)."""
    order = list(SECTIONS)
    entries = [view for section in sections for view in section.entries]
    return sorted(
        entries,
        key=lambda v: (-v.entry.priority_score, order.index(v.kind) if v.kind in order else 99,
                       v.entry.rank),
    )  # fmt: skip


def _agenda(items: list[Item], day: Date, sections: list[SectionView]) -> list[Slot]:
    """Les événements non annulés qui occupent `day`, journées entières d'abord."""
    in_brief = {i: v.uid for s in sections for v in s.entries for i in v.entry.source_item_ids}
    slots = []
    for item in items:
        calendar = item.attributes.calendar if item.attributes else None
        if item.source != "calendar" or not item.start_at or not item.end_at:
            continue
        if calendar is not None and calendar.is_cancelled:
            continue
        start, end = item.start_at.astimezone(PARIS), item.end_at.astimezone(PARIS)
        if not start.date() <= day <= (end - timedelta(seconds=1)).date():
            continue
        whole = start.date() < day or end - start >= timedelta(hours=23)
        slot = Slot(
            "Journée" if whole else _hour(start), item.title, item.snippet, str(item.url),
            in_brief.get(item.item_id),
        )  # fmt: skip
        slots.append(((not whole, start), slot))
    return [slot for _, slot in sorted(slots, key=lambda pair: pair[0])]


def _feeds(items: list[Item]) -> list[Feed]:
    counts = Counter(item.source for item in items)
    return [Feed(label, icon, unit, counts[kind]) for kind, (label, icon, unit) in SOURCES.items()]


def _display_name(items: list[Item], user: str) -> str:
    """Prénom du manager, déduit de ses items (destinataire le plus fréquent)."""
    names = Counter(
        p.name for i in items for p in i.participants if p.role in ("to", "attendee", "assignee")
    )
    return names.most_common(1)[0][0].split()[0] if names else user


def _page(request: Request, user: str, day: Date, error: ApiError | None = None) -> Response:
    state = request.app.state
    brief, sections, hidden = None, [], 0
    try:
        brief = view(state, user, day)
        sections, hidden = _sections(state, user, brief)
    except ApiError as exc:
        if exc.code != "brief_not_found":
            raise
    collected_items = state.items.items_for_day(user, day)
    items = collected_items or []
    collected = max((i.collected_at for i in items), default=None)
    ranked = _ranked(sections)
    context = {
        "user": user,
        "name": _display_name(items, user),
        "day": day,
        "day_label": long_date(day),
        "prev_day": day - timedelta(days=1),
        "next_day": day + timedelta(days=1),
        "is_today": day == _today(),
        "greeting": "Bonsoir" if datetime.now(PARIS).hour >= 18 else "Bonjour",
        "brief": brief,
        "sections": sections,
        "by_kind": {s.kind: s for s in sections},
        "ranked": ranked[:ESSENTIEL_MAX],
        "others": ranked[ESSENTIEL_MAX:],
        "retained": len({i for v in ranked[:ESSENTIEL_MAX] for i in v.entry.source_item_ids}),
        "agenda": _agenda(items, day, sections),
        "feeds": _feeds(items),
        "collected": _hour(collected) if collected else None,
        "has_items": collected_items is not None,
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
