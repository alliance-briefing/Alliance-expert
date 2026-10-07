"""Photographie du monde au moment de la collecte du matin, avec les fenêtres du guide.

- Mails (L1.3)     : reçus dans les 24 dernières heures + mails des 7 derniers jours
                     adressés directement au manager (champ « À ») et sans réponse de sa part.
                     Les expéditeurs automatiques (noreply, newsletters) ne sont pas reportés.
                     Plafond : 500.
- Calendrier (L1.4): événements des 48 prochaines heures + événement en cours.
- Tâches (L1.4)    : en retard (plafond 50, les plus récentes) + dues dans les 24 h
                     + tâches sans échéance créées dans les dernières 24 h.
- Fichiers (L1.5)  : créés ou modifiés dans les 48 dernières heures, plafond 100,
                     jamais un fichier supprimé.
Seuls les objets qui existaient déjà à l'heure de la collecte sont visibles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from urllib.parse import quote

from collect.core.text import make_snippet, truncate
from collect.synth.clock import collect_time, to_paris
from collect.synth.model import SimEvent, SimFile, SimFileVersion, SimMail, SimTask
from collect.synth.people import INTERNAL_DOMAIN, Person, slug
from contracts.models import Item

CONNECTOR_NAME = "demo"
CONNECTOR_VERSION = "0.1.0"
MAIL_CAP, TASK_OVERDUE_CAP, FILE_CAP = 500, 50, 100
SNIPPET_FILE_TYPES = {"txt", "docx", "pdf"}


@dataclass
class Snapshot:
    day: date
    now: datetime
    mails: list[SimMail]
    events: list[SimEvent]
    tasks: list[SimTask]
    files: list[tuple[SimFile, SimFileVersion]]
    deleted_files: list[SimFile] = field(default_factory=list)
    overdue_dropped: int = 0

    def counts(self) -> dict[str, int]:
        return {"mail": len(self.mails), "calendar": len(self.events), "task": len(self.tasks), "file": len(self.files)}


def mail_in_window(m: SimMail, now: datetime, me: Person) -> bool:
    if not (now - timedelta(days=7) < m.sent_at <= now):
        return False
    if m.sent_at > now - timedelta(hours=24):
        return True
    unanswered = m.answered_at is None or m.answered_at > now
    return me in m.to and not m.sender.automated and unanswered


def snapshot(world, day: date) -> Snapshot:
    now = collect_time(day)
    me = world.me

    mails = sorted((m for m in world.mails if mail_in_window(m, now, me)), key=lambda m: (m.sent_at, m.native_id))
    mails = mails[-MAIL_CAP:]

    horizon = now + timedelta(hours=48)
    events = [e for e in world.events if e.created_at <= now and e.end > now and e.start < horizon]
    events.sort(key=lambda e: (e.start, e.native_id))

    overdue, soon = [], []
    for t in world.tasks:
        if t.created_at > now or (t.completed_at is not None and t.completed_at <= now):
            continue
        if t.due_at is None:
            if t.created_at > now - timedelta(hours=24):
                soon.append(t)
        elif t.due_at < now:
            overdue.append(t)
        elif t.due_at < now + timedelta(hours=24):
            soon.append(t)
    overdue.sort(key=lambda t: (t.due_at, t.native_id))
    dropped = max(0, len(overdue) - TASK_OVERDUE_CAP)
    tasks = sorted(overdue[dropped:] + soon, key=lambda t: (t.due_at or t.created_at, t.native_id))

    files, deleted = [], []
    since = now - timedelta(hours=48)
    for f in world.files:
        if f.created_at > now:
            continue
        in_window = [v for v in f.versions if since < v.modified_at <= now]
        if not in_window:
            continue
        if f.deleted_at is not None and f.deleted_at <= now:
            deleted.append(f)
            continue
        files.append((f, in_window[-1]))
    files.sort(key=lambda fv: (fv[1].modified_at, fv[0].native_id))
    files = files[-FILE_CAP:]

    return Snapshot(day, now, mails, events, tasks, files, deleted, dropped)


# ---------------------------------------------------------------------- conversion en Item
def _participant(person: Person, role: str) -> dict[str, str]:
    return {"name": person.name, "email": person.email, "role": role}


def _base(world, source: str, native_id: str, now: datetime) -> dict:
    uid = world.profile.user_id
    return {
        "item_id": f"{source}:{native_id}",
        "source": source,
        "provider": "demo",
        "owner_user_id": uid,
        "content_text": None,  # politique de rétention par défaut : corps non conservé (L1.7)
        "acl": {"visible_to": [uid]},
        "collected_at": now,
        "connector": {"name": CONNECTOR_NAME, "version": CONNECTOR_VERSION},
    }


def mail_item(world, m: SimMail, now: datetime) -> Item:
    people = [_participant(m.sender, "from")]
    people += [_participant(p, "to") for p in m.to]
    people += [_participant(p, "cc") for p in m.cc]
    sent = to_paris(m.sent_at)
    return Item(
        **_base(world, "mail", m.native_id, now),
        title=m.subject,
        snippet=make_snippet(m.html),
        created_at=sent,
        updated_at=sent,
        participants=people,
        thread_id=f"conv:{m.thread_id}",
        native_importance=m.importance,
        is_read=m.read_at is not None and m.read_at <= now,
        url=f"https://outlook.{INTERNAL_DOMAIN}/mail/id/{m.native_id}",
    )


def event_item(world, e: SimEvent, now: datetime) -> Item:
    people = [_participant(e.organizer, "organizer")]
    people += [_participant(p, "attendee") for p in e.attendees if p != e.organizer]
    snippet = " – ".join(part for part in (e.location, e.description) if part)
    return Item(
        **_base(world, "calendar", e.native_id, now),
        title=e.title,
        snippet=truncate(snippet),
        created_at=to_paris(e.created_at),
        updated_at=to_paris(e.updated_at or e.created_at),
        start_at=to_paris(e.start),
        end_at=to_paris(e.end),
        participants=people,
        url=f"https://outlook.{INTERNAL_DOMAIN}/calendar/item/{e.native_id}",
    )


def task_item(world, t: SimTask, now: datetime) -> Item:
    return Item(
        **_base(world, "task", t.native_id, now),
        title=t.title,
        snippet=truncate(t.notes),
        created_at=to_paris(t.created_at),
        updated_at=to_paris(t.created_at),
        due_at=to_paris(t.due_at) if t.due_at else None,
        completed=False,
        participants=[_participant(t.assignee, "assignee")],
        url=f"https://todo.{INTERNAL_DOMAIN}/tasks/id/{t.native_id}",
    )


def file_item(world, f: SimFile, version: SimFileVersion, now: datetime) -> Item:
    site = slug(world.profile.agency).replace(".", "-")
    path = quote(f"{f.folder}/{f.name}", safe="/")
    snippet = truncate(f.excerpt) if f.extension in SNIPPET_FILE_TYPES else ""
    return Item(
        **_base(world, "file", f.native_id, now),
        title=f.name,
        snippet=snippet,
        created_at=to_paris(f.created_at),
        updated_at=to_paris(version.modified_at),
        participants=[_participant(version.modifier, "modifier")],
        url=f"https://sharepoint.{INTERNAL_DOMAIN}/sites/agence-{site}/Documents/{path}",
    )


_SOURCE_ORDER = {"mail": 0, "calendar": 1, "task": 2, "file": 3}


def to_items(world, snap: Snapshot) -> tuple[list[Item], dict[str, object]]:
    """Renvoie les Item triés de façon stable et l'index item_id → objet simulé."""
    items: list[Item] = []
    index: dict[str, object] = {}
    for m in snap.mails:
        items.append(mail_item(world, m, snap.now))
        index[items[-1].item_id] = m
    for e in snap.events:
        items.append(event_item(world, e, snap.now))
        index[items[-1].item_id] = e
    for t in snap.tasks:
        items.append(task_item(world, t, snap.now))
        index[items[-1].item_id] = t
    for f, version in snap.files:
        items.append(file_item(world, f, version, snap.now))
        index[items[-1].item_id] = f
    items.sort(key=lambda it: (_SOURCE_ORDER[it.source], it.created_at, it.item_id))
    return items, index
