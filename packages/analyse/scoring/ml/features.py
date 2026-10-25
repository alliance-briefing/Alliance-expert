"""Fiche d'un mail à l'instant du brief : les caractéristiques lisibles du score appris.

Les candidats d'un matin sont les mails reçus dans les 3 jours précédant 8 h 30 (Paris),
hors bruit et hors envois du manager. Le fil n'est lu que jusqu'à l'instant du brief et sur
30 jours : aucune caractéristique ne regarde le futur.
`flag`, `categories` et `importance` sont exclus : tirés au hasard dans les données de David.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta

import numpy as np

from contracts.models import Item
from packages.analyse.clock import PARIS
from packages.analyse.scoring.ml.deadline import find_deadline
from packages.analyse.scoring.ml.graph_convert import GraphDataset
from packages.analyse.scoring.ml.intents import intent_text, is_alert
from packages.analyse.scoring.ml.noise import noise_reason
from packages.analyse.scoring.ml.persona import SENDER_KINDS, Persona

BRIEF_TIME = time(8, 30)
WINDOW = timedelta(days=3)
HISTORY = timedelta(days=30)
DEADLINE_BINS = ("dépassée", "aujourd'hui ou demain", "≤ 3 jours", "plus tard", "aucune")


def brief_instant(day: date) -> datetime:
    return datetime.combine(day, BRIEF_TIME, tzinfo=PARIS)


@dataclass(frozen=True)
class MailFeatures:
    item_id: str
    sender_kind: str
    action: float  # probabilités apprises (intents.py)
    reponse: float
    incident: float
    alerte: bool
    deadline: str  # une valeur de DEADLINE_BINS
    deadline_phrase: str | None
    ball_in_court: bool  # aucun envoi du manager dans le fil depuis ce mail
    thread_size: int  # messages du fil sur 30 jours, jusqu'à l'instant du brief
    hierarchy_in_thread: bool
    fresh: bool  # reçu il y a moins de 24 h
    meeting_soon: bool  # invitation à une réunion qui commence dans les 48 h

    def as_dict(self) -> dict:
        return asdict(self)


def _sender(item: Item) -> str:
    return next((p.email for p in item.participants if p.role == "from"), "")


def candidates(dataset: GraphDataset, day: date) -> list[Item]:
    """Mails reçus dans les 3 jours avant le brief du jour, hors bruit et hors envois."""
    now = brief_instant(day)
    return sorted(
        (
            item
            for item in dataset.items
            if item.source == "mail"
            and not dataset.extras[item.item_id].is_sent
            and now - WINDOW <= item.created_at < now
            and not noise_reason(item)
        ),
        key=lambda item: item.item_id,
    )


def _deadline_bin(days_left: int | None) -> str:
    if days_left is None:
        return "aucune"
    if days_left < 0:
        return "dépassée"
    if days_left <= 1:
        return "aujourd'hui ou demain"
    return "≤ 3 jours" if days_left <= 3 else "plus tard"


def mail_features(
    items: Sequence[Item],
    dataset: GraphDataset,
    now: datetime,
    persona: Persona,
    probs: Mapping[str, Mapping[str, float]],
) -> list[MailFeatures]:
    """La fiche de chaque mail à l'instant `now`. `probs` : {intention: {item_id: proba}}."""
    threads: dict[str, list[Item]] = defaultdict(list)
    for item in dataset.items:
        if item.source == "mail" and item.thread_id and now - HISTORY <= item.created_at < now:
            threads[item.thread_id].append(item)
    meetings = {
        item.correlation_id: item.start_at
        for item in dataset.items
        if item.source == "calendar" and item.correlation_id and item.start_at
    }
    out = []
    for item in items:
        thread = threads.get(item.thread_id or "", [item])
        sent_after = any(
            dataset.extras[m.item_id].is_sent and m.created_at > item.created_at for m in thread
        )
        text = intent_text(item.title, item.content_text or item.snippet)
        kind = persona.sender_kind(_sender(item))
        deadline = find_deadline(text, item.created_at)
        start = meetings.get(item.correlation_id or "")
        out.append(
            MailFeatures(
                item_id=item.item_id,
                sender_kind=kind,
                action=float(probs["action"][item.item_id]),
                reponse=float(probs["reponse"][item.item_id]),
                incident=float(probs["incident"][item.item_id]),
                alerte=is_alert(text, kind),
                deadline=_deadline_bin(deadline.days_left(now) if deadline else None),
                deadline_phrase=deadline.phrase if deadline else None,
                ball_in_court=not sent_after,
                thread_size=len(thread),
                hierarchy_in_thread=any(
                    persona.sender_kind(_sender(m)) == "hiérarchie" for m in thread
                ),
                fresh=now - item.created_at < timedelta(hours=24),
                meeting_soon=start is not None and now <= start < now + timedelta(hours=48),
            )
        )
    return out


# Codage numérique (référence omise pour les catégories : collègue, échéance « aucune »)
VECTOR_NAMES = (
    *(f"expéditeur : {k}" for k in SENDER_KINDS if k != "collègue"),
    "demande une action",
    "attend une réponse",
    "incident",
    "alerte",
    *(f"échéance : {b}" for b in DEADLINE_BINS if b != "aucune"),
    "balle dans son camp",
    "fil long (≥ 3 messages)",
    "hiérarchie dans le fil",
    "reçu depuis moins de 24 h",
    "réunion dans les 48 h",
)


def vectorize(f: MailFeatures) -> np.ndarray:
    values = [
        *(float(f.sender_kind == k) for k in SENDER_KINDS if k != "collègue"),
        f.action,
        f.reponse,
        f.incident,
        float(f.alerte),
        *(float(f.deadline == b) for b in DEADLINE_BINS if b != "aucune"),
        float(f.ball_in_court),
        float(f.thread_size >= 3),
        float(f.hierarchy_in_thread),
        float(f.fresh),
        float(f.meeting_soon),
    ]
    return np.array(values, dtype=float)
