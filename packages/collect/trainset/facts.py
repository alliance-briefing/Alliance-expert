"""Le fait : tout ce que le monde sait d'un élément, et son étiquette selon le guide.

Le simulateur décide des faits ; l'étiqueteur (`label`) applique au fait les règles du guide
d'annotation (`packages/analyse/scoring/annotations/guide-annotation.md`, figé le 01/11).
L'écrivain n'en montre qu'une partie dans l'`Item` : le modèle doit retrouver le reste.

L'étiquette dépend du jour : `label_on` rend « rien » une fois l'élément traité par le
manager (`resolved_at`) ou la réunion passée, comme le guide (« déjà traité → Rien »).

Règles du 23/11 (guide) : deux d'entre elles dépendent de la formulation choisie par
l'écrivain, pas du fait. `label` reçoit donc les gabarits utilisés (ou les marques posées par
l'écrivain LLM) : « avis ou accord » sur un objet de décision → Décider ; copie dont le texte
dit « pour suivre » → S'informer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from packages.collect.trainset.metiers import DECISION_KINDS, Obj
from packages.collect.trainset.world import Person

ACTIONS = ("repondre", "faire", "decider", "preparer", "relire", "informer", "rien")
# Règle de priorité du guide : Décider > Faire > Répondre > Préparer > Relire > S'informer > Rien
PRIORITY = ("decider", "faire", "repondre", "preparer", "relire", "informer", "rien")
URGENCY = ("urgent", "normal", "sans_urgence")
SOURCES = ("mail", "calendar", "task", "file")


# Gabarits de Répondre qui demandent un avis ou un accord (« te convient », « ton avis », « ta
# position », « ce que tu en penses »…) : sur un objet de décision, ils se lisent Décider.
AVIS_TEMPLATES = frozenset(
    {f"mail.repondre.question.{i}" for i in (0, 1)}
    | {f"mail.repondre.directe.{i}" for i in (0, 2, 3, 7, 8, 9)}
    | {f"mail.repondre.indirecte.{i}" for i in (0, 1, 4)}
)
AVIS = "lecture:avis"  # marque de l'écrivain LLM : la fiche demandait un avis
COPIE_SUIVI = "lecture:copie_suivi"  # marque : le texte dit « en copie pour suivre »


def strongest(*actions: str | None) -> str:
    present = [a for a in actions if a]
    return min(present, key=PRIORITY.index) if present else "rien"


@dataclass
class Fact:
    item_id: str
    source: str
    at: datetime  # visible à partir de (réception, création, modification)
    author: Person  # expéditeur, organisateur, auteur de la tâche ou du fichier
    case_id: str
    case_type: str
    act: str = "rien"  # acte demandé (mail, tâche) ; sens propre à chaque source sinon
    obj: Obj | None = None
    tone: str = "normal"
    alert: bool = False
    deadline: datetime | None = None  # échéance écrite dans le texte
    resolved_at: datetime | None = None  # le manager l'a traité à ce moment
    secondary: str | None = None  # second acte dans le même mail
    form: str = "directe"  # directe, question, indirecte
    to_manager: bool = True  # destinataire direct (sinon en copie)
    addressee: Person | None = None  # la personne à qui la demande s'adresse, si ce n'est pas lui
    recipients: list[Person] = field(default_factory=list)
    sent: bool = False  # envoyé par le manager
    thread_id: str | None = None
    correlation_id: str | None = None
    links: list[str] = field(default_factory=list)  # autres éléments de la même affaire
    trap: str | None = None  # injection, hameconnage, doublon, conflit, contradiction…
    meta: dict = field(default_factory=dict)  # détails propres à la source (voir label)


def label(fact: Fact, wording: Iterable[str] = ()) -> dict:
    """Étiquette propre de l'élément, à la lecture du guide (action, urgence, alerte).
    `wording` : gabarits utilisés par l'écrivain et marques de lecture de l'écrivain LLM."""
    return {"action": _action(fact, set(wording)), "urgence": fact.tone, "alerte": fact.alert}


def _mass_mailing(f: Fact) -> bool:
    return str(f.meta.get("template", "")).startswith("newsletter")


def _action(f: Fact, wording: set[str] | frozenset[str] = frozenset()) -> str:
    m = f.meta
    if f.source == "mail":
        if f.sent or f.trap in ("injection", "doublon"):
            return "rien"
        if not f.to_manager:  # en copie d'une demande adressée à quelqu'un d'autre (23/11)
            return "informer" if COPIE_SUIVI in wording else "rien"
        act = f.act
        if (
            act == "repondre"
            and f.obj is not None
            and f.obj.kind in DECISION_KINDS
            and (AVIS in wording or AVIS_TEMPLATES & wording)
        ):
            act = "decider"  # avis ou accord sur un objet de décision (23/11)
        main = strongest(act, f.secondary)
        if main == "rien" and not _mass_mailing(f):
            return "informer"  # message sans demande, accusé, notification qui informe (23/11)
        return main
    if f.source == "calendar":
        if m.get("cancelled"):
            return "rien"
        prepare = "preparer" if (m.get("organizer_me") or m.get("bring")) else None
        answer = (
            "repondre"
            if (m.get("response") in ("none", "tentative") or m.get("conflict"))
            else None
        )
        return strongest(prepare, answer)
    if f.source == "task":
        return "rien" if m.get("completed") else f.act
    if f.source == "file":
        if m.get("by_me"):
            return "rien"
        if m.get("linked_task_act"):
            return m["linked_task_act"]
        if m.get("linked_meeting"):
            return "preparer"
        if m.get("to_sign"):
            return "decider"
        return "informer"  # à relire : c'est le mail qui le demande qui porte Relire (23/11)
    raise ValueError(f"source inconnue : {f.source}")


def label_on(row: dict, moment: datetime) -> dict | None:
    """Étiquette d'une ligne de scénario au moment du brief ; None si pas encore visible."""
    if datetime.fromisoformat(row["visible_from"]) > moment:
        return None
    labels = {"action": row["action"], "urgence": row["urgence"], "alerte": row["alerte"]}
    done = row.get("resolved_at") and datetime.fromisoformat(row["resolved_at"]) <= moment
    over = row.get("ends_at") and datetime.fromisoformat(row["ends_at"]) <= moment
    if done or over:
        labels["action"] = "rien"
    return labels
