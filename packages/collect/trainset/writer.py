"""L'écrivain par gabarits : un fait → un `Item` du contrat A.

Il ne montre que ce qu'un vrai connecteur porterait : pour une tâche, ni qui l'a assignée
ni son lien avec une affaire ; pour un fichier, ni sa réunion ni sa tâche (seulement son
nom, son auteur, son dossier). Le reste reste dans le scénario : le modèle doit le déduire.

Chaque fait a son propre tirage (graine dérivée de son identifiant) : un écrivain LLM (M3)
peut remplacer celui-ci fait par fait, sans changer le monde.
"""

from __future__ import annotations

import random
import re
from datetime import timedelta

from contracts.models import Item
from packages.collect.core.text import normalize_whitespace, strip_quotes_and_signature, truncate
from packages.collect.trainset import phrases as P
from packages.collect.trainset.facts import Fact
from packages.collect.trainset.world import Person, Persona, fr_day, slug

CONNECTOR = {"name": "trainset", "version": "0.1.0"}
SUBJECT = {
    "decider": ("{S} – validation", "Validation : {s}", "{S} à valider", "{S}"),
    "faire": ("{S}", "Demande : {s}", "{S} – à faire", "Action requise : {s}"),
    "repondre": ("{S}", "Question sur {s}", "{S} – ton avis ?", "Point sur {s}"),
    "preparer": ("Préparation : {s}", "{S} – pour la réunion", "Réunion : {s}"),
    "relire": ("{S} – pour relecture", "Relecture : {s}", "{S} (version de travail)"),
    "informer": ("{S} – pour info", "Info : {s}", "{S}", "Point d'étape : {s}"),
    "rien": ("Re: {s}", "RE : {s}", "{S}"),
}
SENT = (
    "C'est validé, merci.",
    "Bien reçu, je m'en occupe.",
    "Réponse : oui, on part là-dessus.",
    "Voici les éléments demandés.",
    "Relu, c'est bon pour moi.",
    "Je te propose qu'on en parle demain.",
)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _short(fact: Fact) -> str:
    """Le sujet d'un objet sans article : « devis Stockeo »."""
    if fact.obj is None:
        return "dossier en cours"
    return re.sub(r"^(le |la |les |l')", "", fact.obj.phrase)


class Writer:
    def __init__(self, persona: Persona, seed: int = 0) -> None:
        self.p = persona
        self.seed = seed
        self.reserved_ok = persona.split == "tgen"

    def write(self, fact: Fact) -> tuple[Item, list[str]]:
        """L'Item et les identifiants des gabarits utilisés (pour le scénario)."""
        rng = random.Random(f"trainset:writer:{self.seed}:{fact.item_id}")
        used: list[str] = []
        build = {
            "mail": self._mail,
            "calendar": self._event,
            "task": self._task,
            "file": self._file,
        }[fact.source]
        fields = build(fact, rng, used)
        moment = fact.at
        item = Item(
            item_id=fact.item_id,
            source=fact.source,
            provider="demo",
            owner_user_id=self.p.owner,
            created_at=fields.pop("created_at", moment),
            updated_at=moment,
            collected_at=moment,
            url=fields.pop("url"),
            acl={"visible_to": [self.p.owner]},
            connector=CONNECTOR,
            **fields,
        )
        return item, used

    # --- mails -------------------------------------------------------------------------------

    def _slots(self, fact: Fact, rng: random.Random) -> dict:
        deadline = ""
        if fact.deadline:
            deadline = rng.choice(P.DEADLINE).format(d=fr_day(fact.deadline.date()))
        return {
            "o": fact.obj.phrase if fact.obj else "le dossier",
            "O": _cap(fact.obj.phrase) if fact.obj else "Le dossier",
            "quand": deadline,
            "qui": fact.meta.get("qui", ""),
            "jour": fact.meta.get("jour", "demain"),
            "statut": rng.choice(P.STATUTS),
            "prenom": self.p.manager.first,
        }

    def _sentence(self, bank: tuple, rng, used, name, tu: bool, slots: dict) -> str:
        text, tid = P.pick(bank, rng, self.reserved_ok, name)
        used.append(tid)
        return P.resolve(str(text), tu).format(**slots)

    def _mail(self, fact: Fact, rng: random.Random, used: list[str]) -> dict:
        tu = fact.meta.get("tu", True)
        kind = fact.meta.get("kind")
        if kind == "fixed":
            used.append(fact.meta["template"])
            title, body = fact.meta["title"], fact.meta["body"]
        elif kind == "sent":
            title = "Re: " + self._subject(fact, fact.meta.get("opening_act", "informer"))
            body = f"{fact.recipients[0].first},\n\n{rng.choice(SENT)}\n\n{self.p.manager.first}"
        elif kind in ("incident", "incident_suivi", "incident_clos", "contradiction"):
            title, body = self._special(fact, rng, tu)
        else:
            title, body = self._composed(fact, rng, used, tu)
        clean = strip_quotes_and_signature(normalize_whitespace(body)) or body
        people = [{"name": fact.author.name, "email": fact.author.email, "role": "from"}]
        people += [{"name": r.name, "email": r.email, "role": "to"} for r in fact.recipients]
        if not fact.to_manager:
            m = self.p.manager
            people.append({"name": m.name, "email": m.email, "role": "cc"})
        return {
            "title": title,
            "snippet": truncate(re.sub(r"\s+", " ", clean).strip()),
            "content_text": clean,
            "participants": people,
            "thread_id": fact.thread_id,
            "is_read": True if fact.sent else rng.random() < 0.25,
            "url": f"https://mail.{self.p.domain}/owa/{fact.item_id.split(':')[1]}",
        }

    def _subject(self, fact: Fact, act: str | None = None) -> str:
        """Le même objet pour tout le fil : tiré avec la graine de l'affaire."""
        rng = random.Random(f"trainset:subject:{self.seed}:{fact.thread_id}")
        s = _short(fact)
        template = rng.choice(SUBJECT[act or fact.act])
        return template.format(s=s, S=_cap(s))

    def _composed(self, fact: Fact, rng: random.Random, used: list[str], tu: bool) -> tuple:
        slots = self._slots(fact, rng)
        target = fact.addressee or self.p.manager
        lines = []
        greetings = P.GREETING_TU if tu else P.GREETING_VOUS
        greeting = rng.choice(greetings).format(
            prenom=target.first, nom=target.last, civ="Madame" if target.female else "Monsieur"
        )
        if greeting:
            lines.append(greeting)
        relance = fact.meta.get("relance")
        opening = rng.choice(P.RELANCE) if relance else rng.choice(P.CONTEXT)
        main = self._sentence(
            P.MAIL[(fact.act, fact.form)], rng, used, f"mail.{fact.act}.{fact.form}", tu, slots
        )
        if opening:
            main = f"{P.resolve(opening, tu)} {main[:1].lower()}{main[1:]}"
        lines.append(main)
        if fact.secondary:
            lines.append(P.resolve(P.SECOND[fact.secondary], tu).format(**slots))
        if fact.tone != "normal":
            lines.append(
                self._sentence(P.TONE[fact.tone], rng, used, f"tone.{fact.tone}", tu, slots)
            )
        closing = P.resolve(rng.choice(P.CLOSING), tu)
        if closing:
            lines.append(closing)
        if rng.random() < 0.75:
            lines.append(fact.author.first if tu else f"{fact.author.name}\n{fact.author.role}")
        if relance and rng.random() < 0.5:  # message cité, que le nettoyage doit retirer
            lines.append(
                f"Le {fr_day(fact.at.date() - timedelta(days=2))}, {fact.author.name} "
                f"a écrit :\n> {main}"
            )
        subject = self._subject(fact, "rien" if fact.act == "rien" else None)
        if relance:
            subject = rng.choice(("Re: ", "RE : ", "Relance : ")) + subject
        if fact.tone == "urgent" and rng.random() < 0.5:
            subject = "URGENT – " + subject
        return subject, "\n\n".join(lines)

    def _special(self, fact: Fact, rng: random.Random, tu: bool) -> tuple:
        incident = fact.meta.get("incident", "")
        kind = fact.meta["kind"]
        me = self.p.manager.first
        if kind == "incident":
            return (
                f"URGENT – {incident}",
                (
                    f"{me},\n\n{incident} depuis ce matin. Il faut ta "
                    "décision pour faire intervenir le prestataire en urgence (coût hors "
                    f"contrat). Ça bloque l'activité.\n\n{fact.author.first}"
                ),
            )
        if kind == "incident_suivi":
            return (
                f"Re: {incident}",
                f"{me},\n\nOn est dessus, je te tiens au courant.\n\n{fact.author.first}",
            )
        if kind == "incident_clos":
            return (
                f"{incident} – résolu",
                (
                    f"Bonjour {me},\n\nPour info, {incident.lower()} : "
                    f"c'est réglé depuis ce matin. Rien à faire.\n\n{fact.author.first}"
                ),
            )
        meeting = fact.meta["meeting"]
        if fact.meta["step"] == 1:
            return (
                f"{meeting} déplacé",
                f"Bonjour,\n\nLa réunion « {meeting} » est déplacée à 15 h.\n\n{fact.author.first}",
            )
        return (
            f"Re: {meeting} déplacé",
            (
                f"Finalement, on garde l'horaire initial pour "
                f"« {meeting} ». Désolé pour la confusion.\n\n{fact.author.first}"
            ),
        )

    # --- agenda ------------------------------------------------------------------------------

    def _event(self, fact: Fact, rng: random.Random, used: list[str]) -> dict:
        m = fact.meta
        slots = self._slots(fact, rng)
        tu = m.get("tu", True)
        variant = m.get("variant", "invite")
        desc = self._sentence(P.MEETING_DESC[variant], rng, used, f"event.{variant}", tu, slots)
        title = m["topic"]
        if fact.obj and variant != "routine" and rng.random() < 0.6:
            title = f"{title} – {_short(fact)}"
        if m.get("cancelled"):
            title = "Annulé : " + title
        attendees: list[Person] = m.get("attendees", [self.p.manager])
        people = [{"name": fact.author.name, "email": fact.author.email, "role": "organizer"}]
        people += [
            {"name": a.name, "email": a.email, "role": "attendee"}
            for a in attendees
            if a.email != fact.author.email
        ]
        online = rng.random() < 0.4
        n = fact.item_id.split(":")[1]
        return {
            "schema_version": "1.1",
            "title": title,
            "snippet": desc,
            "content_text": desc or None,
            "start_at": m["start"],
            "end_at": m["end"],
            "participants": people,
            "correlation_id": f"evt-{n}",
            "attributes": {
                "calendar": {
                    "location": None
                    if online
                    else rng.choice(
                        ("Salle A", "Salle du conseil", "Bureau 2", "Salle de réunion")
                    ),
                    "online_url": f"https://teams.{self.p.domain}/meet/{n}" if online else None,
                    "is_cancelled": bool(m.get("cancelled")),
                    "response": m.get("response"),
                    "series_id": m.get("series"),
                }
            },
            "url": f"https://calendar.{self.p.domain}/event/{n}",
        }

    # --- tâches ------------------------------------------------------------------------------

    def _task(self, fact: Fact, rng: random.Random, used: list[str]) -> dict:
        m = fact.meta
        if "title" in m:
            title = m["title"]
        else:
            slots = self._slots(fact, rng) | {"reunion": m.get("reunion", "la réunion")}
            title = self._sentence(
                P.TASK_TITLE[fact.act], rng, used, f"task.{fact.act}", True, slots
            )
        if fact.tone == "urgent":
            title = f"URGENT – {title}"
        n = fact.item_id.split(":")[1]
        planner = m.get("origin") == "planner"
        url = (
            f"https://planner.{self.p.domain}/plan/{slug(self.p.metier.folder)}/task/{n}"
            if planner
            else f"https://todo.{self.p.domain}/tasks/id/{n}"
        )
        return {
            "title": title,
            "snippet": "",
            "due_at": m.get("due"),
            "completed": bool(m.get("completed")),
            "participants": [
                {"name": self.p.manager.name, "email": self.p.manager.email, "role": "assignee"}
            ],
            "url": url,
        }

    # --- fichiers ----------------------------------------------------------------------------

    def _file(self, fact: Fact, rng: random.Random, used: list[str]) -> dict:
        m = fact.meta
        if m.get("cr"):
            day = m["meeting_day"]
            name = (
                f"{rng.choice(P.CR_PREFIX)}_{slug(m['topic']).replace('.', '_')}_"
                f"{day.day:02d}-{day.month:02d}.docx"
            )
        elif m.get("linked_meeting"):
            day = m["meeting_day"]
            name = f"{fact.obj.stem}_{day.day:02d}-{day.month:02d}.{fact.obj.ext}"
        else:
            name = f"{fact.obj.stem}_v{m.get('version', 1)}.{fact.obj.ext}"
        if m.get("by_me"):
            url = (
                f"https://onedrive.{self.p.domain}/personal/{slug(self.p.manager.name)}/"
                f"Documents/{name}"
            )
        else:
            url = (
                f"https://sharepoint.{self.p.domain}/sites/{self.p.metier.folder}/Documents/{name}"
            )
        return {
            "title": name,
            "snippet": "",
            "created_at": fact.at - timedelta(days=rng.randint(0, 20), hours=rng.randint(0, 5)),
            "participants": [
                {"name": fact.author.name, "email": fact.author.email, "role": "modifier"}
            ],
            "url": url,
        }
