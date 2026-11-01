"""Le simulateur : des affaires qui traversent les 4 sources, un fond, des pièges.

Il décide des faits (qui, quoi, quand, acte demandé, ton, liens) et du comportement du
manager (traite-t-il la demande, et quand ?). Il n'écrit aucun texte libre : c'est le
rôle de l'écrivain. Seuls les messages d'automates (lettres, tickets, portail RH) et les
pièges sont tirés tels quels des banques, car leur forme fait partie du fait.

Déterministe : même persona, même graine, mêmes faits.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta

from packages.collect.trainset import phrases as P
from packages.collect.trainset.facts import Fact
from packages.collect.trainset.metiers import DECISION_KINDS, REVIEW_KINDS, Obj
from packages.collect.trainset.world import Person, Persona, at, fr_day, working_days

CASE_TYPES = (
    "validation",
    "demande_client",
    "reunion_preparee",
    "invitation",
    "incident",
    "relecture",
    "demande_info",
    "delegation_cc",
    "tache_perso",
    "signature",
)
CASE_WEIGHTS = (14, 12, 11, 9, 5, 8, 10, 7, 9, 5)


class Sim:
    def __init__(self, persona: Persona, seed: int = 0) -> None:
        self.p = persona
        self.rng = random.Random(f"trainset:sim:{persona.key}:{seed}")
        self.days = working_days()
        self.facts: list[Fact] = []
        self._n = 0
        self._cases = 0
        self.reserved_ok = persona.split == "tgen"

    # --- outils ------------------------------------------------------------------------------

    def _id(self, source: str) -> str:
        self._n += 1
        return f"{source}:{self.p.key}-{self._n:05d}"

    def _case(self) -> str:
        self._cases += 1
        return f"{self.p.key}-c{self._cases:03d}"

    def _moment(self, day: date, start: float = 8.0, end: float = 18.5) -> datetime:
        return at(day, self.rng.uniform(start, end))

    def _later(self, day: date, k: int) -> date | None:
        """Le k-ième jour ouvré après `day`, ou None hors de l'horizon."""
        i = self.days.index(day) + k
        return self.days[i] if 0 <= i < len(self.days) else None

    def _handled(self, after: datetime, speed: float = 1.0) -> datetime | None:
        """Quand le manager traite la demande (None : pas dans l'horizon simulé)."""
        day = after.date()
        while day is not None:
            if self.rng.random() < min(0.95, self.p.diligence * speed):
                start = (
                    max(8.0, after.hour + after.minute / 60 + 0.25) if day == after.date() else 8.0
                )
                if start < 19.0:
                    return self._moment(day, start, 19.0)
            day = self._later(day, 1)
        return None

    def _obj(self, kinds: tuple[str, ...] | None = None) -> Obj:
        pool = [o for o in self.p.metier.objects if kinds is None or o.kind in kinds]
        return self.rng.choice(pool)

    def _tone(self, urgent: float = 0.15, calm: float = 0.12) -> str:
        r = self.rng.random()
        return "urgent" if r < urgent else "sans_urgence" if r < urgent + calm else "normal"

    def _deadline(self, day: date, p: float = 0.45) -> datetime | None:
        if self.rng.random() >= p:
            return None
        target = self._later(day, self.rng.randint(1, 4)) or self.days[-1]
        return at(target, self.rng.choice((12.0, 17.0, 18.0)))

    def _form(self, act: str) -> str:
        forms = P.FORMS[act]
        weights = [{"question": 4, "directe": 4, "indirecte": 1}[f] for f in forms]
        return self.rng.choices(forms, weights)[0]

    def _qui(self) -> Person:
        return self.rng.choice(self.p.clients + self.p.suppliers + self.p.team)

    def add(self, **kw) -> Fact:
        source = kw.pop("source")
        fact = Fact(item_id=self._id(source), source=source, **kw)
        self.facts.append(fact)
        return fact

    def mail(self, case: str, ctype: str, moment: datetime, author: Person, act: str, **kw) -> Fact:
        kw.setdefault("thread_id", f"thr-{case}")
        kw.setdefault("form", self._form(act) if act in P.FORMS else "directe")
        kw.setdefault("recipients", [self.p.manager])
        meta = kw.setdefault("meta", {})
        meta.setdefault("qui", self._qui().name)
        meta.setdefault("tu", author.kind in ("hierarchie", "pair", "equipe"))
        return self.add(
            source="mail", at=moment, author=author, case_id=case, case_type=ctype, act=act, **kw
        )

    def reply(self, case: str, ctype: str, moment: datetime, to: Person, opening: Fact) -> Fact:
        """Réponse envoyée par le manager depuis sa boîte (étiquette : rien)."""
        return self.add(
            source="mail",
            at=moment,
            author=self.p.manager,
            case_id=case,
            case_type=ctype,
            act="rien",
            obj=opening.obj,
            sent=True,
            recipients=[to],
            thread_id=opening.thread_id,
            meta={
                "kind": "sent",
                "tu": opening.meta.get("tu", True),
                "opening_act": opening.act,
                "qui": opening.meta.get("qui", ""),
            },
        )

    def _link(self, facts: list[Fact]) -> None:
        ids = [f.item_id for f in facts]
        for f in facts:
            f.links = [i for i in ids if i != f.item_id]

    def _resolve(self, facts: list[Fact], when: datetime | None) -> None:
        for f in facts:
            if not f.sent:
                f.resolved_at = when

    # --- affaires ----------------------------------------------------------------------------

    def validation(self, day: date) -> None:
        case, ct = self._case(), "validation"
        obj = self._obj(DECISION_KINDS)
        who = self.rng.choice(self.p.team + (self.p.boss,) + self.p.peers[:1])
        t0 = self._moment(day, 8.0, 17.5)
        deadline = self._deadline(day)
        done = self._handled(t0, 0.9)
        facts = [
            self.mail(case, ct, t0, who, "decider", obj=obj, tone=self._tone(), deadline=deadline)
        ]
        task = None
        if self.rng.random() < 0.55:
            task = self.add(
                source="task",
                at=t0 + timedelta(minutes=5),
                author=who,
                case_id=case,
                case_type=ct,
                act="decider",
                obj=obj,
                meta={"origin": "planner", "due": deadline, "completed": False},
            )
            facts.append(task)
        if self.rng.random() < 0.65:
            facts.append(
                self.add(
                    source="file",
                    at=t0 - timedelta(minutes=self.rng.randint(5, 90)),
                    author=who,
                    case_id=case,
                    case_type=ct,
                    obj=obj,
                    meta={
                        "linked_task_act": "decider" if task else None,
                        "to_sign": task is None,
                        "version": self.rng.randint(1, 4),
                    },
                )
            )
        later = self._later(day, 2)
        if later and (done is None or done > at(later, 8.0)):
            facts.append(
                self.mail(
                    case,
                    ct,
                    self._moment(later, 8.0, 12.0),
                    who,
                    "decider",
                    obj=obj,
                    tone=self._tone(0.5, 0.0),
                    deadline=deadline,
                    meta={"relance": True},
                )
            )
        self._resolve(facts, done)
        if done:
            facts.append(self.reply(case, ct, done, who, facts[0]))
        self._link(facts)

    def demande_client(self, day: date) -> None:
        case, ct = self._case(), "demande_client"
        obj = self._obj()
        client = self.rng.choice(self.p.clients)
        act = self.rng.choices(("repondre", "faire"), (6, 4))[0]
        t0 = self._moment(day)
        done = self._handled(t0)
        facts = [
            self.mail(
                case,
                ct,
                t0,
                client,
                act,
                obj=obj,
                tone=self._tone(0.2, 0.08),
                deadline=self._deadline(day, 0.4),
            )
        ]
        later = self._later(day, self.rng.randint(1, 3))
        if later and (done is None or done > at(later, 8.0)):
            facts.append(
                self.mail(
                    case,
                    ct,
                    self._moment(later, 7.5, 11.0),
                    client,
                    "repondre",
                    obj=obj,
                    tone=self._tone(0.6, 0.0),
                    form="indirecte",
                    meta={"relance": True},
                )
            )
        self._resolve(facts, done)
        if done:
            facts.append(self.reply(case, ct, done, client, facts[0]))
        self._link(facts)

    def reunion_preparee(self, day: date) -> None:
        case, ct = self._case(), "reunion_preparee"
        meet_day = self._later(day, self.rng.randint(1, 5))
        if meet_day is None:
            return
        obj = self._obj(("presentation", "budget", "rapport", "planning"))
        organizer_me = self.rng.random() < 0.5
        organizer = self.p.manager if organizer_me else self.p.boss
        start = at(meet_day, self.rng.choice((9.0, 10.0, 14.0, 15.0)))
        end = start + timedelta(hours=self.rng.choice((1, 1, 2)))
        topic = self.rng.choice(self.p.metier.meetings)
        t0 = self._moment(day)
        others = self.rng.sample(self.p.team + self.p.peers, 3)
        facts = [
            self.add(
                source="calendar",
                at=t0,
                author=organizer,
                case_id=case,
                case_type=ct,
                obj=obj,
                meta={
                    "start": start,
                    "end": end,
                    "topic": topic,
                    "organizer_me": organizer_me,
                    "bring": not organizer_me,
                    "response": "organizer" if organizer_me else "accepted",
                    "attendees": [self.p.manager, *others],
                    "variant": "organizer" if organizer_me else "bring",
                },
            )
        ]
        if not organizer_me and self.rng.random() < 0.7:
            facts.append(
                self.mail(
                    case,
                    ct,
                    t0 + timedelta(minutes=2),
                    self.p.boss,
                    "preparer",
                    obj=obj,
                    form="directe",
                    meta={"jour": fr_day(meet_day)},
                )
            )
        if self.rng.random() < 0.6:
            prep_day = self._later(meet_day, -1) or day
            prep_day = max(prep_day, day)
            facts.append(
                self.add(
                    source="file",
                    at=self._moment(prep_day),
                    author=self.rng.choice(self.p.team),
                    case_id=case,
                    case_type=ct,
                    obj=obj,
                    meta={
                        "linked_meeting": True,
                        "ends_at": end,
                        "version": 1,
                        "topic": topic,
                        "meeting_day": meet_day,
                    },
                )
            )
        after = self._later(meet_day, 1)
        if after:
            cr_author = self.rng.choice(self.p.team)
            facts.append(
                self.add(
                    source="file",
                    at=self._moment(after, 9.0, 17.0),
                    author=cr_author,
                    case_id=case,
                    case_type=ct,
                    obj=None,
                    meta={"cr": True, "topic": topic, "meeting_day": meet_day},
                )
            )
            if self.rng.random() < 0.5:
                t = self._moment(after, 9.0, 17.0)
                task = self.add(
                    source="task",
                    at=t,
                    author=self.p.boss,
                    case_id=case,
                    case_type=ct,
                    act="faire",
                    obj=obj,
                    meta={
                        "origin": "planner",
                        "completed": False,
                        "due": at(self._later(after, 3) or self.days[-1], 17.0),
                        "qui": self._qui().name,
                    },
                )
                task.resolved_at = self._handled(t)
                facts.append(task)
        self._link(facts)

    def invitation(self, day: date) -> None:
        case, ct = self._case(), "invitation"
        meet_day = self._later(day, self.rng.randint(1, 6))
        if meet_day is None:
            return
        organizer = self.rng.choice(self.p.peers + self.p.team[:2] + self.p.clients[:1])
        conflict = self.rng.random() < 0.3
        start = at(meet_day, self.rng.choice((9.5, 11.0, 14.0, 16.0)))
        t0 = self._moment(day)
        fact = self.add(
            source="calendar",
            at=t0,
            author=organizer,
            case_id=case,
            case_type=ct,
            obj=self._obj(),
            meta={
                "start": start,
                "end": start + timedelta(hours=1),
                "topic": self.rng.choice(self.p.metier.meetings),
                "organizer_me": False,
                "response": "tentative" if conflict else "none",
                "conflict": conflict,
                "attendees": [self.p.manager, organizer],
                "variant": "conflict" if conflict else "invite",
                "tu": organizer.kind != "externe_client",
            },
        )
        fact.resolved_at = self._handled(t0, 1.1)
        if fact.resolved_at and fact.resolved_at > start:
            fact.resolved_at = None

    def incident(self, day: date) -> None:
        case, ct = self._case(), "incident"
        incident = self.rng.choice(self.p.metier.incidents)
        t0 = self._moment(day, 6.5, 17.0)
        facts = []
        if self.rng.random() < 0.5:
            title, body = P.SECURITY[0]
            facts.append(
                self.mail(
                    case,
                    ct,
                    t0,
                    self.p.automates[4],
                    "faire",
                    tone="urgent",
                    alert=True,
                    form="directe",
                    meta={
                        "kind": "fixed",
                        "title": title.format(incident=incident),
                        "body": body.format(incident=incident),
                        "template": "security.0",
                    },
                )
            )
        else:
            facts.append(
                self.mail(
                    case,
                    ct,
                    t0,
                    self.rng.choice(self.p.team),
                    "decider",
                    tone="urgent",
                    alert=True,
                    form="directe",
                    meta={"kind": "incident", "incident": incident},
                )
            )
        done = self._handled(t0, 1.6)
        follow = t0 + timedelta(minutes=self.rng.randint(20, 90))
        if follow.hour < 20:
            facts.append(
                self.mail(
                    case,
                    ct,
                    follow,
                    self.rng.choice(self.p.team),
                    "informer",
                    form="directe",
                    meta={"kind": "incident_suivi", "incident": incident},
                )
            )
        self._resolve(facts, done)
        nxt = self._later(day, 1)
        if nxt:
            facts.append(
                self.mail(
                    case,
                    ct,
                    self._moment(nxt, 9.0, 16.0),
                    self.rng.choice(self.p.team),
                    "informer",
                    form="directe",
                    meta={"kind": "incident_clos", "incident": incident},
                )
            )
        self._link(facts)

    def relecture(self, day: date) -> None:
        case, ct = self._case(), "relecture"
        obj = self._obj(REVIEW_KINDS)
        who = self.rng.choice(self.p.team + self.p.peers[:1])
        t0 = self._moment(day)
        done = self._handled(t0, 0.8)
        facts = [
            self.mail(
                case,
                ct,
                t0,
                who,
                "relire",
                obj=obj,
                tone=self._tone(0.08, 0.3),
                deadline=self._deadline(day, 0.35),
            )
        ]
        if self.rng.random() < 0.7:
            facts.append(
                self.add(
                    source="file",
                    at=t0 - timedelta(minutes=10),
                    author=who,
                    case_id=case,
                    case_type=ct,
                    obj=obj,
                    meta={"review": True, "version": self.rng.randint(2, 5)},
                )
            )
        self._resolve(facts, done)
        if done:
            facts.append(self.reply(case, ct, done, who, facts[0]))
        self._link(facts)

    def demande_info(self, day: date) -> None:
        case, ct = self._case(), "demande_info"
        who = self.rng.choice((self.p.boss,) + self.p.peers + self.p.team[:1])
        act = self.rng.choices(("repondre", "faire"), (7, 3))[0]
        t0 = self._moment(day)
        done = self._handled(t0)
        secondary = (
            self.rng.choice(("faire", "informer", "relire")) if self.rng.random() < 0.15 else None
        )
        facts = [
            self.mail(
                case,
                ct,
                t0,
                who,
                act,
                obj=self._obj(),
                tone=self._tone(0.18 if who is self.p.boss else 0.08, 0.12),
                deadline=self._deadline(day, 0.3),
                secondary=secondary,
                meta={"jour": fr_day(self._later(day, 1) or day)},
            )
        ]
        self._resolve(facts, done)
        if done:
            facts.append(self.reply(case, ct, done, who, facts[0]))
        self._link(facts)

    def delegation_cc(self, day: date) -> None:
        case, ct = self._case(), "delegation_cc"
        member = self.rng.choice(self.p.team)
        author = self.rng.choice((self.p.boss,) + self.p.peers)
        act = self.rng.choice(("faire", "repondre", "decider"))
        self.mail(
            case,
            ct,
            self._moment(day),
            author,
            act,
            obj=self._obj(),
            tone=self._tone(0.12, 0.1),
            to_manager=False,
            addressee=member,
            recipients=[member],
            meta={"tu": True},
        )

    def tache_perso(self, day: date) -> None:
        case, ct = self._case(), "tache_perso"
        act = self.rng.choices(
            ("faire", "repondre", "preparer", "decider", "relire"), (5, 2, 2, 1, 1)
        )[0]
        created = self._later(day, -self.rng.randint(0, 3)) or self.days[0]
        due_day = max(self._later(day, self.rng.randint(-2, 3)) or day, created)
        meta = {
            "origin": "todo",
            "completed": False,
            "due": at(due_day, self.rng.choice((12.0, 17.0, 18.0)))
            if self.rng.random() < 0.8
            else None,
            "qui": self._qui().name,
            "reunion": self.rng.choice(self.p.metier.meetings),
        }
        t0 = self._moment(created, 8.0, 19.0)
        task = self.add(
            source="task",
            at=t0,
            author=self.p.manager,
            case_id=case,
            case_type=ct,
            act=act,
            obj=self._obj(),
            tone="urgent" if self.rng.random() < 0.1 else "normal",
            meta=meta,
        )
        task.resolved_at = self._handled(max(t0, at(day, 8.0)))

    def signature(self, day: date) -> None:
        case, ct = self._case(), "signature"
        t0 = self._moment(day)
        if self.rng.random() < 0.5:
            (title, body), tid = P.pick(P.HR_DECIDE, self.rng, self.reserved_ok, "hr_decide")
            member = self.rng.choice(self.p.team)
            fact = self.mail(
                case,
                ct,
                t0,
                self.p.automates[1],
                "decider",
                form="directe",
                meta={
                    "kind": "fixed",
                    "title": title,
                    "template": tid,
                    "body": body.format(
                        qui=member.name,
                        d1=fr_day(self._later(day, 5) or day),
                        d2=fr_day(self._later(day, 7) or day),
                    ),
                },
            )
        else:
            n = self.rng.randint(2, 6)
            fact = self.add(
                source="task",
                at=t0,
                author=self.rng.choice(self.p.peers),
                case_id=case,
                case_type=ct,
                act="decider",
                obj=None,
                meta={
                    "origin": "planner",
                    "completed": False,
                    "title": f"Signer les bons de commande en attente ({n})",
                    "due": at(day, 18.0) if self.rng.random() < 0.6 else None,
                },
            )
        fact.resolved_at = self._handled(t0)

    # --- fond et pièges ----------------------------------------------------------------------

    def recurring(self) -> None:
        weekday = self.rng.randint(0, 4)
        hour = self.rng.choice((9.0, 9.5, 14.0))
        series = f"serie-{self.p.key}"
        for day in self.days:
            if day.weekday() != weekday:
                continue
            start = at(day, hour)
            visible = self._later(day, -3) or self.days[0]
            self.add(
                source="calendar",
                at=at(visible, 7.0),
                author=self.rng.choice(self.p.team),
                case_id=f"{self.p.key}-recurrent",
                case_type="fond",
                meta={
                    "start": start,
                    "end": start + timedelta(minutes=45),
                    "topic": self.p.metier.recurring,
                    "response": "accepted",
                    "series": series,
                    "attendees": [self.p.manager, *self.p.team],
                    "variant": "routine",
                },
            )

    def background(self, day: date, load: float) -> None:
        r = self.rng
        case = f"{self.p.key}-fond"
        for _ in range(round(r.randint(1, 3) * load)):
            (title, body), tid = P.pick(P.NEWSLETTER, r, self.reserved_ok, "newsletter")
            title = title.format(
                org=self.p.metier.company,
                mois="novembre",
                sem=day.isocalendar().week,
                annee=day.year + 1,
            )
            self.mail(
                case,
                "fond",
                self._moment(day, 6.0, 11.0),
                self.p.automates[3],
                "rien",
                form="directe",
                thread_id=None,
                meta={"kind": "fixed", "title": title, "body": body, "template": tid},
            )
        for _ in range(round(r.randint(5, 9) * load)):
            who = r.choice(self.p.team + self.p.peers)
            act = r.choices(("informer", "rien"), (3, 1))[0]
            self.mail(
                case,
                "fond",
                self._moment(day),
                who,
                act,
                obj=self._obj(),
                thread_id=f"thr-{self.p.key}-info-{len(self.facts)}",
                meta={"jour": fr_day(self._later(day, 1) or day)},
            )
        if r.random() < 0.6 * load:
            title, body, act, tone = P.pick(P.TEAMS, r, self.reserved_ok, "teams")[0]
            tid = f"teams.{P.TEAMS.index((title, body, act, tone))}"
            human = r.choice(self.p.team + self.p.peers)
            self.mail(
                case,
                "fond",
                self._moment(day),
                self.p.automates[0],
                act,
                tone=tone,
                form="directe",
                thread_id=None,
                meta={
                    "kind": "fixed",
                    "template": tid,
                    "human": human.name,
                    "title": title.format(qui=human.name, canal=r.choice(self.p.metier.meetings)),
                    "body": body.format(qui=human.name, prenom=self.p.manager.first),
                },
            )
        if r.random() < 0.35 * load:
            title, body, act = P.pick(P.TICKET, r, self.reserved_ok, "ticket")[0]
            n = r.randint(10000, 99999)
            subject = r.choice(P.TICKET_SUBJECTS)
            self.mail(
                case,
                "fond",
                self._moment(day),
                self.p.automates[2],
                act,
                form="directe",
                thread_id=None,
                meta={
                    "kind": "fixed",
                    "title": title.format(n=n),
                    "body": body.format(sujet=subject),
                    "template": f"ticket.{P.TICKET.index((title, body, act))}",
                },
            )
        if r.random() < 0.25 * load:
            info = r.random() < 0.5
            (title, body), tid = P.pick(P.HR_INFO, r, self.reserved_ok, "hr_info")
            if not info:
                (title, body), tid = P.pick(P.HR_DO, r, self.reserved_ok, "hr_do")
            act = ("rien" if "bulletin" in title else "informer") if info else "faire"
            deadline = None if info else at(self._later(day, 5) or self.days[-1], 18.0)
            self.mail(
                case,
                "fond",
                self._moment(day),
                self.p.automates[1],
                act,
                form="directe",
                thread_id=None,
                deadline=deadline,
                meta={
                    "kind": "fixed",
                    "title": title,
                    "template": tid,
                    "body": body.format(
                        mois="novembre",
                        quand=" avant le " + fr_day(deadline.date()) if deadline else "",
                    ),
                },
            )
        if r.random() < 0.5 * load:
            self.add(
                source="file",
                at=self._moment(day),
                author=self.p.manager,
                case_id=case,
                case_type="fond",
                obj=self._obj(),
                meta={"by_me": True, "version": r.randint(1, 6)},
            )
        if r.random() < 0.35 * load:
            self.add(
                source="file",
                at=self._moment(day),
                author=r.choice(self.p.peers),
                case_id=case,
                case_type="fond",
                obj=None,
                meta={
                    "cr": True,
                    "topic": r.choice(self.p.metier.meetings),
                    "meeting_day": self._later(day, -1) or day,
                },
            )
        if r.random() < 0.15 * load:
            meet_day = self._later(day, r.randint(0, 3)) or day
            start = at(meet_day, r.choice((10.0, 11.0, 15.0)))
            self.add(
                source="calendar",
                at=self._moment(day),
                author=r.choice(self.p.peers),
                case_id=case,
                case_type="fond",
                obj=self._obj(),
                meta={
                    "start": start,
                    "end": start + timedelta(hours=1),
                    "topic": r.choice(self.p.metier.meetings),
                    "cancelled": True,
                    "response": "accepted",
                    "attendees": [self.p.manager],
                    "variant": "invite",
                },
            )
        if r.random() < 0.15 * load:
            self.add(
                source="task",
                at=self._moment(day),
                author=self.p.manager,
                case_id=case,
                case_type="fond",
                act=r.choice(("faire", "repondre")),
                obj=self._obj(),
                meta={"origin": "todo", "completed": True, "due": None, "qui": self._qui().name},
            )

    def traps(self, day: date) -> None:
        r = self.rng
        case = f"{self.p.key}-pieges"
        stranger = Person(
            "Service facturation",
            f"facturation@{r.choice(('paiements-rapides', 'compta-express', 'secure-pay'))}.test",
            "externe_inconnu",
            "expéditeur inconnu",
        )
        if r.random() < 0.22:
            (title, body), tid = P.pick(P.INJECTION, r, self.reserved_ok, "injection")
            self.mail(
                case,
                "piege",
                self._moment(day, 0.0, 23.0),
                stranger,
                "rien",
                form="directe",
                alert=True,
                trap="injection",
                thread_id=None,
                meta={
                    "kind": "fixed",
                    "title": title,
                    "template": tid,
                    "body": body.format(mal=f"archive@{stranger.email.split('@')[1]}"),
                },
            )
        if r.random() < 0.08:
            (title, body), tid = P.pick(P.PHISHING, r, self.reserved_ok, "phishing")
            self.mail(
                case,
                "piege",
                self._moment(day, 0.0, 23.0),
                stranger,
                "faire",
                tone="urgent",
                form="directe",
                alert=True,
                trap="hameconnage",
                thread_id=None,
                meta={"kind": "fixed", "title": title, "body": body, "template": tid},
            )
        mails = [
            f
            for f in self.facts
            if f.source == "mail"
            and not f.sent
            and f.trap is None
            and f.meta.get("kind") is None
            and f.at.date() < day
        ]
        if mails and r.random() < 0.12:
            original = r.choice(mails)
            copy = self.mail(
                original.case_id,
                original.case_type,
                self._moment(day),
                original.author,
                original.act,
                obj=original.obj,
                tone=original.tone,
                form=original.form,
                trap="doublon",
                thread_id=original.thread_id,
                meta=dict(original.meta, duplicate_of=original.item_id),
            )
            copy.links = [original.item_id]
        if r.random() < 0.08:
            meeting = r.choice(self.p.metier.meetings)
            who = r.choice(self.p.peers)
            first = self.mail(
                case,
                "piege",
                self._moment(day, 8.0, 12.0),
                who,
                "informer",
                form="directe",
                trap="contradiction",
                meta={"kind": "contradiction", "meeting": meeting, "step": 1},
            )
            second = self.mail(
                case,
                "piege",
                self._moment(day, 12.5, 18.0),
                who,
                "informer",
                form="directe",
                trap="contradiction",
                thread_id=first.thread_id,
                meta={"kind": "contradiction", "meeting": meeting, "step": 2},
            )
            self._link([first, second])

    # --- la simulation -----------------------------------------------------------------------

    def run(self) -> list[Fact]:
        self.recurring()
        methods = [getattr(self, name) for name in CASE_TYPES]
        for day in self.days:
            r = self.rng.random()
            load = 0.3 if r < 0.1 else 2.0 if r < 0.2 else 1.0  # journée calme, surchargée
            for _ in range(round(self.rng.randint(8, 11) * load)):
                self.rng.choices(methods, CASE_WEIGHTS)[0](day)
            self.background(day, load)
            self.traps(day)
        return sorted(self.facts, key=lambda f: (f.at, f.item_id))
