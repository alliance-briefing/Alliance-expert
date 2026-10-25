"""Journées générées (SPEC.md, niveau 2) : la boîte de cinq managers sur trois semaines.

Chaque persona reçoit des affaires qui vivent (ouverture, relance si pas de réponse, réponse
de la manager envoyée depuis sa boîte, clôture), un flux de fond (informations, demandes à des
tiers, engagements…), des automates (rappels RH, tickets, alertes), du bruit (lettres
d'information) et des invitations avec leur événement d'agenda. Sortie : des Item du contrat
A, validés, un fichier par persona dans `scoring/annotations/synth_days/`.

Puis 160 paires de deux mails du même matin (32 par persona), dont 80 réservées à la mesure.
Annotation à l'aveugle : la carte montre le mail, la fonction de l'expéditeur, l'ancienneté et
le fil, jamais les intentions calculées par un modèle.

`uv run --group embeddings python -m packages.analyse.scoring.ml.synth.days`
"""

from __future__ import annotations

import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from contracts.models import Item
from packages.analyse.clock import PARIS
from packages.analyse.scoring.ml.annotate import SYNTH_PAIRS_PATH as PAIRS_SYNTH_PATH
from packages.analyse.scoring.ml.graph_convert import GraphDataset, dataset_from_items
from packages.analyse.scoring.ml.persona import Persona
from packages.analyse.scoring.ml.sample import ANNOTATIONS, write_sample
from packages.analyse.scoring.ml.synth import phrases as P
from packages.analyse.scoring.ml.synth.generator import (
    ACT_WEIGHTS,
    DEFAULT_SENDERS,
    SENDERS_BY_ACT,
    TITLE_KIND,
    TITLES,
    _cap,
    _contract,
    _Mail,
)
from packages.analyse.scoring.ml.synth.sectors import SECTORS, Sector
from packages.collect.core.text import normalize_whitespace, strip_quotes_and_signature, truncate

DAYS_DIR = ANNOTATIONS / "synth_days"
START = date(2026, 11, 2)  # un lundi
N_WORKING_DAYS = 15
N_CASES = 20
PAIRS_PER_PERSONA, HELDOUT_PER_PERSONA = 32, 16
CONNECTOR = {"name": "synth_lot2", "version": "0.2.0"}
MANAGER_LAST = {
    "conseil": "Vidal",
    "btp": "Mansouri",
    "clinique": "Royer",
    "ecommerce": "Tanguy",
    "collectivite": "Pelletier",
}
MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)
WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
OPENING_ACTS = (
    "demande explicite",
    "demande explicite",
    "demande implicite",
    "question",
    "proposition de créneau",
)
BACKGROUND_ACTS = {a: w for a, w in ACT_WEIGHTS.items() if a not in ("relance", "invitation")}
IT_SUBJECTS = ("Accès VPN", "Imprimante du 2e étage", "Messagerie mobile", "Licence logicielle")
IT_STATUSES = ("pris en charge", "résolu", "en attente de votre réponse", "transmis au prestataire")
NEWSLETTER_TOPICS = (
    "les tendances du secteur",
    "les nouvelles obligations réglementaires",
    "les chiffres du trimestre",
    "nos prochains webinaires",
)


def slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.replace("Mme ", "")).encode("ascii", "ignore")
    return re.sub(r"[^a-z0-9]+", ".", text.decode().lower()).strip(".")


def fr_date(day: date) -> str:
    return f"{WEEKDAYS[day.weekday()]} {day.day} {MONTHS[day.month - 1]}"


def working_days(start: date, count: int) -> list[date]:
    out, day = [], start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


@dataclass(frozen=True)
class Contact:
    name: str
    email: str
    kind: str  # type d'expéditeur (persona.py)
    role: str  # fonction affichée sur la carte, comme dans une vraie boîte


def directory(sector: Sector) -> tuple[Contact, list[Contact]]:
    """(la manager, ses correspondants) avec des adresses `prenom.nom@domaine.test`."""
    d = sector.domain
    first, title = sector.manager
    manager = Contact(
        f"{first} {MANAGER_LAST[sector.key]}",
        f"{slug(first)}.{slug(MANAGER_LAST[sector.key])}@{d}",
        "manager",
        title,
    )
    people = [Contact(sector.boss[0], f"{slug(sector.boss[0])}@{d}", "hiérarchie", sector.boss[1])]
    people += [
        Contact(n, f"{slug(n)}@{d}", "direction", f"Direction, {sector.company}")
        for n in sector.peers
    ]
    people += [
        Contact(n, f"{slug(n)}@{d}", "collègue", f"Équipe, {sector.company}") for n in sector.team
    ]
    people += [
        Contact(n, f"{slug(n)}@{slug(org)}.test", "client", f"Client, {org}")
        for n, org in sector.clients
    ]
    people += [
        Contact(n, f"{slug(n)}@{slug(org)}.test", "fournisseur", f"Fournisseur, {org}")
        for n, org in sector.suppliers
    ]
    return manager, people


def persona(sector: Sector) -> Persona:
    manager, people = directory(sector)
    return Persona(
        address=manager.email,
        internal_domain=sector.domain,
        hierarchy=frozenset(p.email for p in people if p.kind == "hiérarchie"),
        direction=frozenset(p.email for p in people if p.kind == "direction"),
        client_domains=frozenset(p.email.split("@")[1] for p in people if p.kind == "client"),
    )


class World:
    """Le monde d'une persona : une chronologie de mails et d'événements, puis des Item."""

    def __init__(self, sector: Sector, seed: int = 0) -> None:
        self.sector = sector
        self.rng = random.Random(f"synth-days:{sector.key}:{seed}")
        self.manager, self.people = directory(sector)
        self.owner = f"u_synth_{sector.key}"
        self.days = working_days(START, N_WORKING_DAYS)
        self.items: list[Item] = []

    # --- écriture ----------------------------------------------------------------------------

    def _writer(
        self,
        act: str,
        who: Contact,
        obj: tuple[str, tuple[str, ...]] | None = None,
        deadline: date | None = None,
    ) -> _Mail:
        w = _Mail(self.rng, self.sector, act)
        w.kind, w.sender, w.role = who.kind, who.name, who.role
        w.tu = who.kind in ("hiérarchie", "direction", "collègue")
        if obj:
            w.slots |= {"o": obj[0], "O": _cap(obj[0]), "v": self.rng.choice(obj[1])}
        if deadline:
            form = self.rng.choice(
                ("avant le {d}", "d'ici le {d}", "au plus tard le {d}", "pour le {d}")
            )
            w.slots["quand"] = form.format(d=fr_date(deadline))
        return w

    def _sentence(self, w: _Mail, act: str) -> str:
        if act == "relance":
            demand = w.fill(self.rng.choice(P.BANK["demande explicite"]))
            return w.fill(self.rng.choice(P.BANK["relance"])) + demand[:1].lower() + demand[1:]
        return w.fill(self.rng.choice(P.BANK[act]))

    def _body(self, w: _Mail, sentence: str, request: bool) -> str:
        rng = self.rng
        lines = []
        greeting = rng.choice(P.GREETINGS_TU if w.tu else P.GREETINGS_VOUS)
        if greeting:
            lines.append(greeting.format(prenom=self.sector.manager[0]))
        lines.append(sentence)
        closing = rng.choice(P.CLOSINGS_REQUEST if request else P.CLOSINGS)
        if closing:
            lines.append(closing)
        if rng.random() < 0.7:
            lines.append(f"{w.sender}\n{w.role}" if rng.random() < 0.5 else w.sender.split()[0])
        return _contract("\n\n".join(lines))

    def _add_mail(
        self,
        moment: datetime,
        sender: Contact,
        title: str,
        body: str,
        thread: str,
        to: Contact | None = None,
        cc: Contact | None = None,
        correlation: str | None = None,
    ) -> Item:
        n = len(self.items)
        to = to or self.manager
        people = [
            {"name": sender.name, "email": sender.email, "role": "from"},
            {"name": to.name, "email": to.email, "role": "to"},
        ]
        if cc:
            people.append({"name": cc.name, "email": cc.email, "role": "cc"})
        clean = strip_quotes_and_signature(normalize_whitespace(body)) or body
        item = Item(
            item_id=f"mail:{self.sector.key}-{n:04d}",
            source="mail",
            provider="demo",
            owner_user_id=self.owner,
            title=title,
            snippet=truncate(re.sub(r"\s+", " ", clean).strip()),
            content_text=clean,
            created_at=moment,
            updated_at=moment,
            participants=people,
            thread_id=thread,
            correlation_id=correlation,
            url=f"https://mail.{self.sector.key}.test/m/{n}",
            acl={"visible_to": [self.owner]},
            collected_at=moment,
            connector=CONNECTOR,
        )
        self.items.append(item)
        return item

    def _moment(self, day: date, start: float = 8.0, end: float = 18.5) -> datetime:
        hour = self.rng.uniform(start, end)
        return datetime.combine(day, time(int(hour), int(hour % 1 * 60)), tzinfo=PARIS)

    def _contact(self, kinds: tuple[str, ...]) -> Contact:
        return self.rng.choice([p for p in self.people if p.kind in kinds])

    # --- simulation --------------------------------------------------------------------------

    def _case(self, n: int) -> None:
        rng, days = self.rng, self.days
        start = rng.randrange(len(days) - 4)
        act = rng.choice(OPENING_ACTS)
        who = self._contact(DEFAULT_SENDERS)
        obj = rng.choice(self.sector.objects)
        deadline = (
            days[min(start + rng.randint(1, 4), len(days) - 1)] if rng.random() < 0.5 else None
        )
        w = self._writer(act, who, obj, deadline)
        title = w.fill(rng.choice(TITLES[TITLE_KIND[act]]))
        thread = f"{self.sector.key}-affaire-{n}"
        self._add_mail(
            self._moment(days[start]),
            who,
            title,
            self._body(w, self._sentence(w, act), True),
            thread,
        )
        if rng.random() < 0.65:  # la manager répond (mail envoyé) : « déjà répondu »
            reply_day = days[min(start + rng.randint(0, 1), len(days) - 1)]
            r = self._writer("engagement de l'expéditeur", self.manager, obj)
            r.tu = who.kind in ("hiérarchie", "direction", "collègue")
            self._add_mail(
                self._moment(reply_day, 9, 19),
                self.manager,
                f"RE : {title}",
                self._body(r, self._sentence(r, "engagement de l'expéditeur"), False),
                thread,
                to=who,
            )
            if rng.random() < 0.6:
                closing_act = rng.choice(("confirmation", "remerciement"))
                c = self._writer(closing_act, who, obj)
                self._add_mail(
                    self._moment(days[min(start + rng.randint(2, 4), len(days) - 1)]),
                    who,
                    f"RE : {title}",
                    self._body(c, self._sentence(c, closing_act), False),
                    thread,
                )
        elif act != "proposition de créneau" and rng.random() < 0.7:  # relance
            r = self._writer("relance", who, obj, deadline)
            self._add_mail(
                self._moment(days[min(start + rng.randint(2, 3), len(days) - 1)]),
                who,
                f"RE : {title}",
                self._body(r, self._sentence(r, "relance"), True),
                thread,
            )

    def _background(self, day: date, n: int) -> None:
        rng = self.rng
        act = rng.choices(list(BACKGROUND_ACTS), weights=list(BACKGROUND_ACTS.values()))[0]
        who = self._contact(SENDERS_BY_ACT.get(act, DEFAULT_SENDERS))
        w = self._writer(act, who)
        title = w.fill(rng.choice(TITLES[TITLE_KIND[act]]))
        tiers = None
        if act == "demande à un tiers":
            tiers = next((p for p in self.people if p.name.split()[0] == w.slots["tiers"]), None)
        self._add_mail(
            self._moment(day),
            who,
            title,
            self._body(w, self._sentence(w, act), act.startswith("demande")),
            f"{self.sector.key}-fond-{n}",
            to=tiers,
            cc=self.manager if tiers else None,
        )

    def _automates(self, day: date, n: int) -> None:
        rng, d = self.rng, self.sector.domain
        kind = rng.choices(("rh", "ticket", "releve", "alerte"), weights=(2, 4, 2, 1))[0]
        if kind == "rh":
            sender = Contact("Portail RH", f"noreply@rh.{d}", "automate", "Portail RH")
            due = self.days[min(self.days.index(day) + rng.randint(2, 6), len(self.days) - 1)]
            title = rng.choice(
                ("Rappel : validation des notes de frais", "Entretiens annuels : saisie")
            )
            body = (
                f"Ceci est un rappel automatique : la saisie est ouverte jusqu'au {fr_date(due)}. "
                "Merci de valider les demandes de votre équipe."
            )
        elif kind == "ticket":
            sender = Contact(
                "Support informatique", f"support@it.{d}", "automate", "Support informatique"
            )
            status = rng.choice(IT_STATUSES)
            title = f"[INC-{rng.randint(10000, 99999)}] {rng.choice(IT_SUBJECTS)} ({status})"
            body = f"Le statut de votre ticket a changé : {status}. Ce message est généré automatiquement."
        else:
            bank = f"banque-{self.sector.key}.test"
            sender = Contact("Banque", f"alertes@{bank}", "automate", "Banque")
            if kind == "alerte":
                title = "Alerte : mouvement inhabituel"
                body = (
                    f"Un mouvement inhabituel de {rng.randint(2, 9)} {rng.randint(100, 999)} € a "
                    "été détecté sur le compte professionnel. Contactez votre conseiller si vous "
                    "ne le reconnaissez pas."
                )
            else:
                title, body = "Relevé de compte disponible", "Votre relevé mensuel est disponible."
        self._add_mail(self._moment(day, 6, 9), sender, title, body, f"{self.sector.key}-auto-{n}")

    def _newsletter(self, day: date, n: int) -> None:
        source = f"Veille {_cap(self.sector.key)}"
        sender = Contact(source, f"newsletter@veille-{self.sector.key}.test", "automate", source)
        topic = self.rng.choice(NEWSLETTER_TOPICS)
        body = (
            f"Au sommaire cette semaine : {topic}.\n\nPour vous désabonner de cette lettre, "
            "cliquez ici."
        )
        self._add_mail(
            self._moment(day, 6, 8),
            sender,
            f"{source} n° {100 + n} : {topic}",
            body,
            f"{self.sector.key}-news-{n}",
        )

    def _invitation(self, day: date, n: int) -> None:
        rng = self.rng
        who = self._contact(("hiérarchie", "direction", "collègue"))
        meet_day = self.days[min(self.days.index(day) + rng.randint(1, 4), len(self.days) - 1)]
        start = datetime.combine(meet_day, time(rng.choice((9, 10, 14, 16))), tzinfo=PARIS)
        w = self._writer("invitation", who)
        w.slots |= {"jour": fr_date(meet_day), "heure": f"{start.hour} h"}
        correlation = f"meet:{self.sector.key}-{n}"
        title = f"Invitation : {w.slots['reunion']} – {fr_date(meet_day)}"
        self._add_mail(
            self._moment(day),
            who,
            _contract(_cap(title)),
            self._body(w, self._sentence(w, "invitation"), False),
            f"{self.sector.key}-invit-{n}",
            correlation=correlation,
        )
        self.items.append(
            Item(
                item_id=f"calendar:{self.sector.key}-{n:03d}",
                source="calendar",
                provider="demo",
                owner_user_id=self.owner,
                title=_cap(w.slots["reunion"]),
                snippet="",
                created_at=self._moment(day),
                updated_at=self._moment(day, 18.6, 19),
                start_at=start,
                end_at=start + timedelta(hours=1),
                participants=[{"name": who.name, "email": who.email, "role": "organizer"}],
                correlation_id=correlation,
                url=f"https://agenda.{self.sector.key}.test/e/{n}",
                acl={"visible_to": [self.owner]},
                collected_at=start,
                connector=CONNECTOR,
            )
        )

    def build(self) -> list[Item]:
        rng = self.rng
        for n in range(N_CASES):
            self._case(n)
        counter = 0
        for day in self.days:
            for _ in range(rng.randint(3, 6)):
                self._background(day, counter)
                counter += 1
            for _ in range(rng.randint(1, 3)):
                self._automates(day, counter)
                counter += 1
            for _ in range(rng.randint(1, 2)):
                self._newsletter(day, counter)
                counter += 1
            if rng.random() < 0.5:
                self._invitation(day, counter)
                counter += 1
        return sorted(self.items, key=lambda i: (i.created_at, i.item_id))


def write_days(items: list[Item], key: str) -> Path:
    DAYS_DIR.mkdir(parents=True, exist_ok=True)
    path = DAYS_DIR / f"{key}.jsonl"
    lines = (json.dumps(i.model_dump(mode="json"), ensure_ascii=False) for i in items)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def load_days(key: str) -> list[Item]:
    path = DAYS_DIR / f"{key}.jsonl"
    return [Item.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def synth_dataset(sector: Sector) -> GraphDataset:
    return dataset_from_items(load_days(sector.key), persona(sector).address)


# --- paires à l'aveugle ----------------------------------------------------------------------

_FACT_CHIPS = ("reçu il y a", "fil de", "déjà répondu")


def blind(side: dict, roles: dict[str, str], items: dict[str, Item]) -> dict:
    """La carte d'une annotation à l'aveugle : fonction de l'expéditeur et faits visibles dans
    une vraie boîte (ancienneté, fil, réponse déjà envoyée), sans intention calculée."""
    sender = next(p for p in items[side["item_id"]].participants if p.role == "from")
    facts = [c for c in side["chips"] if c.startswith(_FACT_CHIPS)]
    who = roles.get(sender.email, f"{sender.name} · message automatique")
    return side | {"chips": [who, *facts]}


def build_synth_pairs() -> list[dict]:
    from packages.analyse.scoring.ml.pairs import _probs, build_pairs

    out = []
    for i, sector in enumerate(SECTORS):
        dataset = synth_dataset(sector)
        items = {item.item_id: item for item in dataset.items}
        manager, people = directory(sector)
        roles = {p.email: f"{p.name} · {p.role}" for p in people}
        pairs = build_pairs(
            dataset,
            _probs(dataset),
            persona(sector),
            seed=10 + i,
            n_pairs=PAIRS_PER_PERSONA,
            n_heldout=HELDOUT_PER_PERSONA,
            prefix=f"s{i}",
        )
        for p in pairs:
            out.append(
                p
                | {
                    "a": blind(p["a"], roles, items),
                    "b": blind(p["b"], roles, items),
                    "dataset": f"synth-{sector.key}",
                    "blind": True,
                    "manager": manager.name.split()[0],
                }
            )
    return out


def main(argv: list[str]) -> int:
    for sector in SECTORS:
        items = World(sector).build()
        path = write_days(items, sector.key)
        mails = sum(i.source == "mail" for i in items)
        print(f"{sector.key:<13} {len(items)} Item ({mails} mails) → {path.name}")
    pairs = build_synth_pairs()
    write_sample(pairs, PAIRS_SYNTH_PATH)
    print(
        f"{len(pairs)} paires → {PAIRS_SYNTH_PATH.name}, "
        f"{sum(p['heldout'] for p in pairs)} réservées à la mesure"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
