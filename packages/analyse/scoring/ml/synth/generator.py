"""Mails isolés générés : actes de langage × formulations × secteur × forme (SPEC.md, niveau 1).

Chaque mail porte ses étiquettes par construction (`action`, `reponse` : vrai, faux, ou None
quand c'est à juger), calculées à partir des actes qu'il contient. Elles ne sont jamais
montrées aux annotateurs : leurs réponses, comparées à la construction, disent si le
générateur fait ce qu'il prétend. Déterministe : même graine, mêmes mails.

`uv run python -m packages.analyse.scoring.ml.synth.generator` écrit
`scoring/annotations/synth_sample.jsonl` (300 mails, versionné : données fictives).
"""

from __future__ import annotations

import random
import re
import sys
import unicodedata

from packages.analyse.scoring.ml.annotate import SYNTH_PATH
from packages.analyse.scoring.ml.sample import TEXT_MAX_CHARS, write_sample
from packages.analyse.scoring.ml.synth import phrases as P
from packages.analyse.scoring.ml.synth.sectors import SECTORS, Sector
from packages.collect.core.text import normalize_whitespace, strip_quotes_and_signature

N_MAILS = 300
AUTHOR = "claude"
# Part de chaque acte principal : environ 1 mail sur 3 demande une action.
ACT_WEIGHTS = {
    "demande explicite": 16,
    "demande implicite": 12,
    "relance": 6,
    "demande à un tiers": 9,
    "engagement de l'expéditeur": 10,
    "question": 12,
    "proposition de créneau": 6,
    "invitation": 4,
    "information": 12,
    "confirmation": 7,
    "remerciement": 6,
}
SENDERS_BY_ACT = {
    "demande à un tiers": ("hiérarchie", "direction", "collègue"),
    "engagement de l'expéditeur": ("collègue", "client", "fournisseur"),
    "confirmation": ("collègue", "client", "fournisseur"),
    "information": ("hiérarchie", "direction", "collègue", "client", "fournisseur"),
    "remerciement": ("hiérarchie", "direction", "collègue", "client", "fournisseur"),
}
DEFAULT_SENDERS = ("hiérarchie", "direction", "collègue", "client")
TITLES = {
    "demande": ("Validation – {o}", "À traiter : {o}", "{O}", "Urgent", "Demande", "{Projet}"),
    "question": ("Question rapide", "{Projet}", "Une question", "{O}"),
    "créneau": ("Disponibilités", "Invitation : {reunion}", "{Reunion}"),
    "info": ("Info – {projet}", "Pour info", "{O}", "{Projet}"),
    "suite": ("{O}", "{Projet}", "Point rapide", "Suite à ton message"),
}
TITLE_KIND = {
    "demande explicite": "demande",
    "demande implicite": "demande",
    "relance": "demande",
    "demande à un tiers": "demande",
    "question": "question",
    "proposition de créneau": "créneau",
    "invitation": "créneau",
    "information": "info",
    "engagement de l'expéditeur": "suite",
    "confirmation": "suite",
    "remerciement": "suite",
}
ENGLISH_OBJECTS = ("the report", "the quote", "the contract", "the schedule", "the invoice")
_CHOICE = re.compile(r"\[([^|\]]*)\|([^\]]*)\]")
_CONTRACTIONS = (
    (" à le ", " au "),
    (" à les ", " aux "),
    (" de le ", " du "),
    (" de les ", " des "),
    ("(à le ", "(au "),
    ("(de le ", "(du "),
)


def _contract(text: str) -> str:
    for long, short in _CONTRACTIONS:
        text = text.replace(long, short)
    return text


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _register(template: str, tu: bool) -> str:
    return _CHOICE.sub(lambda m: m.group(1) if tu else m.group(2), template)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def labels(acts: list[str]) -> dict[str, bool | None]:
    """« ou » des actes (vrai l'emporte, puis « à juger », puis faux), puis annulations."""
    out: dict[str, bool | None] = {}
    for i, feature in enumerate(("action", "reponse")):
        values = [P.ACTS[a][i] for a in acts if a in P.ACTS]
        out[feature] = True if True in values else (None if None in values else False)
    for modifier, feature in P.CANCELS.items():
        if modifier in acts:
            out[feature] = False
    return out


class _Mail:
    """Les choix d'un mail (secteur, expéditeur, objet, dates) et le remplissage des gabarits."""

    def __init__(self, rng: random.Random, sector: Sector, act: str) -> None:
        self.rng, self.sector = rng, sector
        kinds = SENDERS_BY_ACT.get(act, DEFAULT_SENDERS)
        self.kind = rng.choice(kinds)
        self.sender, self.role = self._person(self.kind)
        self.tu = self.kind in ("hiérarchie", "direction", "collègue")
        obj, verbs = rng.choice(sector.objects)
        days = rng.sample(P.DAYS, 2)
        others = [n for n in sector.team if n != self.sender]
        self.slots = {
            "o": obj,
            "O": _cap(obj),
            "v": rng.choice(verbs),
            "tiers": rng.choice(others).split()[0],
            "jour": days[0],
            "jour2": days[1],
            "heure": rng.choice(P.HOURS),
            "reunion": rng.choice(sector.meetings),
            "projet": rng.choice(sector.projects),
        }
        self.slots |= {
            "Reunion": _cap(self.slots["reunion"]),
            "Projet": _cap(self.slots["projet"]),
            "quand": rng.choice(P.DEADLINES).format(**self.slots),
            "quand_moi": rng.choice(P.SENDER_DELAYS),
            "oe": rng.choice(ENGLISH_OBJECTS),
            "quande": rng.choice(("by Friday", "today", "")),
            "daye": rng.choice(("Monday", "Thursday")),
        }

    def new_object(self) -> None:
        obj, verbs = self.rng.choice([o for o in self.sector.objects if o[0] != self.slots["o"]])
        self.slots |= {"o": obj, "O": _cap(obj), "v": self.rng.choice(verbs)}

    def _person(self, kind: str) -> tuple[str, str]:
        s, rng = self.sector, self.rng
        if kind == "hiérarchie":
            return s.boss
        if kind == "direction":
            return rng.choice(s.peers), "Direction"
        if kind == "collègue":
            return rng.choice(s.team), s.company
        name, org = rng.choice(s.clients if kind == "client" else s.suppliers)
        return name, org

    def fill(self, template: str) -> str:
        text = _register(template, self.tu).format(**self.slots)
        return _cap(re.sub(r"\s+([.,])", r"\1", re.sub(r" {2,}", " ", text)).strip())


def generate_mail(rng: random.Random, index: int, sector: Sector) -> dict:
    act = rng.choices(list(ACT_WEIGHTS), weights=list(ACT_WEIGHTS.values()))[0]
    mail = _Mail(rng, sector, act)
    english = act in P.ENGLISH and rng.random() < 0.08
    acts = [act]
    if english:
        sentences = [mail.fill(rng.choice(P.ENGLISH[act]))]
    elif act == "relance":
        demand = mail.fill(rng.choice(P.BANK["demande explicite"]))
        sentences = [mail.fill(rng.choice(P.BANK["relance"])) + demand[:1].lower() + demand[1:]]
    else:
        sentences = [mail.fill(rng.choice(P.BANK[act]))]
    if not english:
        if rng.random() < 0.25:  # un second acte, avant ou après
            extra = rng.choice(("information", "remerciement", "question", "demande explicite"))
            acts.append(extra)
            mail.new_object()  # un autre sujet : « je m'occupe de X… merci de faire X » est absurde
            sentence = mail.fill(rng.choice(P.BANK[extra]))
            sentences.insert(rng.randrange(2), sentence)
        if act in ("demande explicite", "relance") and rng.random() < 0.08:
            acts.append("annulation")
            sentences.append(mail.fill(rng.choice(P.BANK["annulation"])))
        elif act == "question" and rng.random() < 0.10:
            acts.append("sans réponse")
            sentences.append(mail.fill(rng.choice(P.BANK["sans réponse"])))
    first = mail.sender.split()[0]
    lines = []
    if not english:
        greetings = P.GREETINGS_TU if mail.tu else P.GREETINGS_VOUS
        greeting = rng.choice(greetings).format(prenom=sector.manager[0])
        if greeting:
            lines.append(greeting)
    opener = mail.fill(rng.choice(P.OPENERS)) if not english and rng.random() < 0.3 else ""
    body_lines = sentences
    if rng.random() < 0.10:  # forme dégradée : minuscules, sans accents, abréviations
        body_lines = [_strip_accents(s).lower().replace("s'il te plaît", "stp") for s in sentences]
    if opener:
        body_lines = (
            [opener, body_lines[0][:1].lower() + body_lines[0][1:]] + body_lines[1:]
            if opener.endswith(",")
            else [opener, *body_lines]
        )
    lines.append(" ".join(body_lines))
    closings = P.CLOSINGS_REQUEST if labels(acts)["action"] else P.CLOSINGS
    closing = "" if english else rng.choice(closings)
    if closing:
        lines.append(closing)
    if rng.random() < 0.65:
        lines.append(f"{mail.sender}\n{mail.role}" if rng.random() < 0.5 else first)
        if rng.random() < 0.15:
            lines.append(P.LEGAL)
    if rng.random() < 0.15:  # citation d'un message précédent
        quoted = mail.fill(rng.choice(P.BANK["question"] + P.BANK["information"]))
        lines.append(
            f"Le {mail.slots['jour']} à {mail.slots['heure']}, {sector.manager[0]} a écrit :"
            f"\n> {quoted}"
        )
    body = _contract("\n\n".join(lines))
    title = _contract(mail.fill(rng.choice(TITLES[TITLE_KIND[act]])))
    if english:
        title = rng.choice(("Quick request", "Follow-up", "FYI", "Question"))
    elif rng.random() < 0.25:
        title = rng.choice(("RE : ", "TR : ", "Re: ")) + title
    clean = strip_quotes_and_signature(normalize_whitespace(body)) or body
    return {
        "item_id": f"synth:{index:04d}",
        "sector": sector.key,
        "sender": mail.sender,
        "sender_kind": mail.kind,
        "title": title,
        "text": clean[:TEXT_MAX_CHARS],
        "raw": body,
        "acts": acts,
        "construction": labels(acts),
        "english": english,
        "author": AUTHOR,
        "round": "synth",
    }


def generate(n: int = N_MAILS, seed: int = 0) -> list[dict]:
    rng = random.Random(f"synth-lot2:{seed}")
    return [generate_mail(rng, i, SECTORS[i % len(SECTORS)]) for i in range(n)]


def main(argv: list[str]) -> int:
    mails = generate()
    write_sample(mails, SYNTH_PATH)
    acts: dict[str, int] = {}
    for m in mails:
        acts[m["acts"][0]] = acts.get(m["acts"][0], 0) + 1
    action = sum(m["construction"]["action"] is True for m in mails)
    print(f"{len(mails)} mails → {SYNTH_PATH} ; demandent une action : {action} ; actes : {acts}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
