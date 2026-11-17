"""Outil d'annotation unifié (M4 du plan) : une page locale pour les trois jugements.

- élément : action principale, urgence déclarée, alerte (grille de M0) ;
- paire d'éléments : « ce matin, lequel traiter en premier ? », toutes sources confondues ;
- paire d'affaires : la même question au niveau de l'affaire.

L'outil n'échantillonne rien : il lit une **série**, un fichier JSONL de tâches déjà prêtes
(`scoring/annotations/series/<série>.jsonl`), écrit par le tirage des jeux gelés (M5) ou par
la sélection active (M7). Chaque tâche porte ses cartes, construites ici à partir des `Item`
du contrat A (`card`, `case_card`) : seulement ce qu'un vrai connecteur exposerait, jamais
l'étiquette du simulateur ni un score de modèle (`check_blind` refuse une série qui en
contient : pas de pastille sur une carte de mesure, leçon du 26/10).

Règles reprises de l'ancien outil (`annotate.py`, figé avec l'ancien générateur) : un fichier
par annotateur, en ajout seul (fermer la page ne perd rien, la dernière réponse compte),
environ 10 % de répétitions cachées (une paire répétée revient côtés inversés), kappa entre
annotateurs, puis arbitrage des désaccords.

    uv run python -m packages.analyse.scoring.ml.annoter --creer-essai
    uv run python -m packages.analyse.scoring.ml.annoter --serie essai --annotateur theo
    uv run python -m packages.analyse.scoring.ml.annoter --accord essai
    uv run python -m packages.analyse.scoring.ml.annoter --serie essai --arbitrage

Les affaires d'une paire d'affaires sont les affaires vraies du simulateur (décision du
17/11) : on juge la priorité sans la mêler aux erreurs de regroupement, mesurées à part (M8).
"""

from __future__ import annotations

import json
import random
import re
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse

from contracts.models import Item
from packages.analyse.scoring.ml.sample import ANNOTATIONS

SERIES = ANNOTATIONS / "series"
PAGE = Path(__file__).with_name("annoter.html")
ANNOTATORS = ("theo", "paul", "guillaume")
ARBITRAGE = "arbitrage"  # nom du fichier des décisions d'arbitrage, à côté des annotateurs
KINDS = ("element", "paire", "affaires")
ACTIONS = ("repondre", "faire", "decider", "preparer", "relire", "informer", "rien")
URGENCIES = ("urgent", "normal", "sans_urgence")
PAIR_VALUES = ("A", "B", "=")  # A d'abord, B d'abord, équivalent
REPEAT_SHARE = 0.10
# Clés interdites dans une tâche : étiquettes du simulateur, sorties d'un modèle, ou indices
# qui les trahissent (type d'affaire, piège, gabarit). Une seule suffit à refuser la série.
FORBIDDEN_KEYS = frozenset(
    {
        "action",
        "urgence",
        "alerte",
        "label",
        "labels",
        "score",
        "priority",
        "priority_score",
        "chips",
        "features",
        "case_type",
        "case_id",
        "trap",
        "templates",
        "split",
        "author_kind",
        "resolved_at",
        "deadline",
    }
)


# --- Cartes : ce que voit l'annotateur -------------------------------------------------


@dataclass(frozen=True)
class Viewpoint:
    """Le manager qui lit son brief : qui il est, quand, et qui sont ses correspondants."""

    manager_email: str
    moment: datetime
    people: dict[str, str]  # adresse → « fonction (relation au manager) »
    context: str  # phrase affichée en tête de page


def _hm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def when(dt: datetime, vp: Viewpoint) -> str:
    """Date lue depuis le moment du brief : « hier à 18:05 », « jeudi 5 novembre à 10:00 »."""
    from packages.collect.trainset.world import PARIS, fr_day

    local = dt.astimezone(PARIS)
    delta = (local.date() - vp.moment.astimezone(PARIS).date()).days
    day = {0: "aujourd'hui", -1: "hier", 1: "demain"}.get(delta, fr_day(local.date()))
    return f"{day} à {_hm(local)}"


def who(email: str, name: str, vp: Viewpoint) -> str:
    if email == vp.manager_email:
        return "vous"
    role = vp.people.get(email)
    return f"{name}, {role}" if role else f"{name} <{email}>"


def _role(item: Item, *roles: str) -> list:
    return [p for p in item.participants if p.role in roles]


def _names(people: list, vp: Viewpoint, limit: int = 6) -> str:
    shown = [who(p.email, p.name, vp) for p in people[:limit]]
    more = len(people) - limit
    return "; ".join(shown) + (f" et {more} autre{'s' * (more > 1)}" if more > 0 else "")


RESPONSES = {
    "none": "pas encore répondu",
    "organizer": "vous organisez",
    "accepted": "acceptée",
    "tentative": "provisoire",
    "declined": "refusée",
}


def _place(url: str) -> str:
    """Emplacement d'un fichier lu dans son adresse (SharePoint, OneDrive)."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if "sites" in parts and parts.index("sites") + 1 < len(parts):
        return f"SharePoint, site {parts[parts.index('sites') + 1]}"
    if parsed.hostname and parsed.hostname.startswith("onedrive"):
        return "OneDrive"
    return parsed.hostname or ""


_VERSION = re.compile(r"v\d+|\d{1,2}-\d{1,2}")  # « v4 », « 04-11 » : pas le sujet du fichier
SOURCE_NAMES = {"mail": "mail", "calendar": "réunion", "task": "tâche", "file": "fichier"}


def _plain_words(text: str) -> set[str]:
    """Mots en minuscules, sans accents : « procédure » et « Procedure » se rejoignent."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return set(re.findall(r"[a-z0-9]+", ascii_text))


def file_words(title: str) -> set[str]:
    """Mots du sujet d'un fichier : « Contrat_blanchisserie_v4.pdf » → {contrat, blanchisserie}.

    Sans extension ni version ; au moins deux mots, sinon aucun (« Budget.xlsx » citerait
    tous les mails qui parlent d'un budget : mieux vaut rater un lien que d'en inventer un).
    """
    stem = title.rsplit(".", 1)[0].replace("_", " ")
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode().lower()
    words = {
        w
        for w in re.findall(r"[a-z0-9]+(?:-[0-9]+)?", ascii_stem)
        if len(w) >= 3 and not _VERSION.fullmatch(w)
    }
    return words if len(words) >= 2 else set()


def cited_in(item: Item, known: Sequence[Item], vp: Viewpoint, limit: int = 3) -> str:
    """Les mails, réunions et tâches déjà reçus qui parlent du fichier, les plus récents d'abord.

    Ce que le manager trouverait en cherchant le sujet du fichier dans sa messagerie : tous
    les mots du nom (`file_words`) dans le titre ou le texte. Les fichiers de la simulation
    ne sont cités que par leur sujet (« le contrat Blanchisserie Alpine »), jamais par leur
    nom exact : la règle de L2.1 (nom exact entre guillemets) n'y trouverait rien.
    """
    words = file_words(item.title)
    if not words:
        return ""
    hits = sorted(
        (
            other
            for other in known
            if other.source != "file"
            and other.created_at <= vp.moment
            and words <= _plain_words(f"{other.title} {other.content_text or other.snippet}")
        ),
        key=lambda other: (other.created_at, other.item_id),
        reverse=True,
    )
    shown = [
        f"{SOURCE_NAMES[h.source]} « {h.title} » ({when(h.created_at, vp)})" for h in hits[:limit]
    ]
    more = len(hits) - limit
    return "; ".join(shown) + (f" et {more} autre{'s' * (more > 1)}" if more > 0 else "")


def card(item: Item, vp: Viewpoint, known: Sequence[Item] = ()) -> dict:
    """Carte d'un élément, au format de la page (et du calibrage de M0) :
    source, titre, date, champs [libellé, valeur], texte.

    `known` : les éléments du même utilisateur déjà visibles au moment du brief ; ils
    servent à la ligne « Cité dans » d'un fichier, dont l'action dépend de ses liens (guide).
    """
    fields: list[list[str]] = []
    text = item.content_text or item.snippet
    if item.source == "mail":
        head = f"reçu {when(item.created_at, vp)}"
        sender = _role(item, "from")
        if sender:
            fields.append(["De", who(sender[0].email, sender[0].name, vp)])
        to, cc = _role(item, "to"), _role(item, "cc")
        others = [p for p in to if p.email != vp.manager_email]
        if any(p.email == vp.manager_email for p in to):
            fields.append(
                [
                    "À",
                    "vous (destinataire direct)" + (f", et {_names(others, vp)}" if others else ""),
                ]
            )
        else:
            fields.append(["À", _names(to, vp) or "—"])
            if any(p.email == vp.manager_email for p in cc):
                fields.append(["Copie", "vous"])
        if item.is_read is False:
            fields.append(["Statut", "non lu"])
        if item.native_importance == "high":
            fields.append(["Importance", "haute"])
    elif item.source == "calendar":
        cal = item.attributes.calendar if item.attributes else None
        start = item.start_at or item.created_at
        if cal and cal.is_all_day:
            head = f"{when(start, vp).split(' à ')[0]}, toute la journée"
        else:
            head = when(start, vp) + (
                f" – {_hm(item.end_at.astimezone(start.tzinfo))}" if item.end_at else ""
            )
        organizer = _role(item, "organizer")
        if organizer:
            fields.append(["Organisateur", who(organizer[0].email, organizer[0].name, vp)])
        attendees = [p for p in _role(item, "attendee") if p.email != vp.manager_email]
        if attendees:
            fields.append(["Participants", _names(attendees, vp)])
        if cal and cal.response:
            fields.append(["Votre réponse", RESPONSES[cal.response]])
        if cal and cal.is_cancelled:
            fields.append(["Statut", "annulée"])
        if cal and cal.location:
            fields.append(["Lieu", cal.location])
        elif cal and cal.online_url:
            fields.append(["Lieu", "en ligne"])
    elif item.source == "task":
        head = f"créée {when(item.created_at, vp)}"
        if item.due_at:
            late = item.due_at < vp.moment and not item.completed
            fields.append(["Échéance", when(item.due_at, vp) + (" (en retard)" if late else "")])
        fields.append(["Statut", "terminée" if item.completed else "à faire"])
        host = urlparse(str(item.url)).hostname or ""
        fields.append(
            ["Liste", "Planner (plan d'équipe)" if host.startswith("planner") else "To Do"]
        )
    else:  # file
        head = f"modifié {when(item.updated_at, vp)}"
        modifier = _role(item, "modifier")
        if modifier:
            fields.append(["Modifié par", who(modifier[0].email, modifier[0].name, vp)])
        fields.append(["Emplacement", _place(str(item.url))])
        mentions = cited_in(item, known, vp)
        if mentions:
            fields.append(["Cité dans", mentions])
    return {
        "item_id": item.item_id,
        "source": item.source,
        "title": item.title or "(sans objet)",
        "when": head,
        "fields": fields,
        "text": text or "",
    }


def _author(item: Item, vp: Viewpoint) -> str:
    for role in ("from", "organizer", "modifier", "assignee"):
        found = _role(item, role)
        if found:
            return who(found[0].email, found[0].name, vp)
    return ""


def case_card(items: list[Item], vp: Viewpoint) -> dict:
    """Carte d'une affaire : un titre et ses éléments visibles, du plus ancien au plus récent."""
    items = sorted(items, key=lambda i: (i.created_at, i.item_id))
    mails = [i for i in items if i.source == "mail" and i.title]
    title = (mails[0] if mails else items[0]).title
    for prefix in ("RE : ", "RE: ", "TR : ", "TR: ", "Re : ", "Re: "):
        title = title.removeprefix(prefix)
    rows = []
    for item in items:
        full = card(item, vp)
        rows.append(
            {
                "item_id": item.item_id,
                "source": item.source,
                "title": full["title"],
                "when": full["when"],
                "who": _author(item, vp),
                "extract": full["text"][:220],
            }
        )
    return {"title": title or "(sans titre)", "items": rows}


def check_blind(task: object, path: str = "tâche") -> None:
    """Refuse toute clé d'étiquette ou de modèle, à n'importe quelle profondeur."""
    if isinstance(task, dict):
        for key, value in task.items():
            if key in FORBIDDEN_KEYS:
                raise ValueError(f"{path} : clé interdite sur une carte de mesure : {key!r}")
            check_blind(value, f"{path}.{key}")
    elif isinstance(task, list):
        for n, value in enumerate(task):
            check_blind(value, f"{path}[{n}]")


# --- Séries, ordre et réponses ---------------------------------------------------------


def load_series(path: Path) -> list[dict]:
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    seen: set[str] = set()
    for task in tasks:
        task_id, kind = task.get("task_id"), task.get("kind")
        if not task_id or task_id in seen:
            raise ValueError(f"{path.name} : task_id absent ou en double : {task_id!r}")
        if kind not in KINDS:
            raise ValueError(f"{path.name} : sorte de jugement inconnue : {kind!r}")
        if ("card" not in task) if kind == "element" else ("a" not in task or "b" not in task):
            raise ValueError(f"{path.name} : tâche {task_id} incomplète")
        check_blind(task, task_id)
        seen.add(task_id)
    return tasks


def write_series(tasks: Iterable[dict], path: Path) -> None:
    tasks = list(tasks)
    for task in tasks:
        check_blind(task, task.get("task_id", "tâche"))
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    path.write_text(text, encoding="utf-8", newline="\n")


def order(tasks: list[dict], annotator: str, series: str) -> list[tuple[str, bool]]:
    """Ordre fixe propre à l'annotateur : (task_id, est_une_répétition).

    Seules les tâches qui lui sont destinées (champ « pour » absent ou qui le cite). Environ
    10 % de la première moitié revient, sans l'annoncer, dans la seconde moitié.
    """
    rng = random.Random(f"{series}:{annotator}")
    ids = [t["task_id"] for t in tasks if annotator in t.get("pour", ANNOTATORS)]
    rng.shuffle(ids)
    half = len(ids) // 2
    repeats = rng.sample(ids[:half], round(len(ids) * REPEAT_SHARE)) if half else []
    out: list[tuple[str, bool]] = [(task_id, False) for task_id in ids]
    for n, task_id in enumerate(repeats):
        out.insert(half + n * (len(out) - half) // len(repeats) + n, (task_id, True))
    return out


def check_answer(kind: str, answer: object) -> dict:
    # Une réponse qui n'est pas un objet échoue aux contrôles ci-dessous (ValueError : le
    # serveur la renvoie en 400, comme toute réponse invalide).
    answer = answer if isinstance(answer, dict) else {}
    if kind == "element":
        action, urgency, alert = answer.get("action"), answer.get("urgence"), answer.get("alerte")
        if action not in ACTIONS or urgency not in URGENCIES or not isinstance(alert, bool):
            raise ValueError("réponse invalide : action, urgence et alerte attendues")
        return {"action": action, "urgence": urgency, "alerte": alert}
    if answer.get("choix") not in PAIR_VALUES:
        raise ValueError("réponse invalide : choix A, B ou = attendu")
    return {"choix": answer["choix"]}


def swap(answer: dict) -> dict:
    """Réponse à une paire montrée côtés inversés, ramenée au sens de la série."""
    if "choix" in answer:
        return {"choix": {"A": "B", "B": "A"}.get(answer["choix"], answer["choix"])}
    return answer


class AnswerStore:
    """Réponses en JSONL, ajoutées en fin de fichier ; la dernière pour (tâche, répétition) compte."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.answers: dict[tuple[str, bool], dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    row = json.loads(line)
                    self.answers[(row["task_id"], row["repeat"])] = row["answer"]

    def record(self, task_id: str, repeat: bool, answer: dict) -> None:
        row = {
            "task_id": task_id,
            "repeat": repeat,
            "answer": answer,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.answers[(task_id, repeat)] = answer


def answers_path(series: str, who: str, folder: Path = SERIES) -> Path:
    return folder / f"{series}_{who}.jsonl"


def first_answers(series: str, folder: Path = SERIES) -> dict[str, dict[str, dict]]:
    """{annotateur: {task_id: réponse}}, premières présentations seulement (pas les répétitions)."""
    out = {}
    for annotator in ANNOTATORS:
        path = answers_path(series, annotator, folder)
        if path.exists():
            store = AnswerStore(path)
            out[annotator] = {t: a for (t, rep), a in store.answers.items() if not rep}
    return out


def disagreements(tasks: list[dict], by_who: dict[str, dict[str, dict]]) -> list[dict]:
    """Tâches jugées par au moins deux annotateurs qui ne répondent pas la même chose."""
    out = []
    for task in tasks:
        opinions = {w: a[task["task_id"]] for w, a in by_who.items() if task["task_id"] in a}
        if (
            len(opinions) >= 2
            and len({json.dumps(a, sort_keys=True) for a in opinions.values()}) > 1
        ):
            out.append(task | {"avis": [{"qui": w, "reponse": a} for w, a in opinions.items()]})
    return out


class Session:
    """Ce que la page demande : les tâches dans l'ordre, les réponses déjà données, une réponse.

    En arbitrage, les tâches sont les désaccords, chacune avec l'avis des annotateurs.
    """

    def __init__(self, tasks: list[dict], store: AnswerStore, plan: list[tuple[str, bool]]) -> None:
        self.tasks = {t["task_id"]: t for t in tasks}
        self.store = store
        self.plan = plan

    def _shown(self, task_id: str, repeat: bool) -> dict:
        task = self.tasks[task_id]
        shown = {k: v for k, v in task.items() if k != "pour"}
        if repeat and task["kind"] != "element":
            shown["a"], shown["b"] = task["b"], task["a"]
        return shown | {"key": f"{task_id}|{int(repeat)}"}

    def state(self) -> dict:
        answers = {}
        for task_id, repeat in self.plan:
            answer = self.store.answers.get((task_id, repeat))
            if answer is not None:
                answers[f"{task_id}|{int(repeat)}"] = swap(answer) if repeat else answer
        return {"tasks": [self._shown(t, r) for t, r in self.plan], "answers": answers}

    def answer(self, payload: dict) -> None:
        key = str(payload.get("key", ""))
        task_id, _, flag = key.rpartition("|")
        repeat = flag == "1"
        if (task_id, repeat) not in set(self.plan):
            raise ValueError("tâche inconnue")
        answer = check_answer(self.tasks[task_id]["kind"], payload.get("answer"))
        self.store.record(task_id, repeat, swap(answer) if repeat else answer)


def open_session(
    series: str, annotator: str | None, arbitrage: bool = False, folder: Path = SERIES
) -> tuple[Session, str]:
    tasks = load_series(folder / f"{series}.jsonl")
    if arbitrage:
        conflicts = disagreements(tasks, first_answers(series, folder))
        store = AnswerStore(answers_path(series, ARBITRAGE, folder))
        plan = [(t["task_id"], False) for t in conflicts]
        return Session(conflicts, store, plan), f"Arbitrage de la série « {series} »"
    if annotator not in ANNOTATORS:
        raise ValueError(f"annotateur attendu : {', '.join(ANNOTATORS)}")
    store = AnswerStore(answers_path(series, annotator, folder))
    title = f"Série « {series} », {annotator}"
    return Session(tasks, store, order(tasks, annotator, series)), title


# --- Accord, cohérence, étiquettes finales ---------------------------------------------


def cohen_kappa(a: list, b: list) -> float:
    """Accord entre deux annotateurs corrigé du hasard : 1 parfait, 0 hasard."""
    if not a or len(a) != len(b):
        raise ValueError("deux listes de même longueur, non vides")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    expected = sum((a.count(v) / len(a)) * (b.count(v) / len(b)) for v in set(a) | set(b))
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def _fields(kind: str) -> tuple[str, ...]:
    return ("action", "urgence", "alerte") if kind == "element" else ("choix",)


def agreement_report(series: str, folder: Path = SERIES) -> str:
    """Kappa par paire d'annotateurs et par sorte de jugement, puis cohérence sur les répétitions."""
    tasks = load_series(folder / f"{series}.jsonl")
    kind_of = {t["task_id"]: t["kind"] for t in tasks}
    by_who = first_answers(series, folder)
    lines = []
    for first, second in combinations(sorted(by_who), 2):
        for kind in KINDS:
            common = sorted(
                t for t in set(by_who[first]) & set(by_who[second]) if kind_of.get(t) == kind
            )
            if not common:
                continue
            lines.append(f"{first} / {second}, {kind} : {len(common)} en commun")
            for key in _fields(kind):
                a = [by_who[first][t][key] for t in common]
                b = [by_who[second][t][key] for t in common]
                agree = sum(x == y for x, y in zip(a, b, strict=True)) / len(common)
                lines.append(f"  {key:<8} accord {agree:.0%}  kappa {cohen_kappa(a, b):.2f}")
    for annotator in sorted(by_who):
        store = AnswerStore(answers_path(series, annotator, folder))
        twice = [t for (t, rep) in store.answers if rep and (t, False) in store.answers]
        if twice:
            main_key = {t: _fields(kind_of[t])[0] for t in twice}
            same = sum(
                store.answers[(t, False)][main_key[t]] == store.answers[(t, True)][main_key[t]]
                for t in twice
            )
            lines.append(f"{annotator} : cohérence sur les répétitions {same}/{len(twice)}")
    open_conflicts = [
        t["task_id"]
        for t in disagreements(tasks, by_who)
        if (t["task_id"], False) not in AnswerStore(answers_path(series, ARBITRAGE, folder)).answers
    ]
    lines.append(f"Désaccords à arbitrer : {len(open_conflicts)}")
    return "\n".join(lines)


def final_labels(series: str, folder: Path = SERIES) -> dict[str, dict]:
    """Vérité d'une série : l'arbitrage s'il existe, sinon la réponse commune des annotateurs
    (ou celle de l'unique annotateur). Un désaccord non arbitré n'a pas d'étiquette."""
    tasks = load_series(folder / f"{series}.jsonl")
    by_who = first_answers(series, folder)
    decided = AnswerStore(answers_path(series, ARBITRAGE, folder)).answers
    out = {}
    for task in tasks:
        task_id = task["task_id"]
        if (task_id, False) in decided:
            out[task_id] = decided[(task_id, False)]
            continue
        opinions = [a[task_id] for a in by_who.values() if task_id in a]
        if opinions and all(o == opinions[0] for o in opinions):
            out[task_id] = opinions[0]
    return out


# --- Série d'essai (la vraie sélection est celle de M5 et M7) ---------------------------


def viewpoint(persona_key: str, day: date) -> Viewpoint:
    """Point de vue d'une persona du simulateur, le jour du brief à 8 h 30."""
    from packages.collect.trainset.build import MANIFEST
    from packages.collect.trainset.world import all_personas, brief_moment, fr_day

    seed = json.loads(MANIFEST.read_text(encoding="utf-8"))["seed"]
    persona = next(p for p in all_personas(seed) if p.key == persona_key)
    relation = {
        "hierarchie": "votre N+1",
        "pair": "votre pair",
        "equipe": "votre équipe",
        "automate": "envoi automatique",
    }
    people = {}
    for group in (
        (persona.boss,),
        persona.peers,
        persona.team,
        persona.clients,
        persona.suppliers,
        persona.automates,
    ):
        for p in group:
            people[p.email] = f"{p.role} ({relation[p.kind]})" if p.kind in relation else p.role
    m = persona.manager
    context = (
        f"Vous êtes {m.name}, {m.role.lower()} chez {persona.metier.company} "
        f"(N+1 : {persona.boss.name}, {persona.boss.role.lower()}). "
        f"Brief du {fr_day(day)} à 8 h 30."
    )
    return Viewpoint(m.email, brief_moment(day), people, context)


def sample_corpus(persona_key: str) -> tuple[list[Item], list[dict]]:
    """Éléments et scénario d'une persona de l'échantillon versionné (sans générer le corpus)."""
    from packages.collect.trainset.build import DATA

    folder = DATA / "sample"
    items = [
        Item.model_validate_json(line)
        for line in (folder / f"{persona_key}.items.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    scenario = [
        json.loads(line)
        for line in (folder / f"{persona_key}.scenario.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    return items, scenario


def build_trial(
    items: list[Item],
    scenario: list[dict],
    vp: Viewpoint,
    prefix: str,
    n_elements: int = 12,
    n_pairs: int = 6,
    n_cases: int = 3,
    seed: int = 0,
) -> list[dict]:
    """Série d'essai tirée au hasard : pour éprouver l'outil, pas pour mesurer.

    Visible au moment du brief = `visible_from` passé (le scénario ne sert qu'à ça et au
    regroupement en affaires vraies ; aucune étiquette n'entre dans les tâches).
    """
    rng = random.Random(f"essai:{prefix}:{seed}")
    rows = {r["item_id"]: r for r in scenario}
    horizon = vp.moment - timedelta(days=3)
    known = [
        i for i in items if datetime.fromisoformat(rows[i.item_id]["visible_from"]) <= vp.moment
    ]
    visible = [
        i for i in known if horizon <= datetime.fromisoformat(rows[i.item_id]["visible_from"])
    ]
    tasks: list[dict] = []
    for n, item in enumerate(rng.sample(visible, min(n_elements, len(visible)))):
        tasks.append(
            {"task_id": f"{prefix}-e{n:03d}", "kind": "element", "context": vp.context}
            | {"card": card(item, vp, known)}
        )
    for n in range(min(n_pairs, len(visible) // 2)):
        a, b = rng.sample(visible, 2)
        tasks.append(
            {"task_id": f"{prefix}-p{n:03d}", "kind": "paire", "context": vp.context}
            | {"a": card(a, vp, known), "b": card(b, vp, known)}
        )
    cases: dict[str, list[Item]] = {}
    for item in visible:
        if rows[item.item_id]["case_type"] != "fond":
            cases.setdefault(rows[item.item_id]["case_id"], []).append(item)
    keys = sorted(cases)
    for n in range(min(n_cases, len(keys) // 2)):
        a, b = rng.sample(keys, 2)
        tasks.append(
            {"task_id": f"{prefix}-a{n:03d}", "kind": "affaires", "context": vp.context}
            | {"a": case_card(cases[a], vp), "b": case_card(cases[b], vp)}
        )
    return tasks


def create_trial(folder: Path = SERIES) -> Path:
    """Série « essai » : 2 personas de l'échantillon, au brief du mardi 3 novembre."""
    day = date(2026, 11, 3)
    tasks = []
    for key in ("logistique-1", "hotel-1"):
        items, scenario = sample_corpus(key)
        tasks += build_trial(items, scenario, viewpoint(key, day), prefix=key)
    path = folder / "essai.jsonl"
    write_series(tasks, path)
    return path


# --- Serveur local ----------------------------------------------------------------------


def make_handler(session: Session, title: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, kind: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", f"{kind}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: object, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode())

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html")
            elif path == "/api/state":
                self._json(session.state() | {"title": title})
            else:
                self._json({"error": "introuvable"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/answer":
                self._json({"error": "introuvable"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                session.answer(json.loads(self.rfile.read(length) or b"{}"))
            except (ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True})

        def log_message(self, format: str, *args: object) -> None:
            pass  # pas de journal à chaque touche

    return Handler


def serve(session: Session, title: str, port: int = 8766) -> ThreadingHTTPServer:
    """N'écoute que la machine locale."""
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(session, title))


def _option(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    i = args.index(name)
    value = args[i + 1] if i + 1 < len(args) else ""
    del args[i : i + 2]
    return value


def main(argv: list[str]) -> int:
    args = list(argv)
    if "--creer-essai" in args:
        print(f"Série d'essai écrite : {create_trial()}")
        return 0
    report = _option(args, "--accord")
    if report:
        print(agreement_report(report))
        return 0
    series = _option(args, "--serie")
    who_ = _option(args, "--annotateur")
    port = int(_option(args, "--port") or 8766)
    arbitrage = "--arbitrage" in args
    if not series:
        print(__doc__)
        return 2
    session, title = open_session(series, who_.lower() if who_ else None, arbitrage)
    server = serve(session, title, port)
    print(f"{title} : http://127.0.0.1:{port}  (Ctrl+C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
