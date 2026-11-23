"""M6 : comprendre chaque élément (action principale, urgence déclarée, alerte), 4 sources.

La référence du plan est MiniLM gelé + régression logistique, sur processeur. Elle lit la carte
d'`annoter.card`, exactement ce que voit l'annotateur, et apprend sur les personas « train » du
simulateur (gabarits + LLM). Les jeux gelés ne servent qu'à la mesure finale (G1).

Décisions du 22/11 (plan, §11) :
- la cible est l'action lisible sur la carte : une résolution par le manager avant le brief ne
  se voit pas dans l'`Item`, elle relève de la fiche (« balle dans son camp », M8) ;
- T-démo de M5 n'a que des paires : on tire ici une série gelée de 60 éléments de démo
  (`m6-tdemo-elements`), jugée par Théo seul et à l'aveugle, avant toute mesure d'un modèle
  (P3), pour mesurer le transfert de l'action vers le vrai domaine.

    uv run python -m packages.analyse.scoring.ml.comprendre --tirer-demo
    uv run python -m packages.analyse.scoring.ml.annoter --serie m6-tdemo-elements --annotateur theo
    uv run python -m packages.analyse.scoring.ml.comprendre --geler-demo

La référence (`--entrainer`) :
- **exemples** : chaque élément des personas « train » à son premier brief (fenêtres des
  connecteurs, `jeux_m5.on_brief`), mis en texte par sa carte ; un mail réécrit par le LLM
  donne une seconde carte, une carte identique d'un corpus à l'autre n'est gardée qu'une fois ;
  tout élément d'un jeu gelé est exclu ;
- **validation (G1)** : 7 personas tenues à l'écart, une par métier pour 7 métiers tirés ;
  couper par persona coupe aussi par affaire (une affaire n'appartient qu'à une persona) ;
- **modèle** : une régression logistique multinomiale par cible sur le vecteur MiniLM gelé,
  numpy seul (déterministe, aucune dépendance de plus) ; régularisation et pondération des
  classes choisies sur la validation, à égalité la plus simple (G3) ; seuil d'alerte le plus
  haut qui garde un rappel ≥ 0,95 sur la validation (veto G5) ;
- le modèle livré est celui qu'on a mesuré (appris sans la validation) : ses chiffres de
  validation et son seuil d'alerte sont les siens, pas ceux d'un autre modèle réentraîné.

    uv run python -m packages.analyse.scoring.ml.comprendre --entrainer

Le choix des réglages suit G2 puis G3 depuis la validation dure (`select`) : le plus simple
dont l'écart au meilleur n'est pas significatif. La référence commitée et mesurée sur les jeux
gelés a été choisie avant, au meilleur score strict ; elle n'est pas réentraînée (un nouveau
modèle demanderait une nouvelle lecture des jeux gelés).

La validation dure (`--valider-dur`, étape 0 A du plan d'amélioration) : la validation de 7
personas, une par métier, donne 0,87 sur l'action quand T-gen donne 0,63. Ses trois découpages,
tous sur des métiers jamais vus, se relancent autant qu'on veut pour comparer des leviers sans
relire les jeux gelés (P3) :
1. métiers : 4 plis de 3 métiers entiers sur 12, chaque carte mesurée une fois ;
2. gabarits : en plus, une part des variantes de chaque famille de gabarits retirée de
   l'entraînement, mesure sur les cartes qui les utilisent (effet d'une formulation inconnue) ;
3. écrivain : sur les mails aux deux textes, appris sur un écrivain, mesuré sur chacun.

    uv run python -m packages.analyse.scoring.ml.comprendre --valider-dur

Le jeu de développement réel (étape 0 B, règle M0 modifiée le 22/11) : 104 cartes écrites hors
de nos générateurs (journée de Claire, un élément par modèle de phrase du jeu de David), jugées
par Claude avec le guide, 30 rejugées à l'aveugle par Théo, gelées avant toute mesure.

    uv run python -m packages.analyse.scoring.ml.comprendre --tirer-dev <data de David>
    uv run python -m packages.analyse.scoring.ml.comprendre --geler-dev
    uv run python -m packages.analyse.scoring.ml.comprendre --valider-reel

La mesure finale (`--mesurer`), une seule fois par modèle (G1) : sur T-gen (M5) et la série
démo de M6, refusée si un gel manque ou a changé, ou si le code ou le modèle ne sont pas
commités ; chaque lecture s'ajoute à `scoring/annotations/journal-geles.jsonl`.

    uv run python -m packages.analyse.scoring.ml.comprendre --mesurer
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from contracts.models import Item
from packages.analyse.rag.embeddings import DEFAULT_MODEL, Embedder, FastEmbedder
from packages.analyse.scoring.ml import jeux_m5
from packages.analyse.scoring.ml.annoter import (
    ACTIONS,
    ARBITRAGE,
    SERIES,
    SOURCE_NAMES,
    URGENCIES,
    AnswerStore,
    Viewpoint,
    answers_path,
    card,
    final_labels,
    load_series,
    order,
    viewpoint,
    write_series,
)
from packages.analyse.scoring.ml.intents import VectorCache, cache_path
from packages.analyse.scoring.ml.sample import ANNOTATIONS
from packages.collect.trainset import load
from packages.collect.trainset.build import DATA as GABARITS
from packages.collect.trainset.world import brief_moment, fr_day

DEMO_SERIES = "m6-tdemo-elements"
DEMO_TIRAGE = f"{DEMO_SERIES}.tirage.json"
GEL = "m6-gel.json"
# La moitié en mails, comme T-gen ; 10 de chaque autre source suffisent pour lire le F1 macro
# de l'action sur l'ensemble (pas source par source : trop peu).
DEMO_QUOTA = {"mail": 30, "calendar": 10, "task": 10, "file": 10}
ANNOTATOR = jeux_m5.ANNOTATOR
# Un élément de démo cité dans un fichier d'annotation a déjà été montré à un annotateur
# (paires de M5, mails étiquetés en octobre) : il n'est pas rejugé de mémoire.
_DEMO_ID = re.compile(r"\b(?:mail|calendar|task|file):dm1-[A-Za-z0-9_-]+")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )


# --- T-démo, éléments ------------------------------------------------------------------------


def demo_briefs() -> list[tuple[date, list[Item]]]:
    """Les journées de démo du lot 1 : (jour, éléments collectés pour le brief de ce matin)."""
    from contracts import demo_data

    return [(date.fromisoformat(d), demo_data.load_items(d)) for d in demo_data.available_days()]


def shown_demo_items(annotations: Path = ANNOTATIONS) -> set[str]:
    """Éléments de démo déjà présents dans un fichier d'annotation (séries comprises)."""
    return {
        found
        for path in sorted(annotations.rglob("*.json*"))
        for found in _DEMO_ID.findall(path.read_text(encoding="utf-8"))
    }


def draw_tdemo_elements(
    briefs: list[tuple[date, list[Item]]] | None = None,
    series: Path = SERIES,
    annotations: Path = ANNOTATIONS,
    seed: int = 0,
) -> dict:
    """Tire la série gelée des éléments de démo et écrit tirage et gel. Chaque élément est
    montré au premier brief où il apparaît, avec les éléments déjà visibles ce matin-là (ligne
    « Cité dans » des fichiers). Refusé si le tirage existe : un jeu gelé ne se retire pas."""
    if (series / DEMO_TIRAGE).exists():
        raise FileExistsError(f"{DEMO_TIRAGE} existe : un jeu de mesure ne se retire pas")
    briefs = demo_briefs() if briefs is None else briefs
    rng = random.Random(f"m6:tdemo-elements:{seed}")
    people = jeux_m5.demo_people(i for _, items in briefs for i in items)
    first: dict[str, tuple[date, Item]] = {}
    for day, items in briefs:
        for item in items:
            first.setdefault(item.item_id, (day, item))
    excluded = shown_demo_items(annotations) | jeux_m5.frozen_items(series)
    candidates = sorted(i for i in first if i not in excluded)
    rng.shuffle(candidates)
    by_source: dict[str, list[str]] = {s: [] for s in DEMO_QUOTA}
    for item_id in candidates:
        source = first[item_id][1].source
        if len(by_source[source]) < DEMO_QUOTA[source]:
            by_source[source].append(item_id)
    short = {s: len(v) for s, v in by_source.items() if len(v) < DEMO_QUOTA[s]}
    if short:
        raise ValueError(f"pas assez d'éléments de démo jamais vus : {short}")
    chosen = [i for s in DEMO_QUOTA for i in by_source[s]]
    rng.shuffle(chosen)
    tasks, behind = [], {}
    for n, item_id in enumerate(chosen, 1):
        day, item = first[item_id]
        vp = jeux_m5.demo_viewpoint(day, people)
        known = [
            i
            for d, items in briefs
            if d <= day
            for i in items
            if i.created_at <= vp.moment or i.item_id == item_id
        ]
        known = list({i.item_id: i for i in known}.values())
        task_id = f"d-e{n:03d}"
        tasks.append(
            {"task_id": task_id, "kind": "element", "context": vp.context}
            | {"card": card(item, vp, known)}
        )
        behind[task_id] = item_id
    write_series(tasks, series / f"{DEMO_SERIES}.jsonl")  # refuse toute étiquette
    info = {
        "graine": seed,
        "annotateur": ANNOTATOR,
        "decision": "22/11 (Théo) : T-démo de M5 n'a que des paires ; série d'éléments pour M6",
        "exclus_deja_vus": len(excluded & set(first)),
        "elements": behind,
    }
    _write_json(series / DEMO_TIRAGE, info)
    files = [series / f"{DEMO_SERIES}.jsonl", series / DEMO_TIRAGE]
    _write_json(series / GEL, {"series": {p.name: _sha(p) for p in files}, "reponses": {}})
    return info


def missing_demo_answers(series: Path = SERIES, annotator: str = ANNOTATOR) -> int:
    """Présentations de la série démo encore sans réponse (répétitions cachées comprises)."""
    tasks = load_series(series / f"{DEMO_SERIES}.jsonl")
    store = AnswerStore(answers_path(DEMO_SERIES, annotator, series))
    return sum(k not in store.answers for k in order(tasks, annotator, DEMO_SERIES))


def freeze_demo_answers(series: Path = SERIES, annotator: str = ANNOTATOR) -> dict:
    """Ajoute au gel l'empreinte des réponses, une fois tout annoté. Refusé avant."""
    left = missing_demo_answers(series, annotator)
    if left:
        raise ValueError(f"réponses manquantes : {left}")
    freeze = json.loads((series / GEL).read_text(encoding="utf-8"))
    if freeze["reponses"]:
        raise ValueError("réponses déjà gelées")
    path = answers_path(DEMO_SERIES, annotator, series)
    freeze["reponses"][path.name] = _sha(path)
    _write_json(series / GEL, freeze)
    return freeze


def changed_since_freeze(series: Path = SERIES) -> list[str]:
    """Fichiers gelés (M5 et M6) qui ont changé ou disparu depuis leur gel."""
    changed = jeux_m5.changed_since_freeze(series)
    path = series / GEL
    if path.exists():
        freeze = json.loads(path.read_text(encoding="utf-8"))
        files = {**freeze["series"], **freeze["reponses"]}
        changed += [
            name
            for name, digest in files.items()
            if not (series / name).exists() or _sha(series / name) != digest
        ]
    return sorted(changed)


def frozen_items(series: Path = SERIES) -> set[str]:
    """Tous les éléments des jeux de mesure (M5 et série démo de M6) : jamais à l'entraînement."""
    out = jeux_m5.frozen_items(series)
    path = series / DEMO_TIRAGE
    if path.exists():
        out |= set(json.loads(path.read_text(encoding="utf-8"))["elements"].values())
    return out


# --- exemples d'entraînement ---------------------------------------------------------------------

CORPORA = {"gabarits": GABARITS, "llm": jeux_m5.CORPUS}
TARGETS: dict[str, tuple[str, ...]] = {
    "action": ACTIONS,
    "urgence": URGENCIES,
    "alerte": ("non", "oui"),
}
MODEL_PATH = Path(__file__).with_name("comprendre_reference.npz")
REPORT_PATH = Path(__file__).with_name("comprendre_reference.json")


@dataclass(frozen=True)
class Example:
    item_id: str
    persona: str
    case_id: str
    source: str
    writer: str  # « gabarits » ou « llm »
    text: str
    labels: dict[str, str]  # une valeur de TARGETS par cible
    templates: tuple[str, ...] = ()  # gabarits utilisés par le texte de cette carte
    card: dict = field(default_factory=dict, compare=False, repr=False)  # pour la fiche
    augmented: bool = False  # copie perturbée (levier 2) : apprise, jamais mesurée

    @property
    def metier(self) -> str:
        return metier(self.persona)


def metier(persona: str) -> str:
    """Le métier d'une persona : « daf-3 » → « daf »."""
    return persona.rsplit("-", 1)[0]


def card_text(c: dict) -> str:
    """La carte en texte, dans l'ordre où l'annotateur la lit : source et titre, date, champs
    (expéditeur et relation, réponse, échéance…), puis le texte."""
    lines = [f"{SOURCE_NAMES[c['source']]} : {c['title']}", c["when"]]
    lines += [f"{label} : {value}" for label, value in c["fields"]]
    return "\n".join([*lines, c["text"]]).strip()


def labels_of(row: dict) -> dict[str, str]:
    """La cible : l'action du fait (décision du 22/11), pas « rien » pour un élément traité
    avant le brief, car la carte ne le montre pas ; l'urgence et l'alerte écrites. Les
    règles du guide du 23/11 sont appliquées par l'étiqueteur du simulateur."""
    return {
        "action": row["action"],
        "urgence": row["urgence"],
        "alerte": "oui" if row["alerte"] else "non",
    }


def first_briefs(items: Iterable[Item], rows: dict[str, dict]) -> dict[str, date]:
    """Le jour du premier brief où figure chaque élément (mêmes fenêtres que T-gen)."""
    items = list(items)
    days = jeux_m5._working_days(rows.values())
    first: dict[str, date] = {}
    for n, day in enumerate(days):
        moment = brief_moment(day)
        previous = brief_moment(days[n - 1]) if n else moment - timedelta(days=3)
        for item in items:
            if item.item_id not in first and jeux_m5.on_brief(
                item, rows[item.item_id], moment, previous
            ):
                first[item.item_id] = day
    return first


def persona_examples(
    key: str, corpora: dict[str, Path], frozen: set[str]
) -> tuple[list[Example], int]:
    """Les exemples d'une persona, et le nombre d'éléments écartés parce que gelés. Chaque
    carte prend l'étiquette de son écrivain : depuis le 23/11, une copie « pour suivre » ou une
    demande d'avis se lit dans la formulation, qui diffère d'un écrivain à l'autre."""
    labels_by_writer = {
        writer: {r["item_id"]: r for r in load.load_scenario(key, path)}
        for writer, path in corpora.items()
    }
    rows = labels_by_writer["gabarits"]
    by_writer = {
        writer: {i.item_id: i for i in load.load_items(key, path)}
        for writer, path in corpora.items()
    }
    first = first_briefs(by_writer["gabarits"].values(), rows)
    excluded = sum(item_id in frozen for item_id in first)
    out: list[Example] = []
    views, known = {}, {}
    for item_id, day in sorted(first.items(), key=lambda x: (x[1], x[0])):
        if item_id in frozen:
            continue
        if day not in views:
            views[day] = viewpoint(key, day)
        vp, row, texts = views[day], rows[item_id], set()
        for writer, items in by_writer.items():
            if (writer, day) not in known:  # déjà visibles : la ligne « Cité dans » des fichiers
                known[writer, day] = [
                    i
                    for i in items.values()
                    if datetime.fromisoformat(rows[i.item_id]["visible_from"]) <= vp.moment
                ]
            shown = card(items[item_id], vp, known[writer, day])
            text = card_text(shown)
            if text in texts:  # même carte qu'avec l'autre écrivain : une seule fois
                continue
            texts.add(text)
            own = labels_by_writer[writer][item_id]
            out.append(
                Example(
                    item_id,
                    key,
                    row["case_id"],
                    row["source"],
                    writer,
                    text,
                    labels_of(own),
                    tuple(own.get("templates", ())),
                    shown,
                )
            )
    return out, excluded


def training_examples(
    corpora: dict[str, Path] = CORPORA, frozen: set[str] | None = None
) -> tuple[list[Example], int]:
    """Les exemples des personas « train » seulement, hors jeux gelés."""
    frozen = frozen_items() if frozen is None else frozen
    out, excluded = [], 0
    for key in load.personas("train", data=corpora["gabarits"]):
        examples, n = persona_examples(key, corpora, frozen)
        out += examples
        excluded += n
    return out, excluded


def validation_personas(keys: Sequence[str], n: int = 7, seed: int = 0) -> list[str]:
    """Les personas tenues à l'écart : une par métier, pour `n` métiers tirés. Chaque métier
    de validation garde ses autres personas à l'entraînement : les métiers jamais vus sont le
    rôle de T-gen, pas de la validation."""
    rng = random.Random(f"m6:validation:{seed}")
    by_metier: dict[str, list[str]] = {}
    for key in sorted(keys):
        by_metier.setdefault(metier(key), []).append(key)
    chosen = rng.sample(sorted(by_metier), n)
    return sorted(rng.choice(by_metier[m]) for m in chosen)


def split(examples: list[Example], held_out: Sequence[str]) -> tuple[list[int], list[int]]:
    """Indices d'entraînement et de validation. Vérifie que personas et affaires sont
    disjointes : aucun quasi-doublon d'une même affaire de part et d'autre."""
    val = [n for n, e in enumerate(examples) if e.persona in held_out]
    train = [n for n, e in enumerate(examples) if e.persona not in held_out]
    cases = ({examples[n].case_id for n in train}, {examples[n].case_id for n in val})
    if cases[0] & cases[1]:
        raise ValueError(f"affaires des deux côtés : {sorted(cases[0] & cases[1])[:3]}")
    return train, val


# --- régression logistique multinomiale -------------------------------------------------------

# De la plus régularisée (la plus simple) à la moins. 1e-5 ajouté le 22/11 après la première
# lecture de la validation : les trois cibles y choisissaient 1e-4, le bord de la grille
# (choix permis par G1 sur la validation, signalé au rapport, aucun jeu gelé lu).
L2_CHOICES = (1e-2, 1e-3, 1e-4, 1e-5)
GRID_NOTE = "1e-5 ajouté le 22/11 : 1e-4, bord de la grille, choisi par les 3 cibles au 1er essai"
STEPS = 1000
ALERT_RECALL = 0.95  # veto G5


@dataclass(frozen=True)
class Config:
    l2: float
    balanced: bool

    def as_dict(self) -> dict:
        return {"l2": self.l2, "classes_pondérées": self.balanced}


def configs() -> list[Config]:
    """Dans l'ordre de simplicité (G3) : sans pondération d'abord, la plus régularisée d'abord."""
    return [Config(l2, balanced) for balanced in (False, True) for l2 in L2_CHOICES]


@dataclass
class Softmax:
    """P(classe | x) = softmax(W · (x − moyenne) / écart + b). Les vecteurs sont centrés et
    réduits : sans cela, la descente de gradient avance très lentement sur des coordonnées
    de l'ordre de 0,05."""

    W: np.ndarray  # (dimension, classes)
    b: np.ndarray  # (classes,)
    mean: np.ndarray
    scale: np.ndarray

    def proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax(((X - self.mean) / self.scale) @ self.W + self.b)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_softmax(
    X: np.ndarray, y: np.ndarray, k: int, config: Config, steps: int = STEPS
) -> Softmax:
    """Entropie croisée (pondérée si `balanced` : chaque classe pèse autant au total) + L2 sur
    W, par descente de gradient accélérée (Nesterov) depuis zéro, à pas fixe 1/L (L : borne
    de la courbure). Convexe et sans tirage : mêmes données, mêmes poids."""
    X = np.asarray(X, dtype=np.float64)
    mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = np.hstack([(X - mean) / scale, np.ones((len(X), 1))])
    Y = np.eye(k)[y]
    if config.balanced:
        counts = np.bincount(y, minlength=k).astype(float)
        weight = len(y) / (k * np.maximum(counts, 1.0))[y]
    else:
        weight = np.ones(len(y))
    weight = weight / weight.mean()
    reg = np.r_[np.full(Z.shape[1] - 1, config.l2), 0.0][:, None]  # pas de L2 sur le biais
    curvature = np.linalg.eigvalsh((Z * weight[:, None]).T @ Z / len(Z))[-1]
    step = 1.0 / (0.5 * curvature + config.l2)
    W = V = np.zeros((Z.shape[1], k))
    for t in range(1, steps + 1):
        grad = Z.T @ ((_softmax(Z @ V) - Y) * weight[:, None]) / len(Z) + reg * V
        W_next = V - step * grad
        V = W_next + (t - 1) / (t + 2) * (W_next - W)
        W = W_next
    return Softmax(W[:-1], W[-1], mean, scale)


# --- mesures ----------------------------------------------------------------------------------


def per_class(pred: np.ndarray, truth: np.ndarray, k: int) -> list[tuple[float, float, float, int]]:
    """(précision, rappel, F1, effectif réel) de chaque classe."""
    out = []
    for c in range(k):
        tp = float(((pred == c) & (truth == c)).sum())
        p = tp / max(float((pred == c).sum()), 1.0)
        r = tp / max(float((truth == c).sum()), 1.0)
        out.append((p, r, 2 * p * r / (p + r) if p + r else 0.0, int((truth == c).sum())))
    return out


def f1_macro(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Moyenne des F1 sur les classes présentes dans la vérité ou la prédiction."""
    present = set(truth.tolist()) | set(pred.tolist())
    scores = per_class(pred, truth, k)
    return float(np.mean([scores[c][2] for c in sorted(present)])) if present else 0.0


def bootstrap_ci(
    pred: np.ndarray, truth: np.ndarray, k: int, draws: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Intervalle à 95 % du F1 macro par rééchantillonnage des exemples (graine fixe, G2)."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(truth), len(truth))
        values.append(f1_macro(pred[idx], truth[idx], k))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def alert_threshold(p_yes: np.ndarray, truth: np.ndarray, recall: float = ALERT_RECALL) -> float:
    """Le seuil le plus haut qui garde au moins `recall` des vraies alertes (veto G5) : plus
    haut, moins de fausses alertes ; jamais au prix d'une alerte manquée de trop."""
    positives = np.sort(p_yes[truth == 1])[::-1]
    if not len(positives):
        return 0.5
    return float(positives[math.ceil(recall * len(positives)) - 1])


def predict(model: Softmax, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Classe prédite ; pour une cible binaire avec seuil, « oui » dès p ≥ seuil."""
    proba = model.proba(X)
    if threshold is not None:
        return (proba[:, 1] >= threshold).astype(int)
    return proba.argmax(axis=1)


# --- modèle enregistré ------------------------------------------------------------------------


@dataclass
class Reference:
    """La référence de M6 : une régression par cible, et le seuil d'alerte."""

    models: dict[str, Softmax]
    alert_threshold: float
    embedder_name: str
    fiche: str = "aucune"  # entrée du modèle : FICHES (levier 1)
    texte: str = "brut"  # texte lu par l'encodeur : TEXTES (levier 2)

    def predict(self, X: np.ndarray) -> dict[str, list[str]]:
        out = {}
        for target, classes in TARGETS.items():
            threshold = self.alert_threshold if target == "alerte" else None
            out[target] = [classes[c] for c in predict(self.models[target], X, threshold)]
        return out

    def save(self, path: Path = MODEL_PATH) -> None:
        arrays = {
            f"{part}_{target}": getattr(model, part)
            for target, model in self.models.items()
            for part in ("W", "b", "mean", "scale")
        }
        meta = {
            "embedder": self.embedder_name,
            "seuil_alerte": self.alert_threshold,
            "classes": {t: list(c) for t, c in TARGETS.items()},
        } | ({"fiche": self.fiche} if self.fiche != "aucune" else {})
        meta |= {"texte": self.texte} if self.texte != "brut" else {}
        np.savez(path, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> Reference:
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            models = {
                t: Softmax(*(data[f"{part}_{t}"] for part in ("W", "b", "mean", "scale")))
                for t in TARGETS
            }
        return cls(
            models,
            meta["seuil_alerte"],
            meta["embedder"],
            meta.get("fiche", "aucune"),
            meta.get("texte", "brut"),
        )


# --- entraînement et rapport ------------------------------------------------------------------


def _codes(examples: Sequence[Example], target: str) -> np.ndarray:
    index = {c: n for n, c in enumerate(TARGETS[target])}
    return np.array([index[e.labels[target]] for e in examples], dtype=int)


def _selection_score(target: str, pred: np.ndarray, truth: np.ndarray) -> float:
    """F1 macro (action, urgence) ; F1 des « oui » pour l'alerte, classe rare qui compte."""
    k = len(TARGETS[target])
    return per_class(pred, truth, k)[1][2] if target == "alerte" else f1_macro(pred, truth, k)


DIFF_DRAWS = 1000  # G2 : 1 000 tirages, graine fixe


def paired_diff_ci(
    score: Callable[[np.ndarray, np.ndarray], float],
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    truth: np.ndarray,
    draws: int = DIFF_DRAWS,
    seed: int = 0,
) -> tuple[float, float]:
    """Intervalle à 95 % de score(a) − score(b), les deux prédictions rééchantillonnées sur
    les mêmes exemples (appariées) : l'écart propre aux modèles, sans le hasard du tirage."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        idx = rng.integers(0, len(truth), len(truth))
        diffs.append(score(pred_a[idx], truth[idx]) - score(pred_b[idx], truth[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def select(target: str, preds: Sequence[np.ndarray], truth: np.ndarray) -> tuple[int, list[dict]]:
    """Le réglage retenu parmi `preds`, rangées de la plus simple à la moins simple (G3) : la
    plus simple dont l'écart au meilleur score n'est pas significatif (G2 : l'intervalle
    apparié de l'écart contient zéro). Le meilleur strict gagnait aussi sur un écart de bruit."""

    def score(pred: np.ndarray, y: np.ndarray) -> float:
        return _selection_score(target, pred, y)

    scores = [score(p, truth) for p in preds]
    best = int(np.argmax(scores))  # à score identique, le premier, donc le plus simple
    tried = [{"score": round(s, 4)} for s in scores]
    for rank in range(best):
        low, high = paired_diff_ci(score, preds[best], preds[rank], truth)
        tried[rank]["ecart_au_meilleur_ic95"] = [round(low, 4), round(high, 4)]
        if low <= 0:
            return rank, tried
    return best, tried


def choose(
    target: str, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[Config, Softmax, list[dict]]:
    """La configuration retenue sur la validation par `select` (G2 puis G3)."""
    k, grid = len(TARGETS[target]), configs()
    models = [fit_softmax(X_tr, y_tr, k, config) for config in grid]
    rank, tried = select(target, [predict(m, X_val) for m in models], y_val)
    return grid[rank], models[rank], [c.as_dict() | t for c, t in zip(grid, tried, strict=True)]


def _class_table(target: str, pred: np.ndarray, truth: np.ndarray) -> dict:
    classes = TARGETS[target]
    return {
        classes[c]: {"précision": round(p, 3), "rappel": round(r, 3), "f1": round(f, 3), "n": n}
        for c, (p, r, f, n) in enumerate(per_class(pred, truth, len(classes)))
    }


def train_reference(
    examples: list[Example],
    X: np.ndarray,
    held_out: Sequence[str],
    embedder_name: str,
) -> tuple[Reference, dict]:
    """Choisit et entraîne la référence sur l'entraînement, la mesure sur la validation."""
    train, val = split(examples, held_out)
    ex_val = [examples[n] for n in val]
    report: dict = {
        "validation": {"personas": list(held_out), "exemples": len(val)},
        "entrainement": {"exemples": len(train)},
        "cibles": {},
    }
    models, threshold = {}, 0.5
    for target, classes in TARGETS.items():
        y = _codes(examples, target)
        config, model, tried = choose(target, X[train], y[train], X[val], y[val])
        models[target] = model
        entry: dict = {"configuration": config.as_dict(), "essais": tried}
        if target == "alerte":
            threshold = alert_threshold(model.proba(X[val])[:, 1], y[val])
            entry["seuil"] = round(threshold, 4)
        pred = predict(model, X[val], threshold if target == "alerte" else None)
        k = len(classes)
        entry |= {
            "f1_macro": round(f1_macro(pred, y[val], k), 4),
            "f1_macro_ic95": [round(v, 4) for v in bootstrap_ci(pred, y[val], k)],
            "classes": _class_table(target, pred, y[val]),
            "par_source": {
                s: round(f1_macro(pred[m], y[val][m], k), 4)
                for s in SOURCE_NAMES
                if (m := np.array([e.source == s for e in ex_val])).any()
            },
            "par_ecrivain": {
                w: round(f1_macro(pred[m], y[val][m], k), 4)
                for w in CORPORA
                if (m := np.array([e.writer == w for e in ex_val])).any()
            },
        }
        if target == "action":
            confusion = np.zeros((k, k), dtype=int)
            np.add.at(confusion, (y[val], pred), 1)
            entry["confusion"] = {
                "lignes = vérité, colonnes = prédiction": list(classes),
                "matrice": confusion.tolist(),
            }
        report["cibles"][target] = entry
    report["ecrivain_croise"] = cross_writer(examples, X, train, val)
    return Reference(models, threshold, embedder_name), report


def cross_writer(examples: list[Example], X: np.ndarray, train: list[int], val: list[int]) -> dict:
    """Test croisé du plan (§6) : appris sur les seules cartes des gabarits, mesuré sur les
    mails du LLM de la validation, comparé au même apprentissage avec les deux écrivains."""
    llm_val = [n for n in val if examples[n].writer == "llm"]
    if not llm_val:
        return {}
    y, k = _codes(examples, "action"), len(ACTIONS)
    gab_train = [n for n in train if examples[n].writer == "gabarits"]
    out = {"exemples_llm_validation": len(llm_val)}
    for name, rows in (("gabarits_seuls", gab_train), ("deux_ecrivains", train)):
        config, _, _ = choose("action", X[rows], y[rows], X[val], y[val])
        model = fit_softmax(X[rows], y[rows], k, config)
        out[name] = round(f1_macro(predict(model, X[llm_val]), y[llm_val], k), 4)
    return out


def truncated_share(embedder: Embedder, examples: Sequence[Example]) -> dict | None:
    """Part des cartes coupées par la longueur maximale de l'encodeur, au total et par source
    (fastembed seulement ; None pour un autre encodeur)."""
    if not isinstance(embedder, FastEmbedder):
        return None
    if embedder._model is None:  # vecteurs tous en cache : le modèle n'a pas été chargé
        embedder.embed(["chargement"])
    tokenizer = embedder._model.model.tokenizer
    limit = tokenizer.truncation["max_length"]
    # Le lot est complété jusqu'au plus long : on compte les vrais jetons, pas le remplissage.
    cut = [
        sum(e.attention_mask) >= limit for e in tokenizer.encode_batch([x.text for x in examples])
    ]
    by_source = {
        s: round(
            float(np.mean([c for c, x in zip(cut, examples, strict=True) if x.source == s])), 4
        )
        for s in SOURCE_NAMES
        if any(x.source == s for x in examples)
    }
    return {"jetons_max": limit, "total": round(float(np.mean(cut)), 4), "par_source": by_source}


CHUNK = 1000


def embed_with_progress(
    cache: VectorCache,
    embedder: Embedder,
    texts: Sequence[str],
    chunk: int = CHUNK,
    log: Callable[[str], None] = print,
) -> np.ndarray:
    """Les vecteurs, calculés par paquets : l'avancement s'affiche, et le cache est enregistré
    après chaque paquet, donc un calcul interrompu reprend où il s'était arrêté."""
    start, total = time.monotonic(), len(texts)
    for begin in range(0, total, chunk):
        cache.embed(embedder, texts[begin : begin + chunk])
        done = min(begin + chunk, total)
        elapsed = time.monotonic() - start
        left = elapsed / done * (total - done)
        log(f"vecteurs : {done}/{total} ({elapsed:.0f} s, reste ≈ {left:.0f} s)")
    return cache.embed(embedder, texts)  # tout est en cache : simple lecture


def entrainer(
    corpora: dict[str, Path] = CORPORA,
    embedder: Embedder | None = None,
    cache: VectorCache | None = None,
    out_model: Path = MODEL_PATH,
    out_report: Path = REPORT_PATH,
    n_validation: int = 7,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """Exemples, vecteurs, choix et mesure sur la validation ; écrit le modèle et le rapport."""
    embedder = embedder or FastEmbedder(DEFAULT_MODEL)
    cache = cache or VectorCache(cache_path(embedder))
    log("cartes des personas « train »…")
    examples, excluded = training_examples(corpora)
    log(f"{len(examples)} cartes")
    keys = sorted({e.persona for e in examples})
    held_out = validation_personas(keys, n_validation)
    X = embed_with_progress(cache, embedder, [e.text for e in examples], log=log)
    log("choix des réglages et mesure sur la validation…")
    reference, report = train_reference(examples, X, held_out, embedder.name)
    report = {
        "modele": f"{embedder.name} gelé + régression logistique",
        "grille": {"l2": list(L2_CHOICES), "pas": STEPS, "note": GRID_NOTE},
        "exemples": {
            "total": len(examples),
            "elements_geles_exclus": excluded,
            "par_ecrivain": dict(sorted(Counter(e.writer for e in examples).items())),
            "par_source": dict(sorted(Counter(e.source for e in examples).items())),
            "cartes_tronquees": truncated_share(embedder, examples),
        },
        "corpus": _corpus_ids(corpora),
    } | report
    reference.save(out_model)
    out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def _corpus_ids(corpora: dict[str, Path]) -> dict:
    out = {}
    for writer, path in corpora.items():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        out[writer] = {"graine": manifest.get("seed")} | (
            {"consigne": manifest["consigne"], "textes_sha256": manifest["textes"]["sha256"]}
            if "textes" in manifest
            else {}
        )
    return out


def summary(report: dict) -> str:
    """Les chiffres à lire d'abord."""
    ex, val = report["exemples"], report["validation"]
    lines = [
        (
            f"{ex['total']} exemples ({ex['par_ecrivain']}), {ex['elements_geles_exclus']} "
            f"éléments gelés exclus ; validation : {val['exemples']} exemples "
            f"({', '.join(val['personas'])})"
        ),
        f"cartes tronquées par l'encodeur : {ex['cartes_tronquees']}",
    ]
    for target, entry in report["cibles"].items():
        low, high = entry["f1_macro_ic95"]
        lines.append(
            f"{target:<8} F1 macro {entry['f1_macro']:.3f} [{low:.3f} ; {high:.3f}]  "
            f"{entry['configuration']}  par source {entry['par_source']}"
        )
    alert = report["cibles"]["alerte"]
    lines.append(
        f"alerte : seuil {alert['seuil']}, rappel {alert['classes']['oui']['rappel']}, "
        f"précision {alert['classes']['oui']['précision']}"
    )
    lines.append(f"test croisé des écrivains (action, mails LLM) : {report['ecrivain_croise']}")
    return "\n".join(lines)


# --- validation dure (M6, étape 0 A) -------------------------------------------------------------

HARD_REPORT_PATH = Path(__file__).with_name("comprendre_validation_dure.json")
HARD_FOLDS = 4  # 12 métiers « train » : 4 plis de 3 métiers


def metier_folds(
    metiers: Iterable[str], n_folds: int = HARD_FOLDS, seed: int = 0
) -> list[list[str]]:
    """Les métiers répartis en `n_folds` plis de même taille (à un près), tirage à graine fixe.
    Chaque métier est tenu à l'écart une fois, avec toutes ses personas : un métier jamais vu,
    comme sur T-gen."""
    ms = sorted(set(metiers))
    if not 2 <= n_folds <= len(ms):
        raise ValueError(f"{n_folds} plis pour {len(ms)} métiers")
    random.Random(f"m6:validation-dure:metiers:{seed}").shuffle(ms)
    return [sorted(ms[f::n_folds]) for f in range(n_folds)]


def template_family(template: str) -> str | None:
    """« mail.decider.question.7 » → « mail.decider.question » ; None pour une marque qui n'est
    pas un gabarit (« llm:mail-v8 », « lecture:avis »)."""
    head, _, tail = template.rpartition(".")
    return head if head and tail.isdigit() else None


def template_folds(templates: Iterable[str], n_folds: int, seed: int = 0) -> list[set[str]]:
    """Les variantes de chaque famille de gabarits réparties entre les plis, à partir d'un pli
    tiré au hasard par famille. Une famille d'une seule variante n'est jamais tenue à l'écart :
    on retirerait la formulation, et avec elle une forme entière d'une classe. Avec au moins
    deux variantes, il en reste toujours une à l'entraînement de chaque pli."""
    families: dict[str, list[str]] = {}
    for t in sorted(set(templates)):
        if (family := template_family(t)) is not None:
            families.setdefault(family, []).append(t)
    folds: list[set[str]] = [set() for _ in range(n_folds)]
    for family, variants in sorted(families.items()):
        if len(variants) < 2:
            continue
        rng = random.Random(f"m6:validation-dure:gabarits:{seed}:{family}")
        rng.shuffle(variants)
        start = rng.randrange(n_folds)
        for n, variant in enumerate(variants):
            folds[(start + n) % n_folds].add(variant)
    return folds


@dataclass
class Fold:
    """Un pli : les exemples d'entraînement et un ou plusieurs jeux mesurés."""

    train: list[int]
    measured: dict[str, list[int]]


@dataclass
class OutOfFold:
    """Les prédictions hors pli d'un jeu mesuré, mises bout à bout : `index` (exemples),
    `fold` (numéro du pli), et les probabilités de chaque réglage de la grille."""

    index: np.ndarray
    fold: np.ndarray
    proba: list[np.ndarray]


def out_of_fold(
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Fold],
    log: Callable[[str], None] = lambda _: None,
    grid: Sequence[Config] | None = None,
) -> dict[str, OutOfFold]:
    """Chaque réglage appris sur l'entraînement de chaque pli, appliqué à ses jeux mesurés :
    un seul apprentissage par pli et par réglage, quel que soit le nombre de jeux mesurés."""
    k, grid = len(TARGETS[target]), list(grid or configs())
    names = list(folds[0].measured)
    idx = {n: [np.asarray(f.measured[n], dtype=int) for f in folds] for n in names}
    proba: dict[str, list[list[np.ndarray]]] = {n: [[] for _ in grid] for n in names}
    for f, fold in enumerate(folds):
        if not fold.train:
            raise ValueError(f"pli {f} sans exemple d'entraînement")
        for c, config in enumerate(grid):
            log(f"  {target}, pli {f + 1}/{len(folds)}, réglage {c + 1}/{len(grid)}")
            model = fit_softmax(X[fold.train], y[fold.train], k, config)
            for name in names:
                proba[name][c].append(model.proba(X[idx[name][f]]))
    return {
        name: OutOfFold(
            np.concatenate(idx[name]),
            np.concatenate([np.full(len(i), f) for f, i in enumerate(idx[name])]),
            [np.concatenate(p) for p in proba[name]],
        )
        for name in names
    }


def _entry(target: str, pred: np.ndarray, y: np.ndarray, sources: Sequence[str]) -> dict:
    """F1 macro avec intervalle (G2), par classe et par source ; confusion de l'action."""
    k = len(TARGETS[target])
    entry = {
        "f1_macro": round(f1_macro(pred, y, k), 4),
        "f1_macro_ic95": [round(v, 4) for v in bootstrap_ci(pred, y, k)],
        "classes": _class_table(target, pred, y),
        "par_source": {
            s: round(f1_macro(pred[m], y[m], k), 4)
            for s in SOURCE_NAMES
            if (m := np.array([x == s for x in sources], dtype=bool)).any()
        },
    }
    if target == "action":
        confusion = np.zeros((k, k), dtype=int)
        np.add.at(confusion, (y, pred), 1)
        entry["confusion"] = confusion.tolist()
    return entry


def judge(
    target: str,
    oof: OutOfFold,
    y: np.ndarray,
    examples: Sequence[Example],
    grid: Sequence[Config] | None = None,
) -> tuple[dict, np.ndarray]:
    """Le réglage retenu (G2 puis G3) sur l'ensemble des prédictions hors pli, et ses chiffres.
    Un seul réglage par découpage, comme un levier qu'on fixerait, et non un par pli."""
    truth, k = y[oof.index], len(TARGETS[target])
    preds = [p.argmax(axis=1) for p in oof.proba]
    rank, tried = select(target, preds, truth)
    pred, grid = preds[rank], list(grid or configs())
    entry = {
        "configuration": grid[rank].as_dict(),
        "essais": [c.as_dict() | t for c, t in zip(grid, tried, strict=True)],
        "exemples": len(truth),
    } | _entry(target, pred, truth, [examples[n].source for n in oof.index])
    n_folds = int(oof.fold.max()) + 1 if len(oof.fold) else 0
    entry["par_pli"] = [
        round(f1_macro(pred[m], truth[m], k), 4) if (m := oof.fold == f).any() else None
        for f in range(n_folds)
    ]
    if target == "alerte":  # le seuil du veto G5, posé sur ces mêmes prédictions
        p_yes = oof.proba[rank][:, 1]
        threshold = alert_threshold(p_yes, truth)
        precision, recall, _, _ = per_class((p_yes >= threshold).astype(int), truth, 2)[1]
        entry["seuil_rappel_095"] = {
            "seuil": round(threshold, 4),
            "rappel": round(recall, 3),
            "précision": round(precision, 3),
        }
    return entry, pred


def _f1_action(pred: np.ndarray, truth: np.ndarray) -> float:
    return f1_macro(pred, truth, len(ACTIONS))


def hard_validation(
    examples: list[Example],
    X: np.ndarray,
    n_folds: int = HARD_FOLDS,
    seed: int = 0,
    log: Callable[[str], None] = lambda _: None,
    keep: dict | None = None,
    grid: Sequence[Config] | None = None,
) -> dict:
    """Les trois découpages de la validation dure, tous sur des métiers jamais vus :
    1. métiers : `n_folds` plis de métiers entiers, les trois cibles ;
    2. gabarits : en plus, une part des variantes de chaque famille retirée de l'entraînement,
       mesure sur les cartes qui les utilisent, comparée au découpage 1 sur les mêmes cartes ;
    3. écrivain : mails qui ont les deux textes, appris sur un écrivain, mesurés sur chacun."""
    plis = metier_folds((e.metier for e in examples), n_folds, seed)
    base: list[tuple[list[int], list[int]]] = []
    for held in plis:
        personas = sorted({e.persona for e in examples if e.metier in held})
        tr, va = split(examples, personas)  # vérifie aussi que les affaires sont disjointes
        base.append((tr, [n for n in va if not examples[n].augmented]))  # copies : apprises
    report: dict = {
        "plis_metiers": plis,
        "graine": seed,
        "selection": "G2 puis G3 : le réglage le plus simple à égalité avec le meilleur",
    }

    log("1. métiers tenus à l'écart")
    originals = np.array([n for n, e in enumerate(examples) if not e.augmented], dtype=int)
    first: dict = {"exemples": len(originals), "cibles": {}}
    action_pred = np.full(len(examples), -1)
    for target, classes in TARGETS.items():
        y = _codes(examples, target)
        folds = [Fold(tr, {"v": va}) for tr, va in base]
        oof = out_of_fold(target, X, y, folds, log, grid=grid)["v"]
        entry, pred = judge(target, oof, y, examples, grid)
        k = len(classes)
        entry["par_ecrivain"] = {
            w: round(f1_macro(pred[m], y[oof.index][m], k), 4)
            for w in CORPORA
            if (m := np.array([examples[n].writer == w for n in oof.index], dtype=bool)).any()
        }
        first["cibles"][target] = entry
        if target == "action":
            action_pred[oof.index] = pred
    report["metiers"] = first
    keep = {} if keep is None else keep  # prédictions de l'action, pour les écarts appariés
    keep["metiers"] = (originals, action_pred[originals])

    log("2. gabarits tenus à l'écart")
    y = _codes(examples, "action")
    held_templates = template_folds((t for e in examples for t in e.templates), n_folds, seed)
    folds = [
        Fold(
            [n for n in tr if not held.intersection(examples[n].templates)],
            {"v": [n for n in va if held.intersection(examples[n].templates)]},
        )
        for (tr, va), held in zip(base, held_templates, strict=True)
    ]
    oof = out_of_fold("action", X, y, folds, log, grid=grid)["v"]
    entry, pred = judge("action", oof, y, examples, grid)
    keep["gabarits"] = (oof.index, pred)
    truth, same = y[oof.index], action_pred[oof.index]
    low, high = paired_diff_ci(_f1_action, pred, same, truth)
    families = {template_family(t) for e in examples for t in e.templates} - {None}
    held_families = {template_family(t) for h in held_templates for t in h}
    report["gabarits"] = {
        "familles": len(families),
        "familles_jamais_tenues": len(families - held_families),
        "variantes_tenues_par_pli": [len(h) for h in held_templates],
        "cartes_retirees_de_l_entrainement": [
            len(tr) - len(f.train) for (tr, _), f in zip(base, folds, strict=True)
        ],
        "action": entry,
        "memes_cartes_decoupage_metiers": round(_f1_action(same, truth), 4),
        "ecart_au_decoupage_metiers": round(_f1_action(pred, truth) - _f1_action(same, truth), 4),
        "ecart_ic95": [round(low, 4), round(high, 4)],
    }

    log("3. écrivain tenu à l'écart (mails)")
    by_writer = {
        w: {e.item_id for e in examples if e.source == "mail" and e.writer == w} for w in CORPORA
    }
    both = set.intersection(*by_writer.values())
    mails = {n for n, e in enumerate(examples) if e.source == "mail" and e.item_id in both}
    cells: dict[str, tuple[dict, np.ndarray, OutOfFold]] = {}
    for learn in CORPORA:
        folds = [
            Fold(
                [n for n in tr if n in mails and examples[n].writer == learn],
                {w: [n for n in va if n in mails and examples[n].writer == w] for w in CORPORA},
            )
            for tr, va in base
        ]
        for measure, oof in out_of_fold("action", X, y, folds, log, grid=grid).items():
            entry, pred = judge("action", oof, y, examples, grid)
            cells[f"{learn}→{measure}"] = (entry, pred, oof)
            keep[f"{learn}→{measure}"] = (oof.index, pred)
    keys = ("configuration", "exemples", "f1_macro", "f1_macro_ic95", "par_pli", "classes")
    third: dict = {
        "elements_aux_deux_textes": len(both),
        "cellules": {name: {k: cell[0][k] for k in keys} for name, cell in cells.items()},
        "ecarts": {},
    }
    for measure in CORPORA:
        other = next(w for w in CORPORA if w != measure)
        _, crossed, oof = cells[f"{other}→{measure}"]
        _, same_writer, oof_same = cells[f"{measure}→{measure}"]
        if not np.array_equal(oof.index, oof_same.index):
            raise AssertionError("les deux cellules doivent mesurer les mêmes cartes")
        truth = y[oof.index]
        low, high = paired_diff_ci(_f1_action, crossed, same_writer, truth)
        third["ecarts"][f"mesuré sur {measure}"] = {
            "appris_sur_l_autre_moins_appris_sur_le_meme": round(
                _f1_action(crossed, truth) - _f1_action(same_writer, truth), 4
            ),
            "ic95": [round(low, 4), round(high, 4)],
        }
    report["ecrivain"] = third
    return report


def valider_dur(
    corpora: dict[str, Path] = CORPORA,
    embedder: Embedder | None = None,
    cache: VectorCache | None = None,
    out_report: Path = HARD_REPORT_PATH,
    n_folds: int = HARD_FOLDS,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """La validation dure de la recette de référence (MiniLM gelé + régression) : exemples et
    vecteurs de `--entrainer` (cache), trois découpages, rapport versionné. Aucun jeu gelé n'est
    lu : leurs éléments sont exclus des exemples (`frozen_items`)."""
    embedder = embedder or FastEmbedder(DEFAULT_MODEL)
    cache = cache or VectorCache(cache_path(embedder))
    log("cartes des personas « train »…")
    examples, excluded = training_examples(corpora)
    X = embed_with_progress(cache, embedder, [e.text for e in examples], log=log)
    report = {
        "modele": f"{embedder.name} gelé + régression logistique",
        "grille": {"l2": list(L2_CHOICES), "pas": STEPS},
        "exemples": {
            "total": len(examples),
            "elements_geles_exclus": excluded,
            "par_ecrivain": dict(sorted(Counter(e.writer for e in examples).items())),
        },
        "corpus": _corpus_ids(corpora),
    } | hard_validation(examples, X, n_folds, log=log)
    out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def hard_summary(report: dict) -> str:
    """Les chiffres de la validation dure à lire d'abord."""

    def f1(e: dict) -> str:
        low, high = e["f1_macro_ic95"]
        return f"{e['f1_macro']:.3f} [{low:.3f} ; {high:.3f}]"

    first, second, third = report["metiers"], report["gabarits"], report["ecrivain"]
    lines = [f"plis de métiers : {report['plis_metiers']}", f"1. métiers ({first['exemples']})"]
    for target, e in first["cibles"].items():
        lines.append(
            f"   {target:<8} F1 macro {f1(e)}  par pli {e['par_pli']}  {e['configuration']}"
        )
    lines.append(f"   alerte au rappel 0,95 : {first['cibles']['alerte']['seuil_rappel_095']}")
    action = first["cibles"]["action"]
    lines.append(
        f"   action par écrivain {action['par_ecrivain']}, par source {action['par_source']}"
    )
    e = second["action"]
    lines.append(
        f"2. gabarits ({e['exemples']} cartes) : action F1 macro {f1(e)} ; mêmes cartes au "
        f"découpage 1 : {second['memes_cartes_decoupage_metiers']:.3f} ; écart "
        f"{second['ecart_au_decoupage_metiers']:+.3f} {second['ecart_ic95']}"
    )
    lines.append(f"3. écrivain ({third['elements_aux_deux_textes']} mails aux deux textes)")
    for name, e in third["cellules"].items():
        lines.append(f"   {name:<16} action F1 macro {f1(e)}  {e['configuration']}")
    for name, e in third["ecarts"].items():
        lines.append(
            f"   {name} : appris sur l'autre − appris sur le même "
            f"{e['appris_sur_l_autre_moins_appris_sur_le_meme']:+.3f} {e['ic95']}"
        )
    return "\n".join(lines)


# --- jeu de développement réel (M6, étape 0 B) -----------------------------------------------
#
# Règle M0 modifiée le 22/11, avant tout tirage (plan, §11) : environ 100 cartes de texte écrit
# hors de nos générateurs, consultables autant qu'on veut, jugées par Claude avec le guide,
# dont 30 rejugées à l'aveugle par Théo. Un garde-fou contre un levier qui ne gagnerait que
# sur le simulateur, pas une mesure humaine (P2).

DEV_SERIES = "m6-dev-reel"
DEV_TIRAGE = f"{DEV_SERIES}.tirage.json"
DEV_SAMPLE = f"{DEV_SERIES}-echantillon"
DEV_GEL = f"{DEV_SERIES}.gel.json"
DEV_REPORT_PATH = Path(__file__).with_name("comprendre_dev_reel.json")
CLAUDE = "claude"  # l'annotateur principal du jeu de développement
CLAIRE_ITEMS = Path(__file__).resolve().parents[3] / "collect/demo/u_claire/2026-11-19/items.jsonl"
CLAIRE_DAY = date(2026, 11, 19)
# Un élément par modèle de phrase : 55 mails reçus et 15 réunions (le jeu de David n'a ni
# tâches ni fichiers ; Claire en apporte 9).
DAVID_QUOTA = {"mail": 55, "calendar": 15}
DEV_SAMPLE_SIZE = 30
DEV_AGREEMENT = 0.85  # sous ce seuil d'accord sur l'action, Théo arbitre l'échantillon (§11)
_HELENE_RELATION = {
    "hiérarchie": "votre N+1",
    "direction": "comité de direction",
    "collègue": "collègue",
    "client": "client",
    "fournisseur": "fournisseur",
    "automate": "envoi automatique",
}


def claire_viewpoint(items: Sequence[Item]) -> Viewpoint:
    """Claire Fontaine, consultante, au brief du 19/11 à 8 h 30 (scénario de la journée réelle :
    Victor Mabe est son client, Titouan Synayoko son associé)."""
    relations = {"victor.mabe": "client", "titouan.synayoko": "associé"}
    manager = next(p.email for i in items for p in i.participants if p.role == "to")
    people = {
        p.email: next((r for k, r in relations.items() if p.email.startswith(k)), "contact")
        for i in items
        for p in i.participants
        if p.email != manager
    }
    context = (
        "Vous êtes Claire Fontaine, consultante (associé : Titouan Synayoko ; client : Victor "
        f"Mabe). Brief du {fr_day(CLAIRE_DAY)} à 8 h 30."
    )
    return Viewpoint(manager, brief_moment(CLAIRE_DAY), people, context)


def helene_viewpoint(day: date, items: Iterable[Item]) -> Viewpoint:
    """Hélène Marchand, directrice des opérations (jeu de David), le jour du brief à 8 h 30."""
    from packages.analyse.scoring.ml.persona import GRAPH_HELENE

    people = {
        p.email: _HELENE_RELATION[GRAPH_HELENE.sender_kind(p.email)]
        for i in items
        for p in i.participants
        if p.email.lower() != GRAPH_HELENE.address
    }
    context = (
        "Vous êtes Hélène Marchand, directrice des opérations de Verdier Expertises (N+1 : "
        f"Bruno Lefèvre). Brief du {fr_day(day)} à 8 h 30."
    )
    return Viewpoint(GRAPH_HELENE.address, brief_moment(day), people, context)


def _next_weekday(day: date) -> date:
    day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def brief_day(item: Item) -> date:
    """Le premier brief où l'élément apparaît : le matin ouvré qui suit sa réception ; une
    réunion, le matin de son jour si elle était déjà connue, sinon le suivant."""
    from packages.collect.trainset.world import PARIS

    created = item.created_at.astimezone(PARIS)
    if item.source == "calendar" and item.start_at is not None:
        start = item.start_at.astimezone(PARIS).date()
        if start.weekday() < 5 and item.created_at <= brief_moment(start):
            return start
    return _next_weekday(created.date())


def _claire_items(path: Path = CLAIRE_ITEMS) -> list[Item]:
    return [Item.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def draw_dev(
    david_dir: Path,
    claire: Path = CLAIRE_ITEMS,
    series: Path = SERIES,
    quota: dict[str, int] = DAVID_QUOTA,
    sample_size: int = DEV_SAMPLE_SIZE,
    seed: int = 0,
) -> dict:
    """Tire le jeu de développement réel et l'échantillon de Théo. Refusé si le tirage existe.

    - Claire : ses 34 éléments, tous, au brief du 19/11 ;
    - David : un élément par modèle de phrase (`sample.template` de l'objet), mails reçus et
      réunions ; les modèles tirés sont inscrits au tirage, pour les exclure du futur test
      de transfert sur David (règle M0 modifiée le 22/11) ;
    - l'échantillon : `sample_size` cartes du jeu, numérotées à neuf et dans un autre ordre,
      à juger par Théo sans voir les réponses de Claude."""
    from packages.analyse.scoring.ml.graph_convert import load_graph_dataset
    from packages.analyse.scoring.ml.sample import template

    if (series / DEV_TIRAGE).exists():
        raise FileExistsError(f"{DEV_TIRAGE} existe : un jeu de mesure ne se retire pas")
    rng = random.Random(f"m6:dev-reel:{seed}")
    claire_items = _claire_items(claire)
    vp = claire_viewpoint(claire_items)
    cards = [("claire", i, vp, card(i, vp, claire_items)) for i in claire_items]

    dataset = load_graph_dataset(david_dir)
    pools: dict[str, dict[str, list[Item]]] = {s: {} for s in quota}
    for item in dataset.items:
        extras = dataset.extras.get(item.item_id)
        if item.source in quota and not (extras and extras.is_sent):
            pools[item.source].setdefault(template(item.title), []).append(item)
    templates: dict[str, list[str]] = {}
    for source, n in quota.items():
        keys = sorted(pools[source])
        rng.shuffle(keys)
        templates[source] = sorted(keys[:n])
        for key in keys[:n]:
            item = rng.choice(sorted(pools[source][key], key=lambda i: i.item_id))
            day = brief_day(item)
            known = [i for i in dataset.items if i.created_at <= brief_moment(day)]
            vp = helene_viewpoint(day, [item])
            cards.append(("david", item, vp, card(item, vp, known)))
    rng.shuffle(cards)
    tasks, behind = [], {}
    for n, (origin, item, vp, c) in enumerate(cards, 1):
        task_id = f"r-e{n:03d}"
        tasks.append({"task_id": task_id, "kind": "element", "context": vp.context, "card": c})
        behind[task_id] = {"item_id": item.item_id, "origine": origin}
    write_series(tasks, series / f"{DEV_SERIES}.jsonl")
    chosen = rng.sample([t["task_id"] for t in tasks], sample_size)
    sample = {f"r-x{n:03d}": task_id for n, task_id in enumerate(chosen, 1)}
    by_id = {t["task_id"]: t for t in tasks}
    write_series(
        [by_id[t] | {"task_id": s, "pour": ["theo"]} for s, t in sample.items()],
        series / f"{DEV_SAMPLE}.jsonl",
    )
    info = {
        "graine": seed,
        "decision": "22/11 (Théo) : règle M0 modifiée, jeu de développement réel (plan, §11)",
        "david_modeles_exclus_du_test": templates,
        "elements": behind,
        "echantillon_theo": sample,
    }
    _write_json(series / DEV_TIRAGE, info)
    return info


def dev_labels(
    series: Path = SERIES, annotator: str = CLAUDE, name: str = DEV_SERIES
) -> dict[str, dict]:
    """Les réponses d'un annotateur à une série du jeu de développement (premières
    présentations seulement)."""
    store = AnswerStore(answers_path(name, annotator, series))
    return {t: a for (t, repeat), a in store.answers.items() if not repeat}


def dev_agreement(series: Path = SERIES) -> dict:
    """Accord de Théo (échantillon, à l'aveugle) avec Claude, cible par cible, et désaccords."""
    info = json.loads((series / DEV_TIRAGE).read_text(encoding="utf-8"))
    claude, theo = dev_labels(series), dev_labels(series, "theo", DEV_SAMPLE)
    pairs = [(t, claude[t], theo[s]) for s, t in info["echantillon_theo"].items() if s in theo]
    out: dict = {"cartes": len(pairs)}
    for target in TARGETS:
        same = sum(a[target] == b[target] for _, a, b in pairs)
        out[target] = round(same / len(pairs), 3) if pairs else None
    out["desaccords_action"] = sorted(t for t, a, b in pairs if a["action"] != b["action"])
    return out


def dev_truth(series: Path = SERIES) -> dict[str, dict]:
    """La vérité du jeu : Claude, sauf sur l'échantillon si l'accord sur l'action est sous le
    seuil écrit d'avance : là, l'arbitrage de Théo (`m6-dev-reel_arbitrage.jsonl`)."""
    labels = dev_labels(series)
    agreement = dev_agreement(series)
    if agreement["cartes"] and agreement["action"] < DEV_AGREEMENT:
        labels |= dev_labels(series, ARBITRAGE)
    return labels


def freeze_dev(series: Path = SERIES, annotator: str = CLAUDE) -> dict:
    """Gèle séries, tirage et réponses présentes (Claude d'abord, puis Théo et l'arbitrage),
    avant la première mesure d'un modèle. Refusé si une carte n'a pas de réponse de Claude."""
    tasks = load_series(series / f"{DEV_SERIES}.jsonl")
    left = [t["task_id"] for t in tasks if t["task_id"] not in dev_labels(series, annotator)]
    if left:
        raise ValueError(f"cartes sans réponse de {annotator} : {len(left)}")
    names = [f"{DEV_SERIES}.jsonl", f"{DEV_SAMPLE}.jsonl", DEV_TIRAGE]
    names += [
        answers_path(n, who, series).name
        for n, who in ((DEV_SERIES, CLAUDE), (DEV_SAMPLE, "theo"), (DEV_SERIES, ARBITRAGE))
        if answers_path(n, who, series).exists()
    ]
    freeze = {"fichiers": {name: _sha(series / name) for name in names}}
    _write_json(series / DEV_GEL, freeze)
    return freeze


def dev_changed(series: Path = SERIES) -> list[str]:
    """Fichiers du jeu de développement modifiés depuis leur gel (tous, si pas de gel)."""
    path = series / DEV_GEL
    if not path.exists():
        return [DEV_GEL]
    files = json.loads(path.read_text(encoding="utf-8"))["fichiers"]
    return sorted(n for n, d in files.items() if not (series / n).exists() or _sha(series / n) != d)


def valider_reel(
    series: Path = SERIES,
    model_path: Path = MODEL_PATH,
    embedder: Embedder | None = None,
    out_report: Path | None = DEV_REPORT_PATH,
) -> dict:
    """La mesure d'un modèle sur le jeu de développement réel, refusée si les étiquettes ont
    changé depuis leur gel : on ne corrige pas une étiquette après avoir vu une erreur."""
    cards, origins, truth = _dev_set(series)
    reference = Reference.load(model_path)
    embedder = embedder or FastEmbedder(reference.embedder_name)
    X = design(cards, embedder, reference.fiche, texte=reference.texte)
    sources = [c["source"] for c in cards]
    result = evaluate(reference, X, sources, truth)
    pred = predict(reference.models["action"], X)
    k = len(ACTIONS)
    result["action"]["par_origine"] = {
        origin: round(f1_macro(pred[m], truth["action"][m], k), 4)
        for origin in ("claire", "david")
        if (m := np.array([o == origin for o in origins])).any()
    }
    report = {
        "modele": model_path.name,
        "modele_sha256": _sha(model_path),
        "accord_theo_claude": dev_agreement(series),
        "etiquettes": "Claude (guide d'annotation), échantillon rejugé par Théo à l'aveugle",
    } | result
    if out_report is not None:
        out_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    return report


def dev_summary(report: dict) -> str:
    lines = [f"{report['elements']} cartes ; accord Théo / Claude : {report['accord_theo_claude']}"]
    for target in TARGETS:
        e = report[target]
        low, high = e["f1_macro_ic95"]
        lines.append(
            f"  {target:<8} F1 macro {e['f1_macro']:.3f} [{low:.3f} ; {high:.3f}]  "
            f"par source {e['par_source']}"
        )
    lines.append(f"  action par origine {report['action']['par_origine']}")
    return "\n".join(lines)


# --- levier 1 : fiche hybride ----------------------------------------------------------------
#
# Le vecteur MiniLM lit le style de la carte ; la fiche donne à la régression, telles quelles,
# les informations que le guide utilise : place du manager, relation de l'expéditeur, réponse à
# une invitation, échéance… Elle est lue sur la carte, comme l'annotateur la voit : la même
# fonction sert au simulateur, au jeu réel et aux jeux gelés, sans rien de caché du simulateur.

FICHES = ("aucune", "hybride", "seule")  # vecteur seul, vecteur + fiche, fiche seule
VARIANTS = {"R": "aucune", "H": "hybride", "F": "seule"}
LEVIER_REPORT_PATH = Path(__file__).with_name("comprendre_levier_fiche.json")

# Relation de l'expéditeur, par mots-clés dans le champ « De » : les mondes l'écrivent
# différemment (« Client, Groupe Belfroid », « client », « associé », « votre pair »).
# L'ordre compte : « Client, Direction régionale » est un client.
RELATIONS = (
    ("automate", ("automatique",)),
    ("client", ("client",)),
    ("fournisseur", ("fournisseur", "partenaire", "prestataire")),
    ("hierarchie", ("n+1",)),
    ("direction", ("pair", "direction", "comite")),
    ("interne", ("equipe", "collegue", "associe")),
)
# Verbe de l'intitulé d'une tâche (guide : « le verbe de l'intitulé donne l'action »).
TASK_VERBS = {
    "repondre": ("repondre", "donner"),
    "faire": (
        "appeler", "transmettre", "envoyer", "mettre", "relancer", "renvoyer", "rappeler",
        "faire", "completer", "traiter", "deposer", "organiser", "reserver", "contacter",
        "saisir", "finaliser", "commander", "payer",
    ),
    "decider": ("signer", "approuver", "valider", "arbitrer", "choisir", "decider"),
    "preparer": ("preparer",),
    "relire": ("relire", "verifier", "revoir", "corriger"),
}  # fmt: skip
FEATURE_NAMES = (
    *(f"source={s}" for s in SOURCE_NAMES),
    "mail:destinataire_direct",
    "mail:en_copie",
    *(f"mail:expediteur={r}" for r, _ in RELATIONS),
    "mail:expediteur=inconnu",
    "mail:de_vous",
    "mail:reponse",
    "mail:transfert",
    "mail:importance_haute",
    "mail:non_lu",
    "mail:desabonnement",
    "reunion:organise",
    "reunion:pas_repondu",
    "reunion:acceptee",
    "reunion:refusee",
    "reunion:annulee",
    "reunion:aujourd_hui",
    "reunion:demain",
    "reunion:hier",
    "tache:echeance",
    "tache:en_retard",
    "tache:terminee",
    "tache:planner",
    *(f"tache:verbe={v}" for v in TASK_VERBS),
    "fichier:modifie_par_vous",
    "fichier:cite",
)
_PREFIX_RE = re.compile(r"^\s*(re|tr|fw|fwd)\s?:", re.IGNORECASE)


def _ascii(text: str) -> str:
    """Minuscules sans accents : « Comité » et « comite » se rejoignent."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def relation(sender: str) -> str:
    """La relation lue dans le champ « De » ; « inconnu » pour une adresse brute."""
    if "<" in sender:
        return "inconnu"
    plain = _ascii(sender)
    return next((r for r, words in RELATIONS if any(w in plain for w in words)), "inconnu")


def task_verb(title: str) -> str | None:
    """Le groupe du premier verbe de l'intitulé, après un éventuel « URGENT – »."""
    words = [w for w in re.findall(r"[a-z+]+", _ascii(title)) if w != "urgent"]
    return next((g for g, verbs in TASK_VERBS.items() if words and words[0] in verbs), None)


def card_features(c: dict) -> np.ndarray:
    """La fiche d'une carte : un vecteur de 0 et de 1 dans l'ordre de FEATURE_NAMES."""
    fields = {label: value for label, value in c["fields"]}
    on: set[str] = {f"source={c['source']}"}
    if c["source"] == "mail":
        if fields.get("À", "").startswith("vous"):
            on.add("mail:destinataire_direct")
        if fields.get("Copie") == "vous":
            on.add("mail:en_copie")
        if fields.get("De") == "vous":  # mail envoyé par le manager lui-même
            on.add("mail:de_vous")
        elif "De" in fields:
            on.add(f"mail:expediteur={relation(fields['De'])}")
        prefix = _PREFIX_RE.match(c["title"])
        if prefix:
            on.add("mail:reponse" if prefix.group(1).lower() == "re" else "mail:transfert")
        if fields.get("Importance") == "haute":
            on.add("mail:importance_haute")
        if fields.get("Statut") == "non lu":
            on.add("mail:non_lu")
        plain = _ascii(c["text"])
        if "desabonn" in plain or "unsubscribe" in plain:
            on.add("mail:desabonnement")
    elif c["source"] == "calendar":
        answer = fields.get("Votre réponse", "")
        on |= {
            name
            for name, hit in (
                ("reunion:organise", answer == "vous organisez"),
                ("reunion:pas_repondu", answer == "pas encore répondu"),
                ("reunion:acceptee", answer in ("acceptée", "provisoire")),
                ("reunion:refusee", answer == "refusée"),
                ("reunion:annulee", fields.get("Statut") == "annulée"),
                ("reunion:aujourd_hui", c["when"].startswith("aujourd'hui")),
                ("reunion:demain", c["when"].startswith("demain")),
                ("reunion:hier", c["when"].startswith("hier")),
            )
            if hit
        }
    elif c["source"] == "task":
        due = fields.get("Échéance", "")
        on |= {
            name
            for name, hit in (
                ("tache:echeance", bool(due)),
                ("tache:en_retard", "en retard" in due),
                ("tache:terminee", fields.get("Statut") == "terminée"),
                ("tache:planner", fields.get("Liste", "").startswith("Planner")),
            )
            if hit
        }
        verb = task_verb(c["title"])
        if verb:
            on.add(f"tache:verbe={verb}")
    else:
        if fields.get("Modifié par") == "vous":
            on.add("fichier:modifie_par_vous")
        if "Cité dans" in fields:
            on.add("fichier:cite")
    return np.array([float(name in on) for name in FEATURE_NAMES])


def design(
    cards: Sequence[dict],
    embedder: Embedder,
    fiche: str = "aucune",
    cache: VectorCache | None = None,
    texte: str = "brut",
) -> np.ndarray:
    """L'entrée du modèle : vecteur de la carte (texte brut ou normalisé), fiche, ou les deux
    bout à bout."""
    if fiche not in FICHES:
        raise ValueError(f"fiche inconnue : {fiche!r}")
    parts = []
    if fiche != "seule":
        texts = [encoder_text(c, texte) for c in cards]
        parts.append(cache.embed(embedder, texts) if cache else embedder.embed(texts))
    if fiche != "aucune":
        parts.append(np.array([card_features(c) for c in cards]).reshape(len(cards), -1))
    return np.hstack(parts)


def _variant_input(X: np.ndarray, F: np.ndarray, fiche: str) -> np.ndarray:
    return {"aucune": X, "hybride": np.hstack([X, F]), "seule": F}[fiche]


def _dev_set(series: Path = SERIES) -> tuple[list[dict], list[str], dict]:
    """Cartes, origines et vérité (codes) du jeu de développement réel, gelé."""
    changed = dev_changed(series)
    if changed:
        raise ValueError(f"jeu de développement non gelé ou modifié : {changed}")
    info = json.loads((series / DEV_TIRAGE).read_text(encoding="utf-8"))
    labels = dev_truth(series)
    tasks = [t for t in load_series(series / f"{DEV_SERIES}.jsonl") if t["task_id"] in labels]
    truth = {
        target: np.array(
            [
                TARGETS[target].index(
                    ("oui" if labels[t["task_id"]]["alerte"] else "non")
                    if target == "alerte"
                    else labels[t["task_id"]][target]
                )
                for t in tasks
            ],
            dtype=int,
        )
        for target in TARGETS
    }
    origins = [info["elements"][t["task_id"]]["origine"] for t in tasks]
    return [t["card"] for t in tasks], origins, truth


def fit_from_report(
    examples: Sequence[Example], X: np.ndarray, first: dict, name: str, fiche: str
) -> Reference:
    """Le modèle d'une variante appris sur tout l'entraînement, avec les réglages et le seuil
    d'alerte choisis sur le découpage 1 de la validation dure (G1)."""
    models = {}
    for target, classes in TARGETS.items():
        cfg = first["cibles"][target]["configuration"]
        config = Config(cfg["l2"], cfg["classes_pondérées"])
        models[target] = fit_softmax(X, _codes(examples, target), len(classes), config)
    threshold = first["cibles"]["alerte"]["seuil_rappel_095"]["seuil"]
    return Reference(models, threshold, name, fiche)


def _gap(truth: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    low, high = paired_diff_ci(_f1_action, a, b, truth)
    diff = _f1_action(a, truth) - _f1_action(b, truth)
    return {"ecart": round(diff, 4), "ic95": [round(low, 4), round(high, 4)]}


DECISIVE = ("reel", "gabarits", "gabarits→llm", "llm→gabarits")


def decide(gaps: dict, recall: dict) -> dict:
    """La règle écrite le 22/11 avant la mesure (plan, §11) : H retenue si un gain significatif
    sur une mesure décisive, sans perte significative sur le découpage 1, veto des alertes ;
    F la remplace si F − H n'est significatif nulle part (G3)."""
    hr, fh = gaps["H-R"], gaps["F-H"]
    gain = [m for m in DECISIVE if hr[m]["ic95"][0] > 0]
    loss = hr["metiers"]["ic95"][1] < 0
    veto = recall["H"] < ALERT_RECALL
    adopt = bool(gain) and not loss and not veto
    tie = all(fh[m]["ic95"][0] <= 0 <= fh[m]["ic95"][1] for m in (*DECISIVE, "metiers"))
    return {
        "gains_significatifs_de_H": gain,
        "perte_significative_decoupage_1": loss,
        "veto_alertes": veto,
        "retenue": ("F" if tie else "H") if adopt else "R",
    }


def run_variant(
    examples: list[Example],
    Xv: np.ndarray,
    Xd: np.ndarray,
    dev: tuple[list[dict], list[str], dict],
    fiche: str,
    n_folds: int = HARD_FOLDS,
    log: Callable[[str], None] = lambda _: None,
    grid: Sequence[Config] | None = None,
    texte: str = "brut",
) -> tuple[dict, dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Une variante : validation dure, puis modèle appris sur tout l'entraînement (réglages du
    découpage 1) et mesuré sur le jeu réel. Rend les chiffres, les prédictions de l'action par
    mesure (vérité, prédiction) pour les écarts appariés, et le rappel des alertes (veto G5)."""
    dev_cards, origins, dev_truth_codes = dev
    y_dev = dev_truth_codes["action"]
    keep: dict = {}
    hard = hard_validation(examples, Xv, n_folds, log=log, keep=keep, grid=grid)
    first = hard["metiers"]
    reference = fit_from_report(examples, Xv, first, "", fiche)
    reference.texte = texte
    result = evaluate(reference, Xd, [c["source"] for c in dev_cards], dev_truth_codes)
    keep["reel"] = (np.arange(len(y_dev)), predict(reference.models["action"], Xd))
    alerts = dev_truth_codes["alerte"] == 1
    dev_alert = predict(reference.models["alerte"], Xd, reference.alert_threshold)
    entry = {
        "fiche": fiche,
        "texte": texte,
        "metiers": {
            t: {
                "f1_macro": e["f1_macro"],
                "f1_macro_ic95": e["f1_macro_ic95"],
                "configuration": e["configuration"],
                "par_source": e["par_source"],
            }
            for t, e in first["cibles"].items()
        },
        "alerte_rappel_095": first["cibles"]["alerte"]["seuil_rappel_095"],
        "gabarits": {k: hard["gabarits"]["action"][k] for k in ("f1_macro", "f1_macro_ic95")},
        "ecrivain": {
            c: {k: e[k] for k in ("f1_macro", "f1_macro_ic95")}
            for c, e in hard["ecrivain"]["cellules"].items()
        },
        "reel": {
            t: {k: result[t][k] for k in ("f1_macro", "f1_macro_ic95", "par_source")}
            for t in TARGETS
        }
        | {
            "action_par_origine": {
                o: round(_f1_action(keep["reel"][1][m], y_dev[m]), 4)
                for o in ("claire", "david")
                if (m := np.array([x == o for x in origins])).any()
            },
            "action_classes": result["action"]["classes"],
            "action_confusion": result["action"]["confusion"],
            "alertes_reelles_detectees": f"{int(dev_alert[alerts].sum())}/{int(alerts.sum())}",
        },
    }
    y = _codes(examples, "action")
    preds = {m: (y[idx], p) if m != "reel" else (y_dev, p) for m, (idx, p) in keep.items()}
    return entry, preds, first["cibles"]["alerte"]["seuil_rappel_095"]["rappel"]


def levier_fiche(
    corpora: dict[str, Path] = CORPORA,
    embedder: Embedder | None = None,
    cache: VectorCache | None = None,
    series: Path = SERIES,
    out_report: Path = LEVIER_REPORT_PATH,
    n_folds: int = HARD_FOLDS,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """Le levier 1 : R, H et F sur la validation dure et le jeu réel, écarts appariés, décision
    par la règle écrite d'avance. Aucun jeu gelé n'est lu."""
    embedder = embedder or FastEmbedder(DEFAULT_MODEL)
    cache = cache or VectorCache(cache_path(embedder))
    log("cartes des personas « train »…")
    examples, _ = training_examples(corpora)
    X = embed_with_progress(cache, embedder, [e.text for e in examples], log=log)
    F = np.array([card_features(e.card) for e in examples])
    dev_cards, origins, dev_truth_codes = _dev_set(series)
    X_dev = design(dev_cards, embedder, "aucune", cache)
    F_dev = design(dev_cards, embedder, "seule")
    report: dict = {
        "fiche": list(FEATURE_NAMES),
        "fiche_frequence": {
            n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, F.mean(axis=0), strict=True)
        },
        "variantes": {},
    }
    preds: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    recall = {}
    dev = (dev_cards, origins, dev_truth_codes)
    for name, fiche in VARIANTS.items():
        log(f"variante {name} ({fiche})")
        Xv, Xd = _variant_input(X, F, fiche), _variant_input(X_dev, F_dev, fiche)
        entry, preds[name], recall[name] = run_variant(examples, Xv, Xd, dev, fiche, n_folds, log)
        report["variantes"][name] = entry
    measures = ("metiers", *DECISIVE)
    gaps = {
        f"{a}-{b}": {m: _gap(preds[a][m][0], preds[a][m][1], preds[b][m][1]) for m in measures}
        for a, b in (("H", "R"), ("F", "H"), ("F", "R"))
    }
    report["ecarts_apparies_action"] = gaps
    report["decision"] = decide(gaps, recall)
    out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def levier_summary(report: dict) -> str:
    def f1(e: dict) -> str:
        low, high = e["f1_macro_ic95"]
        return f"{e['f1_macro']:.3f} [{low:.3f} ; {high:.3f}]"

    lines = []
    for name, v in report["variantes"].items():
        cells = ", ".join(f"{c} {e['f1_macro']:.3f}" for c, e in v["ecrivain"].items())
        lines.append(
            f"{name} ({v['fiche']}) : métiers {f1(v['metiers']['action'])} ; gabarits "
            f"{f1(v['gabarits'])} ; {cells} ; réel {f1(v['reel']['action'])} "
            f"(alertes réelles {v['reel']['alertes_reelles_detectees']}, précision des "
            f"alertes au rappel 0,95 : {v['alerte_rappel_095']['précision']})"
        )
    for pair, gaps in report["ecarts_apparies_action"].items():
        lines.append(
            f"{pair} : " + " ; ".join(f"{m} {g['ecart']:+.3f} {g['ic95']}" for m, g in gaps.items())
        )
    lines.append(f"décision : {report['decision']}")
    return "\n".join(lines)


# --- levier 2 : augmentation du texte --------------------------------------------------------
#
# Au levier 1, la fiche seule bat l'hybride sur le jeu réel : le vecteur lit encore le décor
# propre à chaque générateur. Règles écrites le 23/11 avant la mesure (plan, §11), fixées
# d'après le simulateur et le guide, pas d'après les erreurs du jeu réel.

TEXTES = ("brut", "normalise")
TEXT_REPORT_PATH = Path(__file__).with_name("comprendre_levier_texte.json")
SEEDS = (0, 1, 2)  # G4
TEXT_GRID = tuple(Config(l2, False) for l2 in (1e-2, 1e-3, 1e-4))  # plan, §11, 23/11
_TEAMS_BLOCK = re.compile(r"_{8,}\s*Réunion Microsoft Teams.*?Code secret\s*:\s*\S+", re.DOTALL)
_TEAMS_LINE = re.compile(
    r"^\s*(_{8,}|Réunion Microsoft Teams|Rejoindre\s*:|Numéro de réunion|Code secret).*$",
    re.MULTILINE | re.IGNORECASE,
)
_UNSUBSCRIBE = re.compile(r"^[^\n]*(désabonn|unsubscribe)[^\n]*$", re.MULTILINE | re.IGNORECASE)
_CLOSING_LINE = re.compile(
    r"^\s*(cdt|cordialement|bien cordialement|bien à vous|bonne journée|belle journée|merci)"
    r"[\s,.!]*$",
    re.IGNORECASE,
)
_CLOSING_INLINE = re.compile(r"\s(?:Cordialement|Bien à vous|Bien cordialement|Cdt),.*$", re.DOTALL)
_SALUTATION = re.compile(
    r"^(bonjour|bonsoir|salut|hello|madame|monsieur|cher|chère)\b|^\w+,$", re.IGNORECASE
)
_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"\S+@\S+\.\w+")
_DATE_WORDS = re.compile(
    r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|janvier|février|mars|avril|mai|"
    r"juin|juillet|août|septembre|octobre|novembre|décembre)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
PERTURBATIONS = ("objet", "phrase", "ordre", "coupe", "politesse")


def mask(text: str) -> str:
    """Liens, adresses, chiffres, jours et mois masqués : le décor variable, pas la demande."""
    text = _URL.sub("[lien]", text)
    text = _EMAIL.sub("[adresse]", text)
    text = _DATE_WORDS.sub("#", text)
    return re.sub(r"\d+", "#", text)


def drop_signature(text: str) -> str:
    """Le texte avant la formule de fin (seule sur sa ligne, ou en ligne suivie d'une virgule)."""
    lines = text.split("\n")
    cut = next((n for n, line in enumerate(lines) if _CLOSING_LINE.match(line)), len(lines))
    return _CLOSING_INLINE.sub("", "\n".join(lines[:cut]))


def clean_body(text: str) -> str:
    """Le texte sans bloc de réunion Teams, pied de désabonnement ni signature, masqué."""
    text = _TEAMS_LINE.sub("", _TEAMS_BLOCK.sub("", text))
    text = drop_signature(_UNSUBSCRIBE.sub("", text))
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in mask(text).split("\n")]
    return "\n".join(line for line in lines if line)


def encoder_text(c: dict, texte: str = "brut") -> str:
    """Ce que lit l'encodeur : la carte entière (brut), ou « source : objet » et le texte
    nettoyé (normalisé : la date et les champs sont déjà dans la fiche)."""
    if texte == "brut":
        return card_text(c)
    if texte != "normalise":
        raise ValueError(f"texte inconnu : {texte!r}")
    return f"{SOURCE_NAMES[c['source']]} : {mask(c['title'])}\n{clean_body(c['text'])}".strip()


def perturbed_text(c: dict, rng: random.Random, texte: str = "brut") -> str:
    """Une copie perturbée de la carte, même étiquette : une ou deux perturbations tirées
    parmi PERTURBATIONS, sur l'objet et le texte (les champs ne bougent pas)."""
    title, body = c["title"], c["text"]
    if texte == "normalise":
        title, body = mask(title), clean_body(body)
    sentences = [x for x in _SENTENCE.split(body) if x.strip()]
    for op in rng.sample(PERTURBATIONS, rng.choice((1, 2))):
        if op == "objet":
            title = ""
        elif op == "phrase" and len(sentences) >= 2:
            del sentences[rng.randrange(len(sentences))]
        elif op == "ordre":
            rng.shuffle(sentences)
        elif op == "coupe" and len(sentences) >= 2:
            sentences = sentences[: rng.randint(1, len(sentences) - 1)]
        elif op == "politesse":
            if sentences and _SALUTATION.match(sentences[0].strip()):
                sentences = sentences[1:]
            sentences = [x for x in drop_signature("\n".join(sentences)).split("\n") if x.strip()]
    body = " ".join(x.strip() for x in sentences)
    if texte == "normalise":
        return f"{SOURCE_NAMES[c['source']]} : {title}\n{body}".strip()
    return card_text(c | {"title": title, "text": body})


def augmented_examples(examples: Sequence[Example], seed: int, texte: str) -> list[Example]:
    """Une copie perturbée de chaque exemple, tirée par carte (graine, élément, écrivain) :
    le même exemple donne la même copie quel que soit l'ordre."""
    out = []
    for e in examples:
        rng = random.Random(f"m6:augmentation:{seed}:{texte}:{e.item_id}:{e.writer}")
        text = perturbed_text(e.card, rng, texte)
        out.append(replace(e, text=text, augmented=True))
    return out


def text_groups(seeds: Sequence[int] = SEEDS) -> dict[str, tuple[str, ...]]:
    """Les groupes de variantes, du plus simple au plus complexe (G3)."""
    return {"N": ("N",), "A": tuple(f"A{s}" for s in seeds), "NA": tuple(f"NA{s}" for s in seeds)}


def _beats(gaps: Sequence[dict]) -> bool:
    """Gain significatif sur une mesure décisive et aucune perte sur le découpage 1, pour
    chaque graine."""
    return all(
        any(g[m]["ic95"][0] > 0 for m in DECISIVE) and g["metiers"]["ic95"][1] >= 0 for g in gaps
    )


def decide_text(gaps: dict, recall: dict, seeds: Sequence[int] = SEEDS) -> dict:
    """La règle du 23/11 : une variante remplace la retenue si elle la bat (`_beats`) avec
    chacune de ses graines, sous le veto des alertes ; de la plus simple à la plus complexe
    (N, A, N+A), une variante plus complexe doit battre la plus simple déjà retenue (G3)."""
    retained, steps = "H", {}
    for group, names in text_groups(seeds).items():
        veto = any(recall[n] < ALERT_RECALL for n in names)
        pairs = [f"{n}-{_paired_name(n, retained)}" for n in names]
        wins = _beats([gaps[p] for p in pairs])
        steps[group] = {"contre": retained, "bat": wins, "veto_alertes": veto}
        if wins and not veto:
            retained = group
    return {"etapes": steps, "retenue": retained}


def _paired_name(name: str, group: str) -> str:
    """La variante du groupe `group` à comparer à `name` : même graine si elle en a."""
    if group in ("H", "N"):
        return group
    return f"{group}{name[-1]}"


def levier_texte(
    corpora: dict[str, Path] = CORPORA,
    embedder: Embedder | None = None,
    cache: VectorCache | None = None,
    series: Path = SERIES,
    out_report: Path = TEXT_REPORT_PATH,
    n_folds: int = HARD_FOLDS,
    seeds: Sequence[int] = SEEDS,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """Le levier 2 : H, N, A et N+A (3 graines), toutes avec la fiche, sur la validation dure
    et le jeu réel ; écarts appariés et décision par la règle écrite d'avance."""
    embedder = embedder or FastEmbedder(DEFAULT_MODEL)
    cache = cache or VectorCache(cache_path(embedder))
    log("cartes des personas « train »…")
    examples, _ = training_examples(corpora)
    F = np.array([card_features(e.card) for e in examples])
    dev = _dev_set(series)
    F_dev = design(dev[0], embedder, "seule")
    specs: dict[str, tuple[str, int | None]] = {"H": ("brut", None), "N": ("normalise", None)}
    specs |= {f"A{s}": ("brut", s) for s in seeds} | {f"NA{s}": ("normalise", s) for s in seeds}
    report: dict = {"grille": [c.as_dict() for c in TEXT_GRID], "variantes": {}}
    preds: dict = {}
    recall: dict[str, float] = {}
    base: dict[str, np.ndarray] = {}
    measures = ("metiers", *DECISIVE)
    gaps: dict[str, dict] = {}

    def run(name: str) -> None:
        texte, seed = specs[name]
        log(f"variante {name} (texte {texte}, graine {seed})")
        if texte not in base:
            texts = [encoder_text(e.card, texte) for e in examples]
            base[texte] = embed_with_progress(cache, embedder, texts, log=log)
        X, used = base[texte], list(examples)
        if seed is not None:
            extra = augmented_examples(examples, seed, texte)
            vectors = embed_with_progress(cache, embedder, [e.text for e in extra], log=log)
            X, used = np.vstack([X, vectors]), used + extra
        Fv = np.vstack([F] * (len(used) // len(examples)))
        Xd = np.hstack([design(dev[0], embedder, "aucune", cache, texte), F_dev])
        entry, preds[name], recall[name] = run_variant(
            used, np.hstack([X, Fv]), Xd, dev, "hybride", n_folds, log, TEXT_GRID, texte
        )
        copies = used[len(examples) :]
        report["variantes"][name] = entry | {
            "graine": seed,
            "copies": len(copies),
            "copies_identiques": sum(
                c.text == encoder_text(e.card, texte)
                for c, e in zip(copies, examples, strict=False)
            ),
        }

    def gap(a: str, b: str) -> dict:
        key = f"{a}-{b}"
        if key not in gaps:
            gaps[key] = {m: _gap(preds[a][m][0], preds[a][m][1], preds[b][m][1]) for m in measures}
        return gaps[key]

    # Règle du 23/11, appliquée au fil du calcul : un groupe doit gagner avec chacune de ses
    # graines, donc une graine perdante (ou sous le veto) l'élimine et les suivantes ne sont pas
    # calculées. La décision est la même que si toutes l'avaient été (`decide_text`).
    run("H")
    retained, steps = "H", {}
    for group, names in text_groups(seeds).items():
        wins, veto, ran = True, False, 0
        for n in names:
            run(n)
            ran += 1
            veto = veto or recall[n] < ALERT_RECALL
            if veto or not _beats([gap(n, _paired_name(n, retained))]):
                wins = False
                break
        steps[group] = {
            "contre": retained,
            "bat": wins,
            "veto_alertes": veto,
            "graines_calculees": ran,
            "arret_anticipe": ran < len(names),
        }
        if wins:
            retained = group
    done = list(preds)
    for n in done:  # tous les écarts utiles entre variantes calculées
        if n != "H":
            gap(n, "H")
        if n[0] in "AN" and n != "N":
            gap(n, "N")
        if n.startswith("NA") and f"A{n[2:]}" in preds:
            gap(n, f"A{n[2:]}")
    report["ecarts_apparies_action"] = gaps
    report["moyennes_par_graine"] = {
        group: {
            m: round(float(np.mean([gaps[f"{n}-H"][m]["ecart"] for n in members])), 4)
            for m in measures
        }
        | {
            "graines": len(members),
            "ecart_type_reel": round(
                float(np.std([gaps[f"{n}-H"]["reel"]["ecart"] for n in members])), 4
            ),
        }
        for group in ("A", "NA")
        if (members := [n for n in done if n[:-1] == group])
    }
    report["decision"] = {"etapes": steps, "retenue": retained}
    out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def texte_summary(report: dict) -> str:
    lines = []
    for name, v in report["variantes"].items():
        cells = ", ".join(f"{c} {e['f1_macro']:.3f}" for c, e in v["ecrivain"].items())
        lines.append(
            f"{name:<4} métiers {v['metiers']['action']['f1_macro']:.3f} ; gabarits "
            f"{v['gabarits']['f1_macro']:.3f} ; {cells} ; réel {v['reel']['action']['f1_macro']:.3f}"
            f" (alertes réelles {v['reel']['alertes_reelles_detectees']})"
        )
    for pair, gaps in report["ecarts_apparies_action"].items():
        if pair.endswith("-H"):
            lines.append(
                f"{pair} : "
                + " ; ".join(f"{m} {g['ecart']:+.3f} {g['ic95']}" for m, g in gaps.items())
            )
    lines.append(f"moyennes sur les graines (écart à H) : {report['moyennes_par_graine']}")
    lines.append(f"décision : {report['decision']}")
    return "\n".join(lines)


# --- mesure finale sur les jeux gelés (G1) ----------------------------------------------------

# Les jeux d'éléments jugés à la main : T-gen (M5) et la série démo de M6.
FROZEN_SETS = {"T-gen": jeux_m5.NAMES["elements"], "T-démo": DEMO_SERIES}
FREEZES = (jeux_m5.GEL.name, GEL)
JOURNAL = ANNOTATIONS / "journal-geles.jsonl"
REPO = Path(__file__).resolve().parents[4]


def check_frozen(series: Path = SERIES, freezes: Sequence[str] = FREEZES) -> None:
    """Refuse la mesure si un gel manque, si des réponses ne sont pas gelées, ou si un fichier
    gelé a changé : on ne mesure que contre une vérité fixée avant (P3)."""
    for name in freezes:
        path = series / name
        if not path.exists() or not json.loads(path.read_text(encoding="utf-8"))["reponses"]:
            raise ValueError(f"{name} : réponses pas encore gelées")
    changed = changed_since_freeze(series)
    if changed:
        raise ValueError(f"fichiers gelés modifiés : {changed}")


def git_commit(paths: Sequence[Path] = (MODEL_PATH, Path(__file__))) -> str:
    """Le commit du code et du modèle mesurés ; refusé s'ils ont des modifications non
    commitées : le journal doit permettre de refaire la mesure à l'identique."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    if git("status", "--porcelain", "--", *(str(p) for p in paths)):
        raise ValueError("code ou modèle modifiés depuis le dernier commit : commiter d'abord")
    return git("rev-parse", "HEAD")


def series_cards(name: str, series: Path = SERIES) -> list[dict]:
    """Les cartes jugées d'une série, dans l'ordre de `series_truth`."""
    labels = final_labels(name, series)
    return [t["card"] for t in load_series(series / f"{name}.jsonl") if t["task_id"] in labels]


def series_truth(name: str, series: Path = SERIES) -> tuple[list[str], list[str], dict]:
    """Textes, sources et étiquettes (codes par cible) des cartes jugées d'une série."""
    labels = final_labels(name, series)
    tasks = [t for t in load_series(series / f"{name}.jsonl") if t["task_id"] in labels]
    truth = {
        target: np.array(
            [
                TARGETS[target].index(
                    ("oui" if labels[t["task_id"]]["alerte"] else "non")
                    if target == "alerte"
                    else labels[t["task_id"]][target]
                )
                for t in tasks
            ],
            dtype=int,
        )
        for target in TARGETS
    }
    return [card_text(t["card"]) for t in tasks], [t["card"]["source"] for t in tasks], truth


def self_agreement(name: str, series: Path = SERIES, annotator: str = ANNOTATOR) -> dict:
    """Cohérence de l'annotateur sur les répétitions cachées : le plafond qu'on peut espérer
    d'un modèle jugé par lui (avec un seul annotateur, pas de kappa entre annotateurs)."""
    answers = AnswerStore(answers_path(name, annotator, series)).answers
    repeated = [t for (t, rep) in answers if rep and (t, False) in answers]
    same = {
        target: sum(answers[t, True][target] == answers[t, False][target] for t in repeated)
        for target in TARGETS
    }
    return {"repetitions": len(repeated)} | {
        t: round(n / len(repeated), 3) if repeated else None for t, n in same.items()
    }


def evaluate(reference: Reference, X: np.ndarray, sources: list[str], truth: dict) -> dict:
    """Les chiffres d'un jeu : F1 macro avec intervalle (G2) pour chaque cible, par source et
    par classe ; matrice de confusion de l'action ; rappel et précision des alertes."""
    out: dict = {"elements": len(sources)}
    for target in TARGETS:
        threshold = reference.alert_threshold if target == "alerte" else None
        pred = predict(reference.models[target], X, threshold)
        out[target] = _entry(target, pred, truth[target], sources)
    return out


def mesurer(
    series: Path = SERIES,
    model_path: Path = MODEL_PATH,
    embedder: Embedder | None = None,
    journal: Path = JOURNAL,
    commit: Callable[[], str] = git_commit,
    sets: dict[str, str] = FROZEN_SETS,
    freezes: Sequence[str] = FREEZES,
) -> dict:
    """La mesure finale de la référence sur les jeux gelés, une seule fois par modèle (G1) :
    chaque lecture est ajoutée au journal (date, commit, empreinte du modèle, chiffres)."""
    check_frozen(series, freezes)
    model_sha = _sha(model_path)
    if journal.exists() and any(
        json.loads(line)["modele_sha256"] == model_sha
        for line in journal.read_text(encoding="utf-8").splitlines()
        if line
    ):
        raise ValueError("ce modèle a déjà été mesuré sur les jeux gelés : voir le journal")
    revision = commit()
    reference = Reference.load(model_path)
    embedder = embedder or FastEmbedder(reference.embedder_name)
    results = {}
    for label, name in sets.items():
        _, sources, truth = series_truth(name, series)
        X = design(series_cards(name, series), embedder, reference.fiche, texte=reference.texte)
        results[label] = {"serie": name} | evaluate(reference, X, sources, truth)
        results[label]["coherence_annotateur"] = self_agreement(name, series)
    entry = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "tache": "M6, référence",
        "commit": revision,
        "modele": model_path.name,
        "modele_sha256": model_sha,
        "encodeur": reference.embedder_name,
        "seuil_alerte": reference.alert_threshold,
        "jeux": results,
    }
    with journal.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def measure_summary(entry: dict) -> str:
    lines = [f"commit {entry['commit'][:7]}, modèle {entry['modele_sha256'][:12]}"]
    for label, result in entry["jeux"].items():
        lines.append(f"{label} ({result['serie']}, {result['elements']} éléments) :")
        for target in TARGETS:
            e = result[target]
            low, high = e["f1_macro_ic95"]
            lines.append(
                f"  {target:<8} F1 macro {e['f1_macro']:.3f} [{low:.3f} ; {high:.3f}]  "
                f"par source {e['par_source']}"
            )
        alert = result["alerte"]["classes"]["oui"]
        lines.append(
            f"  alertes : {alert['n']}, rappel {alert['rappel']}, précision {alert['précision']}"
        )
        lines.append(f"  cohérence de l'annotateur : {result['coherence_annotateur']}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if argv == ["--entrainer"]:
        report = entrainer(log=lambda line: print(line, flush=True))
        print(summary(report))
        print(f"→ {MODEL_PATH.name}, {REPORT_PATH.name}")
        return 0
    if argv == ["--valider-dur"]:
        report = valider_dur(log=lambda line: print(line, flush=True))
        print(hard_summary(report))
        print(f"→ {HARD_REPORT_PATH.name}")
        return 0
    if len(argv) == 2 and argv[0] == "--tirer-dev":
        info = draw_dev(Path(argv[1]))
        print(
            f"{DEV_SERIES} : {len(info['elements'])} cartes ; échantillon de Théo : "
            f"{len(info['echantillon_theo'])} cartes ({DEV_SAMPLE})"
        )
        return 0
    if argv == ["--geler-dev"]:
        print(f"Gelés : {', '.join(freeze_dev()['fichiers'])}")
        return 0
    if argv == ["--levier-fiche"]:
        print(levier_summary(levier_fiche(log=lambda line: print(line, flush=True))))
        print(f"→ {LEVIER_REPORT_PATH.name}")
        return 0
    if argv == ["--levier-texte"]:
        print(texte_summary(levier_texte(log=lambda line: print(line, flush=True))))
        print(f"→ {TEXT_REPORT_PATH.name}")
        return 0
    if argv == ["--valider-reel"]:
        print(dev_summary(valider_reel()))
        print(f"→ {DEV_REPORT_PATH.name}")
        return 0
    if argv == ["--mesurer"]:
        print(measure_summary(mesurer()))
        print(f"→ {JOURNAL.name}")
        return 0
    if argv == ["--tirer-demo"]:
        info = draw_tdemo_elements()
        print(
            f"{DEMO_SERIES} : {len(info['elements'])} éléments "
            f"({info['exclus_deja_vus']} éléments de démo déjà vus exclus)."
        )
        print(
            "  uv run python -m packages.analyse.scoring.ml.annoter "
            f"--serie {DEMO_SERIES} --annotateur {ANNOTATOR}"
        )
        return 0
    if argv == ["--reste-demo"]:
        print(missing_demo_answers())
        return 0
    if argv == ["--geler-demo"]:
        freeze = freeze_demo_answers()
        print(f"Réponses gelées : {', '.join(freeze['reponses'])}")
        return 0
    print(
        "usage : comprendre --entrainer | --valider-dur | --tirer-dev <data David> | "
        "--geler-dev | --valider-reel | --levier-fiche | --levier-texte | --mesurer | --tirer-demo | --reste-demo | --geler-demo"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
