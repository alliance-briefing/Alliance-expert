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
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from contracts.models import Item
from packages.analyse.rag.embeddings import DEFAULT_MODEL, Embedder, FastEmbedder
from packages.analyse.scoring.ml import jeux_m5
from packages.analyse.scoring.ml.annoter import (
    ACTIONS,
    SERIES,
    SOURCE_NAMES,
    URGENCIES,
    AnswerStore,
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
from packages.collect.trainset.world import brief_moment

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
            text = card_text(card(items[item_id], vp, known[writer, day]))
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
        }
        np.savez(path, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> Reference:
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            models = {
                t: Softmax(*(data[f"{part}_{t}"] for part in ("W", "b", "mean", "scale")))
                for t in TARGETS
            }
        return cls(models, meta["seuil_alerte"], meta["embedder"])


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
) -> dict[str, OutOfFold]:
    """Chaque réglage appris sur l'entraînement de chaque pli, appliqué à ses jeux mesurés :
    un seul apprentissage par pli et par réglage, quel que soit le nombre de jeux mesurés."""
    k, grid = len(TARGETS[target]), configs()
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
    target: str, oof: OutOfFold, y: np.ndarray, examples: Sequence[Example]
) -> tuple[dict, np.ndarray]:
    """Le réglage retenu (G2 puis G3) sur l'ensemble des prédictions hors pli, et ses chiffres.
    Un seul réglage par découpage, comme un levier qu'on fixerait, et non un par pli."""
    truth, k = y[oof.index], len(TARGETS[target])
    preds = [p.argmax(axis=1) for p in oof.proba]
    rank, tried = select(target, preds, truth)
    pred, grid = preds[rank], configs()
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
        base.append(split(examples, personas))  # vérifie aussi que les affaires sont disjointes
    report: dict = {
        "plis_metiers": plis,
        "graine": seed,
        "selection": "G2 puis G3 : le réglage le plus simple à égalité avec le meilleur",
    }

    log("1. métiers tenus à l'écart")
    first: dict = {"exemples": len(examples), "cibles": {}}
    action_pred = np.full(len(examples), -1)
    for target, classes in TARGETS.items():
        y = _codes(examples, target)
        oof = out_of_fold(target, X, y, [Fold(tr, {"v": va}) for tr, va in base], log)["v"]
        entry, pred = judge(target, oof, y, examples)
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
    oof = out_of_fold("action", X, y, folds, log)["v"]
    entry, pred = judge("action", oof, y, examples)
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
        for measure, oof in out_of_fold("action", X, y, folds, log).items():
            entry, pred = judge("action", oof, y, examples)
            cells[f"{learn}→{measure}"] = (entry, pred, oof)
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
        texts, sources, truth = series_truth(name, series)
        X = embedder.embed(texts)
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
        "usage : comprendre --entrainer | --valider-dur | --mesurer | --tirer-demo | "
        "--reste-demo | --geler-demo"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
