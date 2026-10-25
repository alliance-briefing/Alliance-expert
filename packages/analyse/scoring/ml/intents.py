"""Intentions d'un mail, caractéristiques du score appris : apprises ou par règle.

Apprises sur les étiquettes de Théo (181 mails, un par modèle de phrase) : `action` (le mail
demande à Hélène de faire quelque chose), `reponse` (attend une réponse), `incident`
(signale un incident ou un blocage). Modèle : embeddings MiniLM multilingue (`--e5` :
`multilingual-e5-large`, 10 fois plus lourd, même score final) +
régression logistique, avec en option un indice par mots-clés (« ? » pour `reponse`),
retenu seulement si la validation croisée le justifie.
Par règle, car le concept est précis : `alerte` (sécurité, banque) et l'échéance
(`deadline.py`). Le NLI zéro-coup a été écarté : F1 0,30 à 0,40, mesuré le 25/10/2026.

`uv run --group embeddings python -m packages.analyse.scoring.ml.intents [--e5]` : mesure en
validation croisée imbriquée, puis entraîne sur tout l'échantillon et écrit le modèle.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from packages.analyse.rag.embeddings import Embedder, FastEmbedder
from packages.analyse.scoring.ml.deadline import find_deadline
from packages.analyse.scoring.ml.sample import ANNOTATIONS, load_sample

E5_MODEL = "intfloat/multilingual-e5-large"
# Modèle par défaut depuis le 26/10/2026 : MiniLM (0,22 Go, celui du RAG) donne le même accord
# du score final sur les paires réservées qu'e5-large (2,2 Go) : 0,86 dans les deux cas.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LEARNED = ("action", "reponse", "incident")
MODEL_PATH = Path(__file__).with_name("intents_model.npz")
LABELS_PATH = ANNOTATIONS / "intent_labels.jsonl"
CACHE_DIR = Path.home() / ".cache" / "alliance-briefing"  # hors du dépôt et de OneDrive
TEXT_MAX_CHARS = 1200
L2_CHOICES = (0.01, 0.1)
# Questions scindées le 25/10 : les mails non revus héritent du « non » de la question mère
INHERIT = {"incident": "probleme", "alerte": "probleme"}


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def intent_text(title: str, body: str) -> str:
    """Le texte lu par le modèle : objet puis corps, tronqué comme dans l'échantillon."""
    return f"{title}\n{body[:TEXT_MAX_CHARS]}"


# Indices par mots-clés (texte sans accents) ; utilisés seulement s'ils améliorent la mesure.
HINTS = {
    "action": re.compile(
        r"merci de|pouvez-vous|pourriez-vous|peux-tu|pourrais-tu|veuillez|a valider|"
        r"a relire|a signer|merci d'|n'oubliez pas|avant le|traiter"
    ),
    "reponse": re.compile(
        r"\?|merci de (me )?confirmer|votre avis|ton avis|disponible|disponibilite"
    ),
    "incident": re.compile(
        r"incident|probleme|erreur|panne|retard|refus|bloquant|hors service|lent|manque"
    ),
}
_ALERT = re.compile(
    r"inhabituel|nouvel appareil|connexion|mot de passe expire dans|fraude|suspect|compromis"
)


def is_alert(text: str, sender_kind: str) -> bool:
    """Alerte de sécurité ou financière : un automate qui signale un événement anormal.
    Un ticket « Mot de passe expiré » n'en est pas une (c'est une demande traitée)."""
    return sender_kind == "automate" and bool(_ALERT.search(_normalize(text)))


# --- embeddings ------------------------------------------------------------------------------


class PrefixedEmbedder:
    """e5 attend « query: » devant chaque texte ; le préfixe fait partie du nom (du cache)."""

    def __init__(self, inner: Embedder, prefix: str) -> None:
        self.inner, self.prefix = inner, prefix
        self.name = f"{inner.name}|{prefix}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.inner.embed([self.prefix + text for text in texts])


def e5_embedder() -> PrefixedEmbedder:
    return PrefixedEmbedder(FastEmbedder(E5_MODEL), "query: ")


def embedder_for(name: str) -> Embedder:
    """L'embedder d'un modèle d'intentions enregistré (son nom est dans le fichier)."""
    return e5_embedder() if name == E5_MODEL else FastEmbedder(name)


def cache_path(embedder: Embedder) -> Path:
    """Un fichier de cache par modèle : leurs vecteurs n'ont pas la même dimension."""
    return CACHE_DIR / f"vectors_{re.sub(r'[^A-Za-z0-9]+', '_', embedder.name).strip('_')}.npz"


class VectorCache:
    """Vecteurs par (modèle, empreinte du texte), enregistrés dans un fichier .npz.

    La clé est le texte et non l'item : deux mails au texte identique ne sont calculés
    qu'une fois. Le fichier reste sur la machine, hors du dépôt ; un fichier par modèle
    (`cache_path`), sinon les dimensions se mélangent.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.vectors: dict[str, np.ndarray] = {}
        if path is not None and path.exists():
            with np.load(path) as data:
                self.vectors = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))

    @staticmethod
    def key(embedder: Embedder, text: str) -> str:
        return hashlib.sha256(f"{embedder.name}\n{text}".encode()).hexdigest()

    def embed(self, embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
        keys = [self.key(embedder, text) for text in texts]
        missing = sorted(
            {k: t for k, t in zip(keys, texts, strict=True) if k not in self.vectors}.items()
        )
        if missing:
            for (k, _), vector in zip(
                missing, embedder.embed([t for _, t in missing]), strict=True
            ):
                self.vectors[k] = vector
            if len({v.shape for v in self.vectors.values()}) > 1:
                raise ValueError("un cache ne doit servir qu'à un seul modèle (cache_path)")
            self.save()
        return np.array([self.vectors[k] for k in keys], dtype=np.float32)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.vectors)
        np.savez(self.path, keys=np.array(keys), vectors=np.array([self.vectors[k] for k in keys]))


# --- étiquettes -------------------------------------------------------------------------------


def load_labels(feature: str, path: Path = LABELS_PATH) -> dict[str, bool]:
    """{item_id: oui ?} : dernière réponse hors répétitions, « ? » écartés ; pour une
    question scindée, les mails non revus héritent du « non » de la question mère."""
    last: dict[tuple[str, str], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            if not row["repeat"]:
                last[(row["item_id"], row["feature"])] = row["value"]
    labels: dict[str, bool] = {}
    parent = INHERIT.get(feature)
    if parent:
        labels = {i: False for (i, f), v in last.items() if f == parent and v == "non"}
    for (item_id, f), value in last.items():
        if f == feature:
            labels.pop(item_id, None)
            if value in ("oui", "non"):
                labels[item_id] = value == "oui"
    return labels | synth_labels(feature, path.parent)


ANNOTATOR_FILES = ("intent_labels_theo.jsonl", "intent_labels_paul.jsonl")
ARBITRATION_FILE = "intent_labels_arbitrage.jsonl"


def _first_answers(path: Path) -> dict[tuple[str, str], str]:
    last: dict[tuple[str, str], str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if not row["repeat"]:
                    last[(row["item_id"], row["feature"])] = row["value"]
    return last


def synth_labels(feature: str, folder: Path) -> dict[str, bool]:
    """Étiquettes des mails générés, annotés à deux (fichiers de `folder`) : la réponse
    commune quand les deux ont répondu pareil, l'arbitrage quand ils divergent, la réponse
    de l'unique annotateur sinon ; « ? » écartés."""
    answers = [_first_answers(folder / name) for name in ANNOTATOR_FILES]
    arbitration = _first_answers(folder / ARBITRATION_FILE)
    out: dict[str, bool] = {}
    keys = {k for a in answers for k in a if k[1] == feature}
    for key in sorted(keys):
        given = {a[key] for a in answers if key in a and a[key] in ("oui", "non")}
        value = arbitration.get(key) if len(given) > 1 else next(iter(given), None)
        if value in ("oui", "non"):
            out[key[0]] = value == "oui"
    return out


# --- régression logistique et validation croisée -----------------------------------------------


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float, steps: int = 1500) -> np.ndarray:
    """Régression logistique L2 (dernier poids : biais), classes pondérées car les « oui »
    sont rares. Descente de gradient à pas fixe depuis zéro : entièrement déterministe."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    pos = float(y.mean())
    weight = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ w))
        w -= 0.5 * (Xb.T @ ((p - y) * weight) / len(y) + l2 * np.r_[w[:-1], 0] / len(y))
    return w


def proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.hstack([X, np.ones((len(X), 1))]) @ w))


def design(E: np.ndarray, texts: Sequence[str], feature: str, use_hint: bool) -> np.ndarray:
    if not use_hint:
        return E
    hint = HINTS[feature]
    return np.hstack([E, [[float(bool(hint.search(_normalize(t))))] for t in texts]])


def folds(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Numéro de pli de chaque exemple, stratifié (autant de « oui » dans chaque pli)."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        fold[idx] = np.arange(len(idx)) % k
    return fold


def prf(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    tp = float(((pred == 1) & (truth == 1)).sum())
    p = tp / max(float(pred.sum()), 1.0)
    r = tp / max(float(truth.sum()), 1.0)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


@dataclass(frozen=True)
class Config:
    use_hint: bool
    l2: float


def select(E: np.ndarray, texts: Sequence[str], y: np.ndarray, feature: str) -> Config:
    """La configuration au meilleur F1 en validation croisée à 4 plis ; à égalité, la plus
    simple (sans indice, plus régularisée)."""
    best: tuple[float, int, Config] | None = None
    for rank, config in enumerate(
        Config(hint, l2) for hint in (False, True) for l2 in sorted(L2_CHOICES, reverse=True)
    ):
        X = design(E, texts, feature, config.use_hint)
        fold, pred = folds(y, 4, seed=100), np.empty(len(y))
        for k in range(4):
            test = fold == k
            pred[test] = proba(fit_logistic(X[~test], y[~test], config.l2), X[test]) > 0.5
        f1 = prf(pred, y)[2]
        if best is None or f1 > best[0] + 1e-9:
            best = (f1, rank, config)
    assert best is not None
    return best[2]


def nested_cv(
    E: np.ndarray, texts: Sequence[str], y: np.ndarray, feature: str, repeats: int = 5
) -> np.ndarray:
    """(précision, rappel, F1) de chaque répétition : 5 plis externes, et le choix de la
    configuration refait à l'intérieur de chaque pli d'entraînement (pas de fuite)."""
    texts = list(texts)
    scores = []
    for seed in range(repeats):
        fold, pred = folds(y, 5, seed), np.empty(len(y))
        for k in range(5):
            train, test = fold != k, fold == k
            tr_texts = [t for t, keep in zip(texts, train, strict=True) if keep]
            config = select(E[train], tr_texts, y[train], feature)
            X = design(E, texts, feature, config.use_hint)
            pred[test] = proba(fit_logistic(X[train], y[train], config.l2), X[test]) > 0.5
        scores.append(prf(pred, y))
    return np.array(scores)


# --- modèle enregistré ------------------------------------------------------------------------


@dataclass
class IntentModel:
    """Une régression logistique par intention apprise, sur les embeddings e5."""

    weights: dict[str, np.ndarray]
    hints: dict[str, bool]
    embedder_name: str

    def predict(self, embeddings: np.ndarray, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Probabilité de chaque intention apprise, par texte (dans l'ordre de `texts`)."""
        return {
            f: proba(w, design(embeddings, texts, f, self.hints[f]))
            for f, w in self.weights.items()
        }

    def save(self, path: Path = MODEL_PATH) -> None:
        arrays = {f"w_{f}": w for f, w in self.weights.items()}
        meta = json.dumps({"hints": self.hints, "embedder": self.embedder_name})
        np.savez(path, meta=np.array(meta), **arrays)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> IntentModel:
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            weights = {k[2:]: data[k] for k in data.files if k.startswith("w_")}
        return cls(weights, meta["hints"], meta["embedder"])


def train(
    sample: list[dict],
    embed: Callable[[list[str]], np.ndarray],
    labels_path: Path = LABELS_PATH,
    repeats: int = 5,
    embedder_name: str = DEFAULT_MODEL,
) -> tuple[IntentModel, list[str]]:
    """Mesure (validation imbriquée) puis entraînement sur tout l'échantillon.

    `embed` : textes → vecteurs (e5 en vrai, un faux modèle dans les tests). Renvoie le
    modèle et les lignes du rapport.
    """
    by_id = {row["item_id"]: row for row in sample}
    texts_by_id = {i: intent_text(r["title"], r["text"]) for i, r in by_id.items()}
    ids = sorted(by_id)
    E_all = embed([texts_by_id[i] for i in ids])
    row = {item_id: n for n, item_id in enumerate(ids)}
    report = ["caractéristique   méthode                   précision  rappel   F1"]
    weights, hints = {}, {}
    for feature in LEARNED:
        labels = load_labels(feature, labels_path)
        keep = [i for i in ids if i in labels]
        y = np.array([labels[i] for i in keep], dtype=float)
        E, texts = E_all[[row[i] for i in keep]], [texts_by_id[i] for i in keep]
        p, r, f = nested_cv(E, texts, y, feature, repeats).mean(axis=0)
        config = select(E, texts, y, feature)
        method = "embeddings + LR" + (" + indice" if config.use_hint else "")
        report.append(f"{feature:<17} {method:<25} {p:>9.2f} {r:>7.2f} {f:>5.2f}")
        weights[feature] = fit_logistic(design(E, texts, feature, config.use_hint), y, config.l2)
        hints[feature] = config.use_hint
    for feature, predict in (
        (
            "echeance",
            lambda r: (
                find_deadline(
                    intent_text(r["title"], r["text"]), datetime.fromisoformat(r["received_at"])
                )
                is not None
            ),
        ),
        ("alerte", lambda r: is_alert(intent_text(r["title"], r["text"]), r["sender_kind"])),
    ):
        labels = load_labels(feature, labels_path)
        keep = [i for i in ids if i in labels]
        pred = np.array([predict(by_id[i]) for i in keep], dtype=float)
        p, r, f = prf(pred, np.array([labels[i] for i in keep], dtype=float))
        report.append(f"{feature:<17} {'règle':<25} {p:>9.2f} {r:>7.2f} {f:>5.2f}")
    return IntentModel(weights, hints, embedder_name), report


def main(argv: list[str]) -> int:
    name = E5_MODEL if "--e5" in argv else DEFAULT_MODEL
    embedder = embedder_for(name)
    cache = VectorCache(cache_path(embedder))
    from packages.analyse.scoring.ml.annotate import INTENT_R2_PATH, SYNTH_PATH

    extra = [path for path in (INTENT_R2_PATH, SYNTH_PATH) if path.exists()]
    sample = load_sample() + [row for path in extra for row in load_sample(path)]
    model, report = train(sample, lambda texts: cache.embed(embedder, texts), embedder_name=name)
    model.save()
    print("\n".join(report))
    print(f"(apprises : validation croisée imbriquée 5 plis x 5 ; modèle → {MODEL_PATH.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
