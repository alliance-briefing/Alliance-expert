"""Score appris des affaires dans le brief (L2.2 ; plan du modèle, M8).

Chaque élément de l'affaire présent au brief (mail, réunion, tâche, fichier) reçoit le score
de priorité de M8 : sa carte (celle que voit l'annotateur), la compréhension N de M6
(MiniLM gelé sur le texte normalisé, fiche, régression), puis les poids Bradley-Terry appris
sur les paires (`scoring/ml/prioriser.py`, `prioriser_modele.json`). L'affaire prend le score
de son **meilleur élément** (règle du 24/11, mesurée sur les paires d'affaires de T-gen) :
un fil plein de relances sans intérêt ne monte pas, un élément urgent suffit.

Score sur 0-100 : le rang centile du score brut (somme pondérée de la fiche) parmi les cartes
des paires d'entraînement (`prioriser_echelle.json`). 70 veut dire « passe avant 70 % des
éléments d'entraînement » ; 50, l'élément médian, est le seuil de la section « actions ».

Éléments notés, comme au brief des paires d'entraînement (`jeux_m5.on_brief`) : mails et
fichiers non lus ou des 3 derniers jours, réunions des 48 h à venir, tâches en retard ou dues
dans les 24 h, jamais un élément postérieur à `now`, ni une tentative d'injection
(`guard/injection.py`, M9), ni un envoi en masse (`ml/noise.py`, comme l'ancien score des
mails). Une affaire sans élément à noter garde le score par règles.

Ce que le contrat A ne donne pas : la relation de chaque correspondant au manager. Elle vient
de la persona (`persona_for`) : configurée pour les jeux de démo, sinon déduite (adresse du
manager, domaine interne), sans hiérarchie ni clients.

Choix du score (A/B, réunion du 14/10) : variable d'environnement `ALLIANCE_SCORE`,
`appris` (défaut, règle du 24/11 : pas d'écart significativement négatif aux règles) ou
`regles`. Si un modèle manque ou échoue, le brief garde le score par règles et l'annonce dans
un warning : il ne casse jamais.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from typing import TYPE_CHECKING, Literal

from contracts.models import Item
from packages.analyse.aggregate.cases import Case
from packages.analyse.guard.injection import is_injection
from packages.analyse.scoring.ml.noise import noise_reason

if TYPE_CHECKING:
    from packages.analyse.rag.embeddings import Embedder
    from packages.analyse.scoring.ml.annoter import Viewpoint
    from packages.analyse.scoring.ml.comprendre import Reference
    from packages.analyse.scoring.ml.persona import Persona

SCORE_ENV = "ALLIANCE_SCORE"
CACHE_ENV = "FASTEMBED_CACHE_PATH"  # dossier du modèle (rempli à la construction de l'image)
RECENT = timedelta(days=3)
MAX_REASONS = 3
UNAVAILABLE = "score appris indisponible ({}) : score par règles"
# generator.llm d'un brief noté par le score appris ; l'interface affiche « par le modèle … ».
GENERATOR = "MiniLM (score de priorité appris)"

# Relation d'un correspondant, telle que la carte l'écrit et que la fiche la relit
# (`comprendre.relation`), d'après le type d'expéditeur de la persona.
RELATION_LABEL = {
    "hiérarchie": "votre N+1",
    "direction": "direction",
    "collègue": "collègue",
    "client": "client",
    "fournisseur": "partenaire",
    "automate": "envoi automatique",
}

# Raisons lisibles : les caractéristiques de la fiche de priorité qui poussent le score vers
# le haut. Une probabilité de N n'est une raison que si elle dépasse 0,5.
REASONS = {
    "N:alerte": "alerte (sécurité, incident, injection)",
    "N:decider": "décision ou validation attendue",
    "N:faire": "action demandée",
    "N:repondre": "réponse attendue",
    "N:preparer": "à préparer",
    "N:relire": "relecture demandée",
    "N:urgent": "ton urgent",
    "tache:en_retard": "tâche en retard",
    "tache:echeance_aujourdhui": "échéance aujourd'hui",
    "tache:verbe=decider": "tâche de décision",
    "reunion:aujourd_hui": "réunion aujourd'hui",
    "reunion:demain": "réunion demain",
    "reunion:pas_repondu": "invitation sans réponse",
    "reunion:organise": "réunion que vous organisez",
    "mail:expediteur=hierarchie": "message de la hiérarchie",
    "mail:expediteur=direction": "message de la direction",
    "mail:expediteur=client": "message d'un client",
    "mail:importance_haute": "importance haute",
    "mail:non_lu": "non lu",
    "fichier:cite": "fichier cité dans un échange",
}


@dataclass(frozen=True)
class CaseScore:
    """Score d'une affaire, ses raisons, et l'élément qui le porte (affiché dans l'entrée)."""

    score: int
    reasons: list[str]
    best_item_id: str
    model: str  # modèle d'embeddings (generator.llm_version du brief)
    action: str = "rien"  # action que N prévoit pour l'élément qui porte le score (M9)
    alert: bool = False  # un élément de l'affaire est une alerte pour N (probabilité ≥ 0,5)


class LearnedScorer:
    """Note les affaires avec le score de priorité de M8. Modèles chargés une fois."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        understanding: Reference | None = None,
        priority: dict | None = None,
        scale: Sequence[float] | None = None,
    ) -> None:
        from packages.analyse.scoring.ml import comprendre, prioriser

        self.understanding = understanding or comprendre.Reference.load(comprendre.CANDIDATE_PATH)
        self.priority = priority or json.loads(prioriser.MODEL_PATH.read_text(encoding="utf-8"))
        self.scale = (
            scale or json.loads(prioriser.SCALE_PATH.read_text(encoding="utf-8"))["centiles"]
        )
        if embedder is None:
            from packages.analyse.rag.embeddings import FastEmbedder

            embedder = FastEmbedder(self.understanding.embedder_name, os.environ.get(CACHE_ENV))
        self.embedder = embedder

    def score_cases(
        self, cases: Sequence[Case], items: Sequence[Item], user: str, now: datetime
    ) -> dict[str, CaseScore]:
        """{case_id: CaseScore} des affaires qui ont au moins un élément à noter.

        `items` : tous les items de l'utilisateur (contexte des cartes) ; aucun autre.
        """
        import numpy as np

        from packages.analyse.scoring.ml import comprendre, prioriser
        from packages.analyse.scoring.ml.annoter import card

        mine = [item for item in items if item.owner_user_id == user]
        known = [item for item in mine if item.created_at <= now]
        vp = viewpoint_for(persona_for(user, mine), mine, now)
        to_score = {
            case.case_id: [
                i
                for i in case.items
                if i.owner_user_id == user
                and on_brief(i, now)
                and not is_injection(i)
                and noise_reason(i) is None
            ]
            for case in cases
        }
        elements = sorted(
            {i.item_id: i for its in to_score.values() for i in its}.values(),
            key=lambda i: i.item_id,
        )
        if not elements:
            return {}
        cards = [card(i, vp, known) for i in elements]
        u = self.understanding
        X = comprendre.design(cards, self.embedder, u.fiche, None, u.texte)
        signals = np.array([prioriser.card_signals(c) for c in cards])
        F = np.hstack([signals, prioriser.n_columns(u.models, X)])
        raw = prioriser.model_score(self.priority, F)
        on_100 = prioriser.to_100(raw, self.scale)
        actions = comprendre.TARGETS["action"]
        predicted = u.models["action"].proba(X).argmax(axis=1)
        alerts = u.models["alerte"].proba(X)[:, comprendre.TARGETS["alerte"].index("oui")] >= 0.5
        row = {item.item_id: n for n, item in enumerate(elements)}

        out = {}
        for case in cases:
            if not to_score[case.case_id]:
                continue
            # Meilleur élément ; à score égal, le plus petit item_id (déterministe).
            best = min(to_score[case.case_id], key=lambda i: (-raw[row[i.item_id]], i.item_id))
            n = row[best.item_id]
            out[case.case_id] = CaseScore(
                round(float(on_100[n])),
                self.reasons(F[n]),
                best.item_id,
                self.embedder.name,
                actions[int(predicted[n])],
                any(bool(alerts[row[i.item_id]]) for i in to_score[case.case_id]),
            )
        return out

    def reasons(self, fiche: Sequence[float]) -> list[str]:
        """Les plus fortes contributions positives de la fiche, en clair (Bradley-Terry)."""
        from packages.analyse.scoring.ml.prioriser import PRIORITY_NAMES

        if self.priority.get("modele") != "bradley-terry":
            return []
        contributions = sorted(
            (
                (w * v, name)
                for name, w, v in zip(PRIORITY_NAMES, self.priority["poids"], fiche, strict=True)
                if name in REASONS and w * v > 0 and (not name.startswith("N:") or v >= 0.5)
            ),
            key=lambda x: (-x[0], x[1]),
        )
        return [REASONS[name] for _, name in contributions][:MAX_REASONS]


def on_brief(item: Item, now: datetime) -> bool:
    """L'élément figure-t-il au brief de `now` ? Les fenêtres des paires d'entraînement :
    mails et fichiers non lus ou récents, réunions des 48 h, tâches en retard ou dues
    dans les 24 h et non terminées ; jamais un élément pas encore visible."""
    if item.created_at > now:
        return False
    if item.source == "calendar":
        return item.start_at is not None and now <= item.start_at < now + timedelta(days=2)
    if item.source == "task":
        return (
            item.due_at is not None and item.due_at < now + timedelta(days=1) and not item.completed
        )
    stamp = item.updated_at if item.source == "file" else item.created_at
    return item.is_read is False or now - stamp <= RECENT


def viewpoint_for(persona: Persona, items: Sequence[Item], now: datetime) -> Viewpoint:
    """Le point de vue du manager pour les cartes : son adresse, et la relation de chaque
    correspondant selon la persona (ce qu'un annuaire donnerait)."""
    from packages.analyse.scoring.ml.annoter import Viewpoint

    people = {
        p.email: RELATION_LABEL[persona.sender_kind(p.email)]
        for item in items
        for p in item.participants
        if p.email.lower() != persona.address
    }
    return Viewpoint(persona.address, now, people, "")


def persona_for(user: str, items: Sequence[Item]) -> Persona:
    """Qui est qui pour `user`. Le contrat A ne donne ni l'adresse du manager ni son
    organigramme : configuration connue pour les jeux de démo, sinon l'adresse déduite
    (destinataire principal le plus fréquent), sans hiérarchie ni clients."""
    from packages.analyse.scoring.ml.persona import DEMO_CAMILLE, GRAPH_HELENE, Persona

    known = {"u_demo_001": DEMO_CAMILLE, "u_graph_helene": GRAPH_HELENE}
    if user in known:
        return known[user]
    counts = Counter(
        p.email.lower()
        for item in items
        if item.source == "mail"
        for p in item.participants
        if p.role == "to"
    )
    address = min(counts, key=lambda e: (-counts[e], e)) if counts else ""
    empty: frozenset[str] = frozenset()
    return Persona(address, address.partition("@")[2], empty, empty, empty)


@cache
def _default_scorer() -> LearnedScorer:
    return LearnedScorer()


def learned_case_scores(
    cases: Sequence[Case],
    items: Sequence[Item],
    user: str,
    now: datetime,
    scorer: LearnedScorer | Literal["auto"] | None = "auto",
) -> tuple[dict[str, CaseScore] | None, str | None]:
    """(scores appris, warning éventuel). `scorer` : "auto" (selon `ALLIANCE_SCORE`),
    None (score par règles seul) ou un scorer donné (tests). Les scores valent None quand le
    score appris n'est pas utilisé (règles demandées, ou repli sur erreur) et {} quand il
    l'est mais que la journée n'a aucun élément à noter : le brief appris reste alors vide
    plutôt que de retomber sur les règles (M9)."""
    if scorer == "auto":
        if os.environ.get(SCORE_ENV, "appris") == "regles":
            return None, None
        try:
            scorer = _default_scorer()
        except Exception as error:  # noqa: BLE001 — le brief ne casse jamais : repli annoncé
            return None, UNAVAILABLE.format(type(error).__name__)
    if scorer is None:
        return None, None
    try:
        return scorer.score_cases(cases, items, user, now), None
    except Exception as error:  # noqa: BLE001 — modèle absent, téléchargement impossible…
        return None, UNAVAILABLE.format(type(error).__name__)


def download_model() -> None:
    """Télécharge le modèle d'embeddings (construction de l'image Docker)."""
    LearnedScorer().embedder.embed(["préchargement"])
